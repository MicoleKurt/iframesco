"""Orchestrates topic -> research -> script -> voice -> media -> render -> package."""
import json
import os
import re
import time
from typing import Callable

from . import formats, music, relevance, render, research, script, storage, thumbnail, voice, youtube

Progress = Callable[[str, float], None]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "video"


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def chapter_list(plan: dict, starts: list[float]) -> list[tuple[float, str]]:
    """(start, title) per chapter; YouTube needs the first at 0:00, >= 3 chapters, each >= 10 s."""
    out = []
    for s, t in zip(plan["scenes"], starts):
        if s.get("chapter"):
            out.append((t, s["chapter"]))
    if out and out[0][0] > 0.5:
        out.insert(0, (0.0, plan.get("intro_title", "Intro")))
    merged = []
    for t, name in out:
        if merged and t - merged[-1][0] < 10:
            continue
        merged.append((t, name))
    return merged if len(merged) >= 3 else []


def _candidates(scene: dict, subject: str, fmt, use_youtube: bool, work: str, idx: int,
                credits: list) -> list[dict]:
    anchor = scene.get("anchor", True)
    queries = [k if not anchor or subject.lower() in k.lower() else f"{subject} {k}"
               for k in scene.get("keywords") or [subject]]
    cands = []
    if use_youtube:
        try:
            clip = youtube.cc_clip(queries[0], f"{work}/s{idx}_yt", orientation=fmt.orientation)
        except Exception as e:  # network / extractor breakage must not kill the render
            print(f"[youtube] search failed: {e}")
            clip = None
        if clip:
            credits.append({k: v for k, v in clip.items() if k != "path"})
            cands.append({"file": clip["path"], "kind": "video", "source": "YouTube (CC BY)", "preview": ""})
    wiki_only = scene.get("images") == "wiki"
    for raw, q in dict.fromkeys(zip(scene.get("keywords") or [subject], queries)):
        if raw.lower().startswith("local:"):  # a file on disk (e.g. a clip cut with clipsco), used as-is
            path = os.path.expanduser(raw[6:].strip())
            kind = "video" if path.lower().endswith((".mp4", ".mov", ".webm", ".mkv")) else "image"
            cands.append({"file": path, "kind": kind, "source": "local", "preview": ""})
        else:
            try:
                if raw.lower().startswith("file:"):  # an exact file, chosen by a person: used as-is, in order
                    cands += research.file_image(raw[5:].strip())
                elif raw.lower().startswith("wiki:"):  # photos from that Wikipedia article: hand-picked, on topic
                    cands += research.wiki_images(raw[5:].strip())
                elif not wiki_only:
                    cands += research.stock_videos(q, 2, fmt.orientation)
                    cands += research.commons_images(q, 6)
            except Exception as e:  # a rate-limited or failing lookup costs one keyword, not the whole render
                print(f"[media] skip keyword {raw!r}: {e}")
    exact = all(k.lower().startswith(("file:", "local:")) for k in scene.get("keywords") or ["-"])
    if len(cands) < 3 and not exact:  # hand-picked files: never pad with other pictures
        try:
            cands += research.wiki_images(subject) or research.commons_images(subject, 6)
        except Exception as e:
            print(f"[media] skip subject fallback: {e}")
    seen, out = set(), []
    for c in cands:
        key = c.get("file") or c["url"]
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _gather(scene: dict, idx: int, work: str, subject: str, fmt, ranker: relevance.Ranker,
            shots: int, use_youtube: bool, credits: list) -> list[dict]:
    cands = _candidates(scene, subject, fmt, use_youtube, work, idx, credits)
    text = scene["narration"] + " " + " ".join(scene.get("keywords", []))
    chosen: list[dict] = []

    def take(pool: list[dict]) -> None:
        for c in pool:
            if len(chosen) == shots:
                return
            try:
                if "file" in c:
                    path = c["file"]
                else:
                    ext = "mp4" if c["kind"] == "video" else "jpg"
                    path = research.download(c["url"], f"{work}/s{idx}_{len(chosen)}.{ext}")
                chosen.append({**c, "path": path})
                if c.get("source") == "Wikimedia Commons" and c.get("license") not in ("Public domain", "CC0"):
                    credits.append({"title": c.get("title", ""), "channel": c.get("credit", ""),
                                    "url": c.get("page", ""), "license": c.get("license", "")})
            except Exception as e:
                print(f"[media] skip {c.get('url')}: {e}")

    exact = all(k.lower().startswith(("file:", "local:")) for k in scene.get("keywords") or ["-"])
    take(cands if exact else ranker.rank(text, cands, keep=shots + 2))
    if not chosen:  # all duplicates of earlier shots or failed: take search order, not a blank card
        take(cands[:shots + 2])
    return chosen


def generate(topic: str, out_dir: str = "output", fmt: str = "short", scenes: int = 6, minutes: int = 10,
             use_youtube: bool = False, script_file: str | None = None, brand: dict | None = None,
             voice_preset: str | None = None,
             progress: Progress = lambda msg, pct: print(f"[{pct:5.1f}%] {msg}")) -> str:
    t0 = time.time()
    F = formats.get(fmt)
    render.use(F)
    work = os.path.abspath(f"{out_dir}/{slug(topic)}-{int(t0)}")
    os.makedirs(work, exist_ok=True)

    if script_file:  # human-written script: skip research + LLM, keep everything else automated
        with open(script_file, encoding="utf-8") as f:
            plan = json.load(f)
        if plan.get("format"):
            F = formats.get(plan["format"])
            render.use(F)
        progress("Using provided script", 12)
    else:
        progress("Researching topic on Wikipedia", 4)
        facts = research.wiki_facts(topic, full=True)  # Shorts pick early-life sections from the full article
        progress(f"Writing script with {script.model()}", 8)
        if F.name == "long":
            steps = [0]

            def step(msg: str) -> None:  # outline, cold open, N chapters, visuals: 10% -> 24%
                steps[0] += 1
                progress(msg, min(24, 9 + 2 * steps[0]))
            plan = script.write_long(topic, facts, minutes, progress=step)
        else:
            plan = script.write_short(topic, facts, scenes)
    plan["format"] = F.name
    _write_json(f"{work}/script.json", plan)

    # ------------------------------------------------ voice
    progress("Recording voiceover", 25)
    preset = voice_preset or plan.get("voice")
    voice_files, words, offset, durations, starts = [], [], 0.0, [], []
    base_rate = voice.resolve(preset)[2]
    n = len(plan["scenes"])
    for i, s in enumerate(plan["scenes"]):
        progress(f"Recording voiceover {i + 1}/{n}", 25 + 15 * i / n)
        rate = s.get("rate")
        if rate is None and i == 0 and F.name == "short":
            rate = base_rate + 6  # Shorts: the hook lands a touch faster
        raw, timings = voice.synthesize(s["narration"], f"{work}/raw{i}", preset=preset,
                                        rate=int(str(rate).rstrip("%")) if rate is not None else None,
                                        pronounce=plan.get("pronounce"))
        gap = s.get("pause", F.scene_gap + (0.4 if i + 1 < n and plan["scenes"][i + 1].get("chapter") else 0))
        padded = f"v{i}.wav"
        render.ff("-i", raw, "-af", f"apad=pad_dur={gap}", "-ar", "44100", "-ac", "2", padded, cwd=work)
        d = render.duration(f"{work}/{padded}")
        words += [{**w, "start": w["start"] + offset, "end": w["end"] + offset} for w in timings]
        starts.append(offset)
        voice_files.append(padded)
        durations.append(d)
        offset += d
    chapters = chapter_list(plan, starts) if F.name == "long" else []
    _write_json(f"{work}/words.json", words)  # lets a render be re-captioned/re-mixed without re-voicing
    render.captions_ass(words, f"{work}/captions.ass", plan["title"], brand=brand, chapters=chapters,
                       hook=plan.get("hook_text"))

    # ------------------------------------------------ visuals
    progress("Finding footage & images", 42)
    subject = plan.get("subject") or script.subject(topic, plan["title"])
    ranker, credits, media = relevance.Ranker(), [], []
    for i, (sc, d) in enumerate(zip(plan["scenes"], durations)):
        progress(f"Finding footage {i + 1}/{n}", 42 + 18 * i / n)
        shots = F.shots_per_scene if F.name == "short" else max(1, min(3, round(d / 9)))
        if F.name == "short" and i == 0:
            shots += 1  # faster first cut: a new picture by ~1.5 s holds the scroller
        media.append(_gather(sc, i, work, subject, F, ranker, shots, use_youtube, credits))
    plan["credits"] = credits

    # ------------------------------------------------ shots
    clips, slots, shot = [], [], 0
    total_shots = sum(max(1, len(m)) for m in media)
    for i, (files, d) in enumerate(zip(media, durations)):
        progress(f"Animating scene {i + 1}/{n}", 60 + 25 * i / n)
        files = files or [None]
        part = d / len(files)
        for f in files:
            last = shot == total_shots - 1
            length = part + (0 if last else F.crossfade)
            out = f"c{shot:03d}.mp4"
            path = f"{work}/{out}"
            try:
                if f is None:
                    render.title_card(path, length)
                elif f["path"].endswith(".mp4"):
                    render.video_clip(f["path"], path, length)
                else:
                    render.still_clip(f["path"], path, length, shot)
            except RuntimeError as e:  # corrupt/odd media -> gradient card instead of failing the run
                print(f"[render] {e}")
                render.title_card(path, length)
            clips.append(out)
            slots.append(part)
            shot += 1
    progress("Crossfading shots", 86)
    render.join(work, clips, slots, "video.mp4", overlap=F.crossfade)

    # ------------------------------------------------ mix + package
    progress("Mixing music, captions & final encode", 90)
    mood = plan.get("mood", "mysterious" if F.name == "long" else "calm")
    track = music.pick(mood, seed=plan["title"])
    final = os.path.abspath(f"{out_dir}/{slug(plan['title'])}-{F.name}-{int(t0)}.mp4")
    render.assemble(work, "video.mp4", voice_files, final, music_file=track["path"] if track else None, mood=mood)
    render.srt(words, final[:-4] + ".srt")
    if track and track.get("credit"):
        credits.append({"title": track["file"], "channel": track["credit"], "url": "", "license": track.get("license", "")})
    plan["chapters"] = [{"start": round(t, 2), "title": c} for t, c in chapters]
    _write_json(final[:-4] + ".json", plan)
    _package(plan, media, final, brand, credits, chapters, F)
    storage.mirror(final)
    progress(f"Done in {time.time() - t0:.0f}s", 100)
    return final


def _package(plan: dict, media: list[list[dict]], final: str, brand: dict | None, credits: list[dict],
             chapters: list[tuple[float, str]], F) -> None:
    """Thumbnail(s) + upload metadata next to the mp4, so publishing needs no extra editing."""
    stills = [f["path"] for files in media for f in files if not f["path"].endswith(".mp4")]
    thumb = plan.get("thumbnail")
    if thumb and stills:
        idx = thumb.get("scene", 0)
        own = [f["path"] for f in (media[idx] if idx < len(media) else []) if not f["path"].endswith(".mp4")]
        img = (own or stills)[0]
        kicker = brand["name"] if brand else ""
        accent = (brand or {}).get("accent")
        thumbnail.make(img, thumb["hook"], final[:-4] + ".thumb.jpg", kicker, thumb.get("label", ""), accent=accent)
        if F.vertical:
            thumbnail.make_vertical(img, thumb["hook"], final[:-4] + ".cover.jpg", kicker, thumb.get("label", ""),
                                    accent=accent)
    meta = dict(plan.get("youtube") or {"title": plan["title"], "description": "", "tags": []})
    desc = meta.get("description", "")
    if chapters:
        desc += "\n\nChapters:\n" + "\n".join(f"{clock(t)} {name}" for t, name in chapters)
    if credits:
        desc += "\n\nCredits:\n" + "\n".join(
            f"{c['title']} by {c['channel']} {c['url']} ({c['license']})".replace("  ", " ") for c in credits)
    _write_json(final[:-4] + ".youtube.json", {**meta, "description": desc.strip()})
    if credits:  # CC BY requires attribution
        with open(final[:-4] + ".credits.txt", "w", encoding="utf-8") as f:
            f.writelines(f'"{c["title"]}" by {c["channel"]}, {c["url"]} ({c["license"]})\n' for c in credits)
