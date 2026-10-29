"""ffmpeg rendering: Ken Burns stills, stock clips, crossfades, captions/chapter cards, final mix.

Call `use(fmt)` before rendering; one render runs at a time (the job queue guarantees it).
"""
import functools
import json
import os
import subprocess

from .formats import SHORT, Format

FPS = 30
FMT: Format = SHORT
W, H = FMT.w, FMT.h
GAP = FMT.scene_gap


def use(fmt: Format) -> None:
    global FMT, W, H, GAP
    FMT, W, H, GAP = fmt, fmt.w, fmt.h, fmt.scene_gap


def ff(*args: str, cwd: str | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()[-800:]}")


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


@functools.cache
def encoder() -> str:
    """h264_nvenc when an NVIDIA GPU can actually encode (3-5x faster), else libx264."""
    forced = os.getenv("FS_ENCODER")
    if forced:
        return forced
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True)
    return "h264_nvenc" if probe.returncode == 0 else "libx264"


def enc() -> list[str]:
    if encoder() == "h264_nvenc":
        v = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    else:
        v = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    return [*v, "-pix_fmt", "yuv420p", "-r", str(FPS)]


# ---------------------------------------------------------------- shots

def _too_wide(img: str) -> bool:
    """True when cropping the photo to the canvas would cut away more than ~45% of it."""
    try:
        from PIL import Image
        with Image.open(img) as im:
            w, h = im.size
    except Exception:
        return False
    return (w / h) / (W / H) > 1.8


def _too_tall(img: str) -> bool:
    """True when the photo is so much narrower than the canvas that cropping to it would cut off its top and bottom
    (a portrait on a landscape canvas loses the eyes)."""
    try:
        from PIL import Image
        with Image.open(img) as im:
            w, h = im.size
    except Exception:
        return False
    return (w / h) / (W / H) < 0.5


def still_clip(img: str, out: str, dur: float, variant: int) -> None:
    """Slow zoom/pan over a still image (Ken Burns), alternating motion per shot."""
    frames = max(1, round(dur * FPS))
    zoom = f"1+0.12*on/{frames}" if variant % 2 == 0 else f"1.12-0.12*on/{frames}"
    pan_x = ["iw/2-(iw/zoom/2)", "(iw-iw/zoom)*on/{f}", "(iw-iw/zoom)*(1-on/{f})"][variant % 3]
    bw, bh = W * 3 // 2, H * 3 // 2  # oversample so the zoom stays sharp
    frame = f"scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
    if _too_wide(img):  # a wide photo on a tall canvas: show all of it over a blurred copy instead of cropping heads off
        zoom = f"1+0.06*on/{frames}" if variant % 2 == 0 else f"1.06-0.06*on/{frames}"
        frame = (f"split[a][b];[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                 f"boxblur=40:3,eq=brightness=-0.18[bg];[b]scale={bw}:-2[fg];[bg][fg]overlay=0:(H-h)/2,")
    elif _too_tall(img):  # a portrait on a landscape canvas: whole picture over a blurred copy, never cropped
        zoom = f"1+0.06*on/{frames}" if variant % 2 == 0 else f"1.06-0.06*on/{frames}"
        frame = (f"split[a][b];[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                 f"boxblur=40:3,eq=brightness=-0.18[bg];[b]scale=-2:{bh}[fg];[bg][fg]overlay=(W-w)/2:0,")
    vf = (frame +
          f"zoompan=z='{zoom}':x='{pan_x.format(f=frames)}':y='ih/2-(ih/zoom/2)'"
          f":d={frames}:s={W}x{H}:fps={FPS},setsar=1")
    ff("-i", img, "-vf", vf, "-frames:v", str(frames), *enc(), "-an", out)


def video_clip(src: str, out: str, dur: float) -> None:
    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS}"
    ff("-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}", "-vf", vf, *enc(), "-an", out)


def title_card(out: str, dur: float) -> None:
    """Fallback visual when no media was found: slow-drifting dark gradient."""
    ff("-f", "lavfi", "-i", f"gradients=s={W}x{H}:c0=0x1a1612:c1=0x3a2e22:x0=0:y0=0:x1={W}:y1={H}"
       f":speed=0.008:d={dur:.3f}:r={FPS}", *enc(), out)


# ---------------------------------------------------------------- crossfades

def xfade_offsets(durs: list[float]) -> list[float]:
    """Offset of each transition when every clip but the last is rendered `overlap` seconds long
    than its slot: transition k starts at sum(durs[:k]). Total length stays sum(durs)."""
    out, t = [], 0.0
    for d in durs[:-1]:
        t += d
        out.append(t)
    return out


def _xfade_group(work: str, clips: list[str], durs: list[float], o: float, out: str) -> None:
    inputs = [a for c in clips for a in ("-i", c)]
    # normalise timebase/fps/pixel format: stock footage and stills must match for xfade
    chain = [f"[{i}:v]settb=AVTB,fps={FPS},format=yuv420p,setsar=1[n{i}]" for i in range(len(clips))]
    prev = "n0"
    for k, off in enumerate(xfade_offsets(durs), start=1):
        tag = f"x{k}"
        chain.append(f"[{prev}][n{k}]xfade=transition=fade:duration={o:.3f}:offset={off:.3f}[{tag}]")
        prev = tag
    ff(*inputs, "-filter_complex", ";".join(chain), "-map", f"[{prev}]", *enc(), out, cwd=work)


def join(work: str, clips: list[str], durs: list[float], out: str, overlap: float = 0.0,
         group: int = 12, level: int = 0) -> None:
    """Concatenate shots. With `overlap`, consecutive shots crossfade; each clip except the last
    must be rendered `overlap` seconds longer than its slot in `durs`. Long timelines are joined
    in groups so the filter graph stays small, then the groups are crossfaded the same way."""
    if overlap <= 0 or len(clips) == 1:
        with open(f"{work}/clips.txt", "w") as f:
            f.writelines(f"file '{c}'\n" for c in clips)
        ff("-f", "concat", "-safe", "0", "-i", "clips.txt", "-c", "copy", out, cwd=work)
        return
    if len(clips) <= group:
        _xfade_group(work, clips, durs, overlap, out)
        return
    parts, part_durs = [], []
    for g in range(0, len(clips), group):
        name = f"grp{level}_{g // group}.mp4"
        sub_c, sub_d = clips[g:g + group], durs[g:g + group]
        if len(sub_c) == 1:
            os.replace(os.path.join(work, sub_c[0]), os.path.join(work, name))
        else:
            _xfade_group(work, sub_c, sub_d, overlap, name)
        parts.append(name)
        part_durs.append(sum(sub_d))
    join(work, parts, part_durs, out, overlap, group, level + 1)


# ---------------------------------------------------------------- music + mix

def music(out: str, dur: float, mood: str = "calm") -> None:
    """Royalty-free ambient pad synthesized on the fly; minor progressions for dark moods."""
    minor = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (130.81, 196.0, 261.63), (196.0, 246.94, 293.66)]
    major = [(261.63, 329.63, 392.0), (196.0, 246.94, 293.66), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63)]
    chords = major if mood in ("uplifting", "triumphant") else minor
    terms = []
    for i, (a, b, c) in enumerate(chords):
        gate = f"between(mod(t,16),{i*4},{i*4+4})"
        terms.append(f"{gate}*(sin(2*PI*{a}*t)+sin(2*PI*{b}*t)+0.6*sin(2*PI*{c}*t)+0.4*sin(2*PI*{a/2}*t))")
    expr = "0.12*(" + "+".join(terms) + ")*(0.75+0.25*sin(2*PI*0.25*t))"
    ff("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d={dur:.2f}",
       "-af", f"lowpass=f=1400,aecho=0.8:0.7:120|240:0.35|0.2,afade=t=in:d=1.5,"
              f"afade=t=out:st={max(0, dur-2.5):.2f}:d=2.5",
       "-c:a", "aac", "-b:a", "160k", out)


def fit_music(src: str, out: str, dur: float) -> None:
    """Loop/trim a library track to the narration length with fades."""
    ff("-stream_loop", "-1", "-i", src, "-t", f"{dur:.2f}",
       "-af", f"afade=t=in:d=2,afade=t=out:st={max(0, dur-3):.2f}:d=3", "-c:a", "aac", "-b:a", "160k", out)


# ---------------------------------------------------------------- captions

def _ts(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs//360000}:{cs//6000%60:02d}:{cs//100%60:02d}.{cs%100:02d}"


def _ass_colour(hex_rgb: str, alpha: str = "00") -> str:
    h = hex_rgb.lstrip("#")
    return f"&H{alpha}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def captions_ass(words: list[dict], out: str, title: str = "", chunk: int | None = None,
                 brand: dict | None = None, chapters: list[tuple[float, str]] | None = None,
                 hook: str | None = None) -> None:
    """Shorts: karaoke captions (spoken word lit) + title lockup + end card, kept inside the Shorts
    safe zone (above the channel handle / description; see Format.layout). Long-form: lower-third chapter cards + end card; captions only when
    the format burns them in (otherwise they ship as an .srt)."""
    chunk = chunk or FMT.caption_words
    accent = _ass_colour((brand or {}).get("accent", "#C88F3B"))
    serif = (brand or {}).get("font_serif", "Georgia")
    if FMT.vertical:
        lay = FMT.layout()  # safe-zone aware: nothing sits under the Shorts handle/description
        mh = lay["cap_margin_h"]
        styles = f"""Style: Cap,Segoe UI Semibold,{lay['cap_font']},&H00F4F7FA,&H00F4F7FA,&H50000000,&H96000000,0,0,0,0,100,100,0.5,0,1,3,2,2,{mh},{mh},{lay['cap_margin_v']},1
Style: Title,{serif},72,&H00F4F7FA,&H00FFFFFF,&H00000000,&H78000000,0,1,0,0,100,100,0,0,1,0,3,2,120,120,{lay['lockup_v']},1
Style: Kicker,{serif},30,&H00DDE1E5,&H00FFFFFF,&H50000000,&H00000000,0,0,0,0,100,100,12,0,1,2,1,2,{mh},{mh},{lay['kicker_v']},1
Style: Lockup,{serif},60,&H00F4F7FA,&H00FFFFFF,&H50000000,&H78000000,0,0,0,0,100,100,6,0,1,3,2,2,{mh},{mh},{lay['lockup_v']},1
Style: Rule,Arial,20,{accent},{accent},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Hook,{serif},96,&H00F4F7FA,&H00FFFFFF,&H64000000,&H96000000,1,0,0,0,100,100,2,0,1,4,3,8,{mh},{mh},0,1"""
        rule = rf"{{\an7\pos({lay['rule_x']},{lay['rule_y']})\p1}}m 0 0 l 140 0 140 4 0 4{{\p0}}"
        pos_big = pos_small = ""
    else:
        styles = f"""Style: Cap,Segoe UI Semibold,52,&H00F4F7FA,&H00F4F7FA,&H50000000,&H96000000,0,0,0,0,100,100,0.3,0,1,3,2,2,200,200,84,1
Style: Title,{serif},64,&H00F4F7FA,&H00FFFFFF,&H00000000,&H78000000,0,1,0,0,100,100,0,0,1,0,3,1,120,120,120,1
Style: Kicker,{serif},26,{accent},&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,10,0,1,0,2,1,120,120,0,1
Style: Lockup,{serif},58,&H00F4F7FA,&H00FFFFFF,&H00000000,&H78000000,0,0,0,0,100,100,3,0,1,0,3,1,120,120,0,1
Style: Rule,Arial,20,{accent},{accent},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"""
        rule = r"{\an7\pos(120,846)\p1}m 0 0 l 110 0 110 4 0 4{\p0}"
        pos_small, pos_big = r"{\an1\pos(120,900)}", r"{\an1\pos(120,972)}"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    fade = r"{\fad(500,700)}"
    halo = r"{\blur2}"  # soft dark halo behind text keeps it legible on bright photos

    def lockup(start: float, end: float, big: str, small: str) -> None:
        lines.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Rule,,0,0,0,,{fade}{rule}")
        lines.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Lockup,,0,0,0,,{fade}{halo}{pos_big}{big}")
        lines.append(f"Dialogue: 1,{_ts(start)},{_ts(end)},Kicker,,0,0,0,,{fade}{halo}{pos_small}{small}")

    if chapters:  # long-form: lower-third per chapter; nothing over the cold open (it's the hook)
        for i, (start, name) in enumerate(chapters[1:], start=1):
            lockup(start + 0.4, start + 4.6, name.upper(), f"CHAPTER {i}")
    elif title and brand:
        lockup(0.2, 4.2, title.upper(), brand["name"].upper())
    elif title:
        lines.append(f"Dialogue: 1,0:00:00.35,0:00:03.60,Title,,0,0,0,,{fade}{title}")
    if hook and FMT.vertical:  # frame-1 hook: the claim a scroller reads before deciding to swipe
        y = FMT.layout()["cap_top"] - 290  # lower third of the picture: portraits keep their face clear
        pop = r"{\fscx86\fscy86\t(0,160,\fscx100\fscy100)\fad(0,300)}"
        text = hook.upper().replace("\n", r"\N")
        lines.append(f"Dialogue: 2,{_ts(0.0)},{_ts(2.0)},Hook,,0,0,0,,{pop}{halo}{{\\an8\\pos({W // 2},{y})}}{text}")
    if brand and words:  # end card: channel name + tagline
        end = words[-1]["end"] + 0.35
        lockup(end - 2.6, end + 0.2, brand["name"].upper(), brand.get("tagline", "").upper())
    if FMT.burn_captions:
        groups = _phrases(words, chunk)
        for gi, group in enumerate(groups):
            # a phrase lingers briefly after its last word, but never into the next phrase
            linger = min(group[-1]["end"] + 0.15, groups[gi + 1][0]["start"]) if gi + 1 < len(groups) \
                else group[-1]["end"] + 0.15
            for j, w in enumerate(group):
                end = group[j + 1]["start"] if j + 1 < len(group) else linger
                parts = [g["text"] if k == j else r"{\c&H9AA0A6&}" + g["text"] + r"{\c&HFAF7F4&}"
                         for k, g in enumerate(group)]
                lines.append(f"Dialogue: 0,{_ts(w['start'])},{_ts(end)},Cap,,0,0,0,,{halo}{' '.join(parts)}")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def _phrases(words: list[dict], size: int) -> list[list[dict]]:
    """Group words into caption phrases, breaking on pauses so lines never straddle sentences."""
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= size or w["start"] - cur[-1]["end"] > 0.25):
            out.append(cur)
            cur = []
        cur.append(w)
    return out + ([cur] if cur else [])


def srt(words: list[dict], out: str, size: int = 9) -> None:
    """Closed captions for upload (YouTube CC). Phrases of up to `size` words."""
    def t(x: float) -> str:
        ms = int(round(x * 1000))
        return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}"
    with open(out, "w", encoding="utf-8") as f:
        for i, g in enumerate(_phrases(words, size), start=1):
            f.write(f"{i}\n{t(g[0]['start'])} --> {t(g[-1]['end'] + 0.1)}\n{' '.join(w['text'] for w in g)}\n\n")


# ---------------------------------------------------------------- final

def shade(out: str) -> None:
    """Darkening mask so overlays read on any photo: top/bottom bands (Shorts), lower-left (long)."""
    if FMT.vertical:
        alpha = (r"255*(0.72*pow(clip((Y-980)/440\,0\,1)\,1.3)"
                 r"+0.55*pow(clip((520-Y)/520\,0\,1)\,1.6))")
    else:
        alpha = r"255*0.7*pow(clip((Y-620)/460\,0\,1)\,1.5)*clip(1.25-X/1500\,0\,1)"
    ff("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1,format=rgba,geq=r=0:g=0:b=0:a='{alpha}'",
       "-frames:v", "1", out)


def assemble(work: str, video: str, voice: list[str], out: str, music_file: str | None = None,
             mood: str = "calm") -> None:
    """Narration + music (ducked under the voice, -14 LUFS) + film grade + overlays.
    Runs inside `work` so the subtitles filter gets a relative path (Windows drive letters)."""
    with open(f"{work}/voice.txt", "w") as f:
        f.writelines(f"file '{v}'\n" for v in voice)
    ff("-f", "concat", "-safe", "0", "-i", "voice.txt", "-c:a", "pcm_s16le", "narration.wav", cwd=work)
    total = duration(f"{work}/narration.wav")
    if music_file:
        fit_music(music_file, f"{work}/music.m4a", total)
    else:
        music(f"{work}/music.m4a", total, mood)
    shade(f"{work}/shade.png")
    fade_in = "" if FMT.vertical else "fade=t=in:d=0.6,"  # Shorts: frame 1 must already show the picture
    ff("-i", video, "-i", "narration.wav", "-i", "music.m4a", "-loop", "1", "-i", "shade.png",
       "-filter_complex",
       # clearer voice: cut rumble, even out the TTS level, lift presence; music sits lower and ducks harder
       "[1:a]highpass=f=70,acompressor=threshold=-20dB:ratio=3:attack=5:release=90:makeup=3,"
       "equalizer=f=3200:width_type=q:width=1.2:g=2.5,asplit=2[n1][n2];[2:a]volume=0.32[m];"
       "[m][n1]sidechaincompress=threshold=0.03:ratio=8:release=400[duck];"
       "[n2][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5[a];"
       # film look: gentle contrast, muted warm palette, vignette, grain, fades
       "[0:v]eq=contrast=1.07:saturation=0.86:gamma=0.98,"
       "colorbalance=rs=0.04:gs=0.01:bs=-0.04:rh=0.03:bh=-0.03,vignette=PI/5,"
       "noise=alls=5:allf=t[g];[g][3:v]overlay=shortest=1,"
       f"subtitles=captions.ass,{fade_in}fade=t=out:st={total-0.8:.2f}:d=0.8[v]",
       "-map", "[v]", "-map", "[a]", *enc(), "-c:a", "aac", "-b:a", "192k",
       "-movflags", "+faststart", "-shortest", out, cwd=work)
