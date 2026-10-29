"""Core invariants. Run: python -m pytest tests"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iframesco import factcheck, formats, render, voice  # noqa: E402
from iframesco.jobs import Queue  # noqa: E402
from iframesco.pipeline import chapter_list, clock  # noqa: E402


def test_xfade_offsets_keep_total_length():
    durs = [3.0, 4.5, 2.0, 5.0]
    assert render.xfade_offsets(durs) == [3.0, 7.5, 9.5]  # transition k starts where slot k ends


def test_chapters_start_at_zero_and_need_three():
    plan = {"scenes": [{"narration": "a"}, {"narration": "b", "chapter": "One"},
                       {"narration": "c", "chapter": "Two"}, {"narration": "d", "chapter": "Three"}]}
    ch = chapter_list(plan, [0, 30, 95, 160])
    assert ch[0] == (0.0, "Intro") and [c for _, c in ch] == ["Intro", "One", "Two", "Three"]
    assert chapter_list({"scenes": plan["scenes"][:2]}, [0, 30]) == []  # < 3 chapters: YouTube ignores them


def test_chapters_drop_ones_shorter_than_ten_seconds():
    plan = {"scenes": [{"narration": "a", "chapter": "A"}, {"narration": "b", "chapter": "B"},
                       {"narration": "c", "chapter": "C"}, {"narration": "d", "chapter": "D"}]}
    assert [c for _, c in chapter_list(plan, [0, 4, 40, 90])] == ["A", "C", "D"]


def test_clock_format():
    assert clock(0) == "0:00" and clock(75) == "1:15" and clock(3725) == "1:02:05"


def test_respell_maps_back_to_real_spelling():
    text, plan = voice.respell("Haise threw Philip.", {"Haise": "Hay zuh bee lus"})
    assert text == "Hay zuh bee lus threw Philip."
    words = [{"text": w, "start": i, "end": i + 1} for i, w in enumerate(text.split())]
    out = voice._remap(words, plan)
    assert [w["text"] for w in out] == ["Haise", "threw", "Philip"]
    assert out[0]["start"] == 0 and out[0]["end"] == 4


def test_respell_survives_engine_splitting_a_hyphenated_respelling():
    text, plan = voice.respell("It began in Yemen, and spread.", {"Yemen": "Yeh-men"})
    words = [{"text": w, "start": i, "end": i + 1} for i, w in enumerate(["It", "began", "in", "Yeh", "men", "and", "spread"])]
    out = voice._remap(words, plan)
    assert [w["text"] for w in out] == ["It", "began", "in", "Yemen", "and", "spread"]
    assert out[3]["start"] == 3 and out[3]["end"] == 5


def test_respell_survives_engine_merging_words():
    text, plan = voice.respell("It fell. In 1524, Suleiman won.", {"Suleiman": "Soo-lay-mahn"})
    words = [{"text": w, "start": i, "end": i + 1} for i, w in enumerate(["It", "fell", "In 1524", "Soo-lay-mahn", "won"])]
    out = voice._remap(words, plan)
    assert [w["text"] for w in out] == ["It", "fell", "In", "1524", "Suleiman", "won"]
    assert out[4]["start"] == 3 and out[4]["end"] == 4


def test_factcheck_drops_invented_numbers():
    src = "Steamboat Willie premiered in New York in 1928."
    text, dropped = factcheck.clean("It premiered in 1928. It cost 4500 dollars.", src)
    assert text == "It premiered in 1928." and dropped == ["It cost 4500 dollars."]


def test_formats():
    assert formats.get("long").w == 1920 and not formats.get("long").vertical
    assert formats.get("nope").name == "short"


def test_long_captions_have_chapter_cards_and_burned_words():
    render.use(formats.LONG)
    try:
        path = os.path.join(tempfile.mkdtemp(), "c.ass")
        words = [{"text": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(50)]
        render.captions_ass(words, path, "T", chapters=[(0, "Intro"), (5, "One"), (12, "Two")])
        with open(path, encoding="utf-8") as f:
            body = f.read()
        assert "CHAPTER 1" in body and "CHAPTER 2" in body and "Dialogue: 0," in body
    finally:
        render.use(formats.SHORT)


def test_queue_persists_and_requeues_running_jobs():
    db = os.path.join(tempfile.mkdtemp(), "jobs.db")
    q = Queue(db, runner=None)
    jid = q.submit({"topic": "x"})
    assert q._claim()["id"] == jid and q.get(jid)["status"] == "running"
    q2 = Queue(db, runner=None)  # "restart": a running job goes back to queued
    assert q2.get(jid)["status"] == "queued"


def test_queue_runs_jobs():
    db = os.path.join(tempfile.mkdtemp(), "jobs.db")
    q = Queue(db, runner=lambda p, prog: (prog("Working", 50), {"echo": p["topic"]})[1]).start(1)
    jid = q.submit({"topic": "hello"})
    for _ in range(50):
        if q.get(jid)["status"] == "done":
            break
        time.sleep(0.1)
    assert q.get(jid)["result"] == {"echo": "hello"}


def test_join_crossfades_keep_timeline_length():
    import shutil
    if not shutil.which("ffmpeg"):
        return
    work = tempfile.mkdtemp()
    slots, o = [1.0, 1.5, 0.8, 1.2, 0.9], 0.3
    clips = []
    for i, d in enumerate(slots):
        length = d + (o if i < len(slots) - 1 else 0)
        render.ff("-f", "lavfi", "-i", f"color=c=0x{i*40:02x}2040:s=320x240:d={length:.3f}:r=30",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", f"c{i}.mp4", cwd=work)
        clips.append(f"c{i}.mp4")
    render.join(work, clips, slots, "out.mp4", overlap=o, group=2)  # forces the grouped path
    assert abs(render.duration(os.path.join(work, "out.mp4")) - sum(slots)) < 0.12


def test_script_cleanup_drops_headings_labels_repeats_and_filler():
    from iframesco.script import _clean, _paragraphs
    raw = ("Oxygen Tank Exploded\nOn April 13, 1970, an explosion ripped through the service module, venting oxygen.\n\n"
           "The promise: you will learn how three men got home in a crippled spacecraft.\n\n"
           "The crew prepared for the long journey home. Mission Control worked through the night on it.")
    paras = _paragraphs(raw)
    assert paras[0].startswith("On April 13") and paras[1].startswith("you will learn")
    seen = set()
    first = _clean(paras, "April 13, 1970", [], seen)
    assert "journey" not in " ".join(first)
    again = _clean(["On April 13, 1970, an explosion ripped through the service module, venting oxygen. "
                    "Engineers built an adapter from plastic and tape that night."], "April 13, 1970", [], seen)
    assert again == ["Engineers built an adapter from plastic and tape that night."]


def _styles(ass_path: str) -> dict:
    with open(ass_path, encoding="utf-8") as f:
        rows = [ln[len("Style: "):].split(",") for ln in f if ln.startswith("Style: ")]
    # fields: Name, Font, Size, ..., Alignment(18), MarginL(19), MarginR(20), MarginV(21)
    return {r[0]: {"size": int(r[2]), "ml": int(r[19]), "mr": int(r[20]), "mv": int(r[21])} for r in rows}


def test_shorts_text_stays_out_of_platform_ui():
    """Captions, title card and end card must sit above the handle/description zone and clear of
    the right-hand buttons; and the title card must not collide with the captions."""
    F = formats.SHORT
    render.use(F)
    path = os.path.join(tempfile.mkdtemp(), "s.ass")
    render.captions_ass([{"text": "w", "start": 0, "end": 1}], path, "T", brand={"name": "B", "tagline": "t"})
    st = _styles(path)
    lay = F.layout()
    for name in ("Cap", "Kicker", "Lockup"):
        assert F.h - st[name]["mv"] <= F.h - F.safe_bottom, f"{name} bottom edge enters the bottom UI zone"
        assert F.w - st[name]["mr"] <= F.w - F.safe_right, f"{name} runs into the right-hand buttons"
    cap_top = F.h - st["Cap"]["mv"] - 2 * 88
    assert F.h - st["Kicker"]["mv"] < cap_top, "title card overlaps the captions"
    assert lay["rule_y"] >= F.safe_top, "title card enters the top UI zone"
    assert F.h - st["Lockup"]["mv"] - 72 >= lay["rule_y"], "title text sits above its rule"
    assert lay["cap_font"] == st["Cap"]["size"]


def test_safe_zone_is_overridable_and_preview_renders(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setenv("FS_SAFE_BOTTOM", "480")
    assert formats.get("short").safe_bottom == 480 and formats.get("short").layout()["cap_margin_v"] == 500
    assert formats.get("long").safe_bottom == 0  # landscape is untouched
    monkeypatch.delenv("FS_SAFE_BOTTOM")
    assert formats.get("short").safe_bottom == 580
    if not shutil.which("ffmpeg"):
        return
    from PIL import Image
    from iframesco import safezone
    Image.new("RGB", (800, 1200), (90, 100, 110)).save(tmp_path / "p.jpg")
    out = safezone.preview(str(tmp_path / "p.jpg"), str(tmp_path / "o.jpg"))
    with Image.open(out) as im:
        assert im.size == (1080, 1920)
    render.use(formats.SHORT)


def test_blocked_titles():
    from iframesco import relevance
    assert relevance.blocked({"title": "Minstrel show poster 1899.jpg"})
    assert relevance.blocked({"title": "x", "url": "https://upload.wikimedia.org/a/Blackface_actor.jpg"})
    assert not relevance.blocked({"title": "McGraw-Hill Building New York.jpg"})


def test_exact_file_keywords_skip_ranking(monkeypatch, tmp_path):
    from iframesco import pipeline, research
    monkeypatch.setattr(research, "file_image", lambda t: [{"url": f"u/{t}", "kind": "image", "title": t}])
    monkeypatch.setattr(research, "download", lambda url, path: path)

    class NoRank:
        def rank(self, *a, **k):
            raise AssertionError("exact files must not be re-ranked")
    scene = {"narration": "x", "keywords": ["file:B.jpg", "file:A.jpg"], "anchor": False}
    got = pipeline._gather(scene, 0, str(tmp_path), "S", type("F", (), {"orientation": "portrait"})(),
                           NoRank(), 2, False, [])
    assert [c["title"] for c in got] == ["B.jpg", "A.jpg"]


def test_pronoun_from_source():
    from iframesco import script
    assert script.pronoun("She was born in 1900. Her father was a baker. She left school and she sang.") == "her"
    assert script.pronoun("The bridge opened in 1937. It spans the strait.") == "its"


def test_pick_images_maps_numbers_to_exact_files(monkeypatch):
    import json
    from iframesco import research, script
    pool = [{"title": "A.jpg", "description": "school"}, {"title": "B.jpg", "description": "final"}]
    monkeypatch.setattr(research, "wiki_images", lambda a, n=12: pool)
    monkeypatch.setattr(script, "with_queries", lambda paras, subj: [{"narration": p, "keywords": [subj]} for p in paras])
    monkeypatch.setattr(script, "ask", lambda *a, **k: json.dumps({"picks": [[1], [2, 9], []]}))
    scenes = script.pick_images(["one", "two", "three"], "Article", "S")
    assert scenes[0]["keywords"] == ["file:A.jpg", "wiki:Article"]
    assert scenes[1]["keywords"] == ["file:B.jpg", "wiki:Article"]  # 9 is out of range and ignored
    assert scenes[2]["keywords"] == ["S"]  # nothing fit: falls back to search queries


def test_tall_portraits_are_not_cropped_on_a_landscape_canvas(tmp_path, monkeypatch):
    from PIL import Image
    from iframesco import render
    monkeypatch.setattr(render, "W", 1920); monkeypatch.setattr(render, "H", 1080)
    tall, wide = tmp_path / "t.jpg", tmp_path / "w.jpg"
    Image.new("RGB", (600, 900)).save(tall); Image.new("RGB", (1600, 900)).save(wide)
    assert render._too_tall(str(tall)) and not render._too_tall(str(wide))


def test_portrait_scoring_prefers_tall_big_faces():
    from iframesco import portraits
    assert portraits.is_portrait(800, 1100) and not portraits.is_portrait(1600, 900)
    assert portraits.score(0.2, 1000, 1500) > portraits.score(0.02, 1000, 1500)
    assert portraits.score(0.2, 1000, 1500) > portraits.score(0.2, 400, 600)
