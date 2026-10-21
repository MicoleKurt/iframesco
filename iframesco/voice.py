"""Narration with word-level timings for captions. Two free engines:

- kokoro: Kokoro-82M, offline, calm and even (default "soothing" voices). Needs `pip install kokoro soundfile`.
- edge:   Microsoft Edge neural voices via edge-tts (online, no key).

Hard names are respelled for the engine (`pronounce` map) while captions keep the real spelling.
"""
import asyncio
import os
import re
import time

PRESETS = {
    # name: (engine, voice, rate %, pitch)
    "soothing":    ("kokoro", "am_michael", -8, ""),
    "storyteller": ("kokoro", "bm_george", -6, ""),
    "documentary": ("edge", "en-US-AndrewMultilingualNeural", -4, "-2Hz"),
    "warm":        ("edge", "en-US-BrianMultilingualNeural", -6, "-1Hz"),
    "british":     ("edge", "en-GB-RyanNeural", -5, "-1Hz"),
    "energetic":   ("edge", "en-US-AndrewMultilingualNeural", 6, ""),
    "narrator":    ("edge", "en-US-AndrewMultilingualNeural", 4, ""),  # clear, brisk documentary read
}
_EDGE_GAP = 2.0  # seconds between edge requests
_last_edge = [0.0]
DEFAULT = os.getenv("FS_VOICE", "soothing")


def kokoro_available() -> bool:
    from importlib.util import find_spec  # cheap: importing kokoro loads torch
    return find_spec("kokoro") is not None and find_spec("soundfile") is not None


def resolve(preset: str | None) -> tuple[str, str, int, str]:
    engine, voice, rate, pitch = PRESETS.get(preset or DEFAULT, PRESETS["soothing"])
    if engine == "kokoro" and not kokoro_available():  # zero-install fallback
        return PRESETS["documentary"]
    return engine, voice, rate, pitch


# ---------------------------------------------------------------- pronunciation

def respell(text: str, pronounce: dict[str, str] | None) -> tuple[str, list[tuple[int, str, str]]]:
    """Swap hard names for phonetic spellings. Returns the text to speak plus, per original word,
    how many spoken words it became, so timings map back to the real spelling."""
    plan, spoken = [], []
    for word in text.split():
        core = re.sub(r"^\W+|\W+$", "", word)
        alt = (pronounce or {}).get(core)
        say = word.replace(core, alt) if alt else word
        spoken.append(say)
        plan.append((len(say.split()), word, say))
    return " ".join(spoken), plan


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _remap(words: list[dict], plan: list[tuple[int, str, str]]) -> list[dict]:
    """Map the engine's word timings back onto the real spelling. When the engine splits a respelling
    ("Yeh-men" -> "Yeh", "men") the counts differ, so tokens are matched by their letters instead."""
    out, i = [], 0
    for _, display, say in plan:
        want, got, j = _letters(say), "", i
        while j < len(words) and (len(got) < len(want) or j == i):
            got += _letters(words[j]["text"])
            j += 1
        if got != want or j == i:  # the engine said something else: keep its raw timings
            return words
        out.append({"text": re.sub(r"^\W+|\W+$", "", display),
                    "start": words[i]["start"], "end": words[j - 1]["end"]})
        i = j
    return out


# ---------------------------------------------------------------- engines

async def _edge(text: str, path: str, voice: str, rate: int, pitch: str) -> list[dict]:
    import edge_tts
    words = []
    kw = {"pitch": pitch} if pitch else {}
    comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", boundary="WordBoundary", **kw)
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":  # offsets are in 100ns ticks
                words.append({"text": chunk["text"], "start": chunk["offset"] / 1e7,
                              "end": (chunk["offset"] + chunk["duration"]) / 1e7})
    return words


_KOKORO: dict = {}


def _kokoro(text: str, path: str, voice: str, rate: int) -> list[dict]:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    lang = "b" if voice.startswith("b") else "a"
    if lang not in _KOKORO:
        _KOKORO[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    audio, words, offset = [], [], 0.0
    for r in _KOKORO[lang](text, voice=voice, speed=1 + rate / 100):
        a = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)
        audio.append(a)
        for tok in r.tokens or []:
            if tok.start_ts is not None and re.search(r"\w", tok.text):
                words.append({"text": tok.text, "start": offset + tok.start_ts, "end": offset + tok.end_ts})
        offset += len(a) / 24000
    sf.write(path, np.concatenate(audio), 24000)
    return words


def synthesize(text: str, stem: str, preset: str | None = None, rate: int | None = None,
               pronounce: dict[str, str] | None = None, retries: int = 6) -> tuple[str, list[dict]]:
    """Speak `text` to `stem`.wav/.mp3. Returns (audio path, word timings)."""
    engine, voice, base_rate, pitch = resolve(preset)
    rate = base_rate if rate is None else rate
    spoken, plan = respell(text, pronounce)
    if engine == "kokoro":
        path = stem + ".wav"
        return path, _remap(_kokoro(spoken, path, voice, rate), plan)
    import edge_tts
    path = stem + ".mp3"
    for attempt in range(retries):  # edge throttles bursts of requests: pace them, back off hard on a miss
        time.sleep(max(0.0, _EDGE_GAP - (time.time() - _last_edge[0])))
        try:
            return path, _remap(asyncio.run(_edge(spoken, path, voice, rate, pitch)), plan)
        except edge_tts.exceptions.NoAudioReceived:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
        finally:
            _last_edge[0] = time.time()
    raise RuntimeError("unreachable")
