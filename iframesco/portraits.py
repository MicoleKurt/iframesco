"""Find the best vertical portraits of a person on Wikimedia Commons.

Portrait-orientation photos fill a 9:16 Short without blurred bars, and a large clear face holds viewers longer
than a landscape monument or crowd shot. Candidates come from a Commons search plus the person's Wikipedia
article; each is scored on how tall it is, how big and sharp the face is, and its resolution.

    python -m iframesco.portraits "Mahatma Gandhi" 8 --sheet gandhi.jpg
"""
import argparse
import os
import re
import tempfile

import requests

from . import research

MIN_TALL = 1.15  # height / width: below this the photo is landscape or square


def is_portrait(width: int, height: int) -> bool:
    return width > 0 and height / width >= MIN_TALL


def score(face_frac: float, width: int, height: int) -> float:
    """Higher is better: a big face (fraction of the image area) dominates, then resolution, then how tall the
    image is (capped, so a sliver doesn't beat a normal portrait)."""
    tall = min(height / max(width, 1), 1.8) / 1.8
    res = min(height, 2000) / 2000
    return round(4 * min(face_frac, 0.35) / 0.35 + res + 0.5 * tall, 3)


def face_fraction(path: str) -> float:
    """Largest frontal face area / image area, 0 if none found or OpenCV is missing."""
    try:
        import cv2
    except ImportError:
        return 0.0
    img = cv2.imread(path)
    if img is None:
        return 0.0
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    best = 0.0
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        det = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        for x, y, w, h in det.detectMultiScale(gray, 1.1, 4, minSize=(max(40, img.shape[1] // 12),) * 2):
            best = max(best, w * h / (img.shape[0] * img.shape[1]))
    return best


def _commons(query: str, limit: int) -> list[dict]:
    r = research._api(research.COMMONS, {
        "action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|Artist", "iiurlwidth": 800})
    out = []
    for p in r.get("query", {}).get("pages", {}).values():
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        meta = info.get("extmetadata", {})
        out.append({"title": p.get("title", "").removeprefix("File:"), "thumb": info.get("thumburl") or info.get("url"),
                    "width": info.get("width", 0), "height": info.get("height", 0),
                    "license": meta.get("LicenseShortName", {}).get("value", ""),
                    "credit": research._strip_html(meta.get("Artist", {}).get("value", "")),
                    "page": info.get("descriptionurl", "")})
    return out


def best(name: str, limit: int = 8, article: str | None = None) -> list[dict]:
    """The `limit` best vertical portraits of `name`, best first."""
    cands: dict[str, dict] = {}
    for q in (f"{name} portrait", f"{name} photograph", name):
        for c in _commons(q, 40):
            cands.setdefault(c["title"], c)
    if article is not False:
        for w in research.wiki_images(article or name, 60):  # the article's own pictures
            if w["title"] not in cands:
                cands[w["title"]] = {"title": w["title"], "thumb": w["url"], "width": w["width"], "height": 0,
                                     "license": w["license"], "credit": w["credit"], "page": w["page"]}
    tall = [c for c in cands.values() if not any(s in c["title"].lower() for s in research.SKIP_FILE)]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(tall):
            try:
                path = os.path.join(tmp, f"{i}.jpg")
                with open(path, "wb") as f:
                    f.write(requests.get(c["thumb"], headers=research.UA, timeout=25).content)
                import cv2
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if not is_portrait(w, h):
                    continue
                c["face"] = round(face_fraction(path), 3)
                c["score"] = score(c["face"], c["width"] or w, c["height"] or h)
                out.append(c)
            except Exception:
                continue
    return sorted((c for c in out if c["face"] > 0), key=lambda c: -c["score"])[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("limit", nargs="?", type=int, default=8)
    ap.add_argument("--article")
    ap.add_argument("--sheet", help="write a contact sheet of the picks to this path")
    a = ap.parse_args()
    picks = best(a.name, a.limit, a.article)
    for c in picks:
        print(f'{c["score"]:.2f} face={c["face"]:.2f} {c["width"]}x{c["height"]} {c["license"]:<14} {c["title"]}')
    if a.sheet and picks:
        from PIL import Image
        tiles = []
        for c in picks:
            im = Image.open(requests.get(c["thumb"], headers=research.UA, stream=True, timeout=25).raw).convert("RGB")
            im.thumbnail((260, 380))
            tiles.append(im)
        sheet = Image.new("RGB", (270 * len(tiles), 390), (10, 10, 10))
        for i, im in enumerate(tiles):
            sheet.paste(im, (i * 270 + 5, 5))
        sheet.save(a.sheet, quality=88)


if __name__ == "__main__":
    main()
