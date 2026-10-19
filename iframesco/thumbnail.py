"""Thumbnails in the channel's look: dark graded image, big serif hook, amber rule, channel kicker.

Text sits on the darker side of the frame and never over the centre, so a face stays visible.
"""
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# Georgia on Windows/macOS, DejaVu/Liberation Serif on Linux; Pillow's bundled font as a last resort
SERIF = ["C:/Windows/Fonts/georgia.ttf", "/Library/Fonts/Georgia.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
SERIF_BOLD = ["C:/Windows/Fonts/georgiab.ttf", "/Library/Fonts/Georgia Bold.ttf",
              "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"]
INK, MUTED, AMBER = (244, 240, 232), (200, 196, 188), (200, 143, 59)


def _font(paths: list[str] | str, size: int) -> ImageFont.FreeTypeFont:
    for path in [paths] if isinstance(paths, str) else paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _grade(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Color(img).enhance(0.75)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    return ImageEnhance.Brightness(img).enhance(0.82)


def _spaced(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing: int) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing


def _spaced_width(draw, text: str, font, spacing: int) -> float:
    return sum(draw.textlength(ch, font=font) + spacing for ch in text) - spacing


def _rgb(hex_rgb: str | None) -> tuple[int, int, int]:
    h = (hex_rgb or "").lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) if len(h) == 6 else AMBER


def make(image: str, hook: str, out: str, kicker: str = "", label: str = "", size=(1280, 720),
         accent: str | None = None) -> str:
    """Landscape (1280x720) thumbnail: the image fills the right ~60% and feathers into a dark
    panel on the left that holds the hook, so the text never covers the subject."""
    w, h = size
    amber = _rgb(accent)
    iw = int(w * 0.62)
    photo = _grade(ImageOps.fit(Image.open(image).convert("RGB"), (iw, h), centering=(0.5, 0.35)))
    base = Image.new("RGB", size, (12, 11, 10))
    feather = Image.new("L", (iw, h))
    for x in range(iw):  # 0 -> 255 across the first 40% of the photo
        feather.paste(int(255 * min(1.0, x / (iw * 0.4)) ** 1.6), (x, 0, x + 1, h))
    base.paste(photo, (w - iw, 0), feather)
    vignette = Image.new("L", size, 0)
    ImageDraw.Draw(vignette).ellipse((-w * 0.15, -h * 0.25, w * 1.15, h * 1.25), fill=255)
    vignette = ImageOps.invert(vignette.filter(ImageFilter.GaussianBlur(120)))
    base = Image.composite(Image.new("RGB", size, (0, 0, 0)), base, vignette.point(lambda v: v * 0.6))

    d = ImageDraw.Draw(base)
    x = int(w * 0.06)
    lines = hook.upper().split("\n")
    size_px = 112
    big = _font(SERIF_BOLD, size_px)
    while max(d.textlength(l, font=big) for l in lines) > w * 0.5 and size_px > 48:  # fit the panel
        size_px -= 4
        big = _font(SERIF_BOLD, size_px)
    line_h = big.size * 1.08
    top = (h - line_h * len(lines)) / 2 + 40
    if label:  # who the story is about: name recognition above the curiosity hook
        lf = _fit_spaced(d, label.upper(), SERIF_BOLD, 42, 6, w * 0.52)
        _spaced(d, (x, top - 112), label.upper(), lf, amber, 6)
    d.rectangle((x, top - 44, x + 120, top - 39), fill=amber)
    for i, line in enumerate(lines):
        d.text((x + 3, top + i * line_h + 4), line, font=big, fill=(0, 0, 0))  # soft drop shadow
        d.text((x, top + i * line_h), line, font=big, fill=INK)
    if kicker:  # channel name, small, bottom-left
        _spaced(d, (x, h - 70), kicker.upper(), _font(SERIF, 22), MUTED, 8)
    base.save(out, quality=92)
    return out


def _fit_spaced(d, text: str, path, size: int, spacing: int, max_w: float):
    font = _font(path, size)
    while _spaced_width(d, text, font, spacing) > max_w and size > 18:
        size -= 2
        font = _font(path, size)
    return font


def make_vertical(image: str, hook: str, out: str, kicker: str = "", label: str = "",
                  accent: str | None = None) -> str:
    """1080x1920 cover frame for Shorts: hook in the lower band, image above."""
    w, h = 1080, 1920
    amber = _rgb(accent)
    base = _grade(ImageOps.fit(Image.open(image).convert("RGB"), (w, h), centering=(0.5, 0.35)))
    shade = Image.new("L", (w, h))
    for y in range(h):
        shade.paste(int(240 * max(0.0, (y - h * 0.5) / (h * 0.5)) ** 1.2), (0, y, w, y + 1))
    base = Image.composite(Image.new("RGB", (w, h), (10, 9, 8)), base, shade)
    d = ImageDraw.Draw(base)
    lines = hook.upper().split("\n")
    size = 130
    big = _font(SERIF_BOLD, size)
    while size > 70 and max(d.textlength(l, font=big) for l in lines) > w * 0.9:  # long lines shrink, never clip
        size -= 6
        big = _font(SERIF_BOLD, size)
    y = h - 260 - len(lines) * 140
    if label:
        lf = _fit_spaced(d, label.upper(), SERIF_BOLD, 52, 8, w * 0.86)
        _spaced(d, ((w - _spaced_width(d, label.upper(), lf, 8)) / 2, y - 130), label.upper(), lf, amber, 8)
    d.rectangle((w / 2 - 70, y - 50, w / 2 + 70, y - 45), fill=amber)
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=big)
        d.text(((w - tw) / 2, y + i * 140), line, font=big, fill=INK)
    if kicker:
        f = _font(SERIF, 34)
        _spaced(d, ((w - _spaced_width(d, kicker.upper(), f, 11)) / 2, h - 200), kicker.upper(), f, MUTED, 11)
    base.save(out, quality=92)
    return out
