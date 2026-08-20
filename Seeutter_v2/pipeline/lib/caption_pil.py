"""Render one subtitle cue to a transparent PNG with Pillow.

ffmpeg's text renderers can't draw colour emoji, so captions are drawn here
instead: Korean text (Pretendard SemiBold) + emotion colour + colour emoji (Segoe UI
Emoji via embedded_color) + a translucent rectangular box, wrapped to a
caption-safe portion of the frame. The render stage overlays these PNGs onto
the video; the naming UI uses the same function for a WYSIWYG position preview.

A "unit" is ``(string, is_emoji)``. Text units are coloured with ``fill_hex``;
emoji units carry their own colour from the font.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# The bundled font makes the rendered video independent of the machine where
# Streamlit/ffmpeg is launched. Its license is kept beside this asset.
PRETENDARD_SEMIBOLD_FONT = (
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Pretendard-SemiBold.otf"
)
MALGUN_FONT = r"C:\Windows\Fonts\malgun.ttf"   # fallback Korean text (Windows)


def _first_existing(paths):
    for candidate in paths:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


# Colour emoji face: Segoe on Windows, Noto Color Emoji on the Linux cloud
# container (installed via packages.txt). Resolved once at import.
EMOJI_FONT = _first_existing(
    [
        r"C:\Windows\Fonts\seguiemj.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/opentype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji-Regular.ttf",
    ]
)

_MEASURE = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _font(path, size):
    return ImageFont.truetype(path, size)


def _caption_font(size):
    """Use the bundled Pretendard SemiBold caption face when available."""
    path = PRETENDARD_SEMIBOLD_FONT if PRETENDARD_SEMIBOLD_FONT.is_file() else MALGUN_FONT
    return _font(path, size)


def _emoji_font(size):
    """Load the colour-emoji face, degrading gracefully rather than crashing.

    Bitmap colour fonts (Noto Color Emoji) ship a single fixed strike, so an
    arbitrary pixel size can raise; fall back to that strike, and if no emoji
    face exists at all, reuse the caption face so a render never fails.
    """
    if EMOJI_FONT is None:
        return _caption_font(size)
    try:
        return ImageFont.truetype(EMOJI_FONT, size)
    except OSError:
        try:
            return ImageFont.truetype(EMOJI_FONT, 109)
        except OSError:
            return _caption_font(size)


# Pre-rendered colour emoji (baked from Segoe on Windows). Pasting these images
# gives identical emoji on any OS -- PIL cannot draw Linux's Noto Color Emoji
# reliably, so the emotion emoji never renders as a broken/monochrome glyph.
_EMOJI_DIR = Path(__file__).resolve().parents[2] / "assets" / "emoji"
_EMOJI_PNG = {
    "😄": "happy.png",
    "😠": "angry.png",
    "😭": "sad.png",
    "🤢": "disgusted.png",
    "😨": "fearful.png",
    "😲": "surprised.png",
}
_EMOJI_CACHE: dict[str, Image.Image] = {}


def _emoji_asset(char):
    """Cached RGBA image for a known emoji, or None to fall back to the font."""
    name = _EMOJI_PNG.get(char)
    if not name:
        return None
    if char not in _EMOJI_CACHE:
        path = _EMOJI_DIR / name
        if not path.is_file():
            return None
        _EMOJI_CACHE[char] = Image.open(path).convert("RGBA")
    return _EMOJI_CACHE.get(char)


def _emoji_layout(char, fontsize):
    """(scaled image, width) for an emoji drawn as an image, or None for font."""
    base = _emoji_asset(char)
    if base is None:
        return None
    height = max(1, round(fontsize * 1.10))
    width = max(1, round(base.width * height / base.height))
    return base.resize((width, height), Image.LANCZOS), width


def _hex_rgb(hex_color):
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ── Korean-aware line breaking (layout level; no morphological analysis) ──────
# A new line should not START with a word that reads as bonded to the previous
# one: dependent nouns (의존명사) and common auxiliary-verb stems (보조용언).
_KO_BOUND_HEAD = set("수것때줄바데뿐채척터리")   # 있을 |수|도, 먹을 |것|, 갈 |때|
_KO_AUX_HEAD = set("있없싶")                      # 하고 |있|다, 하고 |싶|어
# A line is GOOD to break AFTER these: clause/breath boundaries.
_BREAK_PUNCT = (",", "，", ".", "?", "!", "…")
_KO_MULTI_ENDINGS = ("는데", "지만", "다가", "면서", "거나", "든지", "라서")


def _break_penalty(prev_unit, next_unit):
    """Extra cost for breaking the line between two adjacent units (- = prefer)."""
    core = prev_unit[0].rstrip("]\"')”’")
    head = next_unit[0].lstrip("[\"'(“‘")
    pen = 0.0
    if core.endswith(_BREAK_PUNCT):
        pen -= 0.12
    elif core.endswith(_KO_MULTI_ENDINGS):
        pen -= 0.06
    if head[:1] in _KO_BOUND_HEAD:
        pen += 0.30
    elif head[:1] in _KO_AUX_HEAD:
        pen += 0.20
    return pen


def _wrap_lines(measured, space_w, max_inner):
    """Balanced word-wrap: minimise per-line slack (evenly filled lines, no
    orphan last word) plus Korean break-point penalties. DP over ``measured``
    units ``(s, is_emoji, width, font)``; O(n^2) which is trivial for a cue."""
    n = len(measured)
    if n == 0:
        return []
    widths = [w for _, _, w, _ in measured]
    inf = float("inf")
    dp = [inf] * (n + 1)     # dp[i] = min cost to lay out units[i:]
    nxt = [n] * (n + 1)
    dp[n] = 0.0
    for i in range(n - 1, -1, -1):
        line_w = 0.0
        for j in range(i, n):
            line_w = widths[i] if j == i else line_w + space_w + widths[j]
            if line_w > max_inner and j > i:
                break  # lines only get wider; a single overlong unit is allowed
            slack = max_inner - line_w
            line_cost = 0.0 if slack <= 0 else (slack / max_inner) ** 2
            brk = 0.0 if j == n - 1 else _break_penalty(measured[j], measured[j + 1])
            cost = line_cost + brk + dp[j + 1]
            if cost < dp[i]:
                dp[i] = cost
                nxt[i] = j + 1
    lines, i = [], 0
    while i < n:
        lines.append(measured[i:nxt[i]])
        i = nxt[i]
    return lines


def draw_caption(
    units,
    fill_hex,
    video_w,
    video_h,
    *,
    box_rgba=(0, 0, 0, 175),
    max_width_frac=0.65,
):
    """Draw a caption cue -> RGBA Image sized to its box (transparent margins).

    Text/emoji sizes scale with ``video_h`` so the overlay matches the frame.
    """
    fontsize = max(20, round(video_h * 0.045))
    tfont = _caption_font(fontsize)
    efont = _emoji_font(fontsize)
    fill = _hex_rgb(fill_hex) + (255,)
    # A slightly smaller horizontal inset and a 65%-wide cap make the box less
    # wide without making the text itself smaller.
    pad_x = round(fontsize * 0.35)
    pad_y = round(fontsize * 0.5)
    space_w = _MEASURE.textlength(" ", font=tfont)
    max_inner = int(video_w * max_width_frac) - 2 * pad_x

    # Split units into forced paragraphs at ("\n", False) sentinels, then
    # word-wrap each paragraph independently. This lets one cue carry multiple
    # speaker lines (overlapping speech) while each line still wraps on its own.
    paragraphs = [[]]
    for s, is_emoji in units:
        if s == "\n" and not is_emoji:
            paragraphs.append([])
        else:
            paragraphs[-1].append((s, is_emoji))

    # balanced, Korean-aware word-wrap (even line lengths, no orphan last word)
    lines = []
    for para in paragraphs:
        if not para:
            continue
        measured = []
        for s, is_emoji in para:
            if is_emoji:
                layout = _emoji_layout(s, fontsize)
                width = layout[1] if layout else _MEASURE.textlength(s, font=efont)
                measured.append((s, is_emoji, width, efont))
            else:
                measured.append((s, is_emoji, _MEASURE.textlength(s, font=tfont), tfont))
        lines.extend(_wrap_lines(measured, space_w, max_inner))
    if not lines:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    def line_width(line):
        return sum(w for _, _, w, _ in line) + space_w * (len(line) - 1)

    ascent, descent = tfont.getmetrics()
    line_h = ascent + descent + round(fontsize * 0.25)
    inner_w = max(line_width(l) for l in lines)
    box_w = int(inner_w + 2 * pad_x)
    box_h = int(line_h * len(lines) + 2 * pad_y)

    img = Image.new("RGBA", (max(1, box_w), max(1, box_h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, box_w - 1, box_h - 1], fill=box_rgba)

    y = pad_y
    for line in lines:
        x = (box_w - line_width(line)) / 2  # centre each line
        # ``text((x, y))`` treats y as a font-specific top origin. Segoe UI
        # Emoji's glyph box starts much higher than Malgun Gothic's, so an
        # emoji drawn at the same y visibly floats above Korean text. Draw
        # every unit from the same left-baseline instead.
        baseline_y = y + ascent
        for s, is_emoji, w, f in line:
            if is_emoji:
                layout = _emoji_layout(s, fontsize)
                if layout is not None:
                    emoji_img, _w = layout
                    # Centre the emoji vertically on the text's visual middle.
                    top = int(round(baseline_y - fontsize * 0.35 - emoji_img.height / 2))
                    img.alpha_composite(emoji_img, (int(round(x)), top))
                else:
                    d.text((x, baseline_y), s, font=f, embedded_color=True, anchor="ls")
            else:
                d.text((x, baseline_y), s, font=f, fill=fill, anchor="ls")
            x += w + space_w
        y += line_h
    return img
