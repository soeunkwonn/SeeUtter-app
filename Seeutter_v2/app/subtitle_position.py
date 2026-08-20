"""Subtitle position presets + preview-image generation for the naming UI.

The user picks one of four positions; instead of re-rendering the whole video we
grab a single still frame (ffmpeg) and paint a mock subtitle box at each preset
(PIL). The Streamlit UI just swaps between the four pre-made preview images.

The same preset also maps to an ASS ``Alignment`` value used at final render, so
the preview matches the burned-in result.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# preset key -> label (UI) / ASS numpad alignment / PIL anchors
POSITION_PRESETS: dict[str, dict] = {
    "bottom_center": {"label": "하단 중앙", "ass": 2, "h": "center", "v": "bottom"},
    "top_center":    {"label": "상단 중앙", "ass": 8, "h": "center", "v": "top"},
    "bottom_left":   {"label": "좌하단",    "ass": 1, "h": "left",   "v": "bottom"},
    "bottom_right":  {"label": "우하단",    "ass": 3, "h": "right",  "v": "bottom"},
}
DEFAULT_POSITION = "bottom_center"

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",   # Malgun Gothic (Korean)
    r"C:\Windows\Fonts\gulim.ttc",
]
SAMPLE_TEXT = "예시 자막입니다"


def _font_path() -> str | None:
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    return None


def ass_alignment(position: str) -> int:
    """ASS \\an alignment for a preset (2/8/1/3). Unknown -> bottom-center."""
    return POSITION_PRESETS.get(position, POSITION_PRESETS[DEFAULT_POSITION])["ass"]


def capture_frame(
    video_path: Path,
    out_png: Path,
    *,
    at_sec: float = 1.0,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Grab one still frame from the video at ``at_sec`` seconds."""
    ffmpeg = ffmpeg_bin or os.environ.get("FFMPEG_BIN", "ffmpeg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{max(0.0, at_sec):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        "-loglevel", "error",
        str(out_png),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"frame capture failed: {result.stderr.strip()}")
    return out_png


# A sample cue that exercises the real renderer: speaker + emoji + Korean.
SAMPLE_UNITS = [("[화자]", False), ("\U0001F604", True), ("예시 자막입니다", False)]
PREVIEW_MARGIN = 40  # must match s10_render._paste_xy so preview == burned output


def draw_preview(
    frame_png: Path,
    position: str,
    out_png: Path,
    *,
    units=None,
    fill_hex: str = "#FFFFFF",
) -> Path:
    """Paste a real caption (the SAME renderer as the burn stage) at ``position``.

    Using ``caption_pil.draw_caption`` makes the preview pixel-match the burned
    subtitle (box, font, colour emoji, wrapping).
    """
    from Seeutter_v2.pipeline.lib.caption_pil import draw_caption

    preset = POSITION_PRESETS.get(position, POSITION_PRESETS[DEFAULT_POSITION])
    base = Image.open(frame_png).convert("RGBA")
    W, H = base.size
    box = draw_caption(units or SAMPLE_UNITS, fill_hex, W, H)

    if preset["h"] == "center":
        x = (W - box.width) // 2
    elif preset["h"] == "left":
        x = PREVIEW_MARGIN
    else:  # right
        x = W - box.width - PREVIEW_MARGIN
    y = PREVIEW_MARGIN if preset["v"] == "top" else H - box.height - PREVIEW_MARGIN

    base.alpha_composite(box, (int(max(0, x)), int(max(0, y))))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_png)
    return out_png


def build_position_previews(
    video_path: Path,
    out_dir: Path,
    *,
    at_sec: float = 1.0,
    ffmpeg_bin: str | None = None,
    units=None,
    fill_hex: str = "#FFFFFF",
) -> dict[str, Path]:
    """Capture one frame and render a preview image for every preset.

    Returns {preset_key: png_path}. Cheap enough to call once per artifact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_png = out_dir / "_frame.png"
    capture_frame(video_path, frame_png, at_sec=at_sec, ffmpeg_bin=ffmpeg_bin)

    previews: dict[str, Path] = {}
    for key in POSITION_PRESETS:
        out_png = out_dir / f"preview_{key}.png"
        draw_preview(frame_png, key, out_png, units=units, fill_hex=fill_hex)
        previews[key] = out_png
    return previews
