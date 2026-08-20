from __future__ import annotations
import shutil
import subprocess
import textwrap
from pathlib import Path
from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    read_json,
    resolve_artifact_dir,
    standard_paths,
)

# 초 단위 시간을 SRT 자막 형식으로 변환
def format_timestamp(seconds):
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

EMOTION_EMOJI = {
    "|EMO_UNKNOWN|": "",
    "|NEUTRAL|": "",
    "|HAPPY|": "😄",
    "|ANGRY|": "😠",
    "|SAD|": "😭",
    "|DISGUSTED|": "🤢",
    "|FEARFUL|": "😨",
    "|SURPRISED|": "😲",
}


def speaker_prefix(segment):
    """Return a user-facing speaker name, or ``None`` when none is assigned.

    A prefix appears only once a real display name has been assigned -- i.e.
    the display name differs from the bare ID. This applies uniformly to
    ``FACE_n``, ``AUDIO_n`` and raw numeric diarization labels, so a manually
    named raw speaker (e.g. ``"3" -> "Monica"``) shows in captions too.
    ``UNKNOWN`` never receives a prefix.
    """
    speaker_id = str(segment.get("speaker_id", segment.get("speaker", "UNKNOWN")))
    display_name = str(segment.get("speaker_display", speaker_id)).strip()
    if speaker_id.upper() == "UNKNOWN":
        return None
    if not display_name or display_name.upper() == "UNKNOWN":
        return None
    if display_name == speaker_id:
        # No real name assigned yet (still the raw/auto ID).
        return None
    return display_name


# 자막 본문 앞에 화자 이름과 감정 붙임
def format_caption_text(
    segment,
    include_speaker,
    include_emotion,
    hide_unknown_emotion,
):
    prefix_parts = []
    if include_speaker:
        name = speaker_prefix(segment)
        if name is not None:
            prefix_parts.append(f"[{name}]")

    emotion = segment.get("emotion")
    if include_emotion and emotion:
        emotion_str = str(emotion)
        is_unknown = emotion_str in {"|EMO_UNKNOWN|", "EMO_UNKNOWN", "unknown"}
        if not (hide_unknown_emotion and is_unknown):
            if emotion_str in EMOTION_EMOJI:
                emoji = EMOTION_EMOJI[emotion_str]
                if emoji:
                    prefix_parts.append(emoji)
            elif not is_unknown:
                prefix_parts.append(f"[{emotion_str}]")

    prefix = "".join(prefix_parts)
    # Korean translation (text_ko) wins when present; else the original text.
    text = str(segment.get("text_ko") or segment.get("text", "")).strip()
    return f"{prefix} {text}".strip()


def build_srt(
    segments,
    include_speaker,
    include_emotion,
    hide_unknown_emotion,
):
    blocks = []
    for index, segment in enumerate(segments, 1):
        start = format_timestamp(float(segment["start"]))
        end = format_timestamp(float(segment["end"]))
        text = format_caption_text(segment, include_speaker, include_emotion, hide_unknown_emotion)
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)

# 자막 위치 프리셋 -> ASS \an alignment (app/subtitle_position.py 와 동일)
POSITION_ALIGNMENT = {
    "bottom_center": 2,
    "top_center": 8,
    "bottom_left": 1,
    "bottom_right": 3,
}
DEFAULT_POSITION = "bottom_center"


def _subtitle_force_style(position, margin_v=45):
    """SRT(subtitles 필터)용 force_style: 전역 정렬 + 세로 여백."""
    align = POSITION_ALIGNMENT.get(position, POSITION_ALIGNMENT[DEFAULT_POSITION])
    return f"Alignment={align},MarginV={margin_v}"


def _drawtext_xy(position, margin=40):
    """drawtext용 x/y 식 (프리셋 위치에 맞춰 계산)."""
    pos = position if position in POSITION_ALIGNMENT else DEFAULT_POSITION
    if pos == "bottom_left":
        x = f"{margin}"
    elif pos == "bottom_right":
        x = f"w-text_w-{margin}"
    else:  # bottom_center / top_center
        x = "(w-text_w)/2"
    y = f"{margin}" if pos == "top_center" else f"h-text_h-{margin}"
    return x, y


# 영상화면에 자막 입히는 함수
def has_ffmpeg_filter(ffmpeg_bin, filter_name):
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return True
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == filter_name:
            return True
    return False


def burn_subtitles(video_path, srt_path, out_path, ffmpeg_bin, position=DEFAULT_POSITION):
    video_path = Path(video_path).resolve()
    srt_path = Path(srt_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not has_ffmpeg_filter(ffmpeg_bin, "subtitles"):
        raise RuntimeError(
            "ffmpeg subtitle burn-in failed: this FFmpeg build does not include "
            "the 'subtitles' filter. Use an FFmpeg build with libass support and "
            "set FFMPEG_BIN to that ffmpeg.exe."
        )

    # Keep the subtitles filter path simple. Windows drive letters and spaces
    # inside a filter expression are easy to misparse, so run ffmpeg from the
    # subtitle directory and reference only the stable file name. force_style
    # applies the chosen position (Alignment/MarginV) globally.
    force_style = _subtitle_force_style(position)
    subtitle_filter = f"subtitles=filename={srt_path.name}:force_style='{force_style}'"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        subtitle_filter,
        "-c:a",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(srt_path.parent))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn-in failed: {result.stderr.strip()}")


def escape_drawtext_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_drawtext_files(artifact_dir, segments, include_speaker, include_emotion, hide_unknown_emotion):
    captions_dir = artifact_dir / "_drawtext_captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, segment in enumerate(segments):
        text = format_caption_text(segment, include_speaker, include_emotion, hide_unknown_emotion)
        wrapped = "\n".join(textwrap.wrap(text, width=44, break_long_words=False) or [""])
        caption_path = captions_dir / f"caption_{index:04d}.txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(escape_drawtext_text(wrapped))
        files.append(caption_path)
    return files


def burn_drawtext_captions(
    video_path,
    segments,
    out_path,
    ffmpeg_bin,
    include_speaker,
    include_emotion,
    hide_unknown_emotion,
    position=DEFAULT_POSITION,
    emotion_colors=None,
):
    video_path = Path(video_path).resolve()
    out_path = Path(out_path).resolve()
    artifact_dir = out_path.parent
    caption_files = write_drawtext_files(
        artifact_dir,
        segments,
        include_speaker,
        include_emotion,
        hide_unknown_emotion,
    )

    x_expr, y_expr = _drawtext_xy(position)
    filters = []
    for segment, caption_path in zip(segments, caption_files):
        start = max(0.0, float(segment["start"]))
        end = max(start + 0.05, float(segment["end"]))
        textfile = caption_path.relative_to(artifact_dir).as_posix()
        filters.append(
            "drawtext="
            f"textfile={textfile}:"
            "fontcolor=white:"
            "fontsize=28:"
            "line_spacing=6:"
            "box=1:"
            "boxcolor=black@0.68:"
            "boxborderw=10:"
            f"x={x_expr}:"
            f"y={y_expr}:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )

    graph_path = artifact_dir / "_drawtext_filtergraph.txt"
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write("[0:v]" + ",".join(filters) + "[v]")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-filter_complex_script",
        graph_path.name,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(artifact_dir))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg drawtext subtitle burn-in failed: {result.stderr.strip()}")


# ── Caption text colour is fixed white across all render paths. ──
# ``emotion_colors`` parameters remain for call compatibility, but are ignored.
DEFAULT_EMOTION_COLOR = {
    "|HAPPY|":     "#FFFFFF",
    "|ANGRY|":     "#FFFFFF",
    "|SAD|":       "#FFFFFF",
    "|FEARFUL|":   "#FFFFFF",
    "|DISGUSTED|": "#FFFFFF",
    "|SURPRISED|": "#FFFFFF",
    "|NEUTRAL|":   "#FFFFFF",
    "|EMO_UNKNOWN|": "#FFFFFF",
}


def _ass_ts(seconds):
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, c = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _hex_to_ass(hex_color):
    """#RRGGBB -> ASS \\c 색상(&Hbbggrr, BGR 순서)."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{b}{g}{r}".upper()


def _hex_to_drawtext(hex_color):
    """#RRGGBB -> drawtext fontcolor(0xRRGGBB)."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return "0x" + h.upper()


def _ass_escape(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r\n", "\\N")
        .replace("\n", "\\N")
    )


def _ffprobe_bin(ffmpeg_bin):
    return str(ffmpeg_bin).replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")


def probe_resolution(video_path, ffmpeg_bin, default=(1280, 720)):
    result = subprocess.run(
        [_ffprobe_bin(ffmpeg_bin), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        w, h = result.stdout.strip().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return default


def build_ass(
    segments,
    width,
    height,
    position,
    emotion_colors,
    include_speaker,
    include_emotion,
    hide_unknown_emotion,
    default_color="#FFFFFF",
):
    align = POSITION_ALIGNMENT.get(position, POSITION_ALIGNMENT[DEFAULT_POSITION])
    fontsize = max(20, round(height * 0.045))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Malgun Gothic,{fontsize},&H00FFFFFF,&H000000FF,"
        f"&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,{align},40,40,45,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines = [header]
    for seg in segments:
        text = _ass_escape(
            format_caption_text(seg, include_speaker, include_emotion, hide_unknown_emotion)
        )
        tag = "{\\c&HFFFFFF&}"
        start = _ass_ts(float(seg["start"]))
        end = _ass_ts(float(seg["end"]))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{tag}{text}")
    return "\n".join(lines) + "\n"


def burn_ass(video_path, ass_path, out_path, ffmpeg_bin):
    video_path = Path(video_path).resolve()
    ass_path = Path(ass_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Reference only the file name and run from its directory (Windows drive
    # letters/colons trip up the filtergraph parser). libass reads .ass styling.
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vf", f"subtitles=filename={ass_path.name}",
        "-c:a", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ass_path.parent))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ass burn-in failed: {result.stderr.strip()}")


# ── PIL overlay path (primary): colour emoji + Korean + colour + box, all drawn
# by Pillow, then overlaid onto the video. ffmpeg text filters can't draw colour
# emoji, so this replaces drawtext for the burned output. ──
def _line_units(item, include_speaker, include_emotion, hide_unknown_emotion):
    """Units [(text, is_emoji)] for one speaker line: prefix + emotion + body."""
    units = []
    if include_speaker:
        name = speaker_prefix(item)
        if name is not None:
            units.append((f"[{name}]", False))
    emotion = item.get("emotion")
    if include_emotion and emotion:
        es = str(emotion)
        is_unknown = es in {"|EMO_UNKNOWN|", "EMO_UNKNOWN", "unknown"}
        if not (hide_unknown_emotion and is_unknown):
            if es in EMOTION_EMOJI:
                if EMOTION_EMOJI[es]:
                    units.append((EMOTION_EMOJI[es], True))
            elif not is_unknown:
                units.append((f"[{es}]", False))
    body = str(item.get("text_ko") or item.get("text", "")).strip()
    for word in body.split(" "):
        if word:
            units.append((word, False))
    return units


def caption_units(segment, include_speaker, include_emotion, hide_unknown_emotion):
    """Rich units [(text, is_emoji)] for PIL -- mirrors format_caption_text.

    A segment may carry a ``lines`` list for overlapping speech: each entry is a
    per-speaker line ({speaker_id, speaker_display, text_ko[, emotion]}) drawn on
    its own row with its own speaker prefix. A ``("\\n", False)`` sentinel marks
    the forced break so draw_caption puts each on a separate line.
    """
    lines = segment.get("lines")
    if not lines:
        return _line_units(segment, include_speaker, include_emotion, hide_unknown_emotion)
    out = []
    for k, line in enumerate(lines):
        if k > 0:
            out.append(("\n", False))
        out.extend(_line_units(line, include_speaker, include_emotion, hide_unknown_emotion))
    return out


def probe_duration(video_path, ffmpeg_bin, default=0.0):
    result = subprocess.run(
        [_ffprobe_bin(ffmpeg_bin), "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return default


def _paste_xy(position, box_w, box_h, frame_w, frame_h, margin=40):
    """Pixel top-left for pasting a caption box onto a full frame at a preset."""
    pos = position if position in POSITION_ALIGNMENT else DEFAULT_POSITION
    if pos == "bottom_left":
        x = margin
    elif pos == "bottom_right":
        x = frame_w - box_w - margin
    else:
        x = (frame_w - box_w) // 2
    y = margin if pos == "top_center" else frame_h - box_h - margin
    return int(max(0, x)), int(max(0, y))


# Minimum time (seconds) each caption stays on screen. Short aligned segments
# would otherwise flash by; raise this to hold captions longer.
MIN_CAPTION_SEC = 1.2


def burn_pil_overlays(
    video_path,
    segments,
    out_path,
    ffmpeg_bin,
    position,
    emotion_colors,
    include_speaker,
    include_emotion,
    hide_unknown_emotion,
):
    from ..lib.caption_pil import draw_caption
    from PIL import Image

    video_path = Path(video_path).resolve()
    out_path = Path(out_path).resolve()
    artifact_dir = out_path.parent
    width, height = probe_resolution(video_path, ffmpeg_bin)
    total = probe_duration(video_path, ffmpeg_bin)

    png_dir = artifact_dir / "_caption_pngs"
    shutil.rmtree(png_dir, ignore_errors=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    gap_png = png_dir / "_gap.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(gap_png)

    # Each cue -> a full-frame transparent PNG with its box pasted at `position`.
    cues = []  # (start, end, png_name)
    for idx, segment in enumerate(segments):
        units = caption_units(segment, include_speaker, include_emotion, hide_unknown_emotion)
        if not units:
            continue
        box = draw_caption(units, "#FFFFFF", width, height)
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        px, py = _paste_xy(position, box.width, box.height, width, height)
        frame.alpha_composite(box, (px, py))
        png_path = png_dir / f"caption_{idx:04d}.png"
        frame.save(png_path)
        start = max(0.0, float(segment["start"]))
        end = max(start + 0.05, float(segment["end"]))
        cues.append((start, end, png_path.name))

    if not cues:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(video_path), "-c", "copy", str(out_path)],
            capture_output=True, text=True,
        )
        return
    cues.sort()
    if total <= 0:
        total = cues[-1][1]

    # Hold each caption for at least MIN_CAPTION_SEC so short cues stay readable,
    # without overrunning the next cue (which would desync everything after it).
    held = []
    for i, (start, end, name) in enumerate(cues):
        next_start = cues[i + 1][0] if i + 1 < len(cues) else total
        end = max(end, start + MIN_CAPTION_SEC)
        if next_start > start:
            end = min(end, next_start)
        end = max(end, start + 0.05)
        held.append((start, end, name))
    cues = held

    # One transparent overlay track via the concat demuxer: gap frames fill the
    # silence between cues. This is a SINGLE overlay pass (fast) instead of one
    # looped-image input + overlay per cue (which was pathologically slow).
    entries, t = [], 0.0  # (png_name, duration_sec)
    for start, end, name in cues:
        if start > t + 1e-3:
            entries.append((gap_png.name, start - t))
        s2 = max(start, t)
        e2 = max(end, s2 + 0.05)
        entries.append((name, e2 - s2))
        t = e2
    if total > t + 1e-3:
        entries.append((gap_png.name, total - t))

    lines = ["ffconcat version 1.0"]
    for name, dur in entries:
        lines.append(f"file '{name}'")
        lines.append(f"duration {dur:.3f}")
    lines.append(f"file '{entries[-1][0]}'")  # concat honours the last duration only if repeated
    concat_path = png_dir / "_concat.txt"
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-f", "concat", "-safe", "0", "-i", concat_path.name,
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto:shortest=1[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        # Overlaying RGBA captions promotes the format to yuv444p, which many
        # players (e.g. Windows Media Player: 0x80004005) refuse. Force the
        # widely-compatible 4:2:0 chroma on the final encode.
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(png_dir))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg caption overlay failed: {result.stderr.strip()}")


def run(
    artifact_dir = None,
    video = None,
    out_root = DEFAULT_ARTIFACT_ROOT,
    include_speaker = True,
    include_emotion = True,
    hide_unknown_emotion= True,
    render_video = True,
    ffmpeg_bin = "ffmpeg",
    position = DEFAULT_POSITION,
    emotion_colors = None,
    overwrite = False,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)

    if not paths["final_caption_data"].exists():
        raise FileNotFoundError(
            f"Final caption data not found: {paths['final_caption_data']}. Run apply_speaker_names.py first."
        )
    if paths["subtitles_srt"].exists() and not overwrite and not render_video:
        return {"artifact_dir": artifact_dir, "paths": paths, "rendered_video": False, "skipped": True}

    payload = read_json(paths["final_caption_data"])
    segments = payload.get("segments", [])
    if not segments:
        raise ValueError("No caption segments found in final_caption_data.json")

    srt_text = build_srt(
        segments=segments,
        include_speaker=include_speaker,
        include_emotion=include_emotion,
        hide_unknown_emotion=hide_unknown_emotion,
    )
    paths["subtitles_srt"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["subtitles_srt"], "w", encoding="utf-8") as f:
        f.write(srt_text)

    rendered = False
    if render_video:
        meta = read_json(paths["meta"]) if paths["meta"].exists() else {}
        video_path = video or Path(meta.get("source_video", ""))
        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(
                "Need a source video to burn subtitles. Pass video or make sure meta.json has source_video."
            )
        # All styling -- Korean text, emotion colour, colour emoji, translucent
        # box, position -- is drawn by Pillow and overlaid onto the video.
        # (ffmpeg text filters can't render colour emoji.)
        burn_pil_overlays(
            Path(video_path), segments, paths["rendered_video"], ffmpeg_bin,
            position, emotion_colors, include_speaker, include_emotion,
            hide_unknown_emotion,
        )
        rendered = True

    return {"artifact_dir": artifact_dir, "paths": paths, "rendered_video": rendered, "skipped": False}
