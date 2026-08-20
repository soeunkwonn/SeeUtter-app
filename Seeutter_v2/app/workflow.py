"""Shared state and pipeline helpers for the Streamlit library wizard.

The browser session only holds the selected artifact and navigation state. User
choices are persisted alongside the artifact, so refreshing the page or moving
back and forth never loses a speaker map or caption style.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from Seeutter_v2.pipeline.common import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    read_json,
    standard_paths,
    write_json,
)
from Seeutter_v2.pipeline.stages.s07_speaker_summary import run as run_speaker_summary  # noqa: E402
from Seeutter_v2.pipeline.stages.s10_render import (  # noqa: E402
    DEFAULT_POSITION,
    POSITION_ALIGNMENT,
    caption_units,
    run as run_render,
)
from subtitle_position import POSITION_PRESETS, build_position_previews  # noqa: E402


# Master artifacts (read-only sources the pipeline produced) live here. Each
# experiment participant works on a private copy under ``runs/<pid>`` so that
# concurrent naming/render never overwrites another participant's results.
MASTER_ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT
RUNS_ROOT = PROJECT_DIR / "runs"

# Only these artifacts are exposed to participants and seeded into their run.
EXPERIMENT_ARTIFACTS = ("N1", "N2")

# Files/dirs a participant regenerates while naming, positioning and rendering.
# They are excluded from the seed copy so every participant starts from a clean
# "not yet named" state, and the heavy unused ``_stimuli`` clips are skipped too.
_SEED_IGNORE = shutil.ignore_patterns(
    "_stimuli",
    "_caption_pngs",
    "_ui_position_previews",
    "_speaker_preview_images",
    "rendered_subtitles.mp4",
    "subtitles.srt",
    "subtitles.ass",
    "speaker_map.json",
    "caption_style.json",
    "translation.json",
)

_PID_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")

# Serialize ffmpeg renders across sessions. On Streamlit Community Cloud every
# browser session shares one ~1GB container, so several concurrent renders would
# exhaust memory; this lock makes at most one render run at a time (a waiting
# participant simply queues for a minute). Harmless on a roomy local machine.
RENDER_LOCK = threading.Lock()

TOP_K_SPEAKERS = 5
STYLE_FILE_NAME = "caption_style.json"
READY_FILE_NAME = "library_ready.json"
PREVIEW_RENDER_VERSION = "7"  # bump when caption_pil rendering changes


def _sanitize_pid(raw: Any) -> str | None:
    """Reduce a URL ``pid`` to a safe folder name (never escapes ``runs/``)."""
    cleaned = _PID_UNSAFE.sub("", str(raw or "")).strip("-_")[:64]
    return cleaned or None


def _ensure_participant_seeded(root: Path) -> None:
    """Copy the clean N1/N2 artifacts into a participant's run on first access."""
    for name in EXPERIMENT_ARTIFACTS:
        source = MASTER_ARTIFACT_ROOT / name
        dest = root / name
        if dest.exists() or not source.is_dir():
            continue
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".{name}.seeding"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            shutil.copytree(source, staging, ignore=_SEED_IGNORE)
            staging.rename(dest)
        except OSError:
            # A concurrent rerun for the same participant may have won the race;
            # keep the finished copy and drop our half-built staging directory.
            shutil.rmtree(staging, ignore_errors=True)
            if not dest.exists():
                raise


def artifact_root() -> Path:
    """Return the artifact root for the current browser session.

    With a ``?pid=`` in the URL, results are isolated under ``runs/<pid>``. With
    no pid (the researcher's own local session) the master ``artifacts`` folder
    is used directly, exactly as before.
    """
    pid = st.session_state.get("participant_id")
    if not pid:
        return MASTER_ARTIFACT_ROOT
    root = RUNS_ROOT / pid
    _ensure_participant_seeded(root)
    return root


def resolve_source_video(meta: dict[str, Any]) -> Path:
    """Locate the source video, tolerating a moved deployment.

    Locally, ``meta['source_video']`` is the absolute path recorded by the
    pipeline. On a cloud deploy that machine-specific path no longer exists, so
    we fall back to a same-named file under the repo's ``videos/`` folder.
    """
    raw_str = str(meta.get("source_video") or "")
    raw = Path(raw_str)
    if raw.exists():
        return raw
    # The recorded path may be a Windows path being read on Linux (cloud), where
    # a backslash is not a separator; take the basename in a way that handles both.
    name = PureWindowsPath(raw_str).name
    return PROJECT_DIR / "videos" / name


def _default_ffmpeg_bin() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    # ``python -m streamlit`` is commonly launched without conda activation.
    # Prefer the ffmpeg shipped next to the current conda interpreter.
    conda_ffmpeg = Path(sys.executable).resolve().parent / "Library" / "bin" / "ffmpeg.exe"
    if conda_ffmpeg.exists():
        return str(conda_ffmpeg)
    return "ffmpeg"


FFMPEG_BIN = _default_ffmpeg_bin()


def _sync_participant_id() -> None:
    """Resolve the participant id from the URL, keeping it sticky.

    A page refresh starts a fresh Streamlit session, so the ``?pid=`` in the URL
    is the source of truth. When present we store it; when a later navigation
    drops it from the query string we write it back so the URL stays shareable.
    """
    params = st.query_params
    pid = _sanitize_pid(params.get("pid"))
    if pid:
        st.session_state["participant_id"] = pid
        if params.get("pid") != pid:
            params["pid"] = pid
    else:
        remembered = st.session_state.get("participant_id")
        if remembered:
            params["pid"] = remembered


def _sync_secret_env() -> None:
    """Expose cloud secrets as env vars the pipeline already reads.

    Locally the key comes from ``.env``; on Streamlit Community Cloud there is no
    ``.env``, so mirror ``OPENAI_API_KEY`` from st.secrets into the environment
    (translation reads it from there and never touches the missing file).
    """
    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return
    if key:
        os.environ["OPENAI_API_KEY"] = str(key)


def init_session() -> None:
    """Set the navigation-only state shared by every Streamlit page."""
    _sync_secret_env()
    _sync_participant_id()
    defaults = {
        "library_artifact_id": None,
        "wizard_step": "library",
        "preview_segment_index": 0,
        "rendered_in_session": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _artifact_dir(artifact_id: str) -> Path:
    """Resolve one direct child of ``artifacts`` without accepting traversal."""
    root = artifact_root().resolve()
    candidate = (root / artifact_id).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise ValueError("Unknown library item.")
    return candidate


def library_items() -> list[dict[str, Any]]:
    """Return only artifacts that have completed emotion analysis."""
    root = artifact_root()
    if not root.exists():
        return []

    items: list[dict[str, Any]] = []
    for artifact_dir in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not artifact_dir.is_dir():
            continue
        paths = standard_paths(artifact_dir)
        if not (
            paths["meta"].exists()
            and paths["speaker_table_json"].exists()
            and paths["emotion_segments"].exists()
        ):
            continue
        try:
            meta = read_json(paths["meta"])
            emotion = read_json(paths["emotion_segments"])
        except (OSError, ValueError):
            continue
        source_video = resolve_source_video(meta)
        if not source_video.exists():
            continue
        items.append(
            {
                "id": artifact_dir.name,
                "title": str(meta.get("title") or meta.get("video_id") or artifact_dir.name),
                "source_video": source_video,
                "duration_sec": float((meta.get("audio") or {}).get("duration_sec") or 0),
                "num_segments": int(emotion.get("num_segments") or len(emotion.get("segments") or [])),
                "ready_marker": (artifact_dir / READY_FILE_NAME).exists(),
            }
        )
    return items


def current_participant() -> str | None:
    """Return the id typed by the participant on the entry screen, if any."""
    return st.session_state.get("participant_id")


def set_participant(pid: str) -> bool:
    """Store a participant id (session + URL). False if the id is unusable."""
    clean = _sanitize_pid(pid)
    if not clean:
        return False
    st.session_state["participant_id"] = clean
    st.query_params["pid"] = clean
    return True


def clear_participant() -> None:
    """Hand off to the next participant: forget who this session was."""
    st.session_state.pop("participant_id", None)
    clear_selection()
    st.query_params.clear()


def select_library_item(artifact_id: str) -> Path:
    artifact_dir = _artifact_dir(artifact_id)
    st.session_state.library_artifact_id = artifact_id
    st.session_state.wizard_step = "names"
    st.session_state.preview_segment_index = 0
    st.session_state.rendered_in_session = False
    return artifact_dir


def current_artifact() -> Path | None:
    artifact_id = st.session_state.get("library_artifact_id")
    if not artifact_id:
        return None
    try:
        return _artifact_dir(str(artifact_id))
    except ValueError:
        return None


def clear_selection() -> None:
    st.session_state.library_artifact_id = None
    st.session_state.wizard_step = "library"
    st.session_state.preview_segment_index = 0
    st.session_state.rendered_in_session = False


def require_current_artifact() -> Path:
    artifact_dir = current_artifact()
    if artifact_dir is None:
        st.warning("먼저 영상을 선택해주세요.")
        # switch_page only accepts the main file or pages/*, so route to the
        # entrypoint, which renders the default (home) page.
        st.switch_page("app.py")
        st.stop()
    return artifact_dir


def style_path(artifact_dir: Path) -> Path:
    return artifact_dir / STYLE_FILE_NAME


def default_caption_style() -> dict[str, Any]:
    return {
        "version": 2,
        "position": DEFAULT_POSITION,
    }


def load_caption_style(artifact_dir: Path) -> dict[str, Any]:
    style = default_caption_style()
    path = style_path(artifact_dir)
    if not path.exists():
        return style
    try:
        saved = read_json(path)
    except (OSError, ValueError):
        return style

    position = saved.get("position")
    if position in POSITION_ALIGNMENT:
        style["position"] = position
    return style


def save_caption_style(artifact_dir: Path, style: dict[str, Any]) -> Path:
    normalized = default_caption_style()
    if style.get("position") in POSITION_ALIGNMENT:
        normalized["position"] = style["position"]
    write_json(style_path(artifact_dir), normalized)
    return style_path(artifact_dir)


def load_speaker_map(artifact_dir: Path) -> dict[str, str]:
    path = standard_paths(artifact_dir)["speaker_map"]
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    speakers = payload.get("speakers")
    return {str(key): str(value) for key, value in speakers.items()} if isinstance(speakers, dict) else {}


def _speaker_sort_key(speaker_id: str) -> tuple[int, int]:
    """Order speakers naturally: FACE_0, FACE_1, ... then any AUDIO_ ids."""
    prefix_rank = 0 if speaker_id.startswith("FACE_") else 1
    digits = "".join(ch for ch in speaker_id if ch.isdigit())
    return (prefix_rank, int(digits) if digits else 0)


def speaker_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    """Return the speakers shown in the naming UI.

    The nameable set is taken from ``emotion_segments.json``: its diarization may
    have been hand-corrected, so it -- not the raw aligned segments -- decides
    which faces a participant names.
    """
    paths = standard_paths(artifact_dir)
    run_speaker_summary(
        artifact_dir=artifact_dir,
        top_k=TOP_K_SPEAKERS,
        overwrite=True,
        write_map_template=False,
    )
    emotion = read_json(paths["emotion_segments"])
    nameable_speaker_ids = {
        str(segment.get("speaker", "UNKNOWN"))
        for segment in emotion.get("segments", [])
        # Raw diarization labels ("0", "1", ...) stay internal.  AUDIO_n
        # labels are created only after WeSpeaker could not link a clean
        # audio-only cluster to any face-backed identity.
        if str(segment.get("speaker", "UNKNOWN")).startswith(("FACE_", "AUDIO_"))
    }
    table = read_json(paths["speaker_table_json"])
    rows = [
        row
        for row in table.get("rows", [])
        if str(row.get("speaker_id")) in nameable_speaker_ids
    ]
    # Include any corrected speaker the aligned-based table no longer lists.
    present = {str(row.get("speaker_id")) for row in rows}
    rows.extend({"speaker_id": sid} for sid in nameable_speaker_ids - present)
    rows.sort(key=lambda row: _speaker_sort_key(str(row.get("speaker_id"))))
    return rows


def save_speaker_map(artifact_dir: Path, names: dict[str, str]) -> Path:
    paths = standard_paths(artifact_dir)
    table = read_json(paths["speaker_table_json"])
    cleaned = {
        str(speaker_id): str(name).strip() or str(speaker_id)
        for speaker_id, name in names.items()
    }
    write_json(
        paths["speaker_map"],
        {
            "video_id": table.get("video_id") or artifact_dir.name,
            "source_table": str(paths["speaker_table_json"].resolve()),
            "speakers": cleaned,
        },
    )
    return paths["speaker_map"]


def prepare_caption_data(artifact_dir: Path) -> None:
    """Apply the participant's names onto the pre-tuned captions, nothing else.

    The shipped ``final_caption_data.json`` already carries the researcher's final
    emotions and Korean translations, so naming must only relabel each speaker:
    we rewrite ``speaker_display`` in place and never re-run naming or translation.
    """
    paths = standard_paths(artifact_dir)
    names = load_speaker_map(artifact_dir)
    data = read_json(paths["final_caption_data"])
    for segment in data.get("segments", []):
        speaker_id = str(segment.get("speaker_id") or segment.get("speaker") or "")
        if speaker_id in names:
            segment["speaker_display"] = names[speaker_id]
    write_json(paths["final_caption_data"], data)


def render_caption_video(artifact_dir: Path) -> Path:
    style = load_caption_style(artifact_dir)
    # Pass the resolved video explicitly: on the cloud the path recorded in
    # meta.json (a Windows path) does not exist, so the render stage would
    # otherwise fail to find the source video.
    meta = read_json(standard_paths(artifact_dir)["meta"])
    result = run_render(
        artifact_dir=artifact_dir,
        video=resolve_source_video(meta),
        position=style["position"],
        # Caption text is intentionally fixed white. Emotion is still shown
        # through its emoji, not through a per-emotion text colour.
        emotion_colors=None,
        render_video=True,
        overwrite=True,
        ffmpeg_bin=FFMPEG_BIN,
    )
    return result["paths"]["rendered_video"]


def _capture_video_frame(video_path: Path, at_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{max(0.0, at_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-loglevel",
        "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")


def build_speaker_preview_images(artifact_dir: Path, speaker_ids: list[str]) -> dict[str, Path]:
    """Use one cached still image from each speaker's longest utterance.

    Globally reconciled ``FACE_n`` crops are preferred in the UI. These stills
    are the fallback for audio-only speakers, so the naming page never embeds
    a video player.
    """
    paths = standard_paths(artifact_dir)
    video_path = resolve_source_video(read_json(paths["meta"]))
    # Match the naming list: pick preview frames from the (corrected) emotion segments.
    segments = read_json(paths["emotion_segments"]).get("segments") or []
    best: dict[str, dict[str, Any]] = {}
    for segment in segments:
        speaker_id = str(segment.get("speaker"))
        if speaker_id not in speaker_ids:
            continue
        if speaker_id not in best or float(segment["duration"]) > float(best[speaker_id]["duration"]):
            best[speaker_id] = segment

    images_dir = artifact_dir / "_speaker_preview_images"
    output: dict[str, Path] = {}
    for speaker_id, segment in best.items():
        image_path = images_dir / f"{speaker_id}.jpg"
        if not image_path.exists() or image_path.stat().st_size == 0:
            try:
                _capture_video_frame(
                    video_path,
                    (float(segment["start"]) + float(segment["end"])) / 2,
                    image_path,
                )
            except RuntimeError:
                continue
        output[speaker_id] = image_path
    return output


def speaker_face_paths(artifact_dir: Path, speaker_ids: list[str]) -> dict[str, Path]:
    faces_dir = artifact_dir / "_reconcile_faces"
    return {
        speaker_id: faces_dir / f"{speaker_id}.jpg"
        for speaker_id in speaker_ids
        if speaker_id.startswith("FACE_") and (faces_dir / f"{speaker_id}.jpg").exists()
    }


def speaker_face_hints(artifact_dir: Path, speaker_ids: list[str]) -> dict[str, Path]:
    """Best-guess face crop for undetermined (audio-label) speakers.

    Visual reconcile records, for each speaker it could not confidently match,
    the face it overlapped most (``reconcile.audio_face_hints``). It is a hint,
    not a label -- the naming UI shows it so a human can confirm or rename.
    """
    paths = standard_paths(artifact_dir)
    rec_path = paths.get("reconciled_diarization_json")
    if not rec_path or not rec_path.exists():
        return {}
    hints = (read_json(rec_path).get("reconcile") or {}).get("audio_face_hints") or {}
    faces_dir = artifact_dir / "_reconcile_faces"
    output: dict[str, Path] = {}
    for speaker_id in speaker_ids:
        face_id = hints.get(speaker_id)
        if face_id is None:
            continue
        crop = faces_dir / f"FACE_{int(face_id)}.jpg"
        if crop.exists() and crop.stat().st_size > 0:
            output[speaker_id] = crop
    return output


def caption_segments(artifact_dir: Path) -> list[dict[str, Any]]:
    paths = standard_paths(artifact_dir)
    if not paths["final_caption_data"].exists():
        return []
    return read_json(paths["final_caption_data"]).get("segments") or []


def position_previews(artifact_dir: Path, segment_index: int) -> tuple[dict[str, Path], dict[str, Any]]:
    """Build one four-preset preview set from an actual translated caption."""
    paths = standard_paths(artifact_dir)
    segments = caption_segments(artifact_dir)
    if not segments:
        raise ValueError("Caption data is not ready. Complete the naming step first.")
    index = min(max(0, segment_index), len(segments) - 1)
    segment = segments[index]
    meta = read_json(paths["meta"])
    video_path = resolve_source_video(meta)
    units = caption_units(
        segment,
        include_speaker=True,
        include_emotion=True,
        hide_unknown_emotion=True,
    )
    fill_hex = "#FFFFFF"
    preview_dir = artifact_dir / "_ui_position_previews" / f"segment_{index:04d}"
    needed = [preview_dir / f"preview_{key}.png" for key in POSITION_PRESETS]
    version_path = preview_dir / "_renderer_version.txt"
    preview_is_current = (
        version_path.exists()
        and version_path.read_text(encoding="utf-8").strip() == PREVIEW_RENDER_VERSION
    )
    if not preview_is_current or not all(path.exists() and path.stat().st_size > 0 for path in needed):
        build_position_previews(
            video_path,
            preview_dir,
            at_sec=float(segment.get("start") or 0.0),
            ffmpeg_bin=FFMPEG_BIN,
            units=units,
            fill_hex=fill_hex,
        )
        version_path.write_text(PREVIEW_RENDER_VERSION, encoding="utf-8")
    return {key: preview_dir / f"preview_{key}.png" for key in POSITION_PRESETS}, segment
