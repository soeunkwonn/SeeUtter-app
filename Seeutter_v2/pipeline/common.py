import json
import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_ARTIFACT_ROOT = PROJECT_DIR / "artifacts"
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"
DIARIZEN_PYTHON = Path(
    os.environ.get("SEEUTTER_DIARIZEN_PYTHON", r"D:\conda\zen\python.exe")
)
SENSEVOICE_REMOTE_CODE = PROJECT_DIR / "vendor" / "sensevoice" / "model.py"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1



def load_env_file(path = DEFAULT_ENV_FILE, override = False):
    loaded: dict[str, str] = {}
    # if not path.exists():
    #     return loaded

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def resolve_artifact_dir(
    video_path,
    artifact_dir,
    out_root= DEFAULT_ARTIFACT_ROOT,
):
    if artifact_dir is not None:
        return artifact_dir
    if video_path is None:
        raise ValueError("Need either a video path or an explicit artifact directory.")
    return out_root / video_path.stem


def standard_paths(artifact_dir: Path):
    return {
        "artifact_dir": artifact_dir,
        "audio": artifact_dir / "audio.wav",
        "meta": artifact_dir / "meta.json",
        "asr": artifact_dir / "asr.json",
        "diarization_json": artifact_dir / "diarization.json",
        "reconciled_diarization_json": artifact_dir / "diarization_reconciled.json",
        "visual_reconcile_json": artifact_dir / "visual_reconcile.json",
        "diarization_rttm": artifact_dir / "diarization.rttm",
        "diarization_exclusive_rttm": artifact_dir / "diarization_exclusive.rttm",
        "aligned_segments": artifact_dir / "aligned_segments.json",
        # Boundary repair runs first and normalizes near-zero-gap fragments.
        "gap_corrected_aligned_segments": artifact_dir / "aligned_segments_gap_corrected.json",
        "speaker_boundary_gap": artifact_dir / "speaker_boundary_gap.json",
        # Context repair consumes the boundary-normalized copy.
        "corrected_aligned_segments": artifact_dir / "aligned_segments_corrected.json",
        "speaker_correction": artifact_dir / "speaker_correction.json",
        # Single downstream contract: UI, naming, emotion and rendering prefer this.
        "final_aligned_segments": artifact_dir / "aligned_segments_final.json",
        "speaker_summary": artifact_dir / "speaker_summary.json",
        "speaker_map": artifact_dir / "speaker_map.json",
        "speaker_table_json": artifact_dir / "speaker_table.json",
        "speaker_table_csv": artifact_dir / "speaker_table.csv",
        "emotion_segments": artifact_dir / "emotion_segments.json",
        "final_caption_data": artifact_dir / "final_caption_data.json",
        "subtitles_srt": artifact_dir / "subtitles.srt",
        "subtitles_ass": artifact_dir / "subtitles.ass",
        "rendered_video": artifact_dir / "rendered_subtitles.mp4",
        "emotion_clips_dir": artifact_dir / "_emotion_clips",
    }


def preferred_diarization_path(paths):
    reconciled = paths.get("reconciled_diarization_json")
    if reconciled is not None and reconciled.exists():
        return reconciled
    return paths["diarization_json"]


def preferred_aligned_segments_path(paths):
    for key in (
        "final_aligned_segments",
        "gap_corrected_aligned_segments",
        "corrected_aligned_segments",
    ):
        candidate = paths.get(key)
        if candidate is not None and candidate.exists():
            return candidate
    return paths["aligned_segments"]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def extract_audio(
    video_path,
    audio_path,
    ffmpeg_bin = "ffmpeg",
    sample_rate = 16000,
    channels = 1,
    overwrite = False,
):
    if audio_path.exists() and not overwrite:
        return

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(audio_path),
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}")


def probe_wav(audio_path: Path) -> dict[str, float | int | str]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": float(info.duration),
        "frames": int(info.frames),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }
