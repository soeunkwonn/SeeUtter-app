"""Prepare videos for the SeeUtter Library.

This is the offline/admin half of the product workflow. It performs every
expensive analysis step through SenseVoice emotion detection, but deliberately
stops before a user supplies speaker names, chooses styles, or renders video.

Example (from ``D:\\seeutter``)::

    D:\\conda\\app\\python.exe -m Seeutter_v2.pipeline.prepare_library \
        --input-dir D:\\samples
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Seeutter_v2.pipeline.common import (  # type: ignore
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    SENSEVOICE_REMOTE_CODE,
    extract_audio,
    load_env_file,
    probe_wav,
    standard_paths,
    write_json,
)
from Seeutter_v2.pipeline.stages.s01_asr import run as run_asr  # type: ignore
from Seeutter_v2.pipeline.stages.s02_diarization import run as run_diarization  # type: ignore
from Seeutter_v2.pipeline.stages.s03_visual_reconcile import run as run_visual_reconcile  # type: ignore
from Seeutter_v2.pipeline.stages.s04_alignment import run as run_alignment  # type: ignore
from Seeutter_v2.pipeline.stages.s05_speaker_boundary_gap import run as run_boundary_gap  # type: ignore
from Seeutter_v2.pipeline.stages.s06_speaker_correction import run as run_speaker_correction  # type: ignore
from Seeutter_v2.pipeline.stages.s07_speaker_summary import run as run_speaker_summary  # type: ignore
from Seeutter_v2.pipeline.stages.s09_emotion import run as run_emotion  # type: ignore


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
READY_FILE_NAME = "library_ready.json"


# Single diarization backend: DiariZen base pretrained. Both the segmentation
# model and the speaker-embedding model are pulled from the HuggingFace hub and
# cached locally on first run -- nothing is vendored in-tree. (The subprocess in
# s02_diarization_diarizen.py downloads them when no local path is supplied.)
DIARIZATION_KWARGS = {
    "backend": "diarizen",
    "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    "ahc_threshold": 0.6,
    "fa": 0.07,
    "fb": 0.07,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--video",
        type=Path,
        action="append",
        help="One source video. Repeat the option to prepare several videos.",
    )
    sources.add_argument("--input-dir", type=Path, help="Directory of source videos.")
    parser.add_argument(
        "--glob",
        default="*.mp4",
        help="Filename glob used with --input-dir (default: *.mp4).",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Explicit output directory; valid only when preparing one video.",
    )
    parser.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--asr-model", default="large")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    parser.add_argument("--emotion-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--emotion-remote-code", type=Path, default=SENSEVOICE_REMOTE_CODE)
    parser.add_argument("--emotion-language", default="auto")
    parser.add_argument("--speaker-summary-top-k", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    if args.video:
        videos = [path.resolve() for path in args.video]
    else:
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        videos = sorted(path.resolve() for path in args.input_dir.glob(args.glob) if path.is_file())
    if not videos:
        raise FileNotFoundError("No input videos matched the requested source.")
    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError("Video not found: " + ", ".join(missing))
    if args.artifact_dir is not None and len(videos) != 1:
        raise ValueError("--artifact-dir can only be used with one --video.")
    return videos


def prepare_media(video_path: Path, artifact_dir: Path, args: argparse.Namespace) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = standard_paths(artifact_dir)
    extract_audio(
        video_path=video_path,
        audio_path=paths["audio"],
        ffmpeg_bin=args.ffmpeg_bin,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        overwrite=args.overwrite,
    )
    write_json(
        paths["meta"],
        {
            "video_id": artifact_dir.name,
            "source_video": str(video_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "audio_path": str(paths["audio"].resolve()),
            "diarization_model": DIARIZATION_KWARGS["model"],
            "audio": probe_wav(paths["audio"]),
        },
    )


def prepare_library_item(video_path: Path, artifact_dir: Path, args: argparse.Namespace) -> Path:
    prepare_media(video_path, artifact_dir, args)
    run_asr(artifact_dir=artifact_dir, model=args.asr_model, overwrite=args.overwrite)
    run_diarization(
        artifact_dir=artifact_dir,
        hf_token=args.hf_token,
        overwrite=args.overwrite,
        **DIARIZATION_KWARGS,
    )
    try:
        run_visual_reconcile(artifact_dir=artifact_dir, overwrite=args.overwrite)
    except Exception as exc:  # noqa: BLE001 -- library preparation may fall back to audio labels
        print(f"[{artifact_dir.name}] visual_reconcile failed ({exc}); using raw diarization")
    run_alignment(artifact_dir=artifact_dir, overwrite=args.overwrite)
    run_boundary_gap(artifact_dir=artifact_dir, overwrite=args.overwrite)
    run_speaker_correction(artifact_dir=artifact_dir, overwrite=args.overwrite)
    run_speaker_summary(
        artifact_dir=artifact_dir,
        top_k=args.speaker_summary_top_k,
        overwrite=args.overwrite,
    )
    run_emotion(
        artifact_dir=artifact_dir,
        backend="sensevoice",
        model=args.emotion_model,
        remote_code=args.emotion_remote_code.resolve(),
        language=args.emotion_language,
        overwrite=args.overwrite,
    )

    paths = standard_paths(artifact_dir)
    write_json(
        artifact_dir / READY_FILE_NAME,
        {
            "video_id": artifact_dir.name,
            "source_video": str(video_path.resolve()),
            "required_files": [
                paths["meta"].name,
                paths["speaker_table_json"].name,
                paths["emotion_segments"].name,
            ],
            "status": "ready",
        },
    )
    return artifact_dir


def main() -> None:
    load_env_file()
    args = parse_args()
    videos = resolve_videos(args)
    for video_path in videos:
        artifact_dir = args.artifact_dir or (args.out_root / video_path.stem)
        print(f"\n==> Preparing Library item: {video_path.name}")
        prepared = prepare_library_item(video_path, artifact_dir, args)
        print(f"Library ready: {prepared}")


if __name__ == "__main__":
    main()
