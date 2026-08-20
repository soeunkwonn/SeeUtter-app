
from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from Seeutter_v2.pipeline.common import (  # type: ignore
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_CHANNELS,
        DEFAULT_SAMPLE_RATE,
        extract_audio,
        probe_wav,
        resolve_artifact_dir,
        standard_paths,
        write_json,
    )
else:
    from .common import (
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_CHANNELS,
        DEFAULT_SAMPLE_RATE,
        extract_audio,
        probe_wav,
        resolve_artifact_dir,
        standard_paths,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Explicit artifact directory. Defaults to artifacts/<video_stem>/.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Parent directory for automatically created artifact dirs.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    artifact_dir = resolve_artifact_dir(args.video, args.artifact_dir, args.out_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = standard_paths(artifact_dir)

    extract_audio(
        video_path=args.video,
        audio_path=paths["audio"],
        ffmpeg_bin=args.ffmpeg_bin,
        sample_rate=args.sample_rate,
        channels=args.channels,
        overwrite=args.overwrite,
    )

    audio_info = probe_wav(paths["audio"])
    meta = {
        "video_id": args.video.stem,
        "source_video": str(args.video.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "audio_path": str(paths["audio"].resolve()),
        "audio": audio_info,
    }
    write_json(paths["meta"], meta)

    print(f"Artifact dir: {artifact_dir}")
    print(f"Audio: {paths['audio']}")
    print(
        f"Audio info: {audio_info['sample_rate']} Hz, "
        f"{audio_info['channels']} ch, {audio_info['duration_sec']:.2f} sec"
    )
    print(f"Meta: {paths['meta']}")


if __name__ == "__main__":
    main()
