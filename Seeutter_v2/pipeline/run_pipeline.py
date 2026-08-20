

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
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
    from Seeutter_v2.pipeline.stages.s01_asr import run as run_asr_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s02_diarization import run as run_diarization_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s03_visual_reconcile import run as run_visual_reconcile_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s04_alignment import run as run_align_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s06_speaker_correction import run as run_speaker_correction_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s05_speaker_boundary_gap import run as run_speaker_boundary_gap_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s07_speaker_summary import run as run_speaker_summary_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s09_emotion import run as run_emotion_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s08_speaker_names import run as run_apply_speaker_names_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s11_translate import run as run_translate_step  # type: ignore
    from Seeutter_v2.pipeline.stages.s10_render import run as run_render_subtitles_step  # type: ignore
else:
    from .common import (
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
    from .stages.s01_asr import run as run_asr_step
    from .stages.s02_diarization import run as run_diarization_step
    from .stages.s03_visual_reconcile import run as run_visual_reconcile_step
    from .stages.s04_alignment import run as run_align_step
    from .stages.s06_speaker_correction import run as run_speaker_correction_step
    from .stages.s05_speaker_boundary_gap import run as run_speaker_boundary_gap_step
    from .stages.s07_speaker_summary import run as run_speaker_summary_step
    from .stages.s09_emotion import run as run_emotion_step
    from .stages.s08_speaker_names import run as run_apply_speaker_names_step
    from .stages.s11_translate import run as run_translate_step
    from .stages.s10_render import run as run_render_subtitles_step

load_env_file()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Source video file.")
    parser.add_argument("--audio", type=Path, default=None, help="Source audio file.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Artifact directory. Required when no --video/--audio is provided.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--media-id",
        default=None,
        help="Logical item id stored in meta.json. Defaults to source stem.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)

    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    parser.add_argument("--diarization-model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    parser.add_argument("--diarization-device", default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)

    parser.add_argument("--reconcile-face-dist", type=float, default=0.7)
    parser.add_argument("--reconcile-conf-min", type=float, default=0.2)
    parser.add_argument("--boundary-gap-glue-sec", type=float, default=0.10)
    parser.add_argument("--boundary-gap-pause-sec", type=float, default=0.30)

    parser.add_argument("--asr-model", default="large")
    parser.add_argument("--asr-device", default=None)
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--asr-task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--word-timestamps", action="store_true", default=True)
    parser.add_argument("--no-word-timestamps", dest="word_timestamps", action="store_false")

    parser.add_argument("--align-granularity", choices=["auto", "word", "segment"], default="auto")
    parser.add_argument("--merge-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-nearest-gap-sec", type=float, default=1.0)
    parser.add_argument("--unknown-speaker", default="UNKNOWN")
    parser.add_argument("--merge-across-asr-segments", action="store_true")

    parser.add_argument("--speaker-correction-model", default=None)
    parser.add_argument(
        "--speaker-correction-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
    )
    parser.add_argument(
        "--speaker-correction-env-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".env",
    )
    parser.add_argument("--speaker-correction-max-run-sec", type=float, default=3.0)
    parser.add_argument("--speaker-correction-run-merge-gap-sec", type=float, default=1.5)
    parser.add_argument("--speaker-correction-max-boundary-gap-sec", type=float, default=1.5)
    parser.add_argument(
        "--no-speaker-correction-verification",
        dest="speaker_correction_verification",
        action="store_false",
    )

    parser.add_argument("--emotion-backend", choices=["stub", "sensevoice"], default="stub")
    parser.add_argument("--emotion-model", default="iic/SenseVoiceSmall")
    parser.add_argument(
        "--emotion-remote-code",
        type=Path,
        default=SENSEVOICE_REMOTE_CODE,
    )
    parser.add_argument("--emotion-language", default="auto")
    parser.add_argument("--emotion-pad-end-sec", type=float, default=0.2)
    parser.add_argument("--emotion-min-duration-sec", type=float, default=0.1)
    parser.add_argument("--keep-emotion-clips", action="store_true")

    parser.add_argument("--speaker-summary-top-k", type=int, default=5)
    parser.add_argument("--speaker-map", type=Path, default=None)
    parser.add_argument("--set", dest="inline_sets", action="append", default=[])

    parser.add_argument("--include-speaker", action="store_true", default=True)
    parser.add_argument("--no-include-speaker", dest="include_speaker", action="store_false")
    parser.add_argument("--include-emotion", action="store_true", default=True)
    parser.add_argument("--no-include-emotion", dest="include_emotion", action="store_false")
    parser.add_argument("--hide-unknown-emotion", action="store_true", default=True)
    parser.add_argument("--no-hide-unknown-emotion", dest="hide_unknown_emotion", action="store_false")
    parser.add_argument("--render-video", action="store_true")

    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--skip-visual-reconcile", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-speaker-correction", action="store_true")
    parser.add_argument("--skip-boundary-gap", action="store_true")
    parser.add_argument("--skip-speaker-summary", action="store_true")
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument("--skip-apply-speaker-names", action="store_true")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(speaker_correction_verification=True)
    return parser.parse_args()

# 오디오 메타데이터 
def infer_processed_audio_context(audio_path):
    try:
        parts = list(audio_path.resolve().parts)
    except FileNotFoundError:
        parts = list(audio_path.parts)

    context: dict[str, str] = {}
    if "audio" not in parts:
        return context

    audio_index = parts.index("audio")
    if audio_index + 2 >= len(parts):
        return context

    processed_root = Path(*parts[:audio_index])
    split = parts[audio_index + 1]
    stem = audio_path.stem
    meta_path = processed_root / "meta" / split / f"{stem}.json"
    rttm_path = processed_root / "rttm" / split / f"{stem}.rttm"

    context["dataset_split"] = split
    context["source_audio_original"] = str(audio_path.resolve())
    if meta_path.exists():
        context["source_dataset_meta"] = str(meta_path.resolve())
    if rttm_path.exists():
        context["reference_rttm"] = str(rttm_path.resolve())
    return context


def resolve_artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return args.artifact_dir
    if args.video is not None:
        return args.out_root / args.video.stem
    if args.audio is not None:
        context = infer_processed_audio_context(args.audio)
        split = context.get("dataset_split")
        if split:
            return args.out_root / split / args.audio.stem
        return args.out_root / args.audio.stem
    raise ValueError("Need --artifact-dir or a source via --video/--audio.")

# 입력오디오를 ffmpeg로 표준 포맷으로 변환
def canonicalize_audio(
    audio_path,
    out_path,
    ffmpeg_bin,
    sample_rate,
    channels,
    overwrite=False,
):
    if out_path.exists() and not overwrite:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(out_path),
        "-loglevel",
        "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio canonicalization failed: {result.stderr.strip()}")


def prepare_artifact(args: argparse.Namespace, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = standard_paths(artifact_dir)

    if args.video is not None:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        extract_audio(
            video_path=args.video,
            audio_path=paths["audio"],
            ffmpeg_bin=args.ffmpeg_bin,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
        )
        media_id = args.media_id or args.video.stem
        meta = {
            "video_id": media_id,
            "source_kind": "video",
            "source_video": str(args.video.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "audio_path": str(paths["audio"].resolve()),
            "audio": probe_wav(paths["audio"]),
        }
        write_json(paths["meta"], meta)
        return

    if args.audio is not None:
        if not args.audio.exists():
            raise FileNotFoundError(f"Audio not found: {args.audio}")
        canonicalize_audio(
            audio_path=args.audio,
            out_path=paths["audio"],
            ffmpeg_bin=args.ffmpeg_bin,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=args.overwrite,
        )
        media_id = args.media_id or args.audio.stem
        meta = {
            "video_id": media_id,
            "source_kind": "audio",
            "artifact_dir": str(artifact_dir.resolve()),
            "audio_path": str(paths["audio"].resolve()),
            "audio": probe_wav(paths["audio"]),
        }
        meta.update(infer_processed_audio_context(args.audio))
        write_json(paths["meta"], meta)
        return

    if not paths["audio"].exists():
        raise FileNotFoundError(
            f"Prepared audio not found: {paths['audio']}. Pass --video/--audio or prepare the artifact first."
        )


def main():
    args = parse_args()
    if args.video is not None and args.audio is not None:
        raise ValueError("Use only one of --video or --audio.")

    artifact_dir = resolve_artifact_dir(args)
    prepare_artifact(args, artifact_dir)

    if not args.skip_asr:
        print("\n==> asr")
        run_asr_step(
            artifact_dir=artifact_dir,
            model=args.asr_model,
            device=args.asr_device,
            language=args.asr_language,
            task=args.asr_task,
            word_timestamps=args.word_timestamps,
            overwrite=args.overwrite,
        )

    if not args.skip_diarization:
        print("\n==> diarization")
        try:
            run_diarization_step(
                artifact_dir=artifact_dir,
                model=args.diarization_model,
                hf_token=args.hf_token,
                device=args.diarization_device,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if not args.skip_visual_reconcile:
        print("\n==> visual_reconcile")
        try:
            run_visual_reconcile_step(
                artifact_dir=artifact_dir,
                face_dist=args.reconcile_face_dist,
                conf_min=args.reconcile_conf_min,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 -- degrade to raw diarization
            print(f"visual_reconcile failed ({exc}); using raw diarization")

    if not args.skip_align:
        print("\n==> align")
        run_align_step(
            artifact_dir=artifact_dir,
            granularity=args.align_granularity,
            merge_gap_sec=args.merge_gap_sec,
            max_nearest_gap_sec=args.max_nearest_gap_sec,
            unknown_speaker=args.unknown_speaker,
            merge_across_asr_segments=args.merge_across_asr_segments,
            overwrite=args.overwrite,
        )

    if not args.skip_boundary_gap:
        print("\n==> speaker_boundary_gap")
        run_speaker_boundary_gap_step(
            artifact_dir=artifact_dir,
            glue_gap_sec=args.boundary_gap_glue_sec,
            pause_gap_sec=args.boundary_gap_pause_sec,
            overwrite=args.overwrite,
        )

    if not args.skip_speaker_correction:
        print("\n==> speaker_correction")
        run_speaker_correction_step(
            artifact_dir=artifact_dir,
            model=args.speaker_correction_model,
            reasoning_effort=args.speaker_correction_reasoning_effort,
            env_file=args.speaker_correction_env_file,
            max_candidate_duration_sec=args.speaker_correction_max_run_sec,
            run_merge_gap_sec=args.speaker_correction_run_merge_gap_sec,
            max_boundary_gap_sec=args.speaker_correction_max_boundary_gap_sec,
            verify_adjustments=args.speaker_correction_verification,
            overwrite=args.overwrite,
        )

    if not args.skip_speaker_summary:
        print("\n==> speaker_summary")
        run_speaker_summary_step(
            artifact_dir=artifact_dir,
            top_k=args.speaker_summary_top_k,
            overwrite=args.overwrite,
        )

    if not args.skip_emotion:
        print("\n==> emotion")
        run_emotion_step(
            artifact_dir=artifact_dir,
            backend=args.emotion_backend,
            model=args.emotion_model,
            remote_code=args.emotion_remote_code,
            language=args.emotion_language,
            pad_end_sec=args.emotion_pad_end_sec,
            min_duration_sec=args.emotion_min_duration_sec,
            keep_clips=args.keep_emotion_clips,
            overwrite=args.overwrite,
        )

    if not args.skip_apply_speaker_names:
        print("\n==> apply_speaker_names")
        run_apply_speaker_names_step(
            artifact_dir=artifact_dir,
            speaker_map=args.speaker_map,
            inline_sets=args.inline_sets,
            overwrite=args.overwrite,
        )

    if not args.skip_translate:
        print("\n==> translate")
        run_translate_step(
            artifact_dir=artifact_dir,
            overwrite=args.overwrite,
        )

    if not args.skip_render:
        print("\n==> render_subtitles")
        run_render_subtitles_step(
            artifact_dir=artifact_dir,
            include_speaker=args.include_speaker,
            include_emotion=args.include_emotion,
            hide_unknown_emotion=args.hide_unknown_emotion,
            render_video=args.render_video,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=args.overwrite,
        )

    paths = standard_paths(artifact_dir)
    print("\nPipeline complete")
    print(f"Artifact dir: {artifact_dir}")
    for key in (
        "meta",
        "asr",
        "diarization_json",
        "reconciled_diarization_json",
        "aligned_segments",
        "corrected_aligned_segments",
        "gap_corrected_aligned_segments",
        "speaker_correction",
        "speaker_summary",
        "emotion_segments",
        "final_caption_data",
        "subtitles_srt",
        "rendered_video",
    ):
        if paths[key].exists():
            print(f"{key}: {paths[key]}")


if __name__ == "__main__":
    main()
