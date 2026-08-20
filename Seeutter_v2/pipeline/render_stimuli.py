import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Seeutter_v2.pipeline.stages.s10_render import run as render_step  # type: ignore

# (id, label, include_speaker, include_emotion)
CONDITIONS = [
    (1, "spk+emo", True, True),
    (2, "spk", True, False),
    (3, "emo", False, True),
    (4, "asr", False, False),
]


def _default_ffmpeg() -> str:
    # Prefer the ffmpeg shipped next to the current conda interpreter.
    conda_ffmpeg = Path(sys.executable).resolve().parent / "Library" / "bin" / "ffmpeg.exe"
    return str(conda_ffmpeg) if conda_ffmpeg.exists() else "ffmpeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default=_default_ffmpeg())
    parser.add_argument("--out-subdir", default="_stimuli")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    out_dir = artifact_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for cond, label, include_speaker, include_emotion in CONDITIONS:
        print(f"\n==> cond{cond} ({label}): speaker={include_speaker} emotion={include_emotion}")
        result = render_step(
            artifact_dir=artifact_dir,
            include_speaker=include_speaker,
            include_emotion=include_emotion,
            render_video=True,
            ffmpeg_bin=args.ffmpeg_bin,
            overwrite=True,  # each condition re-renders the fixed output path
        )
        # s10 always burns to the fixed ``rendered_video`` path; copy it out
        # before the next condition overwrites it.
        rendered = Path(result["paths"]["rendered_video"])
        dst = out_dir / f"{artifact_dir.name}_cond{cond}_{label}.mp4"
        shutil.copy2(rendered, dst)
        outputs.append(dst)
        print(f"    -> {dst}")

    print(f"\nDone. {len(outputs)} stimuli in {out_dir}")
    for path in outputs:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
