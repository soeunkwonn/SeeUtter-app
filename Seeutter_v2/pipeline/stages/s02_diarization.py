from __future__ import annotations
import os
import subprocess
from pathlib import Path
from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    DIARIZEN_PYTHON,
    read_json,
    resolve_artifact_dir,
    standard_paths,
)

# 화자분리 엔진: DiariZen base pretrained 단일 백엔드 (별도 conda env 서브프로세스).
DIARIZATION_BACKEND = "diarizen"

# Diarizen subprocess 환경변수 구성 
def build_diarizen_subprocess_env():
    env = os.environ.copy()
    env_root = Path(DIARIZEN_PYTHON).parent
    env_path_parts = [
        env_root,
        env_root / "Library" / "mingw-w64" / "bin",
        env_root / "Library" / "usr" / "bin",
        env_root / "Library" / "bin",
        env_root / "Scripts",
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(str(path) for path in env_path_parts if path.exists())
    if existing_path:
        env["PATH"] += os.pathsep + existing_path
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    zen_cert = next(
        (
            candidate
            for candidate in (
                env_root / "Library" / "ssl" / "cacert.pem",
                env_root / "lib" / "site-packages" / "certifi" / "cacert.pem",
            )
            if candidate.exists()
        ),
        None,
    )
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        current = env.get(var)
        if zen_cert is not None:
            env[var] = str(zen_cert)
        elif current and not Path(current).exists():
            env.pop(var, None)
    return env

# diarizen 실행 
def run_diarizen(
    artifact_dir=None,
    video=None,
    out_root=DEFAULT_ARTIFACT_ROOT,
    model="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    hf_token=None,
    device=None,
    download_root=None,
    diarizen_hub=None,
    embedding_model=None,
    seg_duration=16,
    segmentation_step=0.1,
    batch_size=32,
    apply_median_filtering=True,
    clustering_method="VBxClustering",
    ahc_criterion="distance",
    ahc_threshold=0.6,
    min_cluster_size=13,
    fa=0.07,
    fb=0.8,
    lda_dim=128,
    max_iters=20,
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
    overwrite=False,
    **_ignored_pyannote_only_kwargs,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    if not paths["audio"].exists():
        raise FileNotFoundError(
            f"Prepared audio not found: {paths['audio']}. Run prepare_media.py first."
        )
    if (
        paths["diarization_json"].exists()
        and paths["diarization_rttm"].exists()
        and not overwrite
    ):
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["diarization_json"]),
            "skipped": True,
        }

    diarizen_python = Path(DIARIZEN_PYTHON)
    if not diarizen_python.exists():
        raise FileNotFoundError(f"DiariZen Python not found: {diarizen_python}")

    script_path = Path(__file__).resolve().parent / "s02_diarization_diarizen.py"
    cmd = [
        str(diarizen_python), str(script_path),
        "--artifact-dir", str(artifact_dir),
        "--backend", "pretrained",
        "--model", model,
    ]
    if hf_token:
        cmd += ["--hf-token", hf_token]
    if device:
        cmd += ["--device", device]
    if download_root is not None:
        cmd += ["--download-root", str(download_root)]
    if diarizen_hub is not None:
        cmd += ["--diarizen-hub", str(diarizen_hub)]
    if embedding_model is not None:
        cmd += ["--embedding-model", str(embedding_model)]
    cmd += [
        "--seg-duration", str(seg_duration),
        "--segmentation-step", str(segmentation_step),
        "--batch-size", str(batch_size),
        "--clustering-method", clustering_method,
        "--ahc-criterion", ahc_criterion,
        "--ahc-threshold", str(ahc_threshold),
        "--min-cluster-size", str(min_cluster_size),
        "--fa", str(fa),
        "--fb", str(fb),
        "--lda-dim", str(lda_dim),
        "--max-iters", str(max_iters),
    ]
    cmd.append("--apply-median-filtering" if apply_median_filtering else "--no-apply-median-filtering")
    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    if min_speakers is not None:
        cmd += ["--min-speakers", str(min_speakers)]
    if max_speakers is not None:
        cmd += ["--max-speakers", str(max_speakers)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=build_diarizen_subprocess_env(),
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = "\n".join(
            part
            for part in (
                f"stderr:\n{stderr}" if stderr else "",
                f"stdout:\n{stdout}" if stdout else "",
            )
            if part
        )
        raise RuntimeError(
            f"DiariZen subprocess failed with exit code {result.returncode}:\n"
            f"{details or '(no output)'}"
        )

    if not paths["diarization_json"].exists():
        raise RuntimeError(
            f"DiariZen subprocess exited cleanly but wrote no {paths['diarization_json']}:\n"
            f"{result.stdout.strip()}"
        )

    payload = read_json(paths["diarization_json"])
    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload, "skipped": False}


def run(*args, backend=None, **kwargs):
    resolved_backend = backend or DIARIZATION_BACKEND
    if resolved_backend not in {"diarizen", "diarizen_pretrained"}:
        raise ValueError(
            f"Unsupported diarization backend: {resolved_backend}. "
            "Only DiariZen base pretrained ('diarizen') is available."
        )
    return run_diarizen(*args, **kwargs)
