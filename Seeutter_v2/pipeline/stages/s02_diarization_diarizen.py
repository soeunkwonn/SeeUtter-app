from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DEFAULT_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
PRETRAINED_BACKEND = "pretrained"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=[PRETRAINED_BACKEND], default=PRETRAINED_BACKEND)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--diarizen-hub", type=Path, default=None)
    parser.add_argument("--embedding-model", type=Path, default=None)
    # 추론 관련 옵션
    parser.add_argument("--seg-duration", type=int, default=16)
    parser.add_argument("--segmentation-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--apply-median-filtering", action=argparse.BooleanOptionalAction, default=True)
    # 클러스터링 관련 옵션 
    parser.add_argument("--clustering-method", choices=["VBxClustering", "AgglomerativeClustering"], default="VBxClustering")
    parser.add_argument("--ahc-criterion", default="distance")
    parser.add_argument("--ahc-threshold", type=float, default=0.6)
    parser.add_argument("--min-cluster-size", type=int, default=13)
    parser.add_argument("--fa", type=float, default=0.07)
    parser.add_argument("--fb", type=float, default=0.8)
    parser.add_argument("--lda-dim", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    return parser.parse_args()

# Hugging Face repo ID를 로컬 폴더명으로 변환 
def sanitize_repo_id(repo_id):
    return repo_id.replace("/", "--")

# speaker embedding 모델 경로를 결정
def resolve_embedding_model(args):
    if args.embedding_model is not None:
        # 로컬 모델 경로
        embedding_model = args.embedding_model.expanduser().resolve()
        if not embedding_model.exists():
            raise FileNotFoundError(f"Embedding model not found: {embedding_model}")
        return str(embedding_model)

    # 로컬 모델 없으면 HF에서 다운로드 
    from huggingface_hub import hf_hub_download

    root = args.download_root or (Path(tempfile.gettempdir()) / "diarizen-hf")
    embedding_dir = root / sanitize_repo_id(DEFAULT_EMBEDDING_MODEL)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    download_kwargs = {"token": args.hf_token} if args.hf_token else {}
    return hf_hub_download(
        repo_id=DEFAULT_EMBEDDING_MODEL,
        filename="pytorch_model.bin",
        local_dir=str(embedding_dir),
        **download_kwargs,
    )

# 사전학습된 Diarizen 파이프라인 로드
def load_pretrained_pipeline(args, device):
    from diarizen.pipelines.inference import DiariZenPipeline

    if args.diarizen_hub is not None:
        diarizen_hub = args.diarizen_hub.expanduser().resolve()
        if not (diarizen_hub / "config.toml").exists():
            raise FileNotFoundError(f"DiariZen hub config not found: {diarizen_hub / 'config.toml'}")
        if not (diarizen_hub / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"DiariZen hub checkpoint not found: {diarizen_hub / 'pytorch_model.bin'}")
    else:
        from huggingface_hub import snapshot_download

        root = args.download_root or (Path(tempfile.gettempdir()) / "diarizen-hf")
        model_dir = root / sanitize_repo_id(args.model)
        model_dir.mkdir(parents=True, exist_ok=True)
        download_kwargs = {"token": args.hf_token} if args.hf_token else {}
        diarizen_hub = Path(
            snapshot_download(repo_id=args.model, local_dir=str(model_dir), **download_kwargs)
        ).expanduser().resolve()

    embedding_model = resolve_embedding_model(args)
    config_parse = build_pretrained_config_parse(args)
    pipeline = DiariZenPipeline(
        diarizen_hub=Path(diarizen_hub).expanduser().resolve(),
        embedding_model=embedding_model,
        config_parse=config_parse,
        device=device,
    )
    metadata = {
        "diarizen_hub": str(Path(diarizen_hub).resolve()),
        "embedding_model": str(Path(embedding_model).resolve()) if Path(embedding_model).exists() else embedding_model,
        "inference": config_parse["inference"]["args"],
        "clustering": config_parse["clustering"]["args"],
    }
    return pipeline, metadata

# pretrained DiariZen pipeline에 넘길 inference 및 clustering 설정을 구성
def build_pretrained_config_parse(args):
    min_speakers, max_speakers, _ = resolve_speaker_bounds(args)
    inference_args = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering,
    }
    clustering_args = {
        "method": args.clustering_method,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "ahc_threshold": args.ahc_threshold,
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_args["min_cluster_size"] = args.min_cluster_size
    elif args.clustering_method == "VBxClustering":
        clustering_args.update(
            {
                "ahc_criterion": args.ahc_criterion,
                "Fa": args.fa,
                "Fb": args.fb,
                "lda_dim": args.lda_dim,
                "max_iters": args.max_iters,
            }
        )
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    return {
        "inference": {"args": inference_args},
        "clustering": {"args": clustering_args},
    }

# pyannote Annotation 객체를 JSON 저장 가능한 리스트로 바꿈
def annotation_to_turns(annotation) -> list[dict]:
    turns = []
    for index, (segment, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        turns.append(
            {
                "index": index,
                "speaker": str(speaker),
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.duration),
            }
        )
    return turns

# 화자분리 결과를 표준 RTTM 형식으로 저장
def write_rttm(annotation, uri: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

# 화자 수 조건을 결정 
def resolve_speaker_bounds(args):
    if args.num_speakers is not None:
        applied = {"min_speakers": args.num_speakers, "max_speakers": args.num_speakers}
        return args.num_speakers, args.num_speakers, applied

    min_speakers = args.min_speakers if args.min_speakers is not None else 1
    max_speakers = args.max_speakers if args.max_speakers is not None else 20
    applied = {"min_speakers": min_speakers, "max_speakers": max_speakers}
    return min_speakers, max_speakers, applied


def main():
    args = parse_args()

    # pyannote.audio 3.1-era checkpoints (what DiariZen's bundled fork loads)
    # predate PyTorch 2.6's weights_only=True default.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch

    artifact_dir = args.artifact_dir
    audio_path = artifact_dir / "audio.wav"
    meta_path = artifact_dir / "meta.json"
    diarization_json_path = artifact_dir / "diarization.json"
    diarization_rttm_path = artifact_dir / "diarization.rttm"

    if not audio_path.exists():
        raise FileNotFoundError(f"Prepared audio not found: {audio_path}")

    video_id = artifact_dir.name
    if meta_path.exists():
        with open(meta_path, encoding="utf-8-sig") as f:
            video_id = json.load(f).get("video_id", video_id)

    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    min_speakers, max_speakers, speaker_constraints = resolve_speaker_bounds(args)

    pipeline, model_metadata = load_pretrained_pipeline(args, torch.device(resolved_device))
    pipeline.min_speakers = min_speakers
    pipeline.max_speakers = max_speakers
    annotation = pipeline(str(audio_path), sess_name=video_id)

    turns = annotation_to_turns(annotation)
    write_rttm(annotation, video_id, diarization_rttm_path)

    payload = {
        "video_id": video_id,
        "source_audio": str(audio_path.resolve()),
        "model": args.model,
        "backend": f"diarizen_{args.backend}",
        "device": resolved_device,
        "pipeline_params": speaker_constraints,
        "model_metadata": model_metadata,
        "speaker_count": len({turn["speaker"] for turn in turns}),
        "num_turns": len(turns),
        "alignment_source": "speaker_turns",
        "speaker_turns": turns,
        "exclusive_speaker_turns": [],
        "alignment_turns": turns,
    }
    with open(diarization_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Diarization JSON: {diarization_json_path}")
    print(f"Diarization RTTM: {diarization_rttm_path}")
    print(f"Speakers: {payload['speaker_count']}, turns: {payload['num_turns']}")


if __name__ == "__main__":
    main()
