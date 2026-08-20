from __future__ import annotations
from pathlib import Path
from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)

# Whisper 반환 결과를 JSON 구조로 정리 
def normalize_result(result):
    normalized_segments = [] # 정리된 segment들 담을 리스트 
    total_words = 0 # 전체 단어 개수 
    for segment in result.get("segments", []):
        words = []
        for word in segment.get("words", []) or []:
            words.append(
                {
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "word": str(word["word"]),
                    "probability": float(word["probability"]) if "probability" in word else None, # confidence
                }
            )
        total_words += len(words)
        normalized_segments.append(
            {
                "id": int(segment["id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment["text"]).strip(),
                "avg_logprob": float(segment["avg_logprob"]) if "avg_logprob" in segment else None,
                "no_speech_prob": float(segment["no_speech_prob"]) if "no_speech_prob" in segment else None,
                "words": words,
            }
        )

    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language"),
        "segments": normalized_segments,
        "num_segments": len(normalized_segments),
        "num_words": total_words,
    }

# whisper 실행 
def run(
    artifact_dir= None,
    video = None,
    out_root= DEFAULT_ARTIFACT_ROOT,
    model = "large",
    device= "cuda",
    language= None,
    task = "transcribe",
    word_timestamps = True,
    overwrite = False,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    
    # 오디오 파일 확인 
    if not paths["audio"].exists():
        raise FileNotFoundError(
            f"Prepared audio not found: {paths['audio']}. Run prepare_media.py first."
        )
    if paths["asr"].exists() and not overwrite:
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["asr"]),
            "skipped": True,
        }

    import whisper

    # meta.json이 존재하면 해당 파일 리드 
    meta = read_json(paths["meta"]) if paths["meta"].exists() else {"video_id": artifact_dir.name}

    load_kwargs = {}
    if device:
        load_kwargs["device"] = device
    whisper_model = whisper.load_model(model, **load_kwargs)
    result = whisper_model.transcribe(
        str(paths["audio"]),
        task=task,
        language=language,
        word_timestamps=word_timestamps,
    )
    normalized = normalize_result(result)
    payload = {
        "video_id": meta["video_id"],
        "source_audio": str(paths["audio"].resolve()),
        "model": model,
        "task": task,
        "word_timestamps": word_timestamps,
        **normalized,
    }
    write_json(paths["asr"], payload)
    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload, "skipped": False}
