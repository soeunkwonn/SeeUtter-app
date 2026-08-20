
from __future__ import annotations

import re
from pathlib import Path

from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    SENSEVOICE_REMOTE_CODE as DEFAULT_SENSEVOICE_REMOTE_CODE,
    read_json,
    preferred_aligned_segments_path,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)

# 세그먼트만큼 시간 슬라이싱
def slice_audio(audio_path, start, end, out_path):
    import soundfile as sf

    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    start_index = max(0, int(start * sample_rate))
    end_index = min(len(data), int(end * sample_rate))
    clip = data[start_index:end_index]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sample_rate)

# sensevoice 결과해석
def parse_sensevoice_output(raw_text):
    tags = re.findall(r"<([^<>]+)>", raw_text)
    emotion = "|EMO_UNKNOWN|"
    if len(tags) >= 2:
        emotion = tags[1]
    elif tags:
        emotion = tags[0]

    transcript = re.sub(r"<[^<>]+>", "", raw_text).strip()
    if transcript.startswith(">"):
        transcript = transcript[1:].strip()
    return emotion, transcript


def infer_stub(segments):
    inferred = []
    for segment in segments:
        inferred.append(
            {
                **segment,
                "emotion": "|EMO_UNKNOWN|",
                "emotion_transcript": None,
                "emotion_backend": "stub",
            }
        )
    return inferred


def infer_sensevoice(
    segments,
    audio_path,
    clips_dir,
    model_tag,
    remote_code,
    language,
    pad_end_sec,
    min_duration_sec,
):
    from funasr import AutoModel

    resolved_remote_code = remote_code.resolve().as_posix()
    model = AutoModel(
        model=model_tag,
        trust_remote_code=True,
        remote_code=resolved_remote_code,
        vad_kwargs={"max_single_segment_time": 30000},
        disable_update=True,
    )

    inferred = []
    for segment in segments:
        start = float(segment["start"])
        end = max(start + min_duration_sec, float(segment["end"]) + pad_end_sec)
        clip_path = clips_dir / f"seg_{int(segment['index']):04d}.wav"
        slice_audio(audio_path, start, end, clip_path)

        lang = None if language == "auto" else language
        result = model.generate(
            input=str(clip_path),
            cache={},
            language=lang,
            use_itn=True,
            batch_size_s=1,
            disable_pbar=True,
            ban_emo_unk=True,
        )
        raw_text = result[0]["text"] if result else ""
        emotion, transcript = parse_sensevoice_output(raw_text)
        inferred.append(
            {
                **segment,
                "emotion": emotion,
                "emotion_transcript": transcript,
                "emotion_raw_text": raw_text,
                "emotion_backend": "sensevoice",
            }
        )
    return inferred


def run(
    artifact_dir= None,
    video = None,
    out_root = DEFAULT_ARTIFACT_ROOT,
    backend = "stub",
    model = "iic/SenseVoiceSmall",
    remote_code = DEFAULT_SENSEVOICE_REMOTE_CODE,
    language = "auto",
    pad_end_sec = 0.2,
    min_duration_sec= 0.1,
    keep_clips = False,
    overwrite = False,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    aligned_path = preferred_aligned_segments_path(paths)

    if not aligned_path.exists():
        raise FileNotFoundError(
            f"Aligned JSON not found: {aligned_path}. Run alignment first."
        )
    if paths["emotion_segments"].exists() and not overwrite:
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["emotion_segments"]),
            "skipped": True,
        }

    aligned_payload = read_json(aligned_path)
    segments = aligned_payload.get("segments", [])
    if not segments:
        raise ValueError("No aligned segments found in aligned_segments.json")

    if backend == "stub":
        enriched_segments = infer_stub(segments)
    else:
        if not paths["audio"].exists():
            raise FileNotFoundError(
                f"Prepared audio not found: {paths['audio']}. Run prepare_media.py first."
            )
        if not remote_code.exists():
            raise FileNotFoundError(
                f"SenseVoice remote code not found: {remote_code}"
            )
        enriched_segments = infer_sensevoice(
            segments=segments,
            audio_path=paths["audio"],
            clips_dir=paths["emotion_clips_dir"],
            model_tag=model,
            remote_code=remote_code,
            language=language,
            pad_end_sec=pad_end_sec,
            min_duration_sec=min_duration_sec,
        )

    payload = {
        "video_id": aligned_payload.get("video_id") or artifact_dir.name,
        "source_aligned_segments": str(aligned_path.resolve()),
        "backend": backend,
        "model": model if backend == "sensevoice" else None,
        "num_segments": len(enriched_segments),
        "segments": enriched_segments,
    }
    write_json(paths["emotion_segments"], payload)

    if backend == "sensevoice" and not keep_clips and paths["emotion_clips_dir"].exists():
        for clip_path in paths["emotion_clips_dir"].glob("*.wav"):
            clip_path.unlink()
        try:
            paths["emotion_clips_dir"].rmdir()
        except OSError:
            pass

    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload, "skipped": False}
