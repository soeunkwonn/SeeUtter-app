from __future__ import annotations
from pathlib import Path
from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    preferred_aligned_segments_path,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)


def load_mapping(path, inline_sets):
    mapping: dict[str, str] = {} # 화자ID와 이름을 저장할 빈 딕셔너리 
    if path is not None and path.exists(): 
        payload = read_json(path)
        if isinstance(payload, dict):
            if "speakers" in payload and isinstance(payload["speakers"], dict):
                mapping.update({str(k): str(v) for k, v in payload["speakers"].items()})
            elif "rows" in payload and isinstance(payload["rows"], list):
                for row in payload["rows"]:
                    if not isinstance(row, dict):
                        continue
                    speaker_id = row.get("speaker_id")
                    if speaker_id is None:
                        continue
                    display_name = (
                        row.get("display_name")
                        or row.get("speaker_display")
                        or row.get("speaker_name")
                        or row.get("name")
                        or row.get("edited_name")
                        or speaker_id
                    )
                    mapping[str(speaker_id)] = str(display_name)
            else:
                mapping.update({str(k): str(v) for k, v in payload.items()})

    for item in inline_sets:
        if "=" not in item:
            raise ValueError(f"Invalid --set value: {item}. Expected speaker_id=name")
        speaker, name = item.split("=", 1)
        mapping[speaker.strip()] = name.strip()
    return mapping

# 어떤 세그먼트 파일을 최종 자막의 입력으로 사용할지 결정
def resolve_source_payload(paths: dict[str, Path]) -> tuple[dict, str]:
    # Each optional enrichment stage (emotion, ...) writes out a full copy of the
    # aligned segments plus its own fields, so the richest file that exists is
    # always a superset of the others -- no per-field merging needed.
    if paths["emotion_segments"].exists():
        return read_json(paths["emotion_segments"]), "emotion_segments"
    aligned_path = preferred_aligned_segments_path(paths)
    if aligned_path.exists():
        source_names = {
            paths["final_aligned_segments"]: "final_aligned_segments",
            paths["corrected_aligned_segments"]: "corrected_aligned_segments",
            paths["gap_corrected_aligned_segments"]: "gap_corrected_aligned_segments",
            paths["aligned_segments"]: "aligned_segments",
        }
        source_name = source_names.get(aligned_path, aligned_path.stem)
        return read_json(aligned_path), source_name
    raise FileNotFoundError("Need either emotion_segments.json or aligned_segments.json.")


def run(
    artifact_dir: Path | None = None,
    video: Path | None = None,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    speaker_map: Path | None = None,
    inline_sets: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    inline_sets = inline_sets or []
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)

    if paths["final_caption_data"].exists() and not overwrite:
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["final_caption_data"]),
            "skipped": True,
        }

    source_payload, source_name = resolve_source_payload(paths)
    speaker_map_path = speaker_map or paths["speaker_map"]
    mapping = load_mapping(speaker_map_path, inline_sets)

    final_segments = []
    for segment in source_payload.get("segments", []):
        speaker_id = str(segment.get("speaker", "UNKNOWN"))
        display_name = mapping.get(speaker_id, speaker_id)
        final_segments.append(
            {
                **segment,
                "speaker_id": speaker_id,
                "speaker_display": display_name,
            }
        )

    payload = {
        "video_id": source_payload.get("video_id") or artifact_dir.name,
        "source_segments": source_name,
        "speaker_map_path": str(speaker_map_path.resolve()) if speaker_map_path.exists() else None,
        "num_segments": len(final_segments),
        "segments": final_segments,
    }
    write_json(paths["final_caption_data"], payload)
    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload, "skipped": False}
