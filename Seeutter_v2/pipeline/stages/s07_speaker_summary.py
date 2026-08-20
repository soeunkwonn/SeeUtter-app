from __future__ import annotations
import csv
from pathlib import Path
from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    preferred_aligned_segments_path,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)


def build_speaker_table_rows(speakers: list[dict], top_k: int) -> list[dict]:
    rows = []
    for speaker in speakers:
        rank = int(speaker["rank"])
        rows.append(
            {
                "speaker_id": str(speaker["speaker"]),
                "display_name": str(speaker["speaker"]),
                "rank": rank,
                "selected_for_labeling": rank <= top_k,
                "total_speaking_sec": float(speaker["total_speaking_sec"]),
                "share_of_speaking": float(speaker["share_of_speaking"]),
                "num_turns": int(speaker["num_turns"]),
                "avg_turn_sec": float(speaker["avg_turn_sec"]),
                "max_turn_sec": float(speaker["max_turn_sec"]),
            }
        )
    return rows


def write_speaker_table_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "display_name",
                "rank",
                "selected_for_labeling",
                "total_speaking_sec",
                "share_of_speaking",
                "num_turns",
                "avg_turn_sec",
                "max_turn_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    artifact_dir: Path | None = None,
    video: Path | None = None,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    top_k: int = 8,
    overwrite: bool = False,
    overwrite_map: bool = False,
    write_map_template: bool = True,
) -> dict:
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)

    if paths["speaker_summary"].exists() and not overwrite:
        payload = read_json(paths["speaker_summary"])
        speakers = payload.get("speakers", [])
        resolved_top_k = int(payload.get("top_k", max(0, min(top_k, len(speakers)))))
    else:
        aligned_path = preferred_aligned_segments_path(paths)
        if not aligned_path.exists():
            raise FileNotFoundError(
                f"Aligned JSON not found: {aligned_path}. "
                "Run alignment and speaker repair first."
            )

        aligned_payload = read_json(aligned_path)
        turns = aligned_payload.get("segments") or []
        if not turns:
            raise ValueError(f"No aligned segments found in {aligned_path.name}")

        by_speaker: dict[str, dict] = {}
        total_speaking_sec = 0.0
        unknown_speaking_sec = 0.0
        for turn in turns:
            speaker = str(turn["speaker"])
            duration = float(turn["duration"])
            if speaker.upper() == "UNKNOWN":
                unknown_speaking_sec += duration
                continue
            total_speaking_sec += duration
            stats = by_speaker.setdefault(
                speaker,
                {"speaker": speaker, "total_speaking_sec": 0.0, "num_turns": 0, "max_turn_sec": 0.0},
            )
            stats["total_speaking_sec"] += duration
            stats["num_turns"] += 1
            stats["max_turn_sec"] = max(float(stats["max_turn_sec"]), duration)

        speakers = sorted(
            by_speaker.values(),
            key=lambda item: float(item["total_speaking_sec"]),
            reverse=True,
        )
        for rank, item in enumerate(speakers, 1):
            total = float(item["total_speaking_sec"])
            num_turns = int(item["num_turns"])
            item["rank"] = rank
            item["avg_turn_sec"] = 0.0 if num_turns == 0 else total / num_turns
            item["share_of_speaking"] = 0.0 if total_speaking_sec == 0 else total / total_speaking_sec

        resolved_top_k = max(0, min(top_k, len(speakers)))
        payload = {
            "video_id": aligned_payload.get("video_id") or artifact_dir.name,
            "source_segments": str(aligned_path.resolve()),
            "num_speakers": len(speakers),
            "total_speaking_sec": total_speaking_sec,
            "unknown_speaking_sec": unknown_speaking_sec,
            "top_k": resolved_top_k,
            "speakers": speakers,
            "top_speakers": speakers[:resolved_top_k],
        }
        write_json(paths["speaker_summary"], payload)

    table_rows = build_speaker_table_rows(speakers, resolved_top_k)
    table_payload = {
        "video_id": payload.get("video_id") or artifact_dir.name,
        "source_segments": payload.get("source_segments"),
        "top_k": resolved_top_k,
        "num_rows": len(table_rows),
        "rows": table_rows,
    }
    if overwrite or not paths["speaker_table_json"].exists():
        write_json(paths["speaker_table_json"], table_payload)
    if overwrite or not paths["speaker_table_csv"].exists():
        write_speaker_table_csv(paths["speaker_table_csv"], table_rows)

    # Refreshing summary/table must not erase names already entered by the user.
    if write_map_template and (overwrite_map or not paths["speaker_map"].exists()):
        write_json(
            paths["speaker_map"],
            {
                "video_id": payload.get("video_id") or artifact_dir.name,
                "source_table": str(paths["speaker_table_json"].resolve()),
                "speakers": {
                    row["speaker_id"]: row["display_name"]
                    for row in table_rows
                    if row["selected_for_labeling"]
                },
            },
        )

    return {
        "artifact_dir": artifact_dir,
        "paths": paths,
        "payload": payload,
        "table_payload": table_payload,
        "top_k": resolved_top_k,
    }
