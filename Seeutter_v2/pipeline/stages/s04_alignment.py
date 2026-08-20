"""Align Whisper ASR timestamps with diarization turns."""

from __future__ import annotations

from pathlib import Path

from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    preferred_diarization_path,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)

DEFAULT_AMBIGUOUS_OVERLAP_MARGIN_SEC = 0.05

# 겹치는 구간 계산 
def overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))

# 두 구간의 중심점 거리 계산 
def center_distance(start_a, end_a, start_b, end_b):
    center_a = 0.5 * (start_a + end_a)
    center_b = 0.5 * (start_b + end_b)
    return abs(center_a - center_b)


def interval_gap(start_a, end_a, start_b, end_b):
    """Return the empty time between two intervals (zero when they touch/overlap)."""
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0.0


def choose_speaker(
    start,
    end,
    turns,
    unknown_speaker,
    max_nearest_gap_sec,
    ambiguous_overlap_margin_sec=DEFAULT_AMBIGUOUS_OVERLAP_MARGIN_SEC,
):
    duration = max(1e-6, end - start) #단어 길이 계산 
    overlaps_by_speaker = {} #화자별 겹침 결과 담을 딕셔너리 
    #모든 diarization turn과 비교해서 overlap 계산 
    for turn in turns:
        ov = overlap(start, end, float(turn["start"]), float(turn["end"]))
        # 겹치지 않는 turn은 무시 
        if ov <= 0:
            continue
        # 현재 turn의 화자 라벨 확인 
        speaker = str(turn["speaker"])
        # 같은 화자가 후보에 있는지 확인 
        candidate = overlaps_by_speaker.get(speaker)
        # 후보 저장: coverage=비율 / reconcile source=turn이 결정된 근거 
        if candidate is None:
            overlaps_by_speaker[speaker] = {
                "speaker": speaker,
                "overlap_sec": ov,
                "coverage": ov / duration,
                "turn_start": float(turn["start"]),
                "turn_end": float(turn["end"]),
                "reconcile_source": turn.get("reconcile_source"),
                "reconcile_cov": turn.get("reconcile_cov"),
            }
        # 동일 화자의 여러 turn이 겹치면 합산
        else:
            candidate["overlap_sec"] += ov
            candidate["coverage"] = candidate["overlap_sec"] / duration
    # 화자 후보 겹침이 큰 순서로 정렬 
    candidates = sorted(
        overlaps_by_speaker.values(),
        key=lambda item: float(item["overlap_sec"]),
        reverse=True,
    )
    if candidates:
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        # 1등과 2등의 차이 계산 
        overlap_margin_sec = (
            float(best["overlap_sec"]) - float(second["overlap_sec"])
            if second is not None
            else None
        )
        return {
            "speaker": str(best["speaker"]),
            "overlap_sec": float(best["overlap_sec"]),
            "coverage": float(best["coverage"]),
            "matched_by": "overlap",
            "speaker_candidates": candidates,
            "overlap_margin_sec": overlap_margin_sec,
            # 1등-2등이 0.05초 이하인 경우 TRUE
            "boundary_ambiguous": bool(
                second is not None
                and overlap_margin_sec is not None
                and overlap_margin_sec <= ambiguous_overlap_margin_sec
            ),
            "reconcile_source": best.get("reconcile_source"),
            "reconcile_cov": best.get("reconcile_cov"),
        }
    # 겹치는 화자가 없을 때 
    nearest_turn = None
    nearest_gap = float("inf")
    nearest_center_distance = float("inf")
    # 모든 turn의 시간상 거리 계산 
    for turn in turns:
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        gap = interval_gap(start, end, turn_start, turn_end)
        distance = center_distance(start, end, turn_start, turn_end)
        # interval gap이 작은지 확인 -> gap이 같다면 중심 거리가 더 작은지 확인 
        if (gap, distance) < (nearest_gap, nearest_center_distance):
            nearest_gap = gap
            nearest_center_distance = distance
            nearest_turn = turn
    # 가까운 turn이 허용범위 안이면 선택 (default= gap 3s)
    if nearest_turn is not None and nearest_gap <= max_nearest_gap_sec:
        return {
            "speaker": str(nearest_turn["speaker"]),
            "overlap_sec": 0.0,
            "coverage": 0.0,
            "matched_by": "nearest",
            "nearest_gap_sec": nearest_gap,
            "speaker_candidates": [],
            "overlap_margin_sec": None,
            "boundary_ambiguous": False,
            "reconcile_source": nearest_turn.get("reconcile_source"),
            "reconcile_cov": nearest_turn.get("reconcile_cov"),
        }
    # 가까운 화자도 없으면 Unknown 
    return {
        "speaker": unknown_speaker,
        "overlap_sec": 0.0,
        "coverage": 0.0,
        "matched_by": "unknown",
        "speaker_candidates": [],
        "overlap_margin_sec": None,
        "boundary_ambiguous": False,
        "reconcile_source": None,
        "reconcile_cov": None,
    }

# whisper 결과에서 단어만 추출 
def words_from_asr(asr_payload):
    words = []
    for segment in asr_payload.get("segments", []):
        segment_id = int(segment["id"])
        for word_index, word in enumerate(segment.get("words", []) or []):
            if "start" not in word or "end" not in word:
                continue
            words.append(
                {
                    "asr_segment_id": segment_id,
                    "word_index": word_index,
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "word": str(word["word"]),
                    "probability": word.get("probability"),
                }
            )
    return words

# 단어마다 화자를 붙이고 자막으로 합치기 
def align_words(
    words,
    turns,
    merge_gap_sec,
    max_nearest_gap_sec,
    unknown_speaker,
    merge_across_asr_segments,
    ambiguous_overlap_margin_sec=DEFAULT_AMBIGUOUS_OVERLAP_MARGIN_SEC,
):
    aligned_words = []
    for word in words:
        match = choose_speaker(
            start=word["start"],
            end=word["end"],
            turns=turns,
            unknown_speaker=unknown_speaker,
            max_nearest_gap_sec=max_nearest_gap_sec,
            ambiguous_overlap_margin_sec=ambiguous_overlap_margin_sec,
        )
        aligned_words.append({**word, **match})
    # 첫 번째 단어로 현재 자막 생성 
    segments = []
    current = None
    for word in aligned_words:
        if current is None:
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "text": word["word"],
                "words": [word],
                "overlap_sec": word["overlap_sec"],
                "coverage_sum": word["coverage"],
                "match_methods": {word["matched_by"]},
                "asr_segment_ids": [word["asr_segment_id"]],
            }
            continue

        gap = float(word["start"]) - float(current["end"]) # 현재 자막 끝과 다음 단어 시작의 시간 차이를 계산 
        same_speaker = word["speaker"] == current["speaker"] # 화자가 같은지 확인 
        same_asr_segment = word["asr_segment_id"] == current["asr_segment_ids"][-1] # 같은 whisper 세그먼트였는지 확인 
        can_merge = same_speaker and gap <= merge_gap_sec
        if not merge_across_asr_segments:
            can_merge = can_merge and same_asr_segment

        if can_merge:
            current["end"] = word["end"]
            current["text"] += word["word"]
            current["words"].append(word)
            current["overlap_sec"] += word["overlap_sec"]
            current["coverage_sum"] += word["coverage"]
            current["match_methods"].add(word["matched_by"])
            current["asr_segment_ids"].append(word["asr_segment_id"])
            continue

        segments.append(current)
        current = {
            "start": word["start"],
            "end": word["end"],
            "speaker": word["speaker"],
            "text": word["word"],
            "words": [word],
            "overlap_sec": word["overlap_sec"],
            "coverage_sum": word["coverage"],
            "match_methods": {word["matched_by"]},
            "asr_segment_ids": [word["asr_segment_id"]],
        }

    if current is not None:
        segments.append(current)

    normalized = []
    for index, segment in enumerate(segments):
        duration = float(segment["end"]) - float(segment["start"])
        avg_coverage = segment["coverage_sum"] / max(1, len(segment["words"]))
        normalized.append(
            {
                "index": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "duration": duration,
                "speaker": segment["speaker"],
                "text": str(segment["text"]).strip(),
                "source": "word_alignment",
                "num_words": len(segment["words"]),
                "overlap_sec": float(segment["overlap_sec"]),
                "coverage": avg_coverage,
                "boundary_ambiguous": any(
                    bool(word.get("boundary_ambiguous"))
                    for word in segment["words"]
                ),
                "match_methods": sorted(segment["match_methods"]),
                "asr_segment_ids": sorted(set(segment["asr_segment_ids"])),
                "words": [
                    {
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "word": str(word["word"]),
                        "speaker": str(word["speaker"]),
                        "matched_by": str(word["matched_by"]),
                        "probability": word.get("probability"),
                        "overlap_sec": float(word["overlap_sec"]),
                        "coverage": float(word["coverage"]),
                        "speaker_candidates": word.get("speaker_candidates", []),
                        "overlap_margin_sec": word.get("overlap_margin_sec"),
                        "boundary_ambiguous": bool(
                            word.get("boundary_ambiguous", False)
                        ),
                        "reconcile_source": word.get("reconcile_source"),
                        "reconcile_cov": word.get("reconcile_cov"),
                    }
                    for word in segment["words"]
                ],
            }
        )
    return normalized


# def align_segments(
#     asr_segments,
#     turns,
#     max_nearest_gap_sec,
#     unknown_speaker,
# ):
#     aligned = []
#     for index, segment in enumerate(asr_segments):
#         match = choose_speaker(
#             start=float(segment["start"]),
#             end=float(segment["end"]),
#             turns=turns,
#             unknown_speaker=unknown_speaker,
#             max_nearest_gap_sec=max_nearest_gap_sec,
#         )
#         aligned.append(
#             {
#                 "index": index,
#                 "start": float(segment["start"]),
#                 "end": float(segment["end"]),
#                 "duration": float(segment["end"]) - float(segment["start"]),
#                 "speaker": match["speaker"],
#                 "text": str(segment["text"]).strip(),
#                 "source": "segment_alignment",
#                 "num_words": len(segment.get("words", []) or []),
#                 "overlap_sec": float(match["overlap_sec"]),
#                 "coverage": float(match["coverage"]),
#                 "match_methods": [str(match["matched_by"])],
#                 "asr_segment_ids": [int(segment["id"])],
#                 "words": segment.get("words", []) or [],
#             }
#         )
#     return aligned


def run(
    artifact_dir=None,
    video=None,
    out_root=DEFAULT_ARTIFACT_ROOT,
    granularity="word",
    merge_gap_sec=0.5,
    max_nearest_gap_sec=3.0,
    ambiguous_overlap_margin_sec=DEFAULT_AMBIGUOUS_OVERLAP_MARGIN_SEC,
    unknown_speaker="UNKNOWN",
    merge_across_asr_segments=False,
    overwrite=False,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)

    if not paths["asr"].exists():
        raise FileNotFoundError(f"ASR JSON not found: {paths['asr']}. Run run_asr_whisper.py first.")
    if not paths["diarization_json"].exists():
        raise FileNotFoundError(
            f"Diarization JSON not found: {paths['diarization_json']}. Run run_diarization.py first."
        )
    if paths["aligned_segments"].exists() and not overwrite:
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["aligned_segments"]),
            "skipped": True,
        }

    asr_payload = read_json(paths["asr"])
    diar_payload = read_json(preferred_diarization_path(paths))
    turns = diar_payload.get("alignment_turns") or diar_payload.get("speaker_turns") or []
    if not turns:
        raise ValueError("No diarization turns found in diarization.json")

    word_items = words_from_asr(asr_payload)
    use_word_alignment = granularity == "word" or (
        granularity == "auto" and len(word_items) > 0
    )

    if use_word_alignment:
        segments = align_words(
            words=word_items,
            turns=turns,
            merge_gap_sec=merge_gap_sec,
            max_nearest_gap_sec=max_nearest_gap_sec,
            unknown_speaker=unknown_speaker,
            merge_across_asr_segments=merge_across_asr_segments,
            ambiguous_overlap_margin_sec=ambiguous_overlap_margin_sec,
        )
        granularity_used = "word"
    # else:
    #     segments = align_segments(
    #         asr_segments=asr_payload.get("segments", []),
    #         turns=turns,
    #         max_nearest_gap_sec=max_nearest_gap_sec,
    #         unknown_speaker=unknown_speaker,
    #     )
    #     granularity_used = "segment"

    payload = {
        "video_id": asr_payload.get("video_id") or diar_payload.get("video_id") or artifact_dir.name,
        "source_asr": str(paths["asr"].resolve()),
        "source_diarization": str(paths["diarization_json"].resolve()),
        "granularity": granularity_used,
        "ambiguous_overlap_margin_sec": ambiguous_overlap_margin_sec,
        "num_segments": len(segments),
        "segments": segments,
    }
    write_json(paths["aligned_segments"], payload)
    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload, "skipped": False}
