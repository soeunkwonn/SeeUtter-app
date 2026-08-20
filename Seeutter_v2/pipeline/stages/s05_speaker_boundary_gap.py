import argparse
import copy
import os
from pathlib import Path
from typing import Any

from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)
from .s06_speaker_correction import (
    DEFAULT_API_BASE_URL,
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    OpenAIResponsesBooleanClient,
    load_openai_api_key,
)


GLUE_GAP_SEC = 0.10 # 두 segment 사이 gap이 0.1초 미만이면 거의 붙어있다고 판단 
MAX_GLUE_OVERLAP_SEC = 0.05 # segment끼리 0.05초 겹치는 것도 허용 
PAUSE_GAP_SEC = 0.30
CONTEXT_EXTRA = 2 # 앞뒤로 봄 

GAP_SYSTEM_PROMPT = """\
You audit speaker turns in a transcript. Speaker labels are hidden.
Return true only when the text marked U clearly forms one person's uninterrupted
utterance and its internal speaker-label change is therefore wrong.

Return false when U can plausibly contain a real response, interruption,
interjection, backchannel, collaborative completion, quotation, or new turn.
No timestamp gap alone is not evidence that the speaker stayed the same.
False is the safe default.
"""


def _clean(text):
    return " ".join(str(text or "").split())


def _gap(a, b):
    return float(b["start"]) - float(a["end"])


def _is_near_zero_gap(gap, glue):
    return -MAX_GLUE_OVERLAP_SEC <= gap < glue # -0.05 <= gap < 0.10


def _two_groups(speakers):
    first = speakers[0]
    i = 0
    while i < len(speakers) and speakers[i] == first:
        i += 1
    if i == len(speakers):
        return None
    second = speakers[i]
    if any(s != second for s in speakers[i:]):
        return None
    return first, second, i


def _glued_span(segments, boundary, glue):
    lo = boundary
    while (
        lo - 1 >= 0
        and _is_near_zero_gap(_gap(segments[lo - 1], segments[lo]), glue)
    ):
        lo -= 1
    hi = boundary + 1
    while (
        hi + 1 < len(segments)
        and _is_near_zero_gap(_gap(segments[hi], segments[hi + 1]), glue)
    ):
        hi += 1
    return lo, hi


def find_candidates(
    segments: list[dict],
    *,
    glue: float = GLUE_GAP_SEC,
    pause: float = PAUSE_GAP_SEC,
):
    """Glued spans that straddle a single label boundary and are pause-bounded."""
    n = len(segments)
    candidates: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for i in range(n - 1):
        # 화자가 안 바뀌면 skip 
        if str(segments[i]["speaker"]) == str(segments[i + 1]["speaker"]):
            continue
        # 경계 확인 (아니면 skip)
        if not _is_near_zero_gap(_gap(segments[i], segments[i + 1]), glue):
            continue
        lo, hi = _glued_span(segments, i, glue)
        # 이미 잡은거면 skip
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))

        left_bounded = lo == 0 or _gap(segments[lo - 1], segments[lo]) >= pause
        right_bounded = hi == n - 1 or _gap(segments[hi], segments[hi + 1]) >= pause
        if not (left_bounded and right_bounded):
            continue  # 경계 애매해서 skip
        # 화자가 두 그룹으로 나뉘는지 확인 
        groups = _two_groups([str(segments[j]["speaker"]) for j in range(lo, hi + 1)])
        if groups is None:
            continue  # 여러 번 섞이면 skip
        candidates.append({"lo": lo, "hi": hi, "groups": groups})
    return candidates


def _format_window(segments, lo, hi, before, after,):
    n = len(segments)
    start = max(0, (before if before is not None else lo) - CONTEXT_EXTRA)
    end = min(n, (after if after is not None else hi) + CONTEXT_EXTRA + 1)
    lines = ["Dialogue excerpt (speaker labels hidden; [..s] = pause before the line):"]
    j = start

    while j < end:
        pause = f"[{_gap(segments[j - 1], segments[j]):.2f}s] " if j > start else ""
        if lo <= j <= hi:
            text = " ".join(_clean(segments[k].get("text")) for k in range(lo, hi + 1))
            lines.append(f'  {pause}U >> "{text}"')
            j = hi + 1
            continue
        role = "(BEFORE) " if j == before else "(AFTER) " if j == after else ""
        lines.append(f'  {pause}{role}"{_clean(segments[j].get("text"))}"')
        j += 1
    return "\n".join(lines)


def _unique_anchor_target(
    segments,
    lo,
    hi,
    groups,
):
    internal_labels = {groups[0], groups[1]}
    supported: set[str] = set() # supported = 집합 
    
    if lo > 0:
        before_label = str(segments[lo - 1]["speaker"]) # 이전 speaker label 확인 
        if before_label in internal_labels:
            supported.add(before_label)
    if hi + 1 < len(segments):
        after_label = str(segments[hi + 1]["speaker"]) # 다음 speaker 확인 
        if after_label in internal_labels:
            supported.add(after_label)
    if len(supported) != 1:
        return None, "ambiguous_anchor"
    return next(iter(supported)), "unique_surrounding_anchor" # 유일한 speaker를 target으로 반환 


def _apply_speaker(segment: dict, target: str) -> bool:
    original = str(segment.get("speaker", "UNKNOWN"))
    if original == target:
        return False
    segment["speaker_original"] = original
    segment["speaker"] = target
    segment["correction_source"] = "boundary_continuation"
    for word in segment.get("words", []) or []:
        if str(word.get("speaker", original)) == original:
            word["speaker"] = target
    return True

# Whisper가 원래 한 호흡으로 인식한 걸, diarization이 화자를 잘못 쪼갠 경우  
def remerge_glued(segments, glue= GLUE_GAP_SEC):
    """Merge adjacent same-speaker segments separated by <= glue seconds.

    A relabel can leave two contiguous fragments (e.g. "I" | "would much
    rather...") that Whisper originally had as one breath; this fuses them so
    each subtitle cue is a whole utterance. Only near-zero gaps merge, so real
    pauses stay as separate cues. Segments are re-indexed afterwards.
    """
    if not segments:
        return segments, 0
    merged = [copy.deepcopy(segments[0])]
    n_merged = 0
    for seg in segments[1:]:
        prev = merged[-1]
        gap = float(seg["start"]) - float(prev["end"])
        
        # merge 조건 
        if (
            str(seg["speaker"]) == str(prev["speaker"]) # 같은 화자인지? 
            and -MAX_GLUE_OVERLAP_SEC <= gap <= glue
        ):
            prev["end"] = seg["end"]
            prev["duration"] = float(prev["end"]) - float(prev["start"])
            prev["text"] = f'{_clean(prev.get("text"))} {_clean(seg.get("text"))}'.strip()
            prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
            prev["num_words"] = len(prev["words"])
            prev["overlap_sec"] = float(prev.get("overlap_sec", 0.0)) + float(seg.get("overlap_sec", 0.0))
            prev["asr_segment_ids"] = (prev.get("asr_segment_ids") or []) + (seg.get("asr_segment_ids") or [])
            methods = set(prev.get("match_methods") or []) | set(seg.get("match_methods") or [])
            prev["match_methods"] = sorted(methods)
            if prev.get("speaker_boundary_gap_checked") or seg.get("speaker_boundary_gap_checked"):
                prev["speaker_boundary_gap_checked"] = True
            sources = {
                str(value)
                for value in (prev.get("correction_source"), seg.get("correction_source"))
                if value
            }
            if sources:
                prev["correction_sources"] = sorted(sources)
            n_merged += 1
        else:
            merged.append(copy.deepcopy(seg))
    for i, seg in enumerate(merged):
        seg["index"] = i
    return merged, n_merged


def run(
    artifact_dir = None,
    video = None,
    out_root = DEFAULT_ARTIFACT_ROOT,
    *,
    model = None,
    reasoning_effort= "low",
    env_file = DEFAULT_ENV_FILE,
    glue_gap_sec= GLUE_GAP_SEC,
    pause_gap_sec= PAUSE_GAP_SEC,
    dry_run = False,
    overwrite = False,
    judge = None,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    # Boundary normalization is the first repair stage and always reads raw
    # alignment. Context correction consumes our output afterwards.
    in_path = paths["aligned_segments"]
    out_corrected = paths["gap_corrected_aligned_segments"]
    out_audit = paths["speaker_boundary_gap"]

    if not in_path.exists():
        raise FileNotFoundError(f"Aligned JSON not found: {in_path}. Run alignment first.")
    if out_corrected.exists() and out_audit.exists() and not overwrite and not dry_run:
        return {"artifact_dir": artifact_dir, "payload": read_json(out_corrected),
                "audit": read_json(out_audit), "skipped": True}

    payload = read_json(in_path)
    segments = payload.get("segments", [])
    if not segments:
        raise ValueError("No aligned segments found in aligned_segments.json")

    candidates = find_candidates(segments, glue=glue_gap_sec, pause=pause_gap_sec)

    if dry_run:
        for c in candidates:
            lo, hi = c["lo"], c["hi"]
            text = " ".join(_clean(segments[k].get("text")) for k in range(lo, hi + 1))
            gl = "start" if lo == 0 else f"{_gap(segments[lo - 1], segments[lo]):.2f}s"
            gr = "end" if hi == len(segments) - 1 else f"{_gap(segments[hi], segments[hi + 1]):.2f}s"
            print(f"segs[{lo}..{hi}] labels={[str(segments[j]['speaker']) for j in range(lo, hi + 1)]} "
                  f"pause(L={gl},R={gr}): {text!r}")
        print(f"\n{len(candidates)} candidate(s) detected (dry-run, no LLM, no write).")
        return {"artifact_dir": artifact_dir, "candidates": candidates, "dry_run": True}

    # LLM judge 준비 
    resolved_model = model or os.environ.get("SEEUTTER_SPEAKER_CORRECTION_MODEL", DEFAULT_MODEL)
    has_actionable_candidate = any(
        _unique_anchor_target(segments, c["lo"], c["hi"], c["groups"])[0]
        is not None
        for c in candidates
    )
    if judge is None and has_actionable_candidate:
        judge = OpenAIResponsesBooleanClient(
            api_key=load_openai_api_key(env_file),
            model=resolved_model,
            reasoning_effort=reasoning_effort,
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        )
    judge_model = getattr(judge, "model", resolved_model)

    corrected = copy.deepcopy(payload)
    corrected_segments = corrected["segments"]
    decisions: list[dict] = []
    applied: list[dict] = []

    for c in candidates:
        lo, hi = c["lo"], c["hi"]
        before = lo - 1 if lo > 0 else None
        after = hi + 1 if hi < len(segments) - 1 else None
        window = _format_window(segments, lo, hi, before, after)
        target, anchor_resolution = _unique_anchor_target(
            segments, lo, hi, c["groups"]
        )
        original_indices = [
            segments[j].get("index", j) for j in range(lo, hi + 1)
        ]

        # This near-zero boundary now belongs to this stage. Mark it even when
        # we conservatively keep the labels, so context correction will not
        # independently mutate the same fragment later.
        for j in range(lo, hi + 1):
            corrected_segments[j]["speaker_boundary_gap_checked"] = True

        if target is None:
            decisions.append(
                {
                    "segment_span": [lo, hi],
                    "segment_indices": original_indices,
                    "labels": [
                        str(segments[j]["speaker"]) for j in range(lo, hi + 1)
                    ],
                    "text": " ".join(
                        _clean(segments[j].get("text")) for j in range(lo, hi + 1)
                    ),
                    "same_uninterrupted_utterance": None,
                    "target_speaker": None,
                    "resolution": anchor_resolution,
                }
            )
            continue # target 못 정하면 skip 

        prompt = (
            f"{window}\n\nQuestion: Is U clearly one person's uninterrupted "
            "utterance despite its internal speaker-label change? "
            "Return the schema only."
        )
        if judge is None:
            raise RuntimeError("No boundary judge is available.")
        same_utterance, meta = judge.ask_boolean(
            system_prompt=GAP_SYSTEM_PROMPT,
            user_prompt=prompt,
            field_name="same_uninterrupted_utterance",
            schema_name="near_zero_boundary_judgment",
        )

        # 모든 판정을 기록 
        decision = {
            "segment_span": [lo, hi],
            "segment_indices": original_indices,
            "labels": [str(segments[j]["speaker"]) for j in range(lo, hi + 1)],
            "text": " ".join(_clean(segments[j].get("text")) for j in range(lo, hi + 1)),
            "same_uninterrupted_utterance": same_utterance,
            "target_speaker": target if same_utterance else None,
            "resolution": (
                "boundary_continuation"
                if same_utterance
                else "plausible_speaker_change"
            ),
            **meta,
        }
        decisions.append(decision)

        if not same_utterance:
            continue
        moved = []
        for j in range(lo, hi + 1):
            if _apply_speaker(corrected_segments[j], target): # True일 때 라벨 교체 
                moved.append(segments[j].get("index", j))
        if moved:
            applied.append({
                "segment_span": [lo, hi],
                "moved_segment_indices": moved,
                "to_speaker": target,
                "resolution": "boundary_continuation",
            })

    # Fuse fragments left contiguous by the relabels (e.g. "I" | "would...").
    corrected_segments, num_merged = remerge_glued(corrected_segments, glue_gap_sec)
    corrected["segments"] = corrected_segments

    corrected["source_aligned_segments"] = str(in_path.resolve())
    corrected["speaker_boundary_gap"] = {
        "model": judge_model, "num_moves": len(applied), "num_merged": num_merged,
    }

    audit = {
        "video_id": payload.get("video_id") or artifact_dir.name,
        "source_aligned_segments": str(in_path.resolve()),
        "model": judge_model,
        "reasoning_effort": reasoning_effort,
        "config": {
            "glue_gap_sec": glue_gap_sec,
            "max_glue_overlap_sec": MAX_GLUE_OVERLAP_SEC,
            "pause_gap_sec": pause_gap_sec,
        },
        "num_segments": len(segments),
        "num_candidates": len(candidates),
        "num_moves": len(applied),
        "num_merged": num_merged,
        "decisions": decisions,
        "applied_moves": applied,
    }

    write_json(out_corrected, corrected)
    write_json(out_audit, audit)
    return {"artifact_dir": artifact_dir, "payload": corrected, "audit": audit,
            "skipped": False}


# def parse_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--artifact-dir", type=Path, required=True)
#     parser.add_argument("--model", default=None)
#     parser.add_argument(
#         "--reasoning-effort",
#         choices=["none", "low", "medium", "high", "xhigh", "max"],
#         default="medium",
#     )
#     parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
#     parser.add_argument("--glue-gap-sec", type=float, default=GLUE_GAP_SEC)
#     parser.add_argument("--pause-gap-sec", type=float, default=PAUSE_GAP_SEC)
#     parser.add_argument("--dry-run", action="store_true",
#                         help="Detect candidate boundaries with gaps only; no LLM, no write.")
#     parser.add_argument("--overwrite", action="store_true")
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     result = run(
#         artifact_dir=args.artifact_dir,
#         model=args.model,
#         reasoning_effort=args.reasoning_effort,
#         env_file=args.env_file,
#         glue_gap_sec=args.glue_gap_sec,
#         pause_gap_sec=args.pause_gap_sec,
#         dry_run=args.dry_run,
#         overwrite=args.overwrite,
#     )
#     if result.get("dry_run"):
#         return
#     audit = result["audit"]
#     print(f"Corrected segments: {result['artifact_dir'] / 'aligned_segments_gap_corrected.json'}")
#     print(f"Audit: {result['artifact_dir'] / 'speaker_boundary_gap.json'}")
#     print(f"Candidates: {audit['num_candidates']}, moves: {audit['num_moves']}, "
#           f"merges: {audit['num_merged']}")


# if __name__ == "__main__":
#     main()
