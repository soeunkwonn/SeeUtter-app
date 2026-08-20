
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..common import (
    DEFAULT_ARTIFACT_ROOT,
    read_json,
    resolve_artifact_dir,
    standard_paths,
    write_json,
)


DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"

DEFAULT_MAX_CANDIDATE_DURATION_SEC = 3.0 # 발화 길이가 3초 이하인 run만 수정 후보로 봄 
DURATION_GATE_ENV = "SEEUTTER_SPEAKER_CORRECTION_MAX_DURATION"
BOUNDARY_FRAGMENT_MAX_DURATION_SEC = 0.5
BOUNDARY_FRAGMENT_MAX_WORDS = 2
BOUNDARY_WORD_MAX_COVERAGE = 0.5
DESTINATION_WORD_MIN_COVERAGE = 0.75

SEMANTIC_BOUNDARY_MAX_WORDS = 2
SEMANTIC_BOUNDARY_MAX_DURATION_SEC = 0.65
SEMANTIC_BOUNDARY_MAX_GAP_SEC = 0.10
SEMANTIC_TARGET_MIN_VISUAL_SHARE = 0.75
SENTENCE_ENDINGS = (".", "!", "?", "…", "。", "！", "？")


def resolve_max_candidate_duration(explicit):
    if explicit is not None:
        return explicit
    raw = os.environ.get(DURATION_GATE_ENV, "").strip().lower()
    if raw in {"", "default"}:
        return DEFAULT_MAX_CANDIDATE_DURATION_SEC
    if raw in {"inf", "off", "none", "disabled"}:
        return float("inf")
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MAX_CANDIDATE_DURATION_SEC

BOUNDARY_SYSTEM_PROMPT = """\
당신은 영상의 전사 구간 경계를 검사합니다.
화자 라벨은 잘못되었을 수 있으므로 의도적으로 제공하지 않습니다.

경계 양쪽의 텍스트가 전사 또는 정렬 과정에서 나뉜 한 사람의
끊기지 않은 발화일 가능성이 더 높다면 true를 반환하세요.
문법, 문장 구조, 앞뒤 대화 문맥을 근거로 판단하세요.

두 번째 구간이 다른 사람의 대답, 감탄, 맞장구, 끼어들기,
상대방 발화의 대신 완성, 인용 또는 새로운 발화 턴일 가능성이
합리적으로 존재한다면 false를 반환하세요.

이 단계는 수정 후보를 찾는 1차 판정입니다.
실제 라벨 변경 전 다음 단계에서 더 엄격하게 재검증합니다.
"""

VERIFIER_SYSTEM_PROMPT = """\
당신은 화자 라벨 수정안을 최종 검증하는 보수적인 판정자입니다.
화자 라벨은 의도적으로 제공하지 않습니다.

표시된 TARGET과 ANCHOR 텍스트가 명백히 한 사람의 끊기지 않은
발화일 때만 수정안을 승인하세요.

실제 화자 전환, 대답, 끼어들기, 감탄, 맞장구,
상대방 발화의 대신 완성, 인용 가능성이 있거나 조금이라도
중요한 불확실성이 남아 있다면 거부하세요.

판단이 애매하면 false가 기본값입니다.
"""


@dataclass(frozen=True)
class SpeakerRun:
    index: int
    speaker: str
    segment_positions: tuple[int, ...]
    segment_ids: tuple[Any, ...]
    start: float
    end: float
    speech_duration: float
    text: str

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.index,
            "speaker": self.speaker,
            "segment_indices": list(self.segment_ids),
            "start": self.start,
            "end": self.end,
            "span_duration": max(0.0, self.end - self.start), # run의 전체 시간 범위
            "speech_duration": self.speech_duration, # 실제 세그먼트 길이 합계 
            "text": self.text,
        }

# LLM judge 인터페이스 
class BooleanJudge(Protocol):
    model: str

    def judge_boundary(
        self,
        left: SpeakerRun,
        right: SpeakerRun,
        context: list[SpeakerRun],
    ) -> tuple[bool, dict[str, Any]]:
        ...

    def verify_proposal(
        self,
        target: SpeakerRun,
        anchors: list[SpeakerRun],
        context: list[SpeakerRun],
    ) -> tuple[bool, dict[str, Any]]:
        ...


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def load_openai_api_key(env_file = DEFAULT_ENV_FILE):
    existing = os.environ.get("OPENAI_API_KEY", "").strip()
    if existing:
        return existing

    if env_file is not None and env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                value = _parse_env_value(raw_value)
                if value:
                    return value

    location = f" or {env_file}" if env_file is not None else ""
    raise ValueError(
        "OPENAI_API_KEY is not set. Define it in the environment"
        f"{location} before running speaker correction."
    )


def _response_output_text(response_payload: dict[str, Any]) -> str:
    for output_item in response_payload.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str):
                    return text
            if content_item.get("type") == "refusal":
                refusal = content_item.get("refusal", "Request refused.")
                raise RuntimeError(f"OpenAI refused the classification request: {refusal}")
    raise RuntimeError("OpenAI response contained no output_text item.")


def _api_error_message(raw_body: bytes, status: int) -> str:
    try:
        payload = json.loads(raw_body.decode("utf-8", errors="replace"))
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message:
            return f"OpenAI API error {status}: {message}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"OpenAI API error {status}."


class OpenAIResponsesBooleanClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "medium",
        base_url: str = DEFAULT_API_BASE_URL,
        timeout_sec: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def _request_boolean(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        field_name: str,
        schema_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": {field_name: {"type": "boolean"}},
            "required": [field_name],
            "additionalProperties": False,
        }
        parsed, meta = self._post_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
            schema=schema,
        )
        value = parsed.get(field_name)
        if not isinstance(value, bool):
            raise RuntimeError(f"Structured response field {field_name!r} was not boolean.")
        return value, meta

    def _post_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """POST a structured-output request; return (parsed_json, metadata)."""
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "seeutter-speaker-correction/1.0",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                parsed = json.loads(_response_output_text(response_payload))
                return parsed, {
                    "response_id": response_payload.get("id"),
                    "usage": response_payload.get("usage"),
                }
            except urllib.error.HTTPError as exc:
                raw_body = exc.read()
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(_api_error_message(raw_body, exc.code)) from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                # Read timeouts raise a bare TimeoutError (socket.timeout), which
                # is NOT a URLError -- catch it here so a slow response retries
                # instead of crashing the whole backbone.
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                reason = getattr(exc, "reason", exc)
                raise RuntimeError(f"OpenAI API request failed: {reason}") from exc

        raise RuntimeError("OpenAI API request failed after retries.")

    def post_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Public wrapper: any strict-JSON structured request (e.g. translation)."""
        return self._post_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
            schema=schema,
        )

    def ask_boolean(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        field_name: str,
        schema_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Generic boolean question, reused by the gap-boundary stage."""
        return self._request_boolean(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            field_name=field_name,
            schema_name=schema_name,
        )

    def ask_choice(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        field_name: str,
        choices: list[str],
        schema_name: str,
    ) -> tuple[str, dict[str, Any]]:
        """Forced single-choice question over an enum of ``choices``."""
        schema = {
            "type": "object",
            "properties": {field_name: {"type": "string", "enum": list(choices)}},
            "required": [field_name],
            "additionalProperties": False,
        }
        parsed, meta = self._post_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
            schema=schema,
        )
        value = parsed.get(field_name)
        if value not in choices:
            raise RuntimeError(f"Structured response field {field_name!r} was not one of {choices}.")
        return value, meta

    # 두 인접 run이 한 사람의 연속 발화인지 판단 
    def judge_boundary(
        self,
        left: SpeakerRun,
        right: SpeakerRun,
        context: list[SpeakerRun],
    ) -> tuple[bool, dict[str, Any]]:
        prompt = (
            f"{_format_context(context, target=right)}\n\n"
            f"Target boundary: RUN_{left.index} | RUN_{right.index}\n"
            "Do these two sides clearly belong to one person's uninterrupted "
            "utterance? Return the schema only."
        )
        return self._request_boolean(
            system_prompt=BOUNDARY_SYSTEM_PROMPT,
            user_prompt=prompt,
            field_name="same_uninterrupted_utterance",
            schema_name="speaker_boundary_judgment",
        )
    # 수정 후보를 최종 승인할지 검사 
    def verify_proposal(
        self,
        target: SpeakerRun,
        anchors: list[SpeakerRun],
        context: list[SpeakerRun],
    ) -> tuple[bool, dict[str, Any]]:
        anchor_names = ", ".join(f"RUN_{anchor.index}" for anchor in anchors)
        prompt = (
            f"{_format_context(context, target=target, anchors=anchors)}\n\n"
            f"Proposed repair: treat TARGET RUN_{target.index} and "
            f"ANCHOR {anchor_names} as the same speaker.\n"
            "Approve only if this is clearly one person's uninterrupted utterance. "
            "Return the schema only."
        )
        return self._request_boolean(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=prompt,
            field_name="approve_adjustment",
            schema_name="speaker_adjustment_verdict",
        )


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def build_speaker_runs(
    segments: list[dict[str, Any]],
    *,
    merge_gap_sec: float = 1.5,
) -> list[SpeakerRun]:
    if merge_gap_sec < 0:
        raise ValueError("merge_gap_sec must be non-negative.")

    runs: list[SpeakerRun] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        runs.append(
            SpeakerRun(
                index=len(runs),
                speaker=current["speaker"],
                segment_positions=tuple(current["segment_positions"]),
                segment_ids=tuple(current["segment_ids"]),
                start=current["start"],
                end=current["end"],
                speech_duration=current["speech_duration"],
                text=" ".join(part for part in current["text_parts"] if part).strip(),
            )
        )
        current = None

    for position, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        speaker = str(segment.get("speaker", "UNKNOWN"))
        duration = max(0.0, float(segment.get("duration", end - start)))
        segment_id = segment.get("index", position)
        text = _clean_text(segment.get("text"))

        can_merge = (
            current is not None
            and current["speaker"] == speaker
            and start - current["end"] <= merge_gap_sec
        )
        if not can_merge:
            flush()
            current = {
                "speaker": speaker,
                "segment_positions": [position],
                "segment_ids": [segment_id],
                "start": start,
                "end": end,
                "speech_duration": duration,
                "text_parts": [text],
            }
            continue

        current["segment_positions"].append(position)
        current["segment_ids"].append(segment_id)
        current["end"] = max(current["end"], end)
        current["speech_duration"] += duration
        current["text_parts"].append(text)

    flush()
    return runs

# 후보 run 주변의 문맥 가져옴 
def _context_window(
    runs: list[SpeakerRun],
    target_index: int,
    context_run_count: int,
) -> list[SpeakerRun]:
    if context_run_count < 2:
        raise ValueError("context_run_count must be at least 2.")
    count = min(context_run_count, len(runs))
    left_bias = count // 2
    start = max(0, target_index - left_bias)
    end = min(len(runs), start + count)
    start = max(0, end - count)
    return runs[start:end]


def _format_context(
    context: list[SpeakerRun],
    *,
    target: SpeakerRun | None = None,
    anchors: list[SpeakerRun] | None = None,
) -> str:
    anchor_indices = {anchor.index for anchor in anchors or []}
    lines = ["Transcript context (speaker IDs omitted):"]
    for run in context:
        markers = []
        if target is not None and run.index == target.index:
            markers.append("TARGET")
        if run.index in anchor_indices:
            markers.append("ANCHOR")
        marker = f" [{' '.join(markers)}]" if markers else ""
        lines.append(
            f"RUN_{run.index}{marker} [{run.start:.2f}-{run.end:.2f}]: {run.text}"
        )
    return "\n".join(lines)


def _boundary_is_eligible(
    left: SpeakerRun,
    right: SpeakerRun,
    *,
    max_gap_sec: float,
    max_overlap_sec: float,
) -> tuple[bool, str | None]:
    if left.speaker == right.speaker:
        return False, "same_label"
    gap = right.start - left.end
    if gap > max_gap_sec:
        return False, "gap_too_large"
    if gap < -max_overlap_sec:
        return False, "overlap"
    if not left.text or not right.text:
        return False, "empty_text"
    return True, None


def _resolve_target_speaker(
    runs,
    candidate_index,
    *,
    left_same,
    right_same,
):
    candidate = runs[candidate_index]
    left = runs[candidate_index - 1] if candidate_index > 0 else None
    right = runs[candidate_index + 1] if candidate_index + 1 < len(runs) else None

    if left_same and right_same:
        if left is not None and right is not None and left.speaker == right.speaker:
            return left.speaker, [left.index, right.index], "both_neighbors_agree"
        return None, [], "both_neighbors_conflict"
    if left_same and left is not None:
        return left.speaker, [left.index], "left_continuation"
    if right_same and right is not None:
        return right.speaker, [right.index], "right_continuation"
    return None, [], "no_continuation"


def _word_candidate_speakers(word: dict[str, Any]) -> set[str]:
    speakers = set()
    for candidate in word.get("speaker_candidates", []) or []:
        speaker = candidate.get("speaker")
        if speaker is not None:
            speakers.add(str(speaker))
    return speakers


def _run_has_stable_word_support(
    segments: list[dict[str, Any]],
    run: SpeakerRun,
) -> bool:
    """Whether a run has at least one word that is not a boundary coin flip."""
    for position in run.segment_positions:
        for word in segments[position].get("words", []) or []:
            if str(word.get("speaker", run.speaker)) != run.speaker:
                continue
            if bool(word.get("boundary_ambiguous", False)):
                continue
            if float(word.get("coverage", 0.0) or 0.0) >= DESTINATION_WORD_MIN_COVERAGE:
                return True
    return False


def _run_has_direct_visual_support(
    segments: list[dict[str, Any]],
    run: SpeakerRun,
) -> bool:
    """Whether direct face-speaking evidence supports most of this run.

    The LLM correction stage sees dialogue text only.  It can safely repair a
    weak diarization boundary, but it must not relabel a run whose words already
    have direct visual-ASD support for the current FACE label.
    """
    supported_duration = 0.0
    run_duration = 0.0
    for position in run.segment_positions:
        for word in segments[position].get("words", []) or []:
            if str(word.get("speaker", run.speaker)) != run.speaker:
                continue
            duration = max(
                0.0,
                float(word.get("end", 0.0)) - float(word.get("start", 0.0)),
            )
            run_duration += duration
            if (
                str(word.get("speaker_original", run.speaker)) == run.speaker
                and str(word.get("reconcile_source", "")) == "visual"
            ):
                supported_duration += duration
    return run_duration > 0 and supported_duration / run_duration >= 0.5


def _segment_direct_visual_share(segment: dict[str, Any]) -> float:
    """Share of a segment backed by its own direct visual-ASD evidence."""
    speaker = str(segment.get("speaker", "UNKNOWN"))
    total_duration = 0.0
    visual_duration = 0.0
    for word in segment.get("words", []) or []:
        if str(word.get("speaker", speaker)) != speaker:
            continue
        duration = max(
            0.0,
            float(word.get("end", 0.0)) - float(word.get("start", 0.0)),
        )
        total_duration += duration
        if (
            str(word.get("speaker_original", speaker)) == speaker
            and str(word.get("reconcile_source", "")) == "visual"
        ):
            visual_duration += duration
    return visual_duration / total_duration if total_duration else 0.0


def _refresh_segment_from_words(segment: dict[str, Any]) -> None:
    """Recalculate the aggregate fields after moving a boundary word."""
    words = segment.get("words", []) or []
    if not words:
        return
    segment["start"] = float(words[0]["start"])
    segment["end"] = float(words[-1]["end"])
    segment["duration"] = max(0.0, segment["end"] - segment["start"])
    segment["text"] = "".join(str(word.get("word", "")) for word in words).strip()
    segment["num_words"] = len(words)
    segment["overlap_sec"] = sum(float(word.get("overlap_sec", 0.0) or 0.0) for word in words)
    segment["coverage"] = sum(
        float(word.get("coverage", 0.0) or 0.0) for word in words
    ) / len(words)
    segment["boundary_ambiguous"] = any(
        bool(word.get("boundary_ambiguous", False)) for word in words
    )
    segment["match_methods"] = sorted(
        {str(word.get("matched_by")) for word in words if word.get("matched_by")}
    )
    # Normalized alignment words do not always retain ``asr_segment_id``.
    # Preserve the original aggregate IDs in that case rather than replacing
    # them with an empty list.
    if all("asr_segment_id" in word for word in words):
        segment["asr_segment_ids"] = sorted(
            {int(word["asr_segment_id"]) for word in words}
        )


def _semantic_boundary_tail_size(left_words: list[dict[str, Any]]) -> int:
    """Number of short trailing words that start a new sentence, else zero."""
    if len(left_words) < 2:
        return 0
    max_size = min(SEMANTIC_BOUNDARY_MAX_WORDS, len(left_words) - 1)
    for size in range(1, max_size + 1):
        prefix_last = _clean_text(left_words[-size - 1].get("word"))
        tail = left_words[-size:]
        duration = sum(
            max(0.0, float(word.get("end", 0.0)) - float(word.get("start", 0.0)))
            for word in tail
        )
        if prefix_last.endswith(SENTENCE_ENDINGS) and duration <= SEMANTIC_BOUNDARY_MAX_DURATION_SEC:
            return size
    return 0


def resolve_semantic_boundary_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move a short sentence opener to the next, visually anchored speaker.

    This fixes timestamp spill at a face transition without asking an LLM to
    guess the speaker: the target must already have strong direct visual
    support.  Only the trailing one or two words after ``.!?`` are moved.
    """
    repairs: list[dict[str, Any]] = []
    for position in range(len(segments) - 1):
        left, right = segments[position], segments[position + 1]
        source_speaker = str(left.get("speaker", "UNKNOWN"))
        target_speaker = str(right.get("speaker", "UNKNOWN"))
        if source_speaker == target_speaker or not target_speaker.startswith("FACE_"):
            continue
        gap = float(right.get("start", 0.0)) - float(left.get("end", 0.0))
        if abs(gap) > SEMANTIC_BOUNDARY_MAX_GAP_SEC:
            continue
        target_visual_share = _segment_direct_visual_share(right)
        if target_visual_share < SEMANTIC_TARGET_MIN_VISUAL_SHARE:
            continue
        left_words = list(left.get("words", []) or [])
        if not right.get("words"):
            continue
        tail_size = _semantic_boundary_tail_size(left_words)
        if tail_size == 0:
            continue

        retained_words = left_words[:-tail_size]
        moved_words = left_words[-tail_size:]
        if any(str(word.get("speaker", source_speaker)) != source_speaker for word in moved_words):
            continue

        for word in moved_words:
            original_speaker = str(word.get("speaker", source_speaker))
            word.setdefault("speaker_original", original_speaker)
            word["speaker"] = target_speaker
            word["speaker_correction_action"] = "semantic_boundary_word"
            word["correction_source"] = "semantic_boundary_word"

        left["words"] = retained_words
        right["words"] = moved_words + list(right.get("words", []) or [])
        _refresh_segment_from_words(left)
        _refresh_segment_from_words(right)
        right["semantic_boundary_word_repair"] = True
        repairs.append(
            {
                "left_segment_index": left.get("index", position),
                "right_segment_index": right.get("index", position + 1),
                "from_speaker": source_speaker,
                "to_speaker": target_speaker,
                "gap_sec": round(gap, 3),
                "target_visual_share": round(target_visual_share, 3),
                "moved_words": [_clean_text(word.get("word")) for word in moved_words],
                "sentence_ending": _clean_text(retained_words[-1].get("word"))[-1:],
            }
        )
    return repairs


def _find_weak_boundary_fragment(
    segments: list[dict[str, Any]],
    source_run: SpeakerRun,
    *,
    target_speaker: str,
    at_tail: bool,
) -> dict[str, Any] | None:
    """Find a tiny ambiguous edge segment that can safely move across a boundary."""
    position = (
        source_run.segment_positions[-1]
        if at_tail
        else source_run.segment_positions[0]
    )
    segment = segments[position]
    words = segment.get("words", []) or []
    start = float(segment["start"])
    end = float(segment["end"])
    if not words or len(words) > BOUNDARY_FRAGMENT_MAX_WORDS:
        return None
    if end - start > BOUNDARY_FRAGMENT_MAX_DURATION_SEC:
        return None

    for word in words:
        if str(word.get("speaker", source_run.speaker)) != source_run.speaker:
            return None
        if not bool(word.get("boundary_ambiguous", False)):
            return None
        if float(word.get("coverage", 1.0) or 0.0) > BOUNDARY_WORD_MAX_COVERAGE:
            return None
        if target_speaker not in _word_candidate_speakers(word):
            return None

    return {
        "segment_positions": [position],
        "segment_indices": [segment.get("index", position)],
        "start": start,
        "end": end,
        "text": _clean_text(segment.get("text")),
        "words": [_clean_text(word.get("word")) for word in words],
    }


def _boundary_fragment_proposal(
    segments: list[dict[str, Any]],
    runs: list[SpeakerRun],
    candidate_index: int,
    *,
    left_same: bool,
    right_same: bool,
) -> dict[str, Any] | None:
    """Prefer moving a weak edge fragment over relabeling a well-supported run."""
    candidate = runs[candidate_index]
    if not _run_has_stable_word_support(segments, candidate):
        return None

    left = runs[candidate_index - 1] if candidate_index > 0 else None
    right = runs[candidate_index + 1] if candidate_index + 1 < len(runs) else None

    if left_same and not right_same and left is not None:
        fragment = _find_weak_boundary_fragment(
            segments,
            left,
            target_speaker=candidate.speaker,
            at_tail=True,
        )
        if fragment is not None:
            return {
                "action": "move_boundary_fragment",
                "run_index": left.index,
                "candidate_run_index": candidate.index,
                "segment_positions": fragment["segment_positions"],
                "segment_indices": fragment["segment_indices"],
                "from_speaker": left.speaker,
                "to_speaker": candidate.speaker,
                "anchor_run_indices": [candidate.index],
                "resolution": "move_left_tail_to_candidate",
                "fragment": {
                    key: fragment[key]
                    for key in ("start", "end", "text", "words")
                },
                "verifier_approved": None,
                "status": "proposed",
            }

    if right_same and not left_same and right is not None:
        fragment = _find_weak_boundary_fragment(
            segments,
            right,
            target_speaker=candidate.speaker,
            at_tail=False,
        )
        if fragment is not None:
            return {
                "action": "move_boundary_fragment",
                "run_index": right.index,
                "candidate_run_index": candidate.index,
                "segment_positions": fragment["segment_positions"],
                "segment_indices": fragment["segment_indices"],
                "from_speaker": right.speaker,
                "to_speaker": candidate.speaker,
                "anchor_run_indices": [candidate.index],
                "resolution": "move_right_head_to_candidate",
                "fragment": {
                    key: fragment[key]
                    for key in ("start", "end", "text", "words")
                },
                "verifier_approved": None,
                "status": "proposed",
            }
    return None


def _fragment_run(
    proposal: dict[str, Any],
    source_run: SpeakerRun,
) -> SpeakerRun:
    fragment = proposal["fragment"]
    return SpeakerRun(
        index=source_run.index,
        speaker=source_run.speaker,
        segment_positions=tuple(int(x) for x in proposal["segment_positions"]),
        segment_ids=tuple(proposal["segment_indices"]),
        start=float(fragment["start"]),
        end=float(fragment["end"]),
        speech_duration=max(0.0, float(fragment["end"]) - float(fragment["start"])),
        text=str(fragment["text"]),
    )


def _apply_speaker(
    segments: list[dict[str, Any]],
    run: SpeakerRun,
    target_speaker: str,
    *,
    action: str = "relabel_run",
) -> None:
    for position in run.segment_positions:
        segment = segments[position]
        original_speaker = str(segment.get("speaker", "UNKNOWN"))
        segment.setdefault("speaker_original", original_speaker)
        segment.setdefault("coverage_speaker", original_speaker)
        segment["speaker_correction_action"] = action
        segment["correction_source"] = "context_continuation"
        segment["speaker"] = target_speaker
        for word in segment.get("words", []) or []:
            if str(word.get("speaker", original_speaker)) == original_speaker:
                word.setdefault("speaker_original", original_speaker)
                word["speaker"] = target_speaker


def _boundary_record(
    left: SpeakerRun,
    right: SpeakerRun,
    *,
    same_utterance: bool,
    metadata: dict[str, Any] | None = None,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    record = {
        "left_run_index": left.index,
        "right_run_index": right.index,
        "same_uninterrupted_utterance": same_utterance,
    }
    if skipped_reason is not None:
        record["skipped_reason"] = skipped_reason
    if metadata:
        record.update(metadata)
    return record


def find_candidate_run_indices(
    runs: list[SpeakerRun],
    *,
    max_candidate_duration_sec: float,
) -> list[int]:
    """Return the center run of each stride-1 three-run window.

    Edge runs normally provide context/anchors rather than becoming targets.
    When the whole input has only two runs, the clearly shorter side is used so
    a raw A-A-B sequence (which becomes A-run/B-run) can still be repaired.
    """
    if len(runs) == 2:
        left, right = runs
        if left.speaker == right.speaker:
            return []
        if (
            len(left.segment_positions) > len(right.segment_positions)
            and right.speech_duration <= max_candidate_duration_sec
        ):
            return [right.index]
        if (
            len(right.segment_positions) > len(left.segment_positions)
            and left.speech_duration <= max_candidate_duration_sec
        ):
            return [left.index]
        if (
            left.speech_duration <= max_candidate_duration_sec
            and left.speech_duration < right.speech_duration
        ):
            return [left.index]
        if (
            right.speech_duration <= max_candidate_duration_sec
            and right.speech_duration < left.speech_duration
        ):
            return [right.index]
        return []

    return [
        run.index
        for run in runs[1:-1]
        if run.text
        and run.speech_duration <= max_candidate_duration_sec
        and (
            runs[run.index - 1].speaker != run.speaker
            or runs[run.index + 1].speaker != run.speaker
        )
    ]


def run(
    artifact_dir: Path | None = None,
    video: Path | None = None,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    model: str | None = None,
    reasoning_effort: str = "low",
    env_file: Path | None = DEFAULT_ENV_FILE,
    max_candidate_duration_sec: float | None = None,
    run_merge_gap_sec: float = 1.5,
    max_boundary_gap_sec: float = 1.5,
    max_boundary_overlap_sec: float = 0.05,
    context_run_count: int = 3,
    verify_adjustments: bool = True,
    overwrite: bool = False,
    judge: BooleanJudge | None = None,
) -> dict[str, Any]:
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    source_path = (
        paths["gap_corrected_aligned_segments"]
        if paths["gap_corrected_aligned_segments"].exists()
        else paths["aligned_segments"]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Aligned JSON not found: {source_path}. Run alignment first."
        )
    if (
        paths["corrected_aligned_segments"].exists()
        and paths["final_aligned_segments"].exists()
        and paths["speaker_correction"].exists()
        and not overwrite
    ):
        return {
            "artifact_dir": artifact_dir,
            "paths": paths,
            "payload": read_json(paths["final_aligned_segments"]),
            "audit": read_json(paths["speaker_correction"]),
            "skipped": True,
        }

    max_candidate_duration_sec = resolve_max_candidate_duration(max_candidate_duration_sec)
    if max_candidate_duration_sec <= 0:
        raise ValueError("max_candidate_duration_sec must be positive.")

    aligned_payload = read_json(source_path)
    # Semantic boundary repair is word-level and happens before runs are
    # built.  Keep the original payload untouched for audit provenance.
    working_payload = copy.deepcopy(aligned_payload)
    source_segments = working_payload.get("segments", [])
    if not source_segments:
        raise ValueError(f"No aligned segments found in {source_path.name}")

    semantic_boundary_repairs = resolve_semantic_boundary_words(source_segments)

    runs = build_speaker_runs(source_segments, merge_gap_sec=run_merge_gap_sec)
    candidate_indices = find_candidate_run_indices(
        runs,
        max_candidate_duration_sec=max_candidate_duration_sec,
    )
    boundary_handled_indices = [
        run_index
        for run_index in candidate_indices
        if any(
            bool(source_segments[position].get("speaker_boundary_gap_checked"))
            for position in runs[run_index].segment_positions
        )
    ]
    boundary_handled_set = set(boundary_handled_indices)
    candidate_indices = [
        run_index
        for run_index in candidate_indices
        if run_index not in boundary_handled_set
    ]

    resolved_model = model or os.environ.get(
        "SEEUTTER_SPEAKER_CORRECTION_MODEL", DEFAULT_MODEL
    )
    if judge is None and candidate_indices:
        judge = OpenAIResponsesBooleanClient(
            api_key=load_openai_api_key(env_file),
            model=resolved_model,
            reasoning_effort=reasoning_effort,
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        )
    judge_model = getattr(judge, "model", resolved_model)

    boundary_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def check_boundary(
        left: SpeakerRun | None,
        right: SpeakerRun | None,
        context: list[SpeakerRun],
    ) -> bool:
        if left is None or right is None:
            return False
        key = (left.index, right.index)
        if key in boundary_cache:
            return bool(boundary_cache[key]["same_uninterrupted_utterance"])

        eligible, skipped_reason = _boundary_is_eligible(
            left,
            right,
            max_gap_sec=max_boundary_gap_sec,
            max_overlap_sec=max_boundary_overlap_sec,
        )
        if not eligible:
            record = _boundary_record(
                left,
                right,
                same_utterance=False,
                skipped_reason=skipped_reason,
            )
        else:
            if judge is None:
                raise RuntimeError("No boundary judge is available.")
            verdict, metadata = judge.judge_boundary(left, right, context)
            record = _boundary_record(
                left,
                right,
                same_utterance=verdict,
                metadata=metadata,
            )
        boundary_cache[key] = record
        return bool(record["same_uninterrupted_utterance"])

    proposals: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for candidate_index in candidate_indices:
        candidate = runs[candidate_index]
        left = runs[candidate_index - 1] if candidate_index > 0 else None
        right = (
            runs[candidate_index + 1]
            if candidate_index + 1 < len(runs)
            else None
        )
        context = _context_window(runs, candidate_index, context_run_count)
        left_same = check_boundary(left, candidate, context)
        right_same = check_boundary(candidate, right, context)
        boundary_proposal = _boundary_fragment_proposal(
            source_segments,
            runs,
            candidate_index,
            left_same=left_same,
            right_same=right_same,
        )
        target_speaker, anchor_indices, resolution = _resolve_target_speaker(
            runs,
            candidate_index,
            left_same=left_same,
            right_same=right_same,
        )
        if boundary_proposal is not None:
            resolution = str(boundary_proposal["resolution"])
        visually_anchored = _run_has_direct_visual_support(
            source_segments,
            candidate,
        )

        candidate_record = {
            "run_index": candidate.index,
            "segment_indices": list(candidate.segment_ids),
            "speaker": candidate.speaker,
            "speech_duration": candidate.speech_duration,
            "left_same_uninterrupted_utterance": left_same,
            "right_same_uninterrupted_utterance": right_same,
            "resolution": resolution,
            "direct_visual_support": visually_anchored,
        }
        candidate_records.append(candidate_record)

        if boundary_proposal is not None:
            proposals.append(boundary_proposal)
            continue
        if visually_anchored:
            # A dialogue-only model cannot overrule direct face-speaking
            # evidence.  Keep the FACE label, even when neighbouring text is
            # linguistically continuous with another run.
            candidate_record["resolution"] = "preserve_direct_visual"
            continue
        if target_speaker is None or target_speaker == candidate.speaker:
            continue
        proposals.append(
            {
                "action": "relabel_run",
                "run_index": candidate.index,
                "segment_indices": list(candidate.segment_ids),
                "from_speaker": candidate.speaker,
                "to_speaker": target_speaker,
                "anchor_run_indices": anchor_indices,
                "resolution": resolution,
                "verifier_approved": None,
                "status": "proposed",
            }
        )

    for proposal in proposals:
        if not verify_adjustments:
            proposal["verifier_approved"] = True
            continue
        if judge is None:
            raise RuntimeError("No proposal verifier is available.")
        proposal_run = runs[int(proposal["run_index"])]
        anchors = [runs[index] for index in proposal["anchor_run_indices"]]
        context_index = int(proposal.get("candidate_run_index", proposal["run_index"]))
        context = _context_window(runs, context_index, context_run_count)
        if proposal.get("action") == "move_boundary_fragment":
            target = _fragment_run(proposal, proposal_run)
            context = [
                target if run.index == proposal_run.index else run
                for run in context
            ]
        else:
            target = proposal_run
        approved, metadata = judge.verify_proposal(target, anchors, context)
        proposal["verifier_approved"] = approved
        proposal["verification"] = metadata
        if not approved:
            proposal["status"] = "verifier_rejected"

    approved_by_run = {
        int(proposal["run_index"]): proposal
        for proposal in proposals
        if proposal["verifier_approved"]
    }
    conflicted_run_indices: set[int] = set()
    for run_index, proposal in approved_by_run.items():
        for anchor_index in proposal["anchor_run_indices"]:
            anchor_proposal = approved_by_run.get(int(anchor_index))
            if (
                anchor_proposal is not None
                and anchor_proposal["to_speaker"] != proposal["to_speaker"]
            ):
                conflicted_run_indices.update({run_index, int(anchor_index)})

    corrected_payload = copy.deepcopy(working_payload)
    corrected_segments = corrected_payload["segments"]
    applied_corrections = []
    for proposal in proposals:
        run_index = int(proposal["run_index"])
        if not proposal["verifier_approved"]:
            continue
        if run_index in conflicted_run_indices:
            proposal["status"] = "conflict"
            continue
        proposal["status"] = "applied"
        if proposal.get("action") == "move_boundary_fragment":
            run_to_apply = _fragment_run(proposal, runs[run_index])
        else:
            run_to_apply = runs[run_index]
        _apply_speaker(
            corrected_segments,
            run_to_apply,
            str(proposal["to_speaker"]),
            action=str(proposal.get("action", "relabel_run")),
        )
        applied_corrections.append(
            {
                "action": proposal.get("action", "relabel_run"),
                "run_index": run_index,
                "segment_indices": proposal["segment_indices"],
                "from_speaker": proposal["from_speaker"],
                "to_speaker": proposal["to_speaker"],
                "resolution": proposal["resolution"],
            }
        )

    corrected_payload["source_aligned_segments"] = str(source_path.resolve())
    corrected_payload["speaker_correction"] = {
        "source_audit": str(paths["speaker_correction"].resolve()),
        "model": judge_model,
        "num_corrections": len(applied_corrections),
    }

    audit_payload = {
        "video_id": aligned_payload.get("video_id") or artifact_dir.name,
        "source_aligned_segments": str(source_path.resolve()),
        "corrected_aligned_segments": str(
            paths["corrected_aligned_segments"].resolve()
        ),
        "final_aligned_segments": str(paths["final_aligned_segments"].resolve()),
        "model": judge_model,
        "reasoning_effort": reasoning_effort,
        "config": {
            "max_candidate_duration_sec": (
                "inf"
                if max_candidate_duration_sec == float("inf")
                else max_candidate_duration_sec
            ),
            "run_merge_gap_sec": run_merge_gap_sec,
            "max_boundary_gap_sec": max_boundary_gap_sec,
            "max_boundary_overlap_sec": max_boundary_overlap_sec,
            "context_run_count": context_run_count,
            "verify_adjustments": verify_adjustments,
            "boundary_fragment_max_duration_sec": BOUNDARY_FRAGMENT_MAX_DURATION_SEC,
            "boundary_fragment_max_words": BOUNDARY_FRAGMENT_MAX_WORDS,
            "boundary_word_max_coverage": BOUNDARY_WORD_MAX_COVERAGE,
            "destination_word_min_coverage": DESTINATION_WORD_MIN_COVERAGE,
            "semantic_boundary_max_words": SEMANTIC_BOUNDARY_MAX_WORDS,
            "semantic_boundary_max_duration_sec": SEMANTIC_BOUNDARY_MAX_DURATION_SEC,
            "semantic_boundary_max_gap_sec": SEMANTIC_BOUNDARY_MAX_GAP_SEC,
            "semantic_target_min_visual_share": SEMANTIC_TARGET_MIN_VISUAL_SHARE,
        },
        "semantic_boundary_repairs": semantic_boundary_repairs,
        "num_semantic_boundary_repairs": len(semantic_boundary_repairs),
        "num_runs": len(runs),
        "num_candidates": len(candidate_indices),
        "num_boundary_handled_skips": len(boundary_handled_indices),
        "num_proposals": len(proposals),
        "num_corrections": len(applied_corrections),
        "runs": [run.to_audit_dict() for run in runs],
        "boundary_checks": list(boundary_cache.values()),
        "candidates": candidate_records,
        "proposals": proposals,
        "applied_corrections": applied_corrections,
    }

    write_json(paths["corrected_aligned_segments"], corrected_payload)
    write_json(paths["final_aligned_segments"], corrected_payload)
    write_json(paths["speaker_correction"], audit_payload)
    return {
        "artifact_dir": artifact_dir,
        "paths": paths,
        "payload": corrected_payload,
        "audit": audit_payload,
        "skipped": False,
    }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--artifact-dir", type=Path, required=True)
#     parser.add_argument("--model", default=None)
#     parser.add_argument(
#         "--reasoning-effort",
#         choices=["none", "low", "medium", "high", "xhigh", "max"],
#         default="medium",
#     )
#     parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
#     parser.add_argument("--max-candidate-duration-sec", type=float, default=None)
#     parser.add_argument(
#         "--no-duration-gate",
#         dest="no_duration_gate",
#         action="store_true",
#         help="Judge every run by context regardless of length (disables the duration filter).",
#     )
#     parser.add_argument("--run-merge-gap-sec", type=float, default=1.5)
#     parser.add_argument("--max-boundary-gap-sec", type=float, default=1.5)
#     parser.add_argument("--max-boundary-overlap-sec", type=float, default=0.05)
#     parser.add_argument("--context-run-count", type=int, default=3)
#     parser.add_argument(
#         "--no-verify-adjustments",
#         dest="verify_adjustments",
#         action="store_false",
#     )
#     parser.add_argument("--overwrite", action="store_true")
#     parser.set_defaults(verify_adjustments=True)
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     result = run(
#         artifact_dir=args.artifact_dir,
#         model=args.model,
#         reasoning_effort=args.reasoning_effort,
#         env_file=args.env_file,
#         max_candidate_duration_sec=(
#             float("inf") if args.no_duration_gate else args.max_candidate_duration_sec
#         ),
#         run_merge_gap_sec=args.run_merge_gap_sec,
#         max_boundary_gap_sec=args.max_boundary_gap_sec,
#         max_boundary_overlap_sec=args.max_boundary_overlap_sec,
#         context_run_count=args.context_run_count,
#         verify_adjustments=args.verify_adjustments,
#         overwrite=args.overwrite,
#     )
#     audit = result["audit"]
#     print(f"Corrected segments: {result['paths']['corrected_aligned_segments']}")
#     print(f"Speaker correction audit: {result['paths']['speaker_correction']}")
#     print(
#         f"Candidates: {audit['num_candidates']}, "
#         f"corrections: {audit['num_corrections']}"
#     )


# if __name__ == "__main__":
#     main()
