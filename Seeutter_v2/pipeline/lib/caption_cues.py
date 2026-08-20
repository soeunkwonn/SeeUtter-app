"""Small deterministic caption-cue splitter based on existing word timings.

This intentionally sits between speaker naming and translation.  It never
changes a word, speaker, emotion, or timestamp; it only turns an overly long
speech segment into smaller, readable caption cues at word boundaries.
"""
from __future__ import annotations

import copy
from typing import Any


CAPTION_CUE_VERSION = 1

# Conservative defaults: short reactions remain intact, while long spoken
# sentences get a natural break before they become a very wide/tall box.
MIN_CUE_DURATION_SEC = 0.75
MAX_CUE_DURATION_SEC = 3.50
MAX_WORDS_PER_CUE = 12

_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "e.g.", "i.e.",
}
_SENTENCE_ENDINGS = (".", "!", "?", "…", "！", "？")
_SOFT_ENDINGS = (",", ";", ":", "，", "；", "：")


def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word") or "")


def _join_words(words: list[dict[str, Any]]) -> str:
    """Join Whisper-style tokens without depending on every token's leading space."""
    text = ""
    for word in words:
        token = _word_text(word)
        stripped = token.strip()
        if not stripped:
            continue
        if not text:
            text = stripped
        elif token[:1].isspace() or stripped[:1] in ".,!?:;…)]}":
            text += token
        else:
            text += " " + token
    return text.strip()


def _valid_words(segment: dict[str, Any]) -> list[dict[str, Any]] | None:
    words = segment.get("words")
    if not isinstance(words, list) or len(words) < 2:
        return None
    output: list[dict[str, Any]] = []
    previous_end = float("-inf")
    for word in words:
        if not isinstance(word, dict) or not _word_text(word).strip():
            return None
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if end < start or start < previous_end:
            return None
        previous_end = end
        output.append(word)
    return output


def _duration(words: list[dict[str, Any]], start: int, stop: int) -> float:
    return float(words[stop - 1]["end"]) - float(words[start]["start"])


def _is_sentence_boundary(word: dict[str, Any]) -> bool:
    token = _word_text(word).strip()
    lowered = token.lower()
    return bool(token) and lowered not in _ABBREVIATIONS and token.endswith(_SENTENCE_ENDINGS)


def _is_soft_boundary(word: dict[str, Any]) -> bool:
    token = _word_text(word).strip()
    return _is_sentence_boundary(word) or bool(token) and token.endswith(_SOFT_ENDINGS)


def _tail_is_viable(words: list[dict[str, Any]], cut: int) -> bool:
    """Avoid leaving a lone word or an unreadably brief final cue."""
    remaining_words = len(words) - cut
    if remaining_words == 0:
        return True
    return (
        remaining_words >= 2
        and _duration(words, cut, len(words)) >= MIN_CUE_DURATION_SEC
    )


def _next_cut(words: list[dict[str, Any]], start: int) -> int:
    """Return the exclusive word index of the next readable cue."""
    n_words = len(words)
    for stop in range(start + 1, n_words + 1):
        cue_duration = _duration(words, start, stop)
        cue_word_count = stop - start
        has_tail = stop < n_words

        # A complete sentence is the best break whenever both sides can stay
        # on screen long enough. This is checked before the hard limits.
        if (
            has_tail
            and _is_sentence_boundary(words[stop - 1])
            and cue_duration >= MIN_CUE_DURATION_SEC
            and _tail_is_viable(words, stop)
        ):
            return stop

        if (
            has_tail
            and (cue_duration >= MAX_CUE_DURATION_SEC or cue_word_count >= MAX_WORDS_PER_CUE)
        ):
            # Prefer the latest comma/sentence boundary before the limit. If
            # none exists, use the current word boundary rather than retaining
            # an overly long cue.
            preferred = [
                cut
                for cut in range(start + 1, stop + 1)
                if _is_soft_boundary(words[cut - 1])
                and _duration(words, start, cut) >= MIN_CUE_DURATION_SEC
                and _tail_is_viable(words, cut)
            ]
            if preferred:
                return preferred[-1]
            if _tail_is_viable(words, stop):
                return stop
    return n_words


def _cue_from_words(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    *,
    parent_index: int,
    part: int,
    part_count: int,
) -> dict[str, Any]:
    cue = copy.deepcopy(segment)
    cue.pop("text_ko", None)  # parent translation no longer matches this cue
    cue["words"] = copy.deepcopy(words)
    cue["text"] = _join_words(words)
    cue["start"] = float(words[0]["start"])
    cue["end"] = float(words[-1]["end"])
    cue["duration"] = cue["end"] - cue["start"]
    cue["num_words"] = len(words)
    cue["caption_cue"] = {
        "version": CAPTION_CUE_VERSION,
        "parent_index": parent_index,
        "part": part,
        "part_count": part_count,
    }
    return cue


def split_caption_cues(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split only long source segments; return fresh, timestamped caption cues."""
    cues: list[dict[str, Any]] = []
    for parent_position, segment in enumerate(segments):
        parent_index = int(segment.get("index", parent_position))
        words = _valid_words(segment)
        if words is None:
            cue = copy.deepcopy(segment)
            cue.pop("text_ko", None)
            cue["caption_cue"] = {
                "version": CAPTION_CUE_VERSION,
                "parent_index": parent_index,
                "part": 1,
                "part_count": 1,
            }
            cues.append(cue)
            continue

        boundaries: list[int] = []
        start = 0
        while start < len(words):
            stop = _next_cut(words, start)
            boundaries.append(stop)
            start = stop

        start = 0
        for part, stop in enumerate(boundaries, 1):
            cues.append(
                _cue_from_words(
                    segment,
                    words[start:stop],
                    parent_index=parent_index,
                    part=part,
                    part_count=len(boundaries),
                )
            )
            start = stop

    for index, cue in enumerate(cues):
        cue["index"] = index
    return cues
