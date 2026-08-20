"""Translate caption text to Korean (batched, one API call per ~batch).

Runs after names are applied (final_caption_data.json). A small deterministic
cue pass first splits only long segments at existing word timestamps. The
original English stays in ``text``; the Korean goes in ``text_ko``. The
renderer prefers ``text_ko`` when present, so this is non-destructive and
idempotent (already-translated segments are skipped).

Reuses the OpenAI structured-output client from the speaker-correction stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..lib.caption_cues import CAPTION_CUE_VERSION, split_caption_cues
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

TARGET_LANG = "Korean"
BATCH_SIZE = 40  # subtitle lines per API call (lines are short)

TRANSLATE_SYSTEM_PROMPT = """\
당신은 영상 자막을 자연스러운 한국어로 번역합니다. 각 항목의 text는 한 줄의 대사입니다.
- 주어진 각 항목을 한국어로 번역하고 같은 i 값을 그대로 유지하세요.
- 구어체·대사 톤을 살리되 자막답게 간결하게.
- 항목을 합치거나 나누지 말고 개수를 그대로 유지하세요.
- 이름/고유명사는 자연스럽게 음차하거나 유지하세요.
- 화자 이름이나 이모지 같은 접두사는 주어지지 않습니다. 순수 대사만 번역하세요.
- 문장 뒤에 온 점을 찍지 마세요. 
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "ko": {"type": "string"},
                },
                "required": ["i", "ko"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def _translate_batch(client, items: list[tuple[int, str]]) -> tuple[dict[int, str], dict]:
    payload = [{"i": i, "text": t} for i, t in items]
    user_prompt = (
        "다음 각 항목의 text를 한국어로 번역해 같은 i로 돌려주세요.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    parsed, meta = client.post_structured(
        system_prompt=TRANSLATE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema_name="subtitle_translation",
        schema=_SCHEMA,
    )
    mapping = {int(row["i"]): str(row["ko"]) for row in parsed.get("translations", [])}
    return mapping, meta


def run(
    artifact_dir: Path | None = None,
    video: Path | None = None,
    out_root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    model: str | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
    batch_size: int = BATCH_SIZE,
    reasoning_effort: str = "low",
    overwrite: bool = False,
    client: OpenAIResponsesBooleanClient | None = None,
) -> dict[str, Any]:
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    src = paths["final_caption_data"]
    if not src.exists():
        raise FileNotFoundError(
            f"Final caption data not found: {src}. Run apply_speaker_names.py first."
        )

    payload = read_json(src)
    segments = payload.get("segments", [])
    source_segment_count = len(segments)

    # Speaker naming recreates final_caption_data, so a new version marker
    # means we can safely re-cue and retranslate from untouched source words.
    if payload.get("caption_cue_version") != CAPTION_CUE_VERSION:
        segments = split_caption_cues(segments)
        payload["segments"] = segments
        payload["num_segments"] = len(segments)
        payload["caption_cue_version"] = CAPTION_CUE_VERSION
        payload["caption_cues"] = {
            "source_segment_count": source_segment_count,
            "cue_count": len(segments),
            "source": "word_timestamps",
        }

    # segments that still need a Korean line
    items = [
        (i, str(seg.get("text") or "").strip())
        for i, seg in enumerate(segments)
        if str(seg.get("text") or "").strip()
        and (overwrite or not seg.get("text_ko"))
    ]
    if not items:
        return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload,
                "num_translated": 0, "skipped": True}

    resolved_model = model or os.environ.get("SEEUTTER_SPEAKER_CORRECTION_MODEL", DEFAULT_MODEL)
    if client is None:
        client = OpenAIResponsesBooleanClient(
            api_key=load_openai_api_key(env_file),
            model=resolved_model,
            reasoning_effort=reasoning_effort,
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        )

    n_translated = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        mapping, _meta = _translate_batch(client, chunk)
        for i, _text in chunk:
            if i in mapping:
                segments[i]["text_ko"] = mapping[i]
                n_translated += 1

    payload["segments"] = segments
    payload["translation"] = {
        "model": getattr(client, "model", resolved_model),
        "target_lang": TARGET_LANG,
        "num_segments": len(items),
        "num_translated": n_translated,
        "caption_cue_version": CAPTION_CUE_VERSION,
    }
    write_json(src, payload)
    write_json(artifact_dir / "translation.json", {
        "video_id": payload.get("video_id") or artifact_dir.name,
        "model": getattr(client, "model", resolved_model),
        "target_lang": TARGET_LANG,
        "num_segments": len(items),
        "num_translated": n_translated,
    })
    return {"artifact_dir": artifact_dir, "paths": paths, "payload": payload,
            "num_translated": n_translated, "skipped": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        artifact_dir=args.artifact_dir,
        model=args.model,
        env_file=args.env_file,
        batch_size=args.batch_size,
        reasoning_effort=args.reasoning_effort,
        overwrite=args.overwrite,
    )
    print(f"translated {result['num_translated']} segment(s) -> text_ko")


if __name__ == "__main__":
    main()
