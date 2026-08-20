"""Wizard page: let the user name the diarized speakers."""
from __future__ import annotations

import sys
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import cloud_store
from workflow import (
    build_speaker_preview_images,
    current_participant,
    init_session,
    load_speaker_map,
    prepare_caption_data,
    require_current_artifact,
    save_speaker_map,
    speaker_face_hints,
    speaker_face_paths,
    speaker_rows,
)


init_session()
artifact_dir = require_current_artifact()
st.session_state.wizard_step = "names"

st.title("1. 말하는 사람 이름 정하기")
st.caption("사진을 보고, 말하는 사람의 이름을 적어 주세요.")
st.markdown(
    """
    <style>
    /* Keep every speaker card visually aligned even when face crops and
       representative scene frames have different aspect ratios. */
    [data-testid="stImage"],
    [data-testid="stImage"] > div {
        width: 100% !important;
    }
    [data-testid="stImage"] img {
        display: block;
        width: 100% !important;
        max-width: none !important;
        height: 260px !important;
        object-fit: cover;
        object-position: center;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    rows = speaker_rows(artifact_dir)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

if not rows:
    st.warning("이름을 정할 사람을 찾지 못했어요.")
    if st.button("처음으로"):
        st.switch_page("app.py")
    st.stop()

speaker_ids = [str(row["speaker_id"]) for row in rows]
with st.spinner("사진을 준비하고 있어요..."):
    preview_images = build_speaker_preview_images(artifact_dir, speaker_ids)
    face_paths = speaker_face_paths(artifact_dir, speaker_ids)
    hint_paths = speaker_face_hints(artifact_dir, speaker_ids)

saved_names = load_speaker_map(artifact_dir)
names: dict[str, str] = {}
with st.form("speaker_names"):
    for start in range(0, len(rows), 3):
        for column, row in zip(st.columns(3), rows[start : start + 3]):
            with column:
                speaker_id = str(row["speaker_id"])
                if speaker_id.startswith("AUDIO_"):
                    st.caption("목소리로 구분한 사람")
                face_path = face_paths.get(speaker_id)
                hint_path = hint_paths.get(speaker_id)
                preview_image = preview_images.get(speaker_id)
                if face_path:
                    st.caption("얼굴 사진")
                elif hint_path:
                    st.caption("얼굴 추정 (확인 후 이름 적기)")
                else:
                    st.caption("장면 사진")
                if face_path:
                    st.image(str(face_path), use_container_width=True)
                elif hint_path:
                    st.image(str(hint_path), use_container_width=True)
                elif preview_image:
                    st.image(str(preview_image), use_container_width=True)
                else:
                    st.info("사진을 만들지 못했어요.")
                names[speaker_id] = st.text_input(
                    "이름",
                    value=saved_names.get(speaker_id, speaker_id),
                    key=f"speaker_name_{artifact_dir.name}_{speaker_id}",
                )

    back_col, _spacer, next_col = st.columns((1, 7.5, 1.3))
    with back_col:
        back = st.form_submit_button("처음으로")
    with next_col:
        next_step = st.form_submit_button(
            "저장하고 다음으로",
            type="primary",
            use_container_width=True,
        )

if back:
    st.switch_page("app.py")
if next_step:
    save_speaker_map(artifact_dir, names)
    with st.spinner("이름을 적용하고 자막을 만들고 있어요..."):
        try:
            prepare_caption_data(artifact_dir)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()
    # Persist the naming decisions off the (ephemeral) cloud disk immediately.
    cloud_store.log_event(current_participant(), artifact_dir.name, "names_confirmed", names=names)
    st.session_state.wizard_step = "position"
    st.switch_page("pages/02_caption_position.py")
