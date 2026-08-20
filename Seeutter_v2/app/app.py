
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from workflow import (
    clear_participant,
    clear_selection,
    current_participant,
    init_session,
    library_items,
    select_library_item,
    set_participant,
)


st.set_page_config(page_title="SeeUtter Library", page_icon="🎬", layout="wide")
init_session()

st.title("SeeUtter Library")

# Participant gate: everyone opens the same URL, then identifies themselves.
# The id keeps each person's speaker names / captions / renders isolated.
participant = current_participant()
if not participant:
    st.subheader("참가자 번호를 입력해주세요")
    st.caption("실험 안내에서 받은 번호를 그대로 입력하세요. 예: P07")
    with st.form("participant_gate"):
        entered = st.text_input("참가자 번호", placeholder="P07", max_chars=64)
        submitted = st.form_submit_button("시작하기", type="primary")
    if submitted:
        if set_participant(entered):
            st.rerun()
        else:
            st.error("영문·숫자로 된 번호를 입력해주세요. 예: P07")
    st.stop()

with st.sidebar:
    st.caption(f"참가자: **{participant}**")
    if st.button("다른 참가자로 전환"):
        clear_participant()
        st.rerun()

st.caption("감정 분석까지 끝난 영상을 고르고, 화자명과 자막 스타일을 완성하세요.")

items = library_items()
if not items:
    st.info(
        "아직 준비된 영상이 없습니다. `python -m Seeutter_v2.pipeline.prepare_library`로 "
        "화자 분석과 SenseVoice 감정 분석을 먼저 실행해주세요."
    )
    st.stop()

labels = {
    item["id"]: f"{item['title']}"
    for item in items
}
current_id = st.session_state.get("library_artifact_id")
ids = [item["id"] for item in items]
initial_index = ids.index(current_id) if current_id in ids else 0
selected_id = st.selectbox(
    "영상 고르기",
    ids,
    index=initial_index,
    format_func=lambda item_id: labels[item_id],
)
selected = next(item for item in items if item["id"] == selected_id)

left, right = st.columns((2, 1))
with left:
    st.video(str(selected["source_video"]))
with right:
    duration = selected["duration_sec"]
    if duration:
        st.metric("영상 길이", f"{duration:.1f}초")

next_col, reset_col = st.columns((1, 5))
with next_col:
    if st.button("이 영상으로 시작", type="primary"):
        select_library_item(selected_id)
        st.switch_page("pages/01_speaker_names.py")
with reset_col:
    if st.session_state.get("library_artifact_id") and st.button("선택 해제"):
        clear_selection()
        st.rerun()
