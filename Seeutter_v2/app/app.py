
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


st.set_page_config(page_title="SeeUtter", page_icon="🎬", layout="wide")
init_session()

st.title("SeeUtter")

# Participant gate: everyone opens the same URL, then identifies themselves.
# The id keeps each person's speaker names / captions / renders isolated.
participant = current_participant()
if not participant:
    st.subheader("번호를 입력해 주세요")
    st.caption("받으신 번호를 입력해 주세요. (예: P07)")
    with st.form("participant_gate"):
        entered = st.text_input("번호", placeholder="P07", max_chars=64)
        submitted = st.form_submit_button("시작하기", type="primary")
    if submitted:
        if set_participant(entered):
            st.rerun()
        else:
            st.error("번호를 다시 확인해 주세요. (예: P07)")
    st.stop()

with st.sidebar:
    st.caption(f"번호: **{participant}**")
    if st.button("다음 사람 시작하기"):
        clear_participant()
        st.rerun()

st.caption("영상을 고르고, 화자명을 완성하세요.")

items = library_items()
if not items:
    st.info("지금은 영상을 불러올 수 없어요. 연구자에게 알려 주세요.")
    st.stop()

labels = {
    item["id"]: f"{item['title']}"
    for item in items
}
current_id = st.session_state.get("library_artifact_id")
ids = [item["id"] for item in items]
initial_index = ids.index(current_id) if current_id in ids else 0
selected_id = st.selectbox(
    "영상을 골라 주세요",
    ids,
    index=initial_index,
    format_func=lambda item_id: labels[item_id],
)
selected = next(item for item in items if item["id"] == selected_id)

st.video(str(selected["source_video"]))

next_col, reset_col = st.columns((1, 5))
with next_col:
    if st.button("이 영상으로 시작하기", type="primary"):
        select_library_item(selected_id)
        st.switch_page("pages/01_speaker_names.py")
with reset_col:
    if st.session_state.get("library_artifact_id") and st.button("다시 고르기"):
        clear_selection()
        st.rerun()
