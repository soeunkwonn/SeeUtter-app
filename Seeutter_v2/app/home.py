"""Home page: pick a video to start naming. Rendered inside st.navigation."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from workflow import clear_selection, library_items, select_library_item

st.title("SeeUtter")
st.caption("영상을 고르고, 화자명을 완성하세요.")

items = library_items()
if not items:
    st.info("지금은 영상을 불러올 수 없어요. 연구자에게 알려 주세요.")
    st.stop()

labels = {item["id"]: f"{item['title']}" for item in items}
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
