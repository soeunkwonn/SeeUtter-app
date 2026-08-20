from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from workflow import (
    clear_participant,
    current_participant,
    init_session,
    set_participant,
)


st.set_page_config(page_title="SeeUtter", page_icon="🎬", layout="wide")
init_session()

# Participant gate: everyone opens the same URL, then identifies themselves.
# The number keeps each person's names / captions / renders isolated. Placed
# before st.navigation so it blocks every page until a number is entered.
participant = current_participant()
if not participant:
    st.title("SeeUtter")
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

# Korean-labelled navigation (replaces the English filename-based sidebar).
pages = [
    st.Page("home.py", title="처음", icon="🏠", default=True),
    st.Page("pages/01_speaker_names.py", title="이름 정하기", icon="✏️"),
    st.Page("pages/02_caption_position.py", title="자막 위치", icon="📍"),
    st.Page("pages/04_render.py", title="완성 영상", icon="🎬"),
]
st.navigation(pages).run()
