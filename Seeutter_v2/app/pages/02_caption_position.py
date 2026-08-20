"""Wizard page: choose one of the existing caption-position presets."""
from __future__ import annotations

import sys
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PAGE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from subtitle_position import POSITION_PRESETS
from workflow import (
    caption_segments,
    init_session,
    load_caption_style,
    position_previews,
    require_current_artifact,
    save_caption_style,
)


st.set_page_config(page_title="SeeUtter · 자막 위치", page_icon="📍", layout="wide")
init_session()
artifact_dir = require_current_artifact()
st.session_state.wizard_step = "position"

st.title("2. 자막 위치 정하기")
st.caption("영상에서 자막 위치를 골라 주세요.")

segments = caption_segments(artifact_dir)
if not segments:
    st.warning("먼저 이름을 정해 주세요.")
    if st.button("이름 정하러 가기"):
        st.switch_page("pages/01_speaker_names.py")
    st.stop()

segment_options = list(range(len(segments)))
selected_index = st.selectbox(
    "미리 볼 자막 고르기",
    segment_options,
    index=min(st.session_state.get("preview_segment_index", 0), len(segments) - 1),
    format_func=lambda index: (
        f"{float(segments[index].get('start', 0)):.1f}초 · "
        f"{str(segments[index].get('text_ko') or segments[index].get('text') or '')[:55]}"
    ),
)
st.session_state.preview_segment_index = selected_index

try:
    previews, _segment = position_previews(artifact_dir, selected_index)
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error("미리보기를 만들지 못했어요.")
    st.stop()

style = load_caption_style(artifact_dir)
keys = list(POSITION_PRESETS)
default_index = keys.index(style["position"]) if style["position"] in keys else 0
position = st.radio(
    "자막 위치",
    keys,
    index=default_index,
    horizontal=True,
    format_func=lambda key: POSITION_PRESETS[key]["label"],
)
_preview_left, preview_column, _preview_right = st.columns((1, 8, 1))
with preview_column:
    st.image(
        # Pass the current file bytes rather than only its path.  Streamlit can
        # otherwise keep serving the previous media entry when a preview is
        # regenerated in-place under the same filename.
        previews[position].read_bytes(),
        caption="완성 영상에서 이렇게 보여요",
        use_container_width=True,
    )

back_col, _spacer, next_col = st.columns((1, 7.5, 1.3))
with back_col:
    if st.button("이름 다시 정하기"):
        st.switch_page("pages/01_speaker_names.py")
with next_col:
    if st.button("저장하고 영상 만들기", type="primary", use_container_width=True):
        style["position"] = position
        save_caption_style(artifact_dir, style)
        st.session_state.wizard_step = "render"
        st.switch_page("pages/04_render.py")
