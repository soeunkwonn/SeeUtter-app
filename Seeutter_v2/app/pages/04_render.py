"""Wizard page: render and download the completed video."""
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
    RENDER_LOCK,
    clear_selection,
    current_participant,
    init_session,
    load_caption_style,
    render_caption_video,
    require_current_artifact,
)


st.set_page_config(page_title="SeeUtter · 렌더", page_icon="🎞️", layout="wide")
init_session()
artifact_dir = require_current_artifact()
st.session_state.wizard_step = "render"

st.title("3. 완성 영상 만들기")
style = load_caption_style(artifact_dir)
st.info(
    f"자막 위치: `{style['position']}` · 감정 이모지와 흰색 자막을 적용합니다."
)

if st.button("완성 영상 만들기", type="primary"):
    with st.spinner("자막 PNG를 만들고 영상에 입히고 있습니다... (다른 참가자가 렌더 중이면 잠시 기다립니다)"):
        # One render at a time across all sessions -- see RENDER_LOCK.
        with RENDER_LOCK:
            try:
                rendered_video = render_caption_video(artifact_dir)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
                st.stop()
    st.session_state.rendered_in_session = True
    st.success("완성 영상을 만들었습니다.")
    # On the cloud the disk is wiped on reboot, so persist the finished result.
    pid = current_participant()
    video_link = cloud_store.upload_file(pid, artifact_dir.name, rendered_video, label="video")
    cloud_store.log_event(
        pid,
        artifact_dir.name,
        "rendered",
        position=load_caption_style(artifact_dir).get("position"),
        video_link=video_link,
    )

rendered_video = artifact_dir / "rendered_subtitles.mp4"
if st.session_state.get("rendered_in_session") and rendered_video.exists():
    st.video(str(rendered_video))
    st.download_button(
        "영상 다운로드",
        rendered_video.read_bytes(),
        file_name=rendered_video.name,
        mime="video/mp4",
    )

back_col, _spacer, library_col = st.columns((1, 7.5, 1.3))
with back_col:
    if st.button("자막 위치 다시 정하기"):
        st.switch_page("pages/02_caption_position.py")
with library_col:
    if st.button("Library로 돌아가기", use_container_width=True):
        clear_selection()
        st.switch_page("app.py")
