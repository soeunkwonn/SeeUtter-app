"""Optional Google Drive + Sheets persistence for cloud deployments.

On a local machine there is nothing to configure: every function here quietly
does nothing and results stay on disk as before. On Streamlit Community Cloud
the container's disk is wiped on reboot, so the app pushes each participant's
naming log (to a Google Sheet) and finished video (to a Drive folder) the moment
they are produced. Nothing here may ever raise into the participant's flow -- a
logging failure must not block the experiment.

Configure via st.secrets (see .streamlit/secrets.toml.example):

    [google]
    service_account = '''{ ...service account JSON... }'''
    drive_folder_id = "the Drive folder id to upload videos into"
    sheet_id = "the Google Sheet id to append log rows to"
"""
from __future__ import annotations

import datetime as _dt
import json as _json
from pathlib import Path
from typing import Any

import streamlit as st

_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]


def _config() -> dict[str, Any] | None:
    """Read Google settings from secrets, or None when not configured."""
    try:
        google = st.secrets["google"]
        sa = google["service_account"]
    except Exception:
        return None
    return {
        "service_account": sa if isinstance(sa, str) else _json.dumps(dict(sa)),
        "drive_folder_id": google.get("drive_folder_id"),
        "sheet_id": google.get("sheet_id"),
    }


def is_enabled() -> bool:
    """True when Google persistence is configured (i.e. running on the cloud)."""
    return _config() is not None


@st.cache_resource(show_spinner=False)
def _services(service_account_json: str):
    """Build cached Drive + Sheets clients from a service-account key."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_service_account_info(
        _json.loads(service_account_json), scopes=_SCOPES
    )
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets


def log_event(
    pid: str | None,
    artifact_id: str,
    event: str,
    names: dict[str, str] | None = None,
    position: str | None = None,
    video_link: str | None = None,
) -> bool:
    """Append one row to the log Sheet. No-op (returns False) when unconfigured."""
    config = _config()
    if not config or not config.get("sheet_id"):
        return False
    try:
        _, sheets = _services(config["service_account"])
        row = [
            _dt.datetime.now().isoformat(timespec="seconds"),
            pid or "",
            artifact_id,
            event,
            _json.dumps(names or {}, ensure_ascii=False),
            position or "",
            video_link or "",
        ]
        sheets.spreadsheets().values().append(
            spreadsheetId=config["sheet_id"],
            range="A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()
        return True
    except Exception as exc:
        # Never let a logging hiccup interrupt the participant.
        import traceback
        traceback.print_exc()
        try:
            st.warning(f"[진단] 시트 기록 실패: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return False


def upload_file(
    pid: str | None,
    artifact_id: str,
    path: Path,
    label: str = "result",
) -> str | None:
    """Upload a file to the Drive folder, returning its link, or None if unconfigured/failed."""
    config = _config()
    if not config or not config.get("drive_folder_id"):
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        from googleapiclient.http import MediaFileUpload

        drive, _ = _services(config["service_account"])
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pid or 'anon'}_{artifact_id}_{label}_{timestamp}{path.suffix}"
        metadata = {"name": filename, "parents": [config["drive_folder_id"]]}
        media = MediaFileUpload(str(path), resumable=False)
        created = (
            drive.files()
            .create(body=metadata, media_body=media, fields="id, webViewLink")
            .execute()
        )
        return created.get("webViewLink")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            st.warning(f"[진단] 드라이브 업로드 실패: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return None
