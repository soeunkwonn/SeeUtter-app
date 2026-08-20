"""Optional Google Drive + Sheets persistence for cloud deployments.

On a local machine there is nothing to configure: every function here quietly
does nothing and results stay on disk as before. On Streamlit Community Cloud
the container's disk is wiped on reboot, so the app pushes each participant's
naming log (to a Google Sheet) and finished video (to a Drive folder) the moment
they are produced. Nothing here may ever raise into the participant's flow -- a
logging failure must not block the experiment.

Configure via st.secrets (see .streamlit/secrets.toml.example):

    [google]
    client_id = "OAuth client id"
    client_secret = "OAuth client secret"
    refresh_token = "refresh token from get_oauth_token.py"
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
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]
_TOKEN_URI = "https://oauth2.googleapis.com/token"


def _config() -> dict[str, Any] | None:
    """Read Google OAuth settings from secrets, or None when not configured.

    Uses OAuth *user* credentials (not a service account) so uploads are owned
    by the researcher's own account -- service accounts have no Drive quota.
    """
    try:
        google = st.secrets["google"]
        return {
            "client_id": google["client_id"],
            "client_secret": google["client_secret"],
            "refresh_token": google["refresh_token"],
            "drive_folder_id": google.get("drive_folder_id"),
            "sheet_id": google.get("sheet_id"),
        }
    except Exception:
        return None


def is_enabled() -> bool:
    """True when Google persistence is configured (i.e. running on the cloud)."""
    return _config() is not None


@st.cache_resource(show_spinner=False)
def _services(client_id: str, client_secret: str, refresh_token: str):
    """Build cached Drive + Sheets clients from OAuth user credentials."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=_TOKEN_URI,
        scopes=_SCOPES,
    )
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets


def _with_retries(fn, attempts: int = 4, base_delay: float = 1.0):
    """Run ``fn`` with exponential backoff -- rides out transient network drops
    (BrokenPipeError, SSL EOF) that Community Cloud occasionally throws."""
    import time

    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- retry any transport hiccup
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_error


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
        _, sheets = _services(
            config["client_id"], config["client_secret"], config["refresh_token"]
        )
        row = [
            _dt.datetime.now().isoformat(timespec="seconds"),
            pid or "",
            artifact_id,
            event,
            _json.dumps(names or {}, ensure_ascii=False),
            position or "",
            video_link or "",
        ]
        _with_retries(
            lambda: sheets.spreadsheets().values().append(
                spreadsheetId=config["sheet_id"],
                range="A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": [row]},
            ).execute(num_retries=5)
        )
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

        drive, _ = _services(
            config["client_id"], config["client_secret"], config["refresh_token"]
        )
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pid or 'anon'}_{artifact_id}_{label}_{timestamp}{path.suffix}"
        metadata = {"name": filename, "parents": [config["drive_folder_id"]]}

        def _upload():
            # Resumable, chunked upload survives a dropped connection mid-transfer
            # far better than sending the whole ~32MB video in one request.
            media = MediaFileUpload(str(path), resumable=True, chunksize=5 * 1024 * 1024)
            request = drive.files().create(
                body=metadata, media_body=media, fields="id, webViewLink"
            )
            response = None
            while response is None:
                _status, response = request.next_chunk(num_retries=5)
            return response

        created = _with_retries(_upload)
        return created.get("webViewLink")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            st.warning(f"[진단] 드라이브 업로드 실패: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return None
