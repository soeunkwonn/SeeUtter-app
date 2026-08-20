"""Pick the actually-speaking face inside a single-speaker turn clip.

Each preview clip produced by ``build_speaker_preview_clips`` covers one
audio speaker's turn window. Several faces may be on screen, but only one is
talking. Pipeline per clip:

  1. Decode a handful of evenly sampled frames.
  2. Detect every face with OpenCV YuNet -- robust on the small, off-angle
     faces typical of wide TV shots, where MediaPipe's own short-range
     detector finds nothing.
  3. Track faces across frames by bbox-center proximity.
  4. For tracks whose face is large enough to read reliably, measure lip
     opening per frame with MediaPipe FaceLandmarker (run on the upscaled
     face crop). The track whose opening varies most is the talker.
  5. Crop that track's most-open frame as the thumbnail.

This is a lip-motion heuristic and it is deliberately conservative: when no
face is large/clear enough, or nobody's mouth clearly moves, it returns None
so the caller falls back to the raw clip rather than show a wrong face. It
sits behind ``extract_speaker_face`` / ``build_speaker_face_crops`` so the
whole thing can later be swapped for an audio-visual ASD model (e.g. LR-ASD)
without touching the app layer.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"
LANDMARKER_MODEL = _MODELS / "face_landmarker.task"
YUNET_MODEL = _MODELS / "face_detection_yunet.onnx"

# FaceMesh landmark indices (478-point topology).
LIP_UPPER_INNER = 13
LIP_LOWER_INNER = 14
FACE_TOP = 10        # forehead, between brows
FACE_BOTTOM = 152    # chin

MAX_SAMPLED_FRAMES = 30      # per clip; turns are short
DETECT_SCORE = 0.6           # YuNet detection confidence
LANDMARK_CROP_SIZE = 256     # upscale each face crop to this before landmarks
MIN_FACE_WIDTH = 70          # px; below this, a crop is too small/unreliable
CROP_MARGIN = 0.35           # fraction of bbox size added on each side
MIN_MOTION_STD = 0.022       # min lip-opening std to accept "this face talks"
MIN_MAX_OPENING = 0.13       # the mouth must open clearly at least once
MIN_LANDMARK_FRAMES = 4      # need this many measured frames to trust a track


def make_detector() -> "cv2.FaceDetectorYN":
    return cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (320, 320), DETECT_SCORE)


def make_landmarker():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _sample_frame_indices(total_frames: int, want: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= want:
        return list(range(total_frames))
    return [round(i * (total_frames - 1) / (want - 1)) for i in range(want)]


def _imwrite(path, image) -> bool:
    """cv2.imwrite that tolerates non-ASCII paths (Windows cv2 cannot)."""
    ext = Path(str(path)).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def _read_sampled_frames(clip_path: Path) -> list[np.ndarray]:
    # cv2.VideoCapture can't open non-ASCII paths on Windows; copy to an ASCII
    # temp file first when the path contains non-ASCII characters.
    path = str(clip_path)
    tmp_dir = None
    if not path.isascii():
        import shutil
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="seeutter_vid_"))
        local = tmp_dir / ("clip" + (Path(path).suffix or ".mp4"))
        shutil.copy(path, str(local))
        path = str(local)
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for fi in _sample_frame_indices(total, MAX_SAMPLED_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        return frames
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_faces(detector, frame) -> list[tuple[float, float, float, float]]:
    """Return face bboxes as (x, y, w, h) for one frame."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [tuple(map(float, f[:4])) for f in faces]


def extract_largest_face(clip_path: Path, out_path: Path, detector=None) -> Path | None:
    """Best-effort fallback: crop the largest detected face in the clip.

    No speaking / lip-motion check -- it just guarantees *a* face crop when the
    confident speaker-face methods (LR-ASD, lip-motion) return None, so the UI
    shows a face rather than the whole video clip. Returns None only when no
    face is detected in any sampled frame.
    """
    frames = _read_sampled_frames(clip_path)
    if not frames:
        return None
    detector = detector or make_detector()
    best = None  # (area, frame, bbox)
    for frame in frames:
        for bbox in _detect_faces(detector, frame):
            area = bbox[2] * bbox[3]
            if best is None or area > best[0]:
                best = (area, frame, bbox)
    if best is None:
        return None
    crop = _crop_face(best[1], best[2])
    if crop.size == 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(out_path, crop)
    return out_path if out_path.exists() else None


def _build_tracks(detections: list[list[tuple]]) -> list[dict]:
    """Greedy center-distance tracking across frames.

    detections[i] = list of (x, y, w, h) bboxes in sampled frame i.
    A track collects (frame_index, bbox) sightings of one person.
    """
    tracks: list[dict] = []
    for fi, boxes in enumerate(detections):
        for (x, y, bw, bh) in boxes:
            center = (x + bw / 2.0, y + bh / 2.0)
            gate = bw * 0.5  # within half a face width counts as the same person
            best_idx, best_dist = -1, gate
            for idx, tr in enumerate(tracks):
                px, py = tr["last_center"]
                dist = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx == -1:
                tracks.append({"last_center": center, "sightings": []})
                best_idx = len(tracks) - 1
            tracks[best_idx]["last_center"] = center
            tracks[best_idx]["sightings"].append((fi, (x, y, bw, bh)))
    return tracks


def _lip_opening(landmarker, frame, bbox) -> float | None:
    """Normalized lip gap for the face in ``bbox``, or None if unreadable."""
    import mediapipe as mp

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    mx, my = bw * CROP_MARGIN, bh * CROP_MARGIN
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = LANDMARK_CROP_SIZE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    lip_gap = abs(lms[LIP_LOWER_INNER].y - lms[LIP_UPPER_INNER].y)
    face_h = abs(lms[FACE_BOTTOM].y - lms[FACE_TOP].y) or 1.0
    return lip_gap / face_h  # scale-invariant


def _crop_face(frame, bbox) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    mx, my = bw * CROP_MARGIN, bh * CROP_MARGIN
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    return frame[y0:y1, x0:x1]


def extract_speaker_face(
    clip_path: Path,
    out_path: Path,
    detector=None,
    landmarker=None,
) -> Path | None:
    """Save a crop of the speaking face in ``clip_path``. None on failure.

    Failure (no readable face, nobody clearly talking) is non-fatal: the
    caller should fall back to the raw clip. Pass a shared ``detector`` /
    ``landmarker`` to avoid re-initialising models across many clips.
    """
    owns = detector is None or landmarker is None
    detector = detector or make_detector()
    landmarker = landmarker or make_landmarker()
    try:
        frames = _read_sampled_frames(clip_path)
        if not frames:
            return None

        detections = [_detect_faces(detector, f) for f in frames]
        tracks = _build_tracks(detections)
        if not tracks:
            return None

        # Only measure lip motion on tracks big enough to read reliably.
        best_track, best_motion, best_frame = None, -1.0, None
        for tr in tracks:
            widths = [bb[2] for _, bb in tr["sightings"]]
            if np.median(widths) < MIN_FACE_WIDTH:
                continue
            openings = []
            for fi, bb in tr["sightings"]:
                op = _lip_opening(landmarker, frames[fi], bb)
                if op is not None:
                    openings.append((op, fi, bb))
            if len(openings) < MIN_LANDMARK_FRAMES:
                continue
            motion = float(np.std([o[0] for o in openings]))
            if motion > best_motion:
                best_motion = motion
                best_track = tr
                best_frame = max(openings, key=lambda o: o[0])  # most open

        # Conservative gate: enough motion AND a clearly open mouth at least
        # once. Wide multi-face shots where nobody's lips clearly match tend
        # to fail here, so we fall back to the clip instead of a wrong crop.
        if best_track is None or best_motion < MIN_MOTION_STD:
            return None
        if best_frame[0] < MIN_MAX_OPENING:
            return None

        _, fi, bb = best_frame
        crop = _crop_face(frames[fi], bb)
        if crop.size == 0:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _imwrite(out_path, crop)
        return out_path if out_path.exists() else None
    finally:
        if owns:
            landmarker.close()


def build_speaker_face_crops(
    clip_paths: dict[str, Path], out_dir: Path
) -> dict[str, Path]:
    """Speaking-face crop per speaker. Missing entries fall back to the clip."""
    face_paths: dict[str, Path] = {}
    pending = {}
    for speaker_id, clip_path in clip_paths.items():
        out_path = out_dir / f"{speaker_id}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            face_paths[speaker_id] = out_path
        else:
            pending[speaker_id] = (clip_path, out_path)

    if not pending:
        return face_paths

    detector = make_detector()
    landmarker = make_landmarker()
    try:
        for speaker_id, (clip_path, out_path) in pending.items():
            try:
                result = extract_speaker_face(
                    clip_path, out_path, detector=detector, landmarker=landmarker
                )
            except Exception:
                result = None
            if result is not None:
                face_paths[speaker_id] = result
    finally:
        landmarker.close()
    return face_paths
