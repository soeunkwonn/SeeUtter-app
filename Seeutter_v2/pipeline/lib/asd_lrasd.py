"""Audio-visual active-speaker face selection via LR-ASD.

Same public interface as ``face.py`` (``extract_speaker_face`` /
``build_speaker_face_crops``) but instead of a lip-motion heuristic it runs
the LR-ASD network (S3FD face detection + tracking + an audio-visual ASD
model). Because the model correlates the *audio track* with each face's lip
movement, it picks the face whose motion matches the voice -- which is what
the pure-visual heuristic in ``face.py`` cannot do on wide multi-face shots.

The heavy lifting (preprocessing functions) is adapted from LR-ASD's
``Columbia_test.py``. We vendor the functions here rather than import that
module because it calls ``argparse.parse_args()`` at import time, and we fix
its POSIX-only path handling so it runs on Windows. Models are cached as
module-level singletons so many clips reuse one S3FD + one ASD instance.
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy
import python_speech_features
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[3]  # lib -> pipeline -> Seeutter_v2 -> workspace root
LR_ASD_DIR = REPO_ROOT / "LR-ASD"
PRETRAIN_MODEL = LR_ASD_DIR / "weight" / "pretrain_AVA.model"
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Preprocessing knobs (LR-ASD Columbia_test.py defaults).
FACEDET_SCALE = 0.25
MIN_TRACK = 10
NUM_FAILED_DET = 10
MIN_FACE_SIZE = 1
CROP_SCALE = 0.40
CROP_MARGIN = 0.35  # margin for the final thumbnail crop from the source frame

_DET = None  # S3FD singleton
_ASD = None  # ASD singleton


@contextlib.contextmanager
def _chdir(path: Path):
    """LR-ASD's S3FD resolves its weight via os.getcwd(); run under its dir."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _get_models():
    global _DET, _ASD
    if _DET is None or _ASD is None:
        inserted = str(LR_ASD_DIR) not in sys.path
        if inserted:
            sys.path.insert(0, str(LR_ASD_DIR))
        try:
            from ASD import ASD
            from model.faceDetector.s3fd import S3FD

            with _chdir(LR_ASD_DIR):
                _DET = S3FD(device="cuda" if torch.cuda.is_available() else "cpu")
                asd = ASD()
                asd.loadParameters(str(PRETRAIN_MODEL))
                asd.eval()
                _ASD = asd
        finally:
            if inserted and str(LR_ASD_DIR) in sys.path:
                sys.path.remove(str(LR_ASD_DIR))
    return _DET, _ASD


def _run(cmd):
    subprocess.run(cmd, shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _imwrite(path, image):
    """cv2.imwrite that tolerates non-ASCII paths (Windows cv2 cannot)."""
    ext = Path(str(path)).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def _bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = float(areaA + areaB - inter) or 1.0
    return inter / denom


def _track_shot(scene_faces) -> list[dict]:
    """Greedy IOU tracking (adapted from Columbia_test.track_shot)."""
    iou_thres = 0.5
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if _bb_iou(face["bbox"], track[-1]["bbox"]) > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > MIN_TRACK:
            frame_num = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frame_i = numpy.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = []
            for ij in range(4):
                interpfn = interp1d(frame_num, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = numpy.stack(bboxes_i, axis=1)
            if max(numpy.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   numpy.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > MIN_FACE_SIZE:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def _crop_track(track, flist, audio_path: Path, crop_stub: str) -> None:
    """Write a 224x224 face-track video + its audio (Columbia_test.crop_video)."""
    v_out = cv2.VideoWriter(crop_stub + "t.avi",
                            cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224))
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = CROP_SCALE
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        image = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                          "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi
        mx = dets["x"][fidx] + bsi
        face = image[int(my - bs):int(my + bs * (1 + 2 * cs)),
                     int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        v_out.write(cv2.resize(face, (224, 224)))
    v_out.release()
    audio_tmp = crop_stub + ".wav"
    a_start = track["frame"][0] / 25
    a_end = (track["frame"][-1] + 1) / 25
    _run(f'{FFMPEG_BIN} -y -i "{audio_path.as_posix()}" -async 1 -ac 1 -vn '
         f'-acodec pcm_s16le -ar 16000 -ss {a_start:.3f} -to {a_end:.3f} '
         f'"{audio_tmp}" -loglevel panic')
    _run(f'{FFMPEG_BIN} -y -i "{crop_stub}t.avi" -i "{audio_tmp}" '
         f'-c:v copy -c:a copy "{crop_stub}.avi" -loglevel panic')
    with contextlib.suppress(OSError):
        os.remove(crop_stub + "t.avi")


def _score_track(asd, crop_stub: str) -> numpy.ndarray:
    """Per-frame ASD scores for one cropped track (Columbia_test.evaluate_network)."""
    _, audio = wavfile.read(crop_stub + ".wav")
    audio_feat = python_speech_features.mfcc(
        audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
    video = cv2.VideoCapture(crop_stub + ".avi")
    video_feat = []
    while video.isOpened():
        ret, frames = video.read()
        if not ret:
            break
        face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face[56:168, 56:168]
        video_feat.append(face)
    video.release()
    video_feat = numpy.array(video_feat)
    if len(video_feat) == 0:
        return numpy.array([])
    length = min((audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100,
                 video_feat.shape[0])
    audio_feat = audio_feat[:int(round(length * 100)), :]
    video_feat = video_feat[:int(round(length * 25)), :, :]
    device = next(asd.parameters()).device
    all_score = []
    for duration in {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}:
        batch = int(math.ceil(length / duration))
        scores = []
        with torch.no_grad():
            for i in range(batch):
                a = torch.FloatTensor(
                    audio_feat[i * duration * 100:(i + 1) * duration * 100, :]
                ).unsqueeze(0).to(device)
                v = torch.FloatTensor(
                    video_feat[i * duration * 25:(i + 1) * duration * 25, :, :]
                ).unsqueeze(0).to(device)
                embed_a = asd.model.forward_audio_frontend(a)
                embed_v = asd.model.forward_visual_frontend(v)
                out = asd.model.forward_audio_visual_backend(embed_a, embed_v)
                scores.extend(asd.lossAV.forward(out, labels=None))
        all_score.append(scores)
    return numpy.round(numpy.mean(numpy.array(all_score), axis=0), 1).astype(float)


def _detect_and_track(det, work: Path, clip_path: Path):
    """ffmpeg -> frames -> S3FD detection -> tracks. Returns (tracks, flist)."""
    pyavi = work / "pyavi"
    pyframes = work / "pyframes"
    pycrop = work / "pycrop"
    for d in (pyavi, pyframes, pycrop):
        d.mkdir(parents=True, exist_ok=True)

    video_avi = (pyavi / "video.avi").as_posix()
    audio_wav = pyavi / "audio.wav"
    _run(f'{FFMPEG_BIN} -y -i "{clip_path.as_posix()}" -qscale:v 2 -async 1 '
         f'-r 25 "{video_avi}" -loglevel panic')
    _run(f'{FFMPEG_BIN} -y -i "{video_avi}" -qscale:a 0 -ac 1 -vn -ar 16000 '
         f'"{audio_wav.as_posix()}" -loglevel panic')
    _run(f'{FFMPEG_BIN} -y -i "{video_avi}" -qscale:v 2 -f image2 '
         f'"{(pyframes / "%06d.jpg").as_posix()}" -loglevel panic')

    flist = sorted(str(p) for p in pyframes.glob("*.jpg"))
    if not flist:
        return [], [], pycrop, audio_wav

    faces = []
    for fname in flist:
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        bboxes = det.detect_faces(image, conf_th=0.9, scales=[FACEDET_SCALE])
        faces.append([{"frame": len(faces), "bbox": b[:-1].tolist(), "conf": b[-1]}
                      for b in bboxes])

    # Single-turn clips are short; treat the whole thing as one shot.
    tracks = _track_shot(faces)
    return tracks, flist, pycrop, audio_wav


def extract_speaker_face(clip_path: Path, out_path: Path) -> Path | None:
    """Crop the audio-visually confirmed speaking face. None on failure/fallback."""
    det, asd = _get_models()
    # Absolute paths: the pipeline runs under _chdir(LR_ASD_DIR), so any
    # relative clip/work path would resolve against the wrong directory.
    clip_path = clip_path.resolve()
    out_path = out_path.resolve()
    import shutil
    import tempfile
    # cv2 (imread/VideoCapture/imwrite) cannot handle non-ASCII paths on Windows
    # (e.g. an artifact dir named "...복사본..."), so run entirely inside an
    # ASCII temp dir with the clip copied in.
    work = Path(tempfile.mkdtemp(prefix="seeutter_asd_"))
    try:
        local_clip = work / "input.mp4"
        shutil.copy(str(clip_path), str(local_clip))
        with _chdir(LR_ASD_DIR):
            tracks, flist, pycrop, audio_wav = _detect_and_track(det, work, local_clip)
            if not tracks:
                return None
            best_track, best_mean, best_offset = None, -1e9, -1
            for ii, track in enumerate(tracks):
                stub = (pycrop / f"{ii:05d}").as_posix()
                _crop_track(track, flist, audio_wav, stub)
                scores = _score_track(asd, stub)
                if len(scores) == 0:
                    continue
                mean_score = float(numpy.mean(scores))
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_track = track
                    best_offset = int(numpy.argmax(scores))

        # Positive mean score == this track is speaking during the turn.
        if best_track is None or best_mean <= 0:
            return None

        frame_idx = int(best_track["frame"][best_offset])
        x1, y1, x2, y2 = best_track["bbox"][best_offset]
        image = cv2.imread(flist[frame_idx])
        if image is None:
            return None
        h, w = image.shape[:2]
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        cx0, cy0 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        cx1, cy1 = min(w, int(x2 + mx)), min(h, int(y2 + my))
        crop = image[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _imwrite(out_path, crop)
        return out_path if out_path.exists() else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_speaker_face_crops(
    clip_paths: dict[str, Path], out_dir: Path
) -> dict[str, Path]:
    """Face crop per speaker, with a fallback chain so a crop is almost always
    produced:

      1. LR-ASD -- the face whose lip motion matches the audio (the speaker).
      2. largest detected face -- when LR-ASD finds no confident speaker face
         (e.g. a wide shot), still crop *a* face instead of showing the video.

    Only clips with no detectable face at all are omitted (caller shows video).
    """
    from . import face as face_fallback

    face_paths: dict[str, Path] = {}
    for speaker_id, clip_path in clip_paths.items():
        out_path = out_dir / f"{speaker_id}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            face_paths[speaker_id] = out_path
            continue
        try:
            result = extract_speaker_face(clip_path, out_path)
        except Exception:
            result = None
        if result is None:
            try:
                result = face_fallback.extract_largest_face(clip_path, out_path)
            except Exception:
                result = None
        if result is not None:
            face_paths[speaker_id] = result
    return face_paths
