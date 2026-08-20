from __future__ import annotations
import argparse
import copy
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ..common import DEFAULT_ARTIFACT_ROOT, read_json, resolve_artifact_dir, standard_paths, write_json
from ..lib.asd_lrasd import (
    LR_ASD_DIR,
    _chdir,
    _crop_track,
    _detect_and_track,
    _get_models,
    _imwrite,
    _score_track,
)

FACE_DIST = 0.7        # cosine distance threshold for face clustering(코사인 유사도 임계값)
CONF_MIN = 0.2         # local turn-level face coverage threshold
MIN_SPEAKER_DUR = 2.0  # gates only the global fallback; never blocks local evidence
LOCAL_MARGIN = 0.15
LOCAL_MIN_OVERLAP_SEC = 0.12  # at least three 25-fps ASD frames
GLOBAL_CONF_MIN = 0.65
GLOBAL_MARGIN = 0.25
DET_MIN = 0.55         # InsightFace detection confidence gate(confidence 임계값)
YAW_MAX = 35.0         # degrees; drop strong profiles from embedding(좌우로 35도 이상 돌아간 얼굴은 임베딩에서 제외)
FPS = 25.0             # LR-ASD works at 25 fps

_FACE_APP = None

# InsightFace의 FaceAnalysis 모델 로드 
def _face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        import torch

        ctx = 0 if torch.cuda.is_available() else -1
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if ctx == 0 else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx, det_size=(320, 320)) # 얼굴 검출 크기 320x320
        _FACE_APP = app
    return _FACE_APP

# bounding box 주변을 magin 있게 크롭 
def _crop_bbox(frame, bbox, margin=0.5):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return frame[max(0, int(y1 - my)):min(h, int(y2 + my)),
                 max(0, int(x1 - mx)):min(w, int(x2 + mx))]

# 하나의 얼굴 track에 해당하는 대표 얼굴 임베딩 생성 
def _track_embedding(fa, tr, flist):

    fidx, bboxes = tr["frame"], tr["bbox"]
    n = len(fidx)
    # 8프레임 이상이면 최대 8개 프레임을 균등하게 추출
    picks = [int(k * (n - 1) / 7) for k in range(8)] if n >= 8 else list(range(n))  
    embs, best_q, best_crop = [], -1.0, None
    for pi in picks:
        img = cv2.imread(flist[int(fidx[pi])])
        if img is None:
            continue
        crop = _crop_bbox(img, bboxes[pi])
        if crop.size == 0:
            continue
        # track bbox를 잘라낸 이미지 안에서 다시 얼굴을 검출 
        faces = fa.get(crop) 
        if not faces:
            continue
        # 가장 큰 얼굴 선택 
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])) 
        yaw = abs(float(f.pose[1])) if getattr(f, "pose", None) is not None else 0.0
        
        # 품질 필터링 
        if f.det_score < DET_MIN or yaw > YAW_MAX: #confidence 0.55 미만 + 얼굴 좌우 회전각 35도 초과인 케이스는 탈락 
            continue
        embs.append(f.embedding)
        q = float(f.det_score) - yaw / 90.0
        if q > best_q:
            best_q, best_crop = q, crop
    if not embs:
        return None, None
    e = np.mean(embs, axis=0)
    return e / (np.linalg.norm(e) + 1e-9), best_crop # 평균 임베딩 L2 norm

# 짧게 끊긴 interval들을 합침 
def _merge_intervals(iv, gap=0.2):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]

# 하나의 diarization turn과 얼굴 speaking interval이 겹치는 총 시간을 계산
def _overlap(ts, te, ivs):
    return sum(max(0.0, min(te, e) - max(ts, s)) for s, e in ivs)


def _ranked_face_overlaps(ts, te, face_spk):
    """Return ``[(face_id, overlap_seconds)]`` for one diarization turn."""
    ranked = [
        (face_id, _overlap(ts, te, intervals))
        for face_id, intervals in face_spk.items()
    ]
    return sorted(((face_id, seconds) for face_id, seconds in ranked if seconds > 0),
                  key=lambda item: item[1], reverse=True)


def _confident_winner(ranked, duration, *, coverage_min, margin_min, overlap_min=0.0):
    """Return ``(face_id, coverage, margin)`` only for an unambiguous winner."""
    if duration <= 0 or not ranked:
        return None
    first_face, first_overlap = ranked[0]
    second_overlap = ranked[1][1] if len(ranked) > 1 else 0.0
    coverage = first_overlap / duration
    margin = (first_overlap - second_overlap) / duration
    if (
        first_overlap >= overlap_min
        and coverage >= coverage_min
        and margin >= margin_min
    ):
        return first_face, coverage, margin
    return None


def _reconcile_local_first(r_turns, face_spk, *, local_conf_min):
    """Assign a face only where visual evidence supports that exact turn.

    Audio-cluster voting is deliberately a restricted fallback.  It can fill
    visually blank turns only when all locally resolved turns for that audio
    label agree with the same face and the aggregate evidence is very strong.
    This keeps useful local matches when diarization merges speakers, without
    turning a dominant face into a label magnet.
    """
    audio_to_faces, face_to_audios = defaultdict(set), defaultdict(set)
    audio_duration = defaultdict(float)
    audio_face_overlap = defaultdict(lambda: defaultdict(float))
    total_duration = 0.0
    local_duration = 0.0
    global_duration = 0.0

    # Phase A: direct, per-turn visual match.  This is allowed for short
    # clusters too; the evidence belongs to the concrete subtitle interval.
    for turn in r_turns:
        audio_id = str(turn["speaker"])
        start, end = float(turn["start"]), float(turn["end"])
        duration = max(0.0, end - start)
        turn["speaker_audio"] = audio_id
        total_duration += duration
        audio_duration[audio_id] += duration

        ranked = _ranked_face_overlaps(start, end, face_spk)
        for face_id, overlap in ranked:
            audio_face_overlap[audio_id][face_id] += overlap

        winner = _confident_winner(
            ranked,
            duration,
            coverage_min=local_conf_min,
            margin_min=LOCAL_MARGIN,
            overlap_min=LOCAL_MIN_OVERLAP_SEC,
        )
        if winner is None:
            turn["speaker"] = audio_id # raw id 
            turn["reconcile_source"] = "audio"
            turn.pop("reconcile_cov", None)
            turn.pop("reconcile_margin", None)
            continue

        face_id, coverage, margin = winner
        turn["speaker"] = f"FACE_{face_id}"
        turn["reconcile_source"] = "visual"
        turn["reconcile_cov"] = round(coverage, 3)
        turn["reconcile_margin"] = round(margin, 3)
        local_duration += duration
        audio_to_faces[audio_id].add(face_id)
        face_to_audios[face_id].add(audio_id)

    # Phase B: a cautious fallback for turns with no local visual decision.
    # A mixed diarization label (multiple locally observed faces) is never
    # globally propagated to its unresolved turns.
    global_face = {}
    audio_face_hints = {}
    for audio_id, duration in audio_duration.items():
        ranked = sorted(audio_face_overlap[audio_id].items(),
                        key=lambda item: item[1], reverse=True)
        if ranked:
            audio_face_hints[audio_id] = int(ranked[0][0])
        winner = _confident_winner(
            ranked,
            duration,
            coverage_min=GLOBAL_CONF_MIN,
            margin_min=GLOBAL_MARGIN,
        )
        direct_faces = audio_to_faces.get(audio_id, set())
        if winner is None or duration < MIN_SPEAKER_DUR or len(direct_faces) > 1:
            continue
        face_id, _coverage, _margin = winner
        # Do not overwrite local evidence that identifies another person.
        if direct_faces and face_id not in direct_faces:
            continue
        global_face[audio_id] = face_id

    for turn in r_turns:
        if turn.get("reconcile_source") != "audio":
            continue
        audio_id = turn["speaker_audio"]
        face_id = global_face.get(audio_id)
        if face_id is None:
            continue
        duration = max(0.0, float(turn["end"]) - float(turn["start"]))
        turn["speaker"] = f"FACE_{face_id}"
        turn["reconcile_source"] = "visual_global"
        global_duration += duration
        audio_to_faces[audio_id].add(face_id)
        face_to_audios[face_id].add(audio_id)

    return {
        "audio_to_faces": audio_to_faces,
        "face_to_audios": face_to_audios,
        "audio_face_hints": audio_face_hints,
        "global_face": global_face,
        "total_duration": total_duration,
        "local_duration": local_duration,
        "global_duration": global_duration,
    }


def run(
    artifact_dir = None,
    video = None,
    out_root = DEFAULT_ARTIFACT_ROOT,
    *,
    face_dist = FACE_DIST,
    conf_min = CONF_MIN,
    overwrite = False,
):
    artifact_dir = resolve_artifact_dir(video, artifact_dir, out_root)
    paths = standard_paths(artifact_dir)
    out_diar = paths["reconciled_diarization_json"]
    out_audit = paths["visual_reconcile_json"]

    if not paths["diarization_json"].exists():
        raise FileNotFoundError(f"Diarization JSON not found: {paths['diarization_json']}.")
    if out_diar.exists() and out_audit.exists() and not overwrite:
        return {"artifact_dir": artifact_dir, "payload": read_json(out_diar),
                "audit": read_json(out_audit), "skipped": True}

    diar = read_json(paths["diarization_json"])
    turns = diar.get("alignment_turns") or diar.get("speaker_turns") or []
    video_path = Path(read_json(paths["meta"])["source_video"]).resolve()

    #LR-ASD와 insightFace 모델 준비 
    det, asd = _get_models()
    fa = _face_app()
    work = Path(tempfile.mkdtemp(prefix="seeutter_reconcile_")) 
    faces_dir = artifact_dir / "_reconcile_faces"
    embeds, track_ids, track_crops, track_speak_frames = [], [], [], {} 
    try:
        local = work / "input.mp4"
        shutil.copy(str(video_path), str(local))
        with _chdir(LR_ASD_DIR):
            tracks, flist, pycrop, audio_wav = _detect_and_track(det, work, local)
            # 각 track의 얼굴 임베딩과 발화 프레임 생성 
            for ti, tr in enumerate(tracks):
                emb, crop = _track_embedding(fa, tr, flist)
                if emb is None:
                    continue
                embeds.append(emb)
                track_ids.append(ti)
                track_crops.append(crop)
                # per-frame ASD speaking scores -> speaking frame indices
                stub = (pycrop / ("%05d" % ti)).as_posix()
                _crop_track(tr, flist, audio_wav, stub)
                scores = _score_track(asd, stub)
                fr = tr["frame"]
                track_speak_frames[ti] = [int(fr[i]) for i in range(min(len(scores), len(fr)))
                                          if scores[i] > 0]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if len(embeds) < 1:
        raise RuntimeError("No embeddable face tracks; cannot reconcile.")

    # ── Phase A: cluster embeddings -> global face IDs ──
    if len(embeds) == 1:
        labels = np.array([0])
    else:
        # AHC 클러스터링 
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=face_dist).fit_predict(np.stack(embeds))
    face_of_track = {int(t): int(l) for t, l in zip(track_ids, labels)}

    # save one representative crop per face 
    shutil.rmtree(faces_dir, ignore_errors=True)
    faces_dir.mkdir(parents=True, exist_ok=True)
    
    # 얼굴별 총 발화 프레임 수 (0이면 화면엔 있으나 말 안 한 인물 -> 화자 아님)
    face_speak_frames = defaultdict(int)
    for t, l in zip(track_ids, labels):
        face_speak_frames[int(l)] += len(track_speak_frames.get(t, []))

    # 대표 얼굴 이미지 저장(발화로 판단된 프레임수가 가장 많은 track을 대표로 선택).
    # 발화가 전혀 없는 얼굴(예: 화면 속 침묵하는 방관자)은 화자가 아니므로 저장하지 않음.
    rep = {}  # face -> (nframes, crop)
    for t, l, crop in zip(track_ids, labels, track_crops):
        if face_speak_frames[int(l)] == 0:
            continue
        nf = len(track_speak_frames.get(t, [])) or 1
        if l not in rep or nf > rep[l][0]:
            rep[l] = (nf, crop)
    for l, (_, crop) in rep.items():
        if crop is not None:
            _imwrite(faces_dir / ("FACE_%d.jpg" % int(l)), crop)

    # ── Phase B: face speaking intervals ──
    face_spk = defaultdict(list)
    for t, frames in track_speak_frames.items():
        f = face_of_track.get(t)
        if f is None:
            continue
        face_spk[f].extend((fr / FPS, (fr + 1) / FPS) for fr in frames)
    face_spk = {f: _merge_intervals(iv) for f, iv in face_spk.items()}

    # ── reconcile: assign each turn to best-overlap face ──
    # Local face evidence decides first; a global audio label can only backfill
    # its weak turns when that label is demonstrably a single face.
    # (_reconcile_local_first records each turn's original audio label in
    # ``speaker_audio`` and returns the total speech duration.)
    reconciled = copy.deepcopy(diar)
    r_turns = reconciled.get("alignment_turns") or reconciled.get("speaker_turns") or []
    hybrid = _reconcile_local_first(r_turns, face_spk, local_conf_min=conf_min)
    audio_to_faces = hybrid["audio_to_faces"]
    face_to_audios = hybrid["face_to_audios"]
    total = hybrid["total_duration"]
    local_duration = hybrid["local_duration"]
    global_duration = hybrid["global_duration"]

    # merge = diarization split one person into >=2 audio labels.  split = one
    # audio label has locally resolved turns belonging to >=2 faces.
    n_merges_fixed = sum(1 for speakers in face_to_audios.values() if len(speakers) >= 2)
    n_splits_fixed = sum(1 for faces in audio_to_faces.values() if len(faces) >= 2)
    local_coverage = round(local_duration / total, 3) if total else 0.0
    global_coverage = round(global_duration / total, 3) if total else 0.0

    reconciled["reconcile"] = {
        "num_faces": len(set(labels)),
        # This remains a direct visual metric; global fallback is reported
        # separately so it cannot make the visual score look better than it is.
        "onscreen_coverage": local_coverage,
        "global_backfill_coverage": global_coverage,
        "audio_face_hints": {
            audio_id: face_id
            for audio_id, face_id in hybrid["audio_face_hints"].items()
            if audio_id not in hybrid["global_face"]
        },
    }
    write_json(out_diar, reconciled)

    audit = {
        "video_id": diar.get("video_id") or artifact_dir.name,
        "config": {
            "face_dist": face_dist,
            "local_conf_min": conf_min,
            "local_margin": LOCAL_MARGIN,
            "local_min_overlap_sec": LOCAL_MIN_OVERLAP_SEC,
            "global_conf_min": GLOBAL_CONF_MIN,
            "global_margin": GLOBAL_MARGIN,
            "global_min_speaker_dur": MIN_SPEAKER_DUR,
            "det_min": DET_MIN,
            "yaw_max": YAW_MAX,
        },
        "num_tracks": len(tracks),
        "num_embeddable_tracks": len(track_ids),
        "num_faces": len(set(labels)),
        "onscreen_coverage": local_coverage,
        "global_backfill_coverage": global_coverage,
        "n_merges_fixed": n_merges_fixed,
        "n_splits_fixed": n_splits_fixed,
        "audio_to_faces": {audio_id: sorted(faces) for audio_id, faces in audio_to_faces.items()},
        "face_to_audios": {int(face_id): sorted(speakers) for face_id, speakers in face_to_audios.items()},
    }
    write_json(out_audit, audit)
    return {"artifact_dir": artifact_dir, "payload": reconciled, "audit": audit, "skipped": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--face-dist", type=float, default=FACE_DIST)
    parser.add_argument("--conf-min", type=float, default=CONF_MIN)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run(artifact_dir=args.artifact_dir, face_dist=args.face_dist,
                 conf_min=args.conf_min, overwrite=args.overwrite)
    a = result["audit"]
    print("faces: %d | onscreen coverage: %.0f%% | merges fixed: %d | splits fixed: %d" % (
        a["num_faces"], 100 * a["onscreen_coverage"], a["n_merges_fixed"], a["n_splits_fixed"]))


if __name__ == "__main__":
    main()
