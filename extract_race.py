#!/usr/bin/env python3
"""
Headless race extraction for race_ui.py.

Runs the full pipeline (YOLO + tracker + PassDetector + CLIP + GateDB race matching)
on a video and saves per-frame data + race results to JSON.

Usage:
    python extract_race.py \
        --video  venv/videos/myvideo.mp4 \
        --det-model  current_best_non_vocab.pt \
        --gate-memory  gate_memory.json \
        --output  race_data.json \
        --clip-device  mps
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from pass_detector import PassDetector, detect_camera_edges
from gate_db import GateDB
from collections import deque
from lazy_spotter import (
    TimeTracker,
    ClipEmbedder,
    clamp_bbox,
    _get_yolo_names,
    _cls_to_name,
)


def _crop_padded(frame: np.ndarray, bbox: list, pad_frac: float = 0.5):
    """Crop frame to bbox expanded by pad_frac on each side. Returns (crop, padded_bbox)."""
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(W, x2 + px), min(H, y2 + py)
    return frame[ny1:ny2, nx1:nx2], [nx1, ny1, nx2, ny2]


def run_race_extraction(
    video_path: str,
    det_model_path: str,
    gate_memory_path: str,
    output_json: str,
    det_conf: float = 0.25,
    clip_device: str = "cpu",
    pass_offset_sec: float = 0.0,
    sim_thresh: float = 0.88,
    min_match_margin: float = 0.03,
    g1_sim_thresh: Optional[float] = None,
    g1_margin: Optional[float] = None,
    require_same_type: bool = False,
):
    print(f"Loading detector: {det_model_path}")
    det = YOLO(det_model_path)
    names = _get_yolo_names(det)
    print(f"Classes: {list(names.values())}")

    tracker = TimeTracker()
    passdet = PassDetector()
    clip = ClipEmbedder(device=clip_device)

    gatedb = GateDB(
        sim_thresh=sim_thresh, require_same_type=require_same_type,
        min_lap_gap_sec=6.0, min_gates_between_laps=2,
        min_match_margin=min_match_margin, race_lookahead=3, max_embeds_per_gate=6,
        g1_sim_thresh=g1_sim_thresh, g1_margin=g1_margin,
    )
    gatedb.set_mode("race")
    gatedb.load_memory(gate_memory_path)
    print(f"[GateDB] Loaded {gatedb.memory_size()} gates from {gate_memory_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    if ok:
        left_norm, right_norm = detect_camera_edges(first_frame)
        passdet.set_camera_edges(left_norm, right_norm)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / max(fps, 1.0)
    print(f"Video: {total_frames} frames  {fps:.1f} fps  {duration:.1f}s")
    print("Running race analysis…")

    frame_buffer: deque = deque(maxlen=5)
    bbox_buffer:  deque = deque(maxlen=5)   # {track_id: bbox} per frame, parallel to frame_buffer
    frames_data = []
    all_passes = []
    frame_idx = 0

    query_frames_dir = str(Path(output_json).parent / f"race_query_{Path(video_path).stem}")
    Path(query_frames_dir).mkdir(parents=True, exist_ok=True)

    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        H, W = frame.shape[:2]
        frame_area = float(W * H)

        # ── YOLO ───────────────────────────────────────────────
        res = det(frame, conf=det_conf, verbose=False, max_det=50)[0]
        typed = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = _cls_to_name(cls_id, names)
            gate_type = cls_name if conf >= 0.20 else "NONE"
            typed.append({"bbox": bb, "det_conf": conf, "type": gate_type, "type_score": conf})
        typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:10]

        # ── Track + pass detector ───────────────────────────────
        frame_buffer.append(frame)
        tracks = tracker.update(typed, t)
        passdet.update(tracks, t, frame_w=W, frame_h=H)
        bbox_buffer.append({int(tr.track_id): list(tr.bbox) for tr in tracks})

        st_map = getattr(passdet, "states", {}) or {}

        # ── Per-frame track info ────────────────────────────────
        frame_tracks = []
        for tr in tracks:
            st = st_map.get(int(tr.track_id))
            stage = str(getattr(st, "stage", "idle")) if st else "idle"
            last_area = float(getattr(st, "last_area", 0.0)) if st else 0.0
            last_cx = float(getattr(st, "last_cx", 0.0)) if st else 0.0
            last_cy = float(getattr(st, "last_cy", 0.0)) if st else 0.0
            area_ratio = last_area / max(frame_area, 1.0)
            cdist = (((last_cx / max(W, 1)) - 0.5) ** 2 + ((last_cy / max(H, 1)) - 0.5) ** 2) ** 0.5
            x1, y1, x2, y2 = tr.bbox
            frame_tracks.append({
                "track_id": int(tr.track_id),
                "bbox": [x1, y1, x2, y2],
                "type": str(tr.locked_type),
                "score": round(float(tr.score_ema), 3),
                "stage": stage,
                "area_ratio": round(area_ratio, 4),
                "cdist": round(float(cdist), 4),
            })

        # ── Pass events ─────────────────────────────────────────
        frame_passes = []
        frame_laps = []

        while True:
            evt = passdet.pop_any_passed()
            if evt is None:
                break

            tid = int(evt.get("track_id", -1))
            evt_type = str(evt.get("type", "UNKNOWN"))

            frames  = list(frame_buffer)
            bboxes  = list(bbox_buffer)
            embed_frame = frames[0] if frames else frame   # 4 frames before fire

            # Crop to gate bbox from that same frame; fall back to full frame
            past_bbox = bboxes[0].get(tid) if bboxes else None
            if past_bbox:
                embed_crop, _ = _crop_padded(embed_frame, past_bbox)
            else:
                embed_crop = embed_frame

            emb = clip.embed_bgr(embed_crop)

            q_fname = f"q_{frame_idx:06d}_{int(t * 1000)}.jpg"
            query_img_path = str(Path(query_frames_dir) / q_fname)
            cv2.imwrite(query_img_path, embed_crop)

            prev_race_laps = len(getattr(gatedb, "_race_laps", []))

            gid, sim, source, _s2, _mg, exp_before, _wsz = gatedb.race_match(
                now=t, gate_type=evt_type, emb=emb
            )
            if source != "RACE":
                gid = -1
            else:
                gatedb.on_pass(
                    now=t, gate_id=gid, gate_type=evt_type,
                    sim=sim, reason=str(evt.get("reason", "")),
                    track_id=tid,
                )

            new_race_laps = len(getattr(gatedb, "_race_laps", []))
            if new_race_laps > prev_race_laps:
                closed = gatedb._race_laps[-1]
                frame_laps.append({"lap": int(closed.get("lap", 0)), "t": float(closed.get("t1", t))})

            pass_entry = {
                "t": round(t, 4),
                "gate_id": int(gid),
                "gate_type": evt_type,
                "sim": round(float(sim), 4),
                "source": source,
                "reason": str(evt.get("reason", "")),
                "track_id": tid,
                "exp_before": int(exp_before),
                "query_img":  query_img_path,
                "query_embedding": emb.tolist(),
            }
            frame_passes.append(pass_entry)
            all_passes.append(dict(pass_entry))  # copy — offset shift must not affect frames_data

        # ── Save frame entry (skip empty frames to save space) ──
        entry = {"idx": frame_idx, "t": round(t, 4)}
        if frame_tracks:
            entry["tracks"] = frame_tracks
        if frame_passes:
            entry["passes"] = frame_passes
        if frame_laps:
            entry["laps"] = frame_laps
        frames_data.append(entry)

        if frame_idx % 150 == 0:
            pct = 100.0 * frame_idx / max(1, total_frames)
            print(f"Progress: {pct:.0f}%  frame={frame_idx}/{total_frames}  passes={len(all_passes)}")

    cap.release()

    # Shift pass timestamps back by offset so timeline ticks align with the
    # visual pass moment (PassDetector fires slightly after the actual pass)
    if pass_offset_sec > 0:
        for p in all_passes:
            p["t"] = round(max(0.0, p["t"] - pass_offset_sec), 4)
        print(f"Applied pass offset: -{pass_offset_sec:.2f}s to {len(all_passes)} pass events")

    race_laps = list(getattr(gatedb, "_race_laps", []))
    output = {
        "video": str(video_path),
        "gate_memory": str(gate_memory_path),
        "duration": float(duration),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "passes": all_passes,
        "laps": race_laps,
        "frames": frames_data,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(all_passes)} passes  {len(race_laps)} laps → {output_json}")
    return len(all_passes)


def main():
    parser = argparse.ArgumentParser(description="Headless race extraction for race_ui.py")
    parser.add_argument("--video",             required=True,              help="Path to video file")
    parser.add_argument("--det-model",         required=True,              help="Path to YOLO .pt model")
    parser.add_argument("--gate-memory",       required=True,              help="Path to gate_memory.json")
    parser.add_argument("--output",            default="race_data.json",   help="Output JSON path")
    parser.add_argument("--det-conf",        type=float, default=0.25)
    parser.add_argument("--clip-device",     default="cpu",            help="cpu / mps / cuda")
    parser.add_argument("--pass-offset-sec", type=float, default=0.0,
                        help="Shift pass event timestamps back by this many seconds to align "
                             "timeline ticks with the visual pass moment (default: 0.0)")
    parser.add_argument("--sim-thresh",        type=float, default=0.88,
                        help="Minimum cosine similarity for a gate to count as matched (default: 0.88)")
    parser.add_argument("--min-match-margin",  type=float, default=0.03,
                        help="Minimum gap between best and second-best gate similarity (default: 0.03)")
    parser.add_argument("--g1-sim-thresh",     type=float, default=None,
                        help="Minimum cosine similarity for G1 (start gate); defaults to --sim-thresh")
    parser.add_argument("--g1-margin",         type=float, default=None,
                        help="Minimum margin for G1 (start gate); defaults to --min-match-margin")
    parser.add_argument("--require-same-type", action="store_true", default=False,
                        help="Only match a detected gate against memory slots of the same type")
    args = parser.parse_args()

    run_race_extraction(
        video_path=args.video,
        det_model_path=args.det_model,
        gate_memory_path=args.gate_memory,
        output_json=args.output,
        det_conf=args.det_conf,
        clip_device=args.clip_device,
        pass_offset_sec=args.pass_offset_sec,
        sim_thresh=args.sim_thresh,
        min_match_margin=args.min_match_margin,
        g1_sim_thresh=args.g1_sim_thresh,
        g1_margin=args.g1_margin,
        require_same_type=args.require_same_type,
    )


if __name__ == "__main__":
    main()
