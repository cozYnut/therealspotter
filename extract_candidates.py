#!/usr/bin/env python3
"""
Phase 1 — Headless candidate extraction.

Runs a video through YOLO + tracker + PassDetector + CLIP with no display,
and saves every detected gate-pass candidate to a JSON file for use in
learn_ui.py (Phase 2).

Usage:
    python extract_candidates.py \
        --video ~/races/video1.mp4 \
        --det-model current_best_non_vocab.pt \
        --output candidates.json \
        --crops-dir candidate_crops \
        --clip-device mps

Output JSON schema:
    {
        "video": "<path>",
        "duration": <seconds>,
        "fps": <float>,
        "total_frames": <int>,
        "candidates": [
            {
                "idx": 1,
                "t": 12.4,
                "gate_type": "square",
                "embedding": [<512 floats>],
                "crop_path": "candidate_crops/candidate_0001_square_12400.jpg",
                "bbox": [x1, y1, x2, y2],
                "reason": "aligned->disappear"
            },
            ...
        ]
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pass_detector import PassDetector, detect_camera_edges
from collections import deque
import pipeline_cfg
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


def run_extraction(
    video_path: str,
    det_model_path: str,
    output_json: str,
    crops_dir: str,
    det_conf: float = 0.25,
    clip_device: str = "cpu",
    max_candidates_per_frame: int = 10,
):
    Path(crops_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading detector: {det_model_path}")
    det = YOLO(det_model_path)
    names = _get_yolo_names(det)
    print(f"Classes: {list(names.values())}")

    _cfg = pipeline_cfg.load()
    tracker = TimeTracker(**pipeline_cfg.tracker_kwargs(_cfg))
    passdet = PassDetector(**pipeline_cfg.gates_passdet_kwargs(_cfg))
    clip = ClipEmbedder(device=clip_device)

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
    print("Extracting candidates — this may take a while on long videos...")

    candidates = []
    pass_idx = 0
    frame_buffer: deque = deque(maxlen=5)
    bbox_buffer:  deque = deque(maxlen=5)   # {track_id: bbox} per frame, parallel to frame_buffer
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        now = pos_msec / 1000.0
        H, W = frame.shape[:2]
        frame_buffer.append(frame)

        # ── YOLO detection ─────────────────────────────
        res = det(frame, conf=det_conf, verbose=False, max_det=50)[0]
        typed = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bb = clamp_bbox((x1, y1, x2, y2), W, H)
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = _cls_to_name(cls_id, names)
            gate_type = cls_name if conf >= det_conf else "NONE"
            typed.append({"bbox": bb, "det_conf": conf, "type": gate_type, "type_score": conf})

        typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:max_candidates_per_frame]

        # ── Track + pass detector ───────────────────────
        tracks = tracker.update(typed, now)
        passdet.update(tracks, now, frame_w=W, frame_h=H)
        bbox_buffer.append({int(tr.track_id): list(tr.bbox) for tr in tracks})

        # ── Collect pass events ─────────────────────────
        while True:
            evt = passdet.pop_any_passed()
            if evt is None:
                break

            tid = int(evt.get("track_id", -1))
            evt_type = str(evt.get("type", "UNKNOWN"))

            frames  = list(frame_buffer)
            bboxes  = list(bbox_buffer)
            embed_frame = frames[0] if frames else frame   # 4 frames before fire

            # Use the gate's bbox from that same frame; fall back to full frame
            past_bbox = bboxes[0].get(tid) if bboxes else None
            if past_bbox:
                crop, used_bbox = _crop_padded(embed_frame, past_bbox)
            else:
                crop, used_bbox = embed_frame, []

            emb = clip.embed_bgr(crop)

            pass_idx += 1
            crop_filename = f"candidate_{pass_idx:04d}_{evt_type}_{int(now * 1000)}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            candidates.append({
                "idx": pass_idx,
                "t": float(now),
                "gate_type": evt_type,
                "embedding": emb.tolist(),
                "crop_path": crop_path,
                "bbox": used_bbox,
                "reason": str(evt.get("reason", "")),
            })

            print(f"  [{pass_idx:04d}] t={now:.2f}s  {evt_type}  ({evt.get('reason', '')})")

        if frame_idx % 150 == 0:
            pct = 100.0 * frame_idx / max(1, total_frames)
            print(f"  Progress: {pct:.0f}%  frame={frame_idx}/{total_frames}  candidates={len(candidates)}")

    cap.release()

    # ── Write output JSON ───────────────────────────────
    output = {
        "video": str(video_path),
        "duration": float(duration),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "candidates": candidates,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(candidates)} candidates → {output_json}")
    print(f"Crops → {crops_dir}/")
    return len(candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Extract gate pass candidates from a video (headless, Phase 1). "
                    "Run this before learn_ui.py."
    )
    parser.add_argument("--video",      required=True,              help="Path to input video file")
    parser.add_argument("--det-model",  required=True,              help="Path to YOLO .pt detector")
    parser.add_argument("--output",      default="candidates.json", help="Output JSON path")
    parser.add_argument("--crops-dir",   default="candidate_crops", help="Directory to save gate crop images")
    parser.add_argument("--det-conf",    type=float, default=0.25,  help="YOLO confidence threshold")
    parser.add_argument("--clip-device", default="cpu",             help="Device for CLIP: cpu / mps / cuda")
    args = parser.parse_args()

    run_extraction(
        video_path=args.video,
        det_model_path=args.det_model,
        output_json=args.output,
        crops_dir=args.crops_dir,
        det_conf=args.det_conf,
        clip_device=args.clip_device,
    )


if __name__ == "__main__":
    main()
