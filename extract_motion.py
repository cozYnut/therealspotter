#!/usr/bin/env python3
"""
Per-frame camera motion from optical flow (for pass_scorer.py).

Reads the video once at low resolution, computes dense Farneback flow
between consecutive frames inside the real camera area (black side bars
excluded), and fits a global motion model per frame:

    u = a + d·x − r·y
    v = b + r·x + d·y          (x, y normalised to [-1, 1] around the centre)

    pan_x  = a   yaw rate   (+ = scene moves right → camera turns left)
    pan_y  = b   pitch rate
    expand = d   forward motion (+ = scene expands, flying forward)
    roll   = r   roll rate
    resid  = RMS error of the fit (crash, breakup, close objects)

Units are fractions of the half-frame per frame.  Frame i has time i / fps,
the same clock as race_data.json.

Usage (video inside a track folder → <track>/runs/<video>.motion.json):
    python extract_motion.py --video "<data_root>/track1/test_videos/myvideo.mp4"
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from dataset_paths import track_of
from pass_detector import detect_camera_edges

FLOW_W, FLOW_H = 160, 90
GRID = 4          # use every 4th flow pixel in the fit


def _design(w: int, h: int):
    ys, xs = np.mgrid[0:h:GRID, 0:w:GRID]
    x = (xs.ravel() + 0.5) / w * 2.0 - 1.0
    y = (ys.ravel() + 0.5) / h * 2.0 - 1.0
    n = x.size
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[:n, 0] = 1.0          # a
    A[:n, 2] = x            # d
    A[:n, 3] = -y           # r
    A[n:, 1] = 1.0          # b
    A[n:, 2] = y            # d
    A[n:, 3] = x            # r
    with np.errstate(all="ignore"):
        A_pinv = np.linalg.pinv(A)
    return A, A_pinv, (slice(0, h, GRID), slice(0, w, GRID))


class MotionEstimator:
    """Feed frames in order; collects the per-frame motion fit.
    Frame 0 gets zeros (no previous frame)."""

    KEYS = ("pan_x", "pan_y", "expand", "roll", "resid")

    def __init__(self):
        self.A, self.A_pinv, self.sl = _design(FLOW_W, FLOW_H)
        self.out = {k: [] for k in self.KEYS}
        self.prev = None
        self.x0 = self.x1 = None
        self.cam = (0.0, 1.0)

    def _prep(self, f):
        return cv2.cvtColor(cv2.resize(f[:, self.x0:self.x1], (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)

    def update(self, frame: np.ndarray):
        if self.prev is None:
            left, right = detect_camera_edges(frame)
            W = frame.shape[1]
            self.cam = (left, right)
            self.x0, self.x1 = int(left * W), max(int(left * W) + 2, int(right * W) + 1)
            self.prev = self._prep(frame)
            for k in self.KEYS:
                self.out[k].append(0.0)
            return
        cur = self._prep(frame)
        flow = cv2.calcOpticalFlowFarneback(self.prev, cur, None, 0.5, 3, 9, 3, 5, 1.1, 0)
        u = flow[..., 0][self.sl].ravel() / (FLOW_W / 2.0)
        v = flow[..., 1][self.sl].ravel() / (FLOW_H / 2.0)
        uv = np.concatenate([u, v]).astype(np.float64)
        with np.errstate(all="ignore"):     # macOS Accelerate matmul raises spurious FP warnings
            p = self.A_pinv @ uv
            res = uv - self.A @ p
        for k, val in zip(("pan_x", "pan_y", "expand", "roll"), p):
            self.out[k].append(round(float(val), 5))
        self.out["resid"].append(round(float(np.sqrt(np.mean(res * res))), 5))
        self.prev = cur

    def result(self, video_path: str, fps: float) -> dict:
        return {"video": str(video_path), "fps": float(fps), "frames": len(self.out["resid"]),
                "camera_x": [round(self.cam[0], 4), round(self.cam[1], 4)], **self.out}


def save_motion(data: dict, output_json: str):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f)


def extract_motion(video_path: str, output_json: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    est = MotionEstimator()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        est.update(frame)
    cap.release()
    data = est.result(video_path, fps)
    save_motion(data, output_json)
    return data


def main():
    ap = argparse.ArgumentParser(description="Per-frame camera motion from optical flow")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--output", default=None, help="Output JSON (default: <track>/runs/<video>.motion.json)")
    args = ap.parse_args()
    if args.output is None:
        tp = track_of(args.video)
        if tp is None:
            ap.error("video is not inside a track folder — pass --output")
        args.output = str(tp.motion(Path(args.video).stem))
    d = extract_motion(args.video, args.output)
    print(f"Done. {d['frames']} frames → {args.output}")


if __name__ == "__main__":
    main()
