#!/usr/bin/env python3
"""
Make a 2×2 test video like the race director's HDMI output, from
single-pilot videos: gray when no drone is connected, optional static noise,
a too-short run and a brief dropout — plus a ground-truth file saying where
every run really is (to test the splitter / run detector and map results
back to reviewed videos).

    python -m race_day.simulate out.mp4 --heat "a.mp4,b.mp4,c.mp4,d.mp4" --heat "e.mp4,,f.mp4,g.mp4"
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

W, H, FPS = 960, 540, 30.0
GRAY = 128


def _frames(path: str):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != W or f.shape[0] != H:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        yield f
    cap.release()


def make(out: str, heats: List[List[Optional[str]]], gap_s: float = 15.0, stagger_s: float = 1.5,
         static: Optional[dict] = None, dropout: Optional[dict] = None, short: Optional[dict] = None,
         max_s: Optional[float] = None, seed: int = 0) -> dict:
    """heats: per heat, 4 video paths (TL, TR, BL, BR; None = empty channel).
    static  = {"quad": i, "at_s": t, "dur_s": d}   noise on an idle channel
    dropout = {"heat": h, "quad": i, "at_s": t, "dur_s": d}   gray inside a run
    short   = {"quad": i, "at_s": t, "src": path, "dur_s": d}   a too-short run
    """
    rng = np.random.default_rng(seed)
    gray = np.full((H, W, 3), GRAY, np.uint8)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (2 * W, 2 * H))
    truth = {"fps": FPS, "runs": [], "static": static, "short": short}
    t = 0.0

    def write(frames_per_quad: List[Optional[np.ndarray]], n: int = 1):
        nonlocal t
        q = [f if f is not None else gray for f in frames_per_quad]
        writer.write(np.vstack([np.hstack(q[:2]), np.hstack(q[2:])]))
        t += 1 / FPS

    def idle(seconds: float):
        for _ in range(int(seconds * FPS)):
            quads: List[Optional[np.ndarray]] = [None] * 4
            if static and static["at_s"] <= t < static["at_s"] + static["dur_s"]:
                quads[static["quad"]] = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            if short and short["at_s"] <= t < short["at_s"] + short["dur_s"]:
                quads[short["quad"]] = next(short_iter, None)
            write(quads)

    short_iter = _frames(short["src"]) if short else iter(())
    idle(gap_s)
    for hi, heat in enumerate(heats):
        iters, starts = [], []
        for qi, p in enumerate(heat):
            iters.append(_frames(p) if p else None)
            starts.append(t + qi * stagger_s if p else None)
        for qi, p in enumerate(heat):
            if p:
                truth["runs"].append({"heat_index": hi, "quad": qi, "src": str(p), "file_start_s": round(starts[qi], 4)})
        done = [it is None for it in iters]
        while not all(done):
            quads = []
            for qi, it in enumerate(iters):
                f = None
                if it is not None and not done[qi] and t >= starts[qi]:
                    f = next(it, None)
                    if f is None:
                        done[qi] = True
                    elif (dropout and dropout["heat"] == hi and dropout["quad"] == qi
                          and dropout["at_s"] <= t - starts[qi] < dropout["at_s"] + dropout["dur_s"]):
                        f = None                              # brief signal loss, video time goes on
                quads.append(f)
            write(quads)
            if max_s and t > max_s:
                break
        idle(gap_s)
    writer.release()
    Path(out).with_suffix(".truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main():
    ap = argparse.ArgumentParser(description="Make a 2×2 test video from single-pilot videos")
    ap.add_argument("out")
    ap.add_argument("--heat", action="append", required=True,
                    help="4 comma-separated videos TL,TR,BL,BR (empty = no pilot)")
    ap.add_argument("--gap", type=float, default=15.0, help="gray seconds between heats")
    args = ap.parse_args()
    heats = [[p or None for p in h.split(",")] for h in args.heat]
    make(args.out, heats, gap_s=args.gap)
    print(f"Wrote {args.out} (+ .truth.json)")


if __name__ == "__main__":
    main()
