"""
The live video's detections, kept for the heat analysis.

During a race the live video (race_day/livevideo.py) runs YOLO on each pilot's
picture. With "Keep the detections for the heat analysis" on, it keeps them —
per pilot, per analysed frame: the clock time, the boxes YOLO gave (before any
filtering) and a tiny thumbnail of the picture — and saves them when the race
ends: <event data>/live/<start>_<channel>.npz (a heat replayed to the announcer:
live/<heat id>/…, since a replay runs on today's clock).

The heat analysis (race_day/heat_analysis.py) then takes YOLO's boxes from
there for every frame the live video analysed, and runs YOLO itself only for
the frames it didn't (dropped, a gray picture, before the race started). The
rest — the tracker, pass detector, camera motion, pass scorer, CLIP, the lap
check — runs on the recorded video as always.

Which recorded frame is which live frame is never taken on trust:

  calibrate()  decodes every frame of a second of the race from the recording
               and finds the live frames that match one recorded frame clearly
               better than the frames next to it. From those: the shift between
               the live clock and the recording's video time (FPVTrackside's
               per-frame clock wobbles by ~10 ms, the video's own times don't),
               how close a matching picture is, and which of the recording's
               frames the live video analysed (a 60 fps card at 30: the even or
               the odd ones). The analysis keeps its own frames all the same
               (the frames its laps were checked on): only the live frames that
               fall on them are used.
  lookup()     a recorded frame's boxes: only from a live frame at the same
               moment (within half a 60 fps frame) whose thumbnail matches the
               recorded picture; otherwise None, and YOLO runs.
"""

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

THUMB = (16, 9)           # the thumbnail: 16×9 gray
THUMB_TOL = 6.0           # mean difference (0–255) of two thumbnails of the same picture (compression ≈ 1–3)
CAL_S = (3.0, 4.0)        # calibrated on this part of the race (s after the start): the pilots are flying
SUBDIR = "live"


def thumb(img) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA)


# ── the live side ─────────────────────────────────────────────────

class Recorder:
    """A pilot's detections in one race (the live video's process)."""

    def __init__(self):
        self.t: List[float] = []
        self.thumbs: List[np.ndarray] = []
        self.boxes: List[np.ndarray] = []

    def add(self, t: float, img, res):
        b = res.boxes
        if len(b):
            arr = np.concatenate([b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.asarray(b.xyxy),
                                  np.asarray(b.conf.cpu() if hasattr(b.conf, "cpu") else b.conf).reshape(-1, 1),
                                  np.asarray(b.cls.cpu() if hasattr(b.cls, "cpu") else b.cls).reshape(-1, 1)],
                                 axis=1).astype(np.float32)
        else:
            arr = np.zeros((0, 6), np.float32)
        self.t.append(t)
        self.thumbs.append(thumb(img))
        self.boxes.append(arr)

    def save(self, folder: str, channel: str) -> Optional[str]:
        if not self.t:
            return None
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder) / f"{int(self.t[0])}_{re.sub(r'[^A-Za-z0-9]+', '', channel)}.npz"
        counts = np.array([len(b) for b in self.boxes], np.int32)
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, t=np.array(self.t), thumbs=np.stack(self.thumbs), counts=counts,
                            boxes=np.concatenate(self.boxes) if self.boxes else np.zeros((0, 6), np.float32),
                            channel=np.array(channel))
        os.replace(tmp, path)
        return str(path)


# ── the analysis side ─────────────────────────────────────────────

def find(folder, channel: str, start: float, end: float) -> Optional[str]:
    """The kept detections of `channel` that cover a race from `start` to `end` (clock s), or None."""
    tag = re.sub(r"[^A-Za-z0-9]+", "", channel)
    best = None
    for p in glob.glob(str(Path(folder) / f"*_{tag}.npz")):
        try:
            t0 = int(Path(p).name.split("_")[0])
        except ValueError:
            continue
        if start - 60 <= t0 <= end and (best is None or abs(t0 - start) < abs(best[0] - start)):
            best = (t0, p)
    return best[1] if best else None


class _Box:
    def __init__(self, row):
        self.xyxy, self.conf, self.cls = row[None, :4], row[None, 4:5], row[None, 5:6]


class _Result:
    """What extract_race reads from a YOLO result: .boxes, each with xyxy[0], conf[0], cls[0]."""

    def __init__(self, rows):
        self.boxes = [_Box(r) for r in rows]


class LiveDets:
    """One pilot's kept detections, as the heat analysis looks them up (by the recording's video time)."""

    def __init__(self, path: str):
        d = np.load(path)
        self.t, self.thumbs = d["t"], d["thumbs"].astype(np.float32)
        ends = np.cumsum(d["counts"])
        self.rows = np.split(d["boxes"], ends[:-1]) if len(ends) else []
        self.shift: Optional[float] = None       # video s − live clock s (calibrate)
        self.tol, self.half = THUMB_TOL, 0.008   # thumbnail tolerance; half a recorded frame (s)
        self.hits = self.misses = 0

    def nearest(self, video_t: float) -> Optional[int]:
        if self.shift is None or not len(self.t):
            return None
        x = video_t - self.shift
        j = int(np.searchsorted(self.t, x))
        k = min((k for k in (j - 1, j) if 0 <= k < len(self.t)), key=lambda k: abs(self.t[k] - x), default=None)
        return k if k is not None and abs(self.t[k] - x) <= self.half else None

    def lookup(self, video_t: float, crop) -> Optional[_Result]:
        k = self.nearest(video_t)
        if k is not None and float(np.abs(thumb(crop).astype(np.float32) - self.thumbs[k]).mean()) <= self.tol:
            self.hits += 1
            return _Result(self.rows[k])
        self.misses += 1
        return None

    def diffs(self, th: np.ndarray) -> np.ndarray:
        return np.abs(self.thumbs - th.astype(np.float32)).mean(axis=(1, 2))


def calibrate(video: str, rects: List[tuple], dets: Dict[int, LiveDets], secs: np.ndarray, race_start_v: float,
              step: int, log=print) -> Optional[int]:
    """Line the live frames up with the recording (see the module doc), from every recorded frame of a second
    of the race: a live frame is the same picture as a recorded one when it matches it clearly better than the
    recorded frames next to it. Sets each LiveDets' shift and tolerance → the phase (0…step-1) of the recorded
    frames the live video analysed, or None (no clear match: the analysis runs YOLO everywhere)."""
    import cv2
    from race_day.capture import cut_quad
    if not dets:
        return None
    cap = cv2.VideoCapture(str(video))
    vt, thumbs = [], {pi: [] for pi in dets}
    try:
        i = int(np.searchsorted(secs, race_start_v + CAL_S[0]))
        i_end = min(int(np.searchsorted(secs, race_start_v + CAL_S[1])), len(secs) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        idx = []
        while i <= i_end:
            ok, frame = cap.read()
            if not ok:
                break
            idx.append(i)
            vt.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            for pi in dets:
                thumbs[pi].append(thumb(cut_quad(frame, rects[pi], (960, 540))).astype(np.float32))
            i += 1
    finally:
        cap.release()
    if len(idx) < 3:
        return None
    pairs = []                                   # (recorded frame index, video s, live clock s, diff)
    for pi, d in dets.items():
        rec = np.stack(thumbs[pi])               # recorded × 9 × 16
        for k in range(len(d.t)):
            dd = np.abs(rec - d.thumbs[k]).mean(axis=(1, 2))
            j = int(np.argmin(dd))
            if dd[j] > THUMB_TOL or j in (0, len(dd) - 1):
                continue
            second = min(dd[j - 1], dd[j + 1])   # the recorded frames next to it
            if dd[j] <= 0.6 * second:
                pairs.append((idx[j], vt[j], float(d.t[k]), float(dd[j])))
    if len(pairs) < 5:
        log(f"live detections: no clear match with the recording ({len(pairs)} frames) — YOLO runs on every frame")
        return None
    shifts = np.array([v - t for _, v, t, _ in pairs])
    shift = float(np.median(shifts))
    ok = [p for p, s in zip(pairs, shifts) if abs(s - shift) <= 0.008]
    if len(ok) < 0.6 * len(pairs):
        log("live detections: the match with the recording isn't consistent — YOLO runs on every frame")
        return None
    tol = float(min(THUMB_TOL, max(1.5, 3 * np.median([p[3] for p in ok]))))
    for d in dets.values():
        d.shift, d.tol = shift, tol
    phase = int(np.argmax(np.bincount([p[0] % step for p in ok], minlength=step)))
    log(f"live detections: lined up with the recording ({len(ok)} clear matches; picture tolerance {tol:.1f}; "
        f"frame phase {phase} of {step})")
    return phase
