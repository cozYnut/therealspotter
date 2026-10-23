"""
Capture for race day: the capture card (or a video file played as if live),
recording heats on demand, and reading one pilot's quadrant of a heat video.

  HeatRecorder   shows the feed; Record → <track>/videos/heat_NNN.mp4 (the
                 whole feed: one pilot or a 2×2), until Stop or the time limit
  QuadReader     one pilot's picture from a heat video, read in place
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# Order of the 4 quadrants of a 2×2 feed: top-left, top-right, bottom-left, bottom-right
QUADS = ("TL", "TR", "BL", "BR")


# ──────────────────────────────────────────────────────────────
# Frame sources
# ──────────────────────────────────────────────────────────────

class FileSource:
    """A 2×2 video file. Times count from the file start; `realtime` paces
    reading to the video's fps (to simulate a live capture)."""

    def __init__(self, path: str, realtime: bool = False, wall_start: Optional[float] = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.realtime = realtime
        self.wall_start = wall_start if wall_start is not None else os.path.getmtime(self.path) - self.frames / self.fps
        self.stopped = False

    def __iter__(self) -> Iterator[Tuple[float, float, np.ndarray]]:
        i, t0 = 0, time.time()
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = i / self.fps
            if self.realtime:
                lag = t - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            yield self.wall_start + t, t, frame
            i += 1
        self.cap.release()

    def stop(self):
        self.stopped = True


def open_device(index: int) -> cv2.VideoCapture:
    """OpenCV capture by OpenCV's device index (see list_devices)."""
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def list_devices() -> List[tuple]:
    """[(opencv_index, name)] for the connected cameras.

    Names come from Qt. On macOS, OpenCV numbers the cameras in the order of
    their unique IDs, not in the order Qt (or macOS) lists them — e.g. a USB
    capture card (ID 0x1100…) is 0 and the built-in camera (6C70…) is 1 — so
    Qt's list is sorted by ID to give each name its OpenCV index."""
    from PyQt6.QtMultimedia import QMediaDevices
    devs = [(bytes(d.id()).decode(errors="ignore"), d.description()) for d in QMediaDevices.videoInputs()]
    if sys.platform == "darwin":
        devs.sort(key=lambda x: x[0])
    return [(i, name) for i, (_, name) in enumerate(devs)]


class DeviceSource:
    """An HDMI capture card (UVC) through OpenCV."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080, fps: float = 30.0):
        self.cap = open_device(index)
        if not self.cap.isOpened():
            raise IOError(f"cannot open capture device {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        # the size the device actually gives (a wrong camera shows up here)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.stopped = False

    def __iter__(self):
        t0 = time.time()
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.time()
            yield now, now - t0, frame
        self.cap.release()

    def stop(self):
        self.stopped = True


def split_quads(frame: np.ndarray) -> List[np.ndarray]:
    """TL, TR, BL, BR (no borders on the 2×2 output)."""
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    return [frame[:h2, :w2], frame[:h2, w2:2 * w2], frame[h2:2 * h2, :w2], frame[h2:2 * h2, w2:2 * w2]]


NO_PICTURE_STD = 8.0      # a feed this flat (FPVTrackside's gray "no signal") has no picture


def no_picture(img) -> bool:
    """A flat gray feed (measured inside the picture: the 2×2 grid's divider
    lines at the edges would make a flat feed look busy). Gray feeds measure
    ~1–3, real pictures 40+."""
    h, w = img.shape[:2]
    return img[h // 20: h - h // 20: 4, w // 20: w - w // 20: 4].std() < NO_PICTURE_STD


def gray_intervals(times: List[float], step: float, min_len: float = 1.0) -> List[List[float]]:
    """Times with no picture → [start, end] intervals at least min_len long."""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= 2 * step:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 2), round(b + step, 2)] for a, b in out if b + step - a >= min_len]


def sample_no_picture(path, rect, every: float = 1.0) -> dict:
    """No-picture intervals of one quadrant by sampling a frame every `every` s
    (for runs analysed before the analysis recorded it)."""
    cap = cv2.VideoCapture(str(path))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    gray, t = [], 0.0
    x, y, w, h = rect
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if ok:
            H, W = f.shape[:2]
            c = f[int(round(y * H)):int(round((y + h) * H)), int(round(x * W)):int(round((x + w) * W))]
            if no_picture(c):
                gray.append(t)
        t += every
    cap.release()
    return {"duration": round(dur, 2), "gray": gray_intervals(gray, every)}


class QuadReader:
    """One pilot's picture from a video, read in place (no copy is made):
    the quadrant `rect` (x, y, w, h as fractions of the frame) scaled to
    `size`, at most `max_fps` frames a second (a 60 fps heat video gives
    every 2nd frame). Same use as extract_race's reader: read() → (ok,
    frame), then .t is that frame's time in the video (seconds)."""

    def __init__(self, path, rect=(0.0, 0.0, 1.0, 1.0), size=(960, 540), max_fps: float = 31.0):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(src_fps / max_fps)))
        self.fps = src_fps / self.step
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.step
        self.rect, self.size = tuple(rect), tuple(size) if size else None
        self.t = 0.0
        self.gray_t: List[float] = []            # times with no picture (a flat gray feed)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        self.t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for _ in range(self.step - 1):            # skipped frames: decoded, not converted
            self.cap.grab()
        H, W = frame.shape[:2]
        x, y, w, h = self.rect
        crop = frame[int(round(y * H)):int(round((y + h) * H)), int(round(x * W)):int(round((x + w) * W))]
        if self.size and (crop.shape[1], crop.shape[0]) != self.size:
            crop = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        if no_picture(crop):
            self.gray_t.append(self.t)
        return True, crop

    def release(self):
        self.cap.release()


# ──────────────────────────────────────────────────────────────
# Recording heats
# ──────────────────────────────────────────────────────────────

def finalize_mp4(tmp: Path, out: Path, time_scale: float = 1.0):
    """Move the moov atom to the front so phone browsers can stream it.
    time_scale ≠ 1 stretches the timestamps (a card that delivered fewer
    frames a second than it said: the video would play too fast)."""
    ff = shutil.which("ffmpeg")
    if ff:
        scale = ["-itsscale", f"{time_scale:.6f}"] if abs(time_scale - 1.0) > 0.01 else []
        r = subprocess.run([ff, "-y", "-loglevel", "error", *scale, "-i", str(tmp), "-c", "copy",
                            "-movflags", "+faststart", str(out)], capture_output=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            return
    tmp.replace(out)


class HeatRecorder:
    """Reads the feed on its own thread: the latest frame for the screen, and
    after record(name) every frame goes to <out_dir>/<name>.mp4 until
    stop_recording() or max_s. on_saved(path, info) is called (on a helper
    thread) once the file is complete."""

    def __init__(self, source, out_dir: Path, tmp_dir: Path, max_s: float = 180.0,
                 on_saved: Optional[Callable[[Path, dict], None]] = None):
        self.source, self.out_dir, self.tmp_dir = source, Path(out_dir), Path(tmp_dir)
        self.max_s, self.on_saved = max_s, on_saved
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ended = False                       # the source stopped (file played out, card unplugged)
        self.error: Optional[str] = None
        self._want: Optional[str] = None         # start recording this at the next frame
        self._stop_rec = False
        self._rec: Optional[dict] = None         # the recording in progress
        self._saving = 0                         # files being finished
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "HeatRecorder":
        self._thread.start()
        return self

    def latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame

    def record(self, name: str):
        with self.lock:
            if self._rec is None and not self.ended:
                self._want, self._stop_rec = name, False

    def stop_recording(self):
        with self.lock:
            self._stop_rec = True

    def status(self) -> dict:
        with self.lock:
            r = self._rec
            return {"recording": r is not None or self._want is not None,
                    "name": r["name"] if r else self._want,
                    "seconds": round(time.time() - r["wall0"], 1) if r else 0.0,
                    "max_s": self.max_s, "saving": self._saving, "ended": self.ended, "error": self.error}

    def close(self):
        self.stop_recording()
        self.source.stop()
        self._thread.join(timeout=5)

    def _run(self):
        try:
            for wall, _t, frame in self.source:
                with self.lock:
                    self.frame = frame
                    want, stop = self._want, self._stop_rec
                if want and self._rec is None:
                    self._begin(want, wall, frame)
                if self._rec is not None:
                    self._rec["writer"].write(frame)
                    self._rec["n"] += 1
                    self._rec["last"] = wall
                    if stop or wall - self._rec["wall0"] >= self.max_s:
                        self._end()
        except Exception as e:                      # the card went away: keep what was recorded
            self.error = f"{type(e).__name__}: {e}"
        if self._rec is not None:
            self._end()
        with self.lock:
            self.ended = True

    def _begin(self, name: str, wall: float, frame: np.ndarray):
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.tmp_dir / f"{name}.recording.mp4"
        h, w = frame.shape[:2]
        fps = float(getattr(self.source, "fps", 30.0) or 30.0)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        with self.lock:
            self._rec = {"name": name, "tmp": tmp, "writer": writer, "fps": fps, "wall0": wall, "last": wall, "n": 0}
            self._want = None

    def _end(self):
        with self.lock:
            rec, self._rec, self._stop_rec = self._rec, None, False
            self._saving += 1
        rec["writer"].release()
        # the frames' real rate: n frames over the time they came in (plus one frame's time)
        secs = (rec["last"] - rec["wall0"]) + 1.0 / rec["fps"]
        real_fps = rec["n"] / secs if rec["n"] > 1 else rec["fps"]
        out = self.out_dir / f"{rec['name']}.mp4"
        info = {"recorded_at": round(rec["wall0"], 3), "seconds": round(secs, 2), "frames": rec["n"],
                "fps": round(real_fps, 2)}

        def finish():                               # off the capture thread: the feed keeps flowing
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                done = rec["tmp"].with_name(f"{rec['name']}.mp4")
                finalize_mp4(rec["tmp"], done, rec["fps"] / real_fps)
                os.replace(done, out)                     # appears in out_dir only when complete
                if self.on_saved:
                    self.on_saved(out, info)
            except Exception as e:
                self.error = f"saving {out.name} failed: {e}"
            finally:
                with self.lock:
                    self._saving -= 1
        threading.Thread(target=finish, daemon=True).start()
