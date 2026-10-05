"""
Split a 2×2 feed into its 4 channels and cut each channel into runs.

The same code runs on the live HDMI capture and on 2×2 video files (Replays).

Per channel, a small state machine:

  idle  ── feed becomes live ──►  live  ── gray for end_gray_s ──►  saved / discarded
            (pre-roll is kept)        ◄── live again (dropout) ──

A feed is live while its image has texture, changes between frames and is
not static noise (noise has no frame-to-frame correlation). A run ends after
end_gray_s of gray — or, while other pilots of the heat are still flying,
after heat_gray_s (a crash or blackout mid-heat doesn't cut the run). A run is kept
only if it was live for at least min_run_s: its video (H.264 .mp4, playable
in phone browsers) and its <run>.meta.json.  Runs that are live at the same
time belong to one heat; files are named h<heat>_<channel>.
"""

import json
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# Order of the 4 quadrants in `layout`: top-left, top-right, bottom-left, bottom-right
QUADS = ("TL", "TR", "BL", "BR")


@dataclass
class CaptureConfig:
    layout: List[str] = field(default_factory=lambda: ["R1", "R3", "R6", "R8"])
    min_run_s: float = 40.0        # shorter runs are discarded
    end_gray_s: float = 10.0       # gray this long ends a run (shorter = dropout)
    heat_gray_s: float = 45.0      # …but while others in the heat still fly, wait this long
                                   # (a crash or video blackout mid-heat, then the pilot goes on)
    preroll_s: float = 2.0         # kept from before the feed was detected live
    live_on_s: float = 0.5         # live this long (mostly) to start a run
    # live detector (on a 96×54 grayscale thumbnail of the quadrant)
    min_texture: float = 8.0       # pixel std: flat gray / black screens are below
    min_corr: float = 0.25         # frame-to-frame correlation: static noise is ~0
    min_change: float = 0.15       # mean abs difference: a frozen frame is ~0


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


class DeviceSource:
    """An HDMI capture card (UVC) through OpenCV."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080, fps: float = 30.0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"cannot open capture device {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
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


# ──────────────────────────────────────────────────────────────
# Live detection
# ──────────────────────────────────────────────────────────────

class LiveDetector:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.prev: Optional[np.ndarray] = None

    def __call__(self, quad: np.ndarray) -> bool:
        g = cv2.cvtColor(cv2.resize(quad, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
        prev, self.prev = self.prev, g
        if prev is None or g.std() < self.cfg.min_texture:
            return False
        a, b = g - g.mean(), prev - prev.mean()
        corr = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
        change = float(np.abs(g - prev).mean())
        return corr >= self.cfg.min_corr and change >= self.cfg.min_change


# ──────────────────────────────────────────────────────────────
# Per-channel recorder
# ──────────────────────────────────────────────────────────────

def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def _unjpeg(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def finalize_mp4(tmp: Path, out: Path):
    """Move the moov atom to the front so phone browsers can stream it."""
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c", "copy",
                            "-movflags", "+faststart", str(out)], capture_output=True)
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            return
    tmp.replace(out)


@dataclass
class RunInfo:
    stem: str
    heat: int
    channel: str
    quad: str
    video: str                 # path of the .mp4
    wall_start: float          # first frame in the file (incl. pre-roll)
    live_start_s: float        # seconds into the file when the feed went live
    live_s: float              # seconds the feed was live (first to last live frame)
    fps: float
    source: str                # "live" or the replayed file


class ChannelRecorder:
    def __init__(self, quad: str, channel: str, cfg: CaptureConfig, fps: float):
        self.quad, self.channel, self.cfg, self.fps = quad, channel, cfg, fps
        self.detect = LiveDetector(cfg)
        self.state = "idle"                          # idle | live | ending
        self.recent: deque = deque(maxlen=max(1, int(cfg.live_on_s * fps)))
        self.preroll: deque = deque(maxlen=max(1, int(cfg.preroll_s * fps)))
        self.held: List[Tuple[float, bytes]] = []    # gray frames during a possible dropout
        self.writer = None
        self.tmp: Optional[Path] = None
        self.info: Optional[RunInfo] = None
        self.n_written = 0
        self.last_live = 0.0
        self.gray_since = 0.0

    @property
    def live_now(self) -> bool:
        return self.state in ("live", "ending")

    def status(self, t: float) -> dict:
        if self.state == "idle":
            return {"state": "waiting"}
        return {"state": "live" if self.state == "live" else "dropout",
                "seconds": round(self.last_live - self.info.wall_start - self.info.live_start_s, 1)}

    def _write(self, img: np.ndarray):
        self.writer.write(img)
        self.n_written += 1

    def feed(self, wall: float, quad: np.ndarray, out_dir: Path, start_run: Callable[[], Tuple[int, str]],
             source: str, others_live: bool = False) -> Optional[Tuple[RunInfo, bool]]:
        """Returns (run, kept) when a run ends."""
        live = self.detect(quad)
        self.recent.append(live)
        if self.state == "idle":
            self.preroll.append((wall, _jpeg(quad)))
            if len(self.recent) == self.recent.maxlen and sum(self.recent) >= 0.8 * len(self.recent):
                heat, stem = start_run()
                h, w = quad.shape[:2]
                self.tmp = out_dir / f"{stem}.tmp.mp4"
                self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"avc1"), self.fps, (w, h))
                self.n_written = 0
                first = self.preroll[0][0]
                for _, b in self.preroll:
                    self._write(_unjpeg(b))
                self.preroll.clear()
                live_start = wall - first - self.cfg.live_on_s
                self.info = RunInfo(stem, heat, self.channel, self.quad, str(out_dir / f"{stem}.mp4"), first,
                                    max(0.0, live_start), 0.0, self.fps, source)
                self.state, self.last_live = "live", wall
            return None
        # recording
        if live:
            if self.state == "ending":                # dropout over: keep the held frames
                for _, b in self.held:
                    self._write(_unjpeg(b))
                self.held = []
            self.state, self.last_live = "live", wall
            self._write(quad)
            return None
        if self.state == "live":
            self.state, self.gray_since = "ending", wall
        self.held.append((wall, _jpeg(quad)))
        gray = wall - self.gray_since
        if gray < self.cfg.end_gray_s or (others_live and gray < self.cfg.heat_gray_s):
            return None
        return self.finish(out_dir)

    def finish(self, out_dir: Path) -> Optional[Tuple[RunInfo, bool]]:
        """End the current run (gray long enough, or the source ended)."""
        if self.state == "idle":
            return None
        self.writer.release()
        info = self.info
        info.live_s = round(self.last_live - info.wall_start - info.live_start_s, 2)
        kept = info.live_s >= self.cfg.min_run_s
        if kept:
            finalize_mp4(self.tmp, Path(info.video))
            meta = asdict(info)
            meta.update(frames=self.n_written, video=Path(info.video).name)
            (out_dir / f"{info.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            self.tmp.unlink(missing_ok=True)
        self.state, self.writer, self.tmp, self.held = "idle", None, None, []
        self.recent.clear()
        self.preroll.clear()
        return info, kept


# ──────────────────────────────────────────────────────────────
# Session: all 4 channels + heats
# ──────────────────────────────────────────────────────────────

class CaptureSession:
    """Feeds frames from a source into the 4 channel recorders.

    on_run(info, kept)      a run ended (kept = saved, ≥ min_run_s)
    on_status(dict)         ~5×/s: per channel state, current heat
    on_preview(frame)       ~10×/s: the full 2×2 frame for the UI
    """

    def __init__(self, out_dir: Path, cfg: CaptureConfig, next_heat: int = 1,
                 on_run=None, on_status=None, on_preview=None, on_heat=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.next_heat = next_heat
        self.heat: Optional[int] = None
        self.on_run, self.on_status, self.on_preview, self.on_heat = on_run, on_status, on_preview, on_heat
        self.recorders: List[ChannelRecorder] = []
        self.source = None
        self.used = set()

    def _start_run(self, rec: ChannelRecorder, wall: float):
        if self.heat is None:                        # first channel to go live opens a heat
            self.heat = self.next_heat
            self.next_heat += 1
            if self.on_heat:
                self.on_heat(self.heat, wall)
        stem = f"h{self.heat:03d}_{rec.channel}"
        k = 0
        while stem in self.used or (self.out_dir / f"{stem}.mp4").exists():   # never overwrite a run
            k += 1
            stem = f"h{self.heat:03d}_{rec.channel}{chr(ord('a') + k)}"
        self.used.add(stem)
        return self.heat, stem

    def run(self, source, source_name: str = "live"):
        self.source = source
        self.recorders = [ChannelRecorder(q, ch, self.cfg, source.fps) for q, ch in zip(QUADS, self.cfg.layout)]
        last_status = last_preview = 0.0
        wall = 0.0
        for wall, t, frame in source:
            flying = [r.state == "live" for r in self.recorders]
            for i, (rec, quad) in enumerate(zip(self.recorders, split_quads(frame))):
                others = any(f for j, f in enumerate(flying) if j != i)
                done = rec.feed(wall, quad, self.out_dir, lambda r=rec, w=wall: self._start_run(r, w), source_name, others)
                if done:
                    self._ended(*done)
            if self.heat is not None and not any(r.live_now for r in self.recorders):
                self.heat = None                     # everyone is back on the ground
            if self.on_status and wall - last_status >= 0.2:
                last_status = wall
                self.on_status(self.status(wall))
            if self.on_preview and wall - last_preview >= 0.1:
                last_preview = wall
                self.on_preview(frame)
        for rec in self.recorders:                   # source ended mid-run
            done = rec.finish(self.out_dir)
            if done:
                self._ended(*done)
        if self.on_status:
            self.on_status(self.status(wall))

    def _ended(self, info: RunInfo, kept: bool):
        if self.on_run:
            self.on_run(info, kept)

    def status(self, wall: float) -> dict:
        return {"heat": self.heat,
                "channels": {r.channel: {"quad": r.quad, **r.status(wall)} for r in self.recorders}}

    def stop(self):
        if self.source:
            self.source.stop()
