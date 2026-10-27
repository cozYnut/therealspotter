"""
A heat's pilots analysed together, in one process.

An FPVTrackside heat (or a 2×2 recording) is ONE video with up to four pilots
in it. It is analysed in three stages that run at the same time:

  decoder   reads the video once — with the Mac's hardware decoder
            (VideoToolbox, through ffmpeg) when there is one, so the CPU is
            left to the pilots — and cuts every kept frame into the pilots'
            pictures (race_day/capture.cut_quad, what QuadReader gives one
            pilot). Only the frames QuadReader would keep (at most 30 a second,
            counted from the video's first frame), and for an FPVTrackside heat
            only the race (from a few seconds before the start): the staging
            before it isn't analysed.
  detector  runs YOLO on each pilot's picture, a few moments ahead of the
            pilots, so the detector and the pilots' work overlap. A pilot
            whose picture has been gray (no signal) or still (crashed, landed,
            waiting on the pad) for IDLE_S isn't run through YOLO until the
            picture moves again: their tracker just sees no gates then.
  pilots    one thread per pilot, each the unchanged
            extract_race.run_race_extraction (its own tracker, pass detector,
            camera motion and pass scorer), given its picture and its
            detections. CLIP is loaded once and shared.

YOLO is the Core ML copy on the Neural Engine (race_day/models.py), one call
per picture; with the .pt on the GPU the pilots' pictures of a moment go in
one call instead.

  analyse_heat(video, runs, det_model, n_gates, device, clip_device, window=…)
      runs: [{"stem", "rect", "out_json", "log"}] — the pilots to analyse
      window: (start, end) in video seconds — the part to analyse (None: all)
      → {stem: {"seconds", "device", "picture"} or {"error"}}

Each pilot keeps its own log (<stem>.analysis.log) and its own result file; a
pilot whose analysis fails doesn't stop the others.
"""

import bisect
import contextlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from race_day.capture import cut_quad, frame_step, gray_intervals, no_picture

QUEUE_FRAMES = 8          # pictures waiting per pilot (a few MB each)
AHEAD = 4                 # moments the decoder keeps ready for the detector
DET_CONF, DET_MAX = 0.25, 50      # what extract_race asks the detector for
IDLE_S = 2.0              # gray or still this long → no YOLO until the picture moves again
STILL_DIFF = 1.5          # mean change of a 160×90 gray picture below this: still (flying is 10+)


# ── reading the video ─────────────────────────────────────────────

def frame_times(video: str) -> Optional[List[float]]:
    """Every frame's time (s from the first frame, as OpenCV's
    CAP_PROP_POS_MSEC gives it), from the container's index (ffprobe, no
    decoding). None without ffprobe."""
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True, timeout=120)
        pts = sorted(float(x) for x in r.stdout.split() if x.strip() not in ("", "N/A"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return [round(p - pts[0], 6) for p in pts] if pts else None


def _kept(n: int, times: Optional[List[float]], src_fps: float, step: int, window) -> Tuple[int, int]:
    """(first, last) frame index to analyse: every step-th frame counted from
    frame 0, as QuadReader keeps them, inside the window."""
    if not window:
        return 0, n - 1
    t0, t1 = window
    if times:
        i0, i1 = bisect.bisect_left(times, t0), bisect.bisect_right(times, t1) - 1
    else:
        i0, i1 = int(t0 * src_fps), int(t1 * src_fps)
    i0 = max(0, i0 + (-i0) % step)                 # the next kept frame
    return i0, min(n - 1, max(i0, i1))


def _ffmpeg_frames(video: str, i0: int, i1: int, step: int, size: Tuple[int, int], hw: bool, times: List[float]):
    """(t, frame) for frames i0, i0+step … i1, decoded by ffmpeg
    (VideoToolbox when hw), converted like OpenCV does (BT.601). Frames before
    i0 are decoded but not converted or sent. Raises if ffmpeg gives nothing."""
    w, h = size
    sel = f"select='between(n\\,{i0}\\,{i1})*not(mod(n-{i0}\\,{step}))'"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hw:
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(video), "-an", "-vf", f"{sel},scale=in_color_matrix=bt601:in_range=tv",
            "-fps_mode", "passthrough", "-frames:v", str((i1 - i0) // step + 1),
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3)
    sent = 0
    try:
        nbytes = w * h * 3
        i = i0
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield times[i], np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            sent += 1
            i += step
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
    if not sent:
        raise IOError(f"ffmpeg decoded nothing: {p.stderr.read().decode('utf-8', 'ignore')[-300:]}")


def _opencv_frames(video: str, i0: int, i1: int, step: int):
    """(t, frame), read the way QuadReader reads (the fallback without ffmpeg)."""
    cap = cv2.VideoCapture(str(video))
    try:
        if i0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        i = i0
        while i <= i1:
            ok, frame = cap.read()
            if not ok:
                return
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame
            for _ in range(step - 1):
                cap.grab()
            i += step
    finally:
        cap.release()


# ── the stages ────────────────────────────────────────────────────

class _NoBoxes:
    """YOLO's result for a picture it wasn't run on: no gates."""
    boxes = ()


class Feed:
    """One pilot's pictures and detections, read like QuadReader:
    read() → (ok, frame), then .t; .fps, .frame_count, .rect, .step, .gray_t."""

    def __init__(self, rect, fps: float, frame_count: int, step: int):
        self.q: "queue.Queue" = queue.Queue(maxsize=QUEUE_FRAMES)
        self.rect, self.fps, self.frame_count, self.step = tuple(rect), fps, frame_count, step
        self.t = 0.0
        self.gray_t: List[float] = []
        self.result = _NoBoxes()                 # the detections of the picture read last
        self.closed = threading.Event()          # the pilot's pipeline stopped: no more pictures for it
        # the detector's view of this pilot (only the detector thread uses these)
        self.prev_small = None
        self.idle_since: Optional[float] = None

    def read(self):
        item = self.q.get()
        if item is None:
            return False, None
        if isinstance(item, BaseException):      # the decoder or detector failed: so does this pilot
            raise item
        self.t, crop, gray, self.result = item
        if gray:
            self.gray_t.append(self.t)
        return True, crop

    def put(self, item) -> None:
        while not self.closed.is_set():          # never blocks on a pipeline that has stopped
            try:
                self.q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def release(self):
        self.closed.set()

    def idle(self, crop, gray: bool, t: float) -> bool:
        """Gray or still for IDLE_S: the detector skips this picture."""
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA)
        still = gray or (self.prev_small is not None
                         and float(cv2.absdiff(small, self.prev_small).mean()) < STILL_DIFF)
        self.prev_small = small
        if not still:
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = t
        return t - self.idle_since >= IDLE_S


class PilotDetector:
    """What a pilot's pipeline calls as its YOLO: the detections the detector
    stage already made for the picture it just read."""

    def __init__(self, feed: Feed, model):
        self.feed, self.model = feed, model         # .model.names, as extract_race reads it

    def __call__(self, frame, **kw):
        return [self.feed.result]


class Detector:
    """YOLO for every pilot's picture of every moment (see the module doc)."""

    def __init__(self, model_path: str, device):
        from ultralytics import YOLO
        self.yolo = YOLO(model_path, task="detect")
        # what extract_race reads the class names from (det.model.names). A Core ML YOLO's .model is only its
        # path, and without the names every gate would be "class0"… — and a flagpole, which the pass detector
        # and the pass scorer treat apart, would pass for a gate.
        self.model = types.SimpleNamespace(names=dict(self.yolo.names))
        self.coreml = str(model_path).endswith(".mlpackage")
        self.kw = dict(conf=DET_CONF, max_det=DET_MAX, verbose=False)
        if device and not self.coreml:
            self.kw["device"] = device

    def detect(self, crops: List[np.ndarray]) -> list:
        if not crops:
            return []
        if self.coreml:                          # the Neural Engine: one picture per call (a fixed shape)
            return [self.yolo(c, **self.kw)[0] for c in crops]
        return list(self.yolo(crops, **self.kw))  # the GPU: the moment's pictures in one call

    def run(self, moments: "queue.Queue", feeds: List[Feed], stopped: threading.Event):
        try:
            while True:
                m = moments.get()
                if m is None or isinstance(m, BaseException):
                    end = m
                    break
                t, crops = m
                live = [i for i, f in enumerate(feeds) if not f.closed.is_set()]
                if not live:                     # every pilot stopped: nothing left to detect for
                    end = None
                    break
                grays = {i: no_picture(crops[i]) for i in live}
                todo = [i for i in live if not feeds[i].idle(crops[i], grays[i], t)]
                results = dict(zip(todo, self.detect([crops[i] for i in todo])))
                for i in live:
                    feeds[i].put((t, crops[i], grays[i], results.get(i, _NoBoxes())))
        except BaseException as e:               # every pilot gets the failure
            end = e
        finally:
            stopped.set()                        # the decoder stops too
        for f in feeds:
            f.put(end)


def decode(video: str, rects: List[tuple], span: Tuple[int, int], step: int, times: Optional[List[float]],
           moments: "queue.Queue", stopped: threading.Event, size=(960, 540)):
    """Read the kept frames once; each moment's pilot pictures go to the
    detector. The hardware decoder when there is ffmpeg (and the frame index),
    else OpenCV — also if ffmpeg fails before its first frame."""
    def put(item) -> bool:
        while not stopped.is_set():
            try:
                moments.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    i0, i1 = span
    try:
        sent = False
        if shutil.which("ffmpeg") and times:
            try:
                for t, frame in _ffmpeg_frames(video, i0, i1, step, _size(video), sys.platform == "darwin", times):
                    sent = True
                    if not put((t, [cut_quad(frame, r, size) for r in rects])):
                        return
            except Exception:
                if sent:                         # failed halfway: the analysis can't be trusted
                    raise
        if not sent:
            for t, frame in _opencv_frames(video, i0, i1, step):
                if not put((t, [cut_quad(frame, r, size) for r in rects])):
                    return
        put(None)
    except BaseException as e:
        put(e)


def _size(video: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


class SharedClip:
    """One CLIP embedder for all the pilots (embed_bgr, one at a time)."""

    def __init__(self, device: str):
        from tracking import ClipEmbedder
        self.clip, self.lock = ClipEmbedder(device=device), threading.Lock()

    def embed_bgr(self, bgr):
        with self.lock:
            return self.clip.embed_bgr(bgr)


class _PerThreadOutput:
    """stdout/stderr that each pilot's thread writes to its own log file
    (redirect_stdout would send every pilot's output to one place)."""

    def __init__(self, fallback):
        self.fallback, self.local = fallback, threading.local()

    def write(self, s):
        out = getattr(self.local, "file", None) or self.fallback
        return out.write(s)

    def flush(self):
        out = getattr(self.local, "file", None) or self.fallback
        out.flush()


def analyse_heat(video: str, runs: List[dict], det_model: str, n_gates: int, device="mps",
                 clip_device: str = "mps", blank_memory=None, window=None) -> Dict[str, dict]:
    """Analyse the pilots `runs` of one video together (see the module doc).
    blank_memory(n_gates) → a gate memory file for one pilot (deleted after)."""
    from extract_race import run_race_extraction
    t0 = time.time()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    step = frame_step(src_fps)
    times = frame_times(video)
    if times and len(times) != n_frames:            # the index and the decoder disagree: OpenCV reads it
        times = None
    i0, i1 = _kept(n_frames, times, src_fps, step, window)
    feeds = [Feed(r["rect"], src_fps / step, (i1 - i0) // step + 1, step) for r in runs]
    det = Detector(det_model, device)
    clip = SharedClip(clip_device)
    results: Dict[str, dict] = {}
    out, err = _PerThreadOutput(sys.stdout), _PerThreadOutput(sys.stderr)
    moments: "queue.Queue" = queue.Queue(maxsize=AHEAD)
    stopped = threading.Event()

    def pilot(run: dict, feed: Feed):
        mem = None
        try:
            mem = blank_memory(n_gates) if blank_memory else None
            with open(run["log"], "w") as log:
                out.local.file = err.local.file = log
                try:
                    run_race_extraction(str(video), det_model, mem, run["out_json"], clip_device=clip_device,
                                        gate_id_logic="greedy", det_device=device, reader=feed,
                                        stem=run["stem"], det=PilotDetector(feed, det.model), clip=clip)
                    results[run["stem"]] = {"seconds": round(time.time() - t0, 1),
                                            "device": "coreml" if det.coreml else device,
                                            "picture": {"duration": round(n_frames // step / feed.fps, 2),
                                                        "gray": gray_intervals(feed.gray_t, 1.0 / feed.fps)}}
                except Exception:
                    traceback.print_exc()                 # into this pilot's log
                    raise
                finally:
                    out.local.file = err.local.file = None
        except Exception as e:                            # this pilot only: the others go on
            results[run["stem"]] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            feed.release()
            if mem:
                os.unlink(mem)

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        threads = [threading.Thread(target=pilot, args=(r, f), daemon=True) for r, f in zip(runs, feeds)]
        threads.append(threading.Thread(target=det.run, args=(moments, feeds, stopped), daemon=True))
        for th in threads:
            th.start()
        try:
            decode(video, [f.rect for f in feeds], (i0, i1), step, times, moments, stopped)
        finally:
            for th in threads:
                th.join()
    return results
