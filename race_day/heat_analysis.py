"""
A heat's pilots analysed together, in one process.

An FPVTrackside heat (or a 2×2 recording) is ONE video with up to four
pilots in it. Analysing each pilot in its own worker meant decoding the same
video four times and loading PyTorch, YOLO and CLIP four times (~3 GB each).
Here the video is decoded once: every frame is cut into the pilots' pictures
(race_day/capture.cut_quad — exactly what QuadReader gives one pilot) and
each picture goes to that pilot's own pipeline — the unchanged
extract_race.run_race_extraction, one thread per pilot, with its own
tracker, pass detector, camera motion and pass scorer. YOLO and CLIP are
loaded once and shared, and YOLO looks at the pilots' pictures of the same
moment in one call (a batch of up to four), which is cheaper on the GPU than
four calls. A pilot's result is the same as analysing them alone.

Measured on 2027Race1 heat 2 (four pilots, M5 Pro): 295 s and 3.15 GB, where
four worker processes took 250 s and 11.6 GB and one took 498 s and 3.4 GB;
all three found the same 228 passes.

  analyse_heat(video, runs, det_model, n_gates, device, clip_device)
      runs: [{"stem", "rect", "out_json", "log"}] — the pilots to analyse
      → {stem: {"seconds", "device"} or {"error"}}

Each pilot keeps its own log (<stem>.analysis.log) and its own result file; a
pilot whose analysis fails doesn't stop the others.
"""

import contextlib
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from typing import Dict, List

import cv2

from race_day.capture import cut_quad, frame_step, gray_intervals, no_picture

QUEUE_FRAMES = 8          # pictures waiting per pilot (a few MB each): the decoder stays a little ahead
BATCH_WAIT_S = 0.03       # how long the detector waits for the other pilots' pictures of a moment


class Feed:
    """One pilot's pictures from the shared decoder, read like QuadReader:
    read() → (ok, frame), then .t; .fps, .frame_count, .rect, .step, .gray_t."""

    def __init__(self, rect, fps: float, frame_count: int, step: int):
        self.q: "queue.Queue" = queue.Queue(maxsize=QUEUE_FRAMES)
        self.rect, self.fps, self.frame_count, self.step = tuple(rect), fps, frame_count, step
        self.t = 0.0
        self.gray_t: List[float] = []
        self.closed = threading.Event()          # the pilot's pipeline stopped: no more pictures for it

    def read(self):
        item = self.q.get()
        if item is None:
            return False, None
        self.t, crop = item
        if no_picture(crop):
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


def decode(video: str, feeds: List[Feed], size=(960, 540)):
    """Read the video once; each frame's pilot pictures go to their feeds."""
    cap = cv2.VideoCapture(str(video))
    step = feeds[0].step
    try:
        while not all(f.closed.is_set() for f in feeds):
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            for _ in range(step - 1):             # skipped frames: decoded, not converted (as QuadReader)
                cap.grab()
            for f in feeds:
                if not f.closed.is_set():
                    f.put((t, cut_quad(frame, f.rect, size)))
    finally:
        cap.release()
        for f in feeds:
            f.put(None)


class SharedDetector:
    """One YOLO for all the pilots, called like YOLO (det(frame, **kw)[0]).
    With batch, the pictures the pilots ask about at the same moment go to
    the GPU in one call; each pilot gets its own picture's result."""

    def __init__(self, model_path: str, device, n: int, batch: bool = True):
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model                 # .model.names, as extract_race reads it
        self.device, self.batch = device, batch and n > 1
        self.active = n
        self.lock = threading.Lock()
        self.requests: "queue.Queue" = queue.Queue()
        if self.batch:
            threading.Thread(target=self._serve, daemon=True).start()

    def __call__(self, frame, **kw):
        if self.device:
            kw["device"] = self.device
        if not self.batch:
            with self.lock:
                return self.yolo(frame, **kw)
        fut: Future = Future()
        self.requests.put((frame, kw, fut))
        return [fut.result()]

    def pilot_done(self):
        """A pilot's pipeline finished: the batch no longer waits for it."""
        with self.lock:
            self.active -= 1

    def close(self):
        if self.batch:
            self.requests.put(None)

    def _serve(self):
        while True:
            first = self.requests.get()
            if first is None:
                return
            items = [first]
            deadline = time.time() + BATCH_WAIT_S
            while len(items) < max(1, self.active):
                try:
                    item = self.requests.get(timeout=max(0.0, deadline - time.time()))
                except queue.Empty:
                    break
                if item is None:
                    self.requests.put(None)          # handled after this batch
                    break
                items.append(item)
            try:
                results = self.yolo([f for f, _, _ in items], **items[0][1])
                for (_, _, fut), r in zip(items, results):
                    fut.set_result(r)
            except Exception as e:                   # every pilot waiting on this batch gets the error
                for _, _, fut in items:
                    fut.set_exception(e)


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
                 clip_device: str = "mps", blank_memory=None, batch: bool = True) -> Dict[str, dict]:
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
    feeds = [Feed(r["rect"], src_fps / step, n_frames // step, step) for r in runs]
    det = SharedDetector(det_model, device, len(runs), batch)
    clip = SharedClip(clip_device)
    results: Dict[str, dict] = {}
    out, err = _PerThreadOutput(sys.stdout), _PerThreadOutput(sys.stderr)

    def pilot(run: dict, feed: Feed):
        mem = None
        try:
            mem = blank_memory(n_gates) if blank_memory else None
            with open(run["log"], "w") as log:
                out.local.file = err.local.file = log
                try:
                    run_race_extraction(str(video), det_model, mem, run["out_json"], clip_device=clip_device,
                                        gate_id_logic="greedy", det_device=device, reader=feed,
                                        stem=run["stem"], det=det, clip=clip)
                    results[run["stem"]] = {"seconds": round(time.time() - t0, 1), "device": device,
                                            "picture": {"duration": round(feed.frame_count / feed.fps, 2),
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
            det.pilot_done()
            if mem:
                os.unlink(mem)

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        threads = [threading.Thread(target=pilot, args=(r, f), daemon=True) for r, f in zip(runs, feeds)]
        for th in threads:
            th.start()
        try:
            decode(video, feeds)
        finally:
            for th in threads:
                th.join()
            det.close()
    return results
