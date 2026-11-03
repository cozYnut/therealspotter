"""
The live video: where each pilot is mid-lap, while the race is flown, for the
announcer (race_day/announcer.py).

FPVTrackside's capture card is opened a second time (macOS lets two apps read
one camera; the card's mode is left as FPVTrackside set it). Each frame's 2×2
picture is cut into the pilots' pictures with the layout FPVTrackside records
for the event (a heat's recordinfo channel bounds), and each pilot gets the
light pipeline — not the full analysis (no camera motion, no pass scorer):

    YOLO (Core ML, Neural Engine) → tracker → pass detector → a pass
    a pass → CLIP of the gate (a few frames before it fired) → online_decoder → G1…GN

All in one process of its own, so the app never waits on it. It analyses
`fps` frames a second, always the NEWEST frame: a slower Mac drops frames (and
says how many), so a gate is named a little late but the delay never grows.
Clock times are the Mac's own, the same clock FPVTrackside stamps its events
with. While no race runs, frames are read but not analysed.

What it sends back (on_event(dict)):
    {"type": "gate", "channel", "gate", "t", "conf", "laps"}   a pass named (gate 0: judged not a real pass)
    {"type": "picture", "channel", "gray"}                     the pilot's picture lost / back (2 s)
    {"type": "stats", "fps", "dropped", "delay_ms", "detector_ms"}
    {"type": "error", "error"}

A recorded heat video can stand in for the card (source = its path, with
`clock`: which video second is which clock time): the app's "Replay a heat".
"""

import math
import multiprocessing as mp
import queue
import threading
import time
import traceback
from collections import deque
from typing import Callable, Dict, Optional

FPS = 15                  # frames analysed a second, per pilot (the card gives 30–60)
EMBED_BACK = 4            # the gate's picture is taken this many 30-fps frames before the pass fired, as offline
GRAY_HOLD_S = 2.0         # a picture gray this long is lost (and back once it isn't)
STATS_EVERY = 2.0
LIVE_PASSDET = {"high_conf_top_tol": 0.2}   # a gate flown through fast and centred (gilboash's live finding)


def frame_step_kwargs(tracker_kw: dict, passdet_kw: dict, k: float):
    """The tracker and pass detector are tuned for every frame at 30 fps; with every k-th frame, time-outs grow k
    times, frame counts shrink k times, and per-frame smoothing (EMA) becomes its k-frame equivalent.
    (gilboash/therealspotter, online_spotter.py.)"""
    if k == 1:
        return tracker_kw, passdet_kw
    ema = lambda a: 1.0 - (1.0 - a) ** k
    t = {**tracker_kw, "ttl_seconds": tracker_kw["ttl_seconds"] * max(1.0, k), "ema_alpha": ema(tracker_kw["ema_alpha"])}
    p = {**passdet_kw, "disappear_timeout": passdet_kw["disappear_timeout"] * max(1.0, k),
         "min_aligned_frames": max(1, math.ceil(passdet_kw["min_aligned_frames"] / k - 1e-9)),
         "area_vel_ema_alpha": ema(passdet_kw["area_vel_ema_alpha"]),
         "approach_score_alpha": ema(passdet_kw["approach_score_alpha"])}
    return t, p


# ── the process ────────────────────────────────────────────────

class _Feed:
    """The newest frame of the card (or of a video played in real time), stamped with the clock."""

    def __init__(self, source, clock: Optional[dict]):
        import cv2
        self.cv2 = cv2
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(str(source))
            if clock:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * clock["from"])
        if not self.cap.isOpened():
            raise IOError(f"can't open the video source {source}")
        self.file, self.clock = not isinstance(source, int), clock
        self.new, self.read, self.dropped, self.ended = None, 0, 0, False
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cv2 = self.cv2
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.file:
                    break
                time.sleep(0.01)
                continue
            if self.file:                        # a video: at the moment its second plays on the clock
                vt = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                t = self.clock["wall"] + (vt - self.clock["video"]) if self.clock else time.time()
                if t > time.time():
                    time.sleep(t - time.time())
            else:
                t = time.time()
            with self.cond:
                if self.new is not None:
                    self.dropped += 1
                self.new = (frame, t)
                self.read += 1
                self.cond.notify_all()
        with self.cond:
            self.ended = True
            self.cond.notify_all()

    def get(self, timeout: float = 1.0):
        with self.cond:
            if self.new is None and not self.ended:
                self.cond.wait(timeout)
            item, self.new = self.new, None
            return item


def _main(cfg: dict, out_q, ctl_q):
    import contextlib
    import os
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):   # the pass detector's chatter
            _run(cfg, out_q, ctl_q)
    except Exception as e:                       # noqa: BLE001 — the app shows it
        out_q.put({"type": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=4)})


def _run(cfg: dict, out_q, ctl_q):
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from ultralytics import YOLO
    import online_decoder as od
    import pipeline_cfg
    from extract_race import _crop_padded
    from pass_detector import PassDetector, detect_camera_edges
    from race_day.capture import cut_quad, no_picture
    from tracking import ClipEmbedder, TimeTracker, _cls_to_name, clamp_bbox

    fps = float(cfg.get("fps") or FPS)
    det = YOLO(cfg["model"], task="detect")
    names = dict(det.names)
    det(np.zeros((540, 960, 3), np.uint8), conf=0.25, max_det=50, verbose=False)          # warm-up
    clip = ClipEmbedder(device="cpu")
    feed = _Feed(cfg["source"], cfg.get("clock"))
    pc = pipeline_cfg.load()
    tkw, pkw = frame_step_kwargs(pipeline_cfg.tracker_kwargs(pc), {**pipeline_cfg.gates_passdet_kwargs(pc), **LIVE_PASSDET},
                                 30.0 / fps)
    rects: Dict[str, tuple] = cfg["rects"]
    back = max(1, round(EMBED_BACK * fps / 30)) + 1

    def fresh():
        return {ch: {"tracker": TimeTracker(**tkw), "passdet": PassDetector(**pkw), "edges": False,
                     "recent": deque(maxlen=back), "decoder": od.OnlineDecoder(cfg["memory"], cfg.get("timing"))}
                for ch in rects}

    pilots, racing = fresh(), bool(cfg.get("racing"))
    gray = {ch: [False, None] for ch in rects}   # shown state, gray since
    jobs: "queue.Queue" = queue.Queue()

    def namer():                                 # CLIP + the decoder, off the detector's way
        torch.set_num_threads(1)
        while True:
            job = jobs.get()
            if job is None:
                return
            ch, dec, t, crop, gtype = job
            emb = clip.embed_bgr(crop) if crop is not None and crop.size else None
            evs = dec.add(t, emb, gtype)
            out_q.put({"type": "gate", "channel": ch, "gate": evs[0]["gate_id"], "t": t, "conf": evs[0]["conf"],
                       "laps": len(dec.laps)})
    threading.Thread(target=namer, daemon=True).start()
    out_q.put({"type": "ready", "names": list(names.values())[:5]})

    done, det_s, lags, next_t, w0 = 0, 0.0, deque(maxlen=60), 0.0, time.time()
    w_read, w_drop, w_done = 0, 0, 0
    while True:
        try:
            while True:
                c = ctl_q.get_nowait()
                if c[0] == "stop":
                    jobs.put(None)
                    return
                if c[0] == "race":
                    racing = bool(c[1])
                    if racing:
                        pilots = fresh()
        except queue.Empty:
            pass
        item = feed.get(0.5)
        if item is None:
            if feed.ended:
                out_q.put({"type": "ended"})
                jobs.put(None)
                return
            continue
        frame, t = item
        if t < next_t:                           # faster than fps: this one isn't analysed
            continue
        next_t = t + 1.0 / fps
        for ch, rect in rects.items():
            img = cut_quad(frame, rect, (960, 540))
            g = no_picture(img)
            st = gray[ch]
            st[1] = (st[1] or t) if g else None
            lost = g and t - st[1] >= GRAY_HOLD_S
            if lost != st[0] and (lost or not g):
                st[0] = lost
                out_q.put({"type": "picture", "channel": ch, "gray": lost, "t": t})
            if not racing or g:
                continue
            q = pilots[ch]
            if not q["edges"]:
                q["passdet"].set_camera_edges(*detect_camera_edges(img))
                q["edges"] = True
            t0 = time.time()
            res = det(img, conf=0.25, max_det=50, verbose=False)[0]
            det_s += time.time() - t0
            typed = []
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                typed.append({"bbox": clamp_bbox((x1, y1, x2, y2), 960, 540), "det_conf": conf,
                              "type": _cls_to_name(int(b.cls[0]), names) if conf >= 0.20 else "NONE",
                              "type_score": conf})
            typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:10]
            tracks = q["tracker"].update(typed, t)
            q["passdet"].update(tracks, t, frame_w=960, frame_h=540)
            q["recent"].append((img, {int(tr.track_id): list(tr.bbox) for tr in tracks}))
            while True:
                evt = q["passdet"].pop_any_passed()
                if evt is None:
                    break
                tp = min(t, float(evt.get("last_seen", t)) + 1.0 / 30)
                old, boxes = q["recent"][0]
                box = boxes.get(int(evt.get("track_id", -1)))
                jobs.put((ch, q["decoder"], tp, _crop_padded(old, box)[0] if box else None, str(evt.get("type", ""))))
        done += 1
        lags.append(time.time() - t)
        now = time.time()
        if now - w0 >= STATS_EVERY:
            dt = now - w0
            out_q.put({"type": "stats", "fps": round((done - w_done) / dt, 1), "racing": racing,
                       "dropped": round(100 * (feed.dropped - w_drop) / max(1, feed.read - w_read)),
                       "delay_ms": round(1000 * float(np.median(lags))) if lags else None,
                       "detector_ms": round(1000 * det_s / max(1, (done - w_done) * len(rects)))})
            w0, w_read, w_drop, w_done, det_s = now, feed.read, feed.dropped, done, 0.0


# ── the app's side ─────────────────────────────────────────────

class LiveVideo:
    """The live-video process, started and stopped by the app; its events go to on_event (on its own thread)."""

    def __init__(self, on_event: Callable[[dict], None], log: Callable[[str], None] = print):
        self.on_event, self.log = on_event, log
        self.proc = None
        self.status = {"running": False, "error": None, "stats": None, "source": None}

    def start(self, source, rects: Dict[str, tuple], model: str, memory: str, timing: Optional[str],
              fps: float = FPS, clock: Optional[dict] = None, racing: bool = False):
        self.stop()
        ctx = mp.get_context("spawn")
        self.out_q, self.ctl_q = ctx.Queue(), ctx.Queue()
        cfg = {"source": source, "rects": {k: tuple(v) for k, v in rects.items()}, "model": model, "memory": memory,
               "timing": timing, "fps": fps, "clock": clock, "racing": racing}
        self.proc = ctx.Process(target=_main, args=(cfg, self.out_q, self.ctl_q), daemon=True, name="live-video")
        self.proc.start()
        self.status = {"running": True, "error": None, "stats": None, "source": source, "ready": False}
        threading.Thread(target=self._pump, args=(self.proc, self.out_q), daemon=True).start()

    def _pump(self, proc, out_q):
        while proc.is_alive() or not out_q.empty():
            try:
                e = out_q.get(timeout=0.5)
            except (queue.Empty, OSError, EOFError):
                continue
            if e.get("type") == "stats":
                self.status["stats"] = e
            elif e.get("type") == "ready":
                self.status["ready"] = True
            elif e.get("type") == "error":
                self.status["error"] = e["error"]
                self.log(f"live video: {e['error']}")
            elif e.get("type") == "ended":
                self.status["running"] = False
            try:
                self.on_event(e)
            except Exception as ex:              # noqa: BLE001
                self.log(f"live video: {type(ex).__name__}: {ex}")
        self.status["running"] = False

    def set_race(self, on: bool):
        """A race started (on: everything starts afresh) or ended (frames are no longer analysed)."""
        if self.proc and self.proc.is_alive():
            self.ctl_q.put(("race", bool(on)))

    def stop(self):
        if self.proc and self.proc.is_alive():
            self.ctl_q.put(("stop",))
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
        self.status["running"] = False
