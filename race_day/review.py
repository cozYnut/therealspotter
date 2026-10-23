"""
Reviewing one pilot's run: confirm or fix each gate pass the system found,
add the ones it missed, mark gates the pilot skipped. Saved after every change
as <data>/gt/<run>.gt.json — the file score.py, pass_scorer and gate_decoder
read (the same format learn_ui's REVIEW mode wrote).

A review starts from the run's analysis (race_data passes, each a pending
mark with the system's gate ID) or from an earlier review of it.

    r = Review(tp, stem, video, crop)      # loads the gt.json, or prefills from race_data
    r.select(r.next_pending()); r.set_gate(3); r.accept(playhead)
    r.reject(); r.add("pass", t); r.cycle_tag(); r.undo()

Mark states: pending · accepted · fixed_id (accepted with another gate ID) ·
added (a pass the system missed) · skipped (a gate the pilot missed) ·
rejected (no real pass: saved under false_passes).
"""

import bisect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dataset_paths import TrackPaths, abs_path, rel

TAGS = [None, "crash_clipped", "bad_video", "reverse", "unsure"]
TAG_LABELS = {None: "", "crash_clipped": "crash/clipped", "bad_video": "bad video",
              "reverse": "reverse", "unsure": "unsure"}     # reverse: flown from the wrong side
STATES = ("pending", "accepted", "fixed_id", "added", "skipped", "rejected")


@dataclass
class Mark:
    t: float
    gate_id: int                      # -1 = not set yet
    kind: str = "pass"                # "pass" | "skipped" (the pilot missed this gate)
    origin: str = "prefill"           # "prefill" (the system's pass) | "added" (the system missed it)
    state: str = "pending"
    tag: Optional[str] = None
    time_adjusted: bool = False
    lap: int = 0
    pred: Optional[dict] = None       # the system's pass, for prefill marks
    track_at_pass: Optional[dict] = None


class Review:
    def __init__(self, tp: TrackPaths, stem: str, video: str, crop=None, track_name: str = ""):
        self.tp, self.stem, self.video = tp, stem, str(video)
        self.crop = list(crop) if crop else None
        self.track_name = track_name or tp.dir.name
        self.gt_path = tp.gt(stem)
        self.race_path = tp.race_data(stem)
        self.mem_path = tp.gate_memory
        mem = json.loads(self.mem_path.read_text(encoding="utf-8")).get("memory", []) if self.mem_path.exists() else []
        self.gates = sorted(mem, key=lambda g: int(g.get("order_idx", 0)))
        self.race = json.loads(self.race_path.read_text(encoding="utf-8")) if self.race_path.exists() else None
        self._race_ts = [float(f.get("t", 0.0)) for f in (self.race or {}).get("frames", [])]
        self.fps = float((self.race or {}).get("fps") or 30.0)
        self.duration = float((self.race or {}).get("duration") or 0.0)
        self.det_model = (self.race or {}).get("det_model")
        self.marks: List[Mark] = []
        self.sel: Optional[Mark] = None
        if self.gt_path.exists():
            self._load()
        elif self.race is not None:
            self._prefill()

    @property
    def n_gates(self) -> int:
        return max(1, len(self.gates))

    # ── loading ────────────────────────────────────────────────

    def _prefill(self):
        """Every pass the system found becomes a pending mark (marks added by hand are kept)."""
        marks = [m for m in self.marks if m.origin == "added"]
        prev = 0
        for i, p in enumerate(self.race.get("passes", [])):
            pred = {"idx": i, "t": round(float(p.get("t", 0.0)), 4), "gate_id": int(p.get("gate_id", -1)),
                    "gate_type": p.get("gate_type"), "source": p.get("source"), "sim": p.get("sim"),
                    "reason": p.get("reason"), "track_id": p.get("track_id"),
                    "query_img": rel(p.get("query_img", "")) if p.get("query_img") else ""}
            gid = pred["gate_id"] if pred["gate_id"] >= 1 else (prev % self.n_gates + 1 if prev else 1)
            if pred["source"] != "DUP":
                prev = gid
            marks.append(Mark(t=pred["t"], gate_id=gid, pred=pred,
                              track_at_pass=self.track_at(pred["t"], pred["track_id"])))
        self.marks = marks
        self._changed(save=False)

    def _load(self):
        data = json.loads(self.gt_path.read_text(encoding="utf-8"))
        marks = []
        for d in data.get("marks", []):
            state = d.get("verdict", "pending")
            marks.append(Mark(t=float(d.get("t", 0.0)), gate_id=int(d.get("gate_id", -1)), kind=d.get("kind", "pass"),
                              origin=d.get("origin", "prefill"),
                              state=state if state in STATES and state != "rejected" else "pending",
                              tag=d.get("tag") if d.get("tag") in TAGS else None,
                              time_adjusted=bool(d.get("time_adjusted", False)), lap=int(d.get("lap", 0)),
                              pred=d.get("pred"), track_at_pass=d.get("track_at_pass")))
        for d in data.get("false_passes", []):
            pred = d.get("pred") or {}
            marks.append(Mark(t=float(d.get("t", pred.get("t", 0.0))), gate_id=int(pred.get("gate_id", -1)),
                              state="rejected", tag=d.get("tag") if d.get("tag") in TAGS else None,
                              pred=d.get("pred"), track_at_pass=d.get("track_at_pass")))
        self.marks = marks
        self.det_model = data.get("det_model") or self.det_model
        if data.get("fps"):
            self.fps = float(data["fps"])
        self._changed(save=False)

    def track_at(self, t: float, track_id: Optional[int]) -> Optional[dict]:
        """Tracker state at the pass: the pass's track (or the biggest box) at the
        nearest saved frame, looking up to 15 frames back if it already vanished."""
        frames = (self.race or {}).get("frames", [])
        if not frames:
            return None
        i = bisect.bisect_left(self._race_ts, t)
        if i >= len(frames) or (i > 0 and abs(self._race_ts[i - 1] - t) <= abs(self._race_ts[i] - t)):
            i -= 1
        for off in range(16):
            j = i - off
            if j < 0:
                break
            tracks = frames[j].get("tracks", []) or []
            if track_id is not None:
                match = [tr for tr in tracks if tr.get("track_id") == track_id]
            else:
                match = sorted(tracks, key=lambda tr: (tr["bbox"][2] - tr["bbox"][0]) * (tr["bbox"][3] - tr["bbox"][1]),
                               reverse=True)
            if match:
                return dict(match[0], frame_offset=-off)
        return None

    # ── state ──────────────────────────────────────────────────

    @property
    def done(self) -> int:
        return sum(1 for m in self.marks if m.state != "pending")

    @property
    def complete(self) -> bool:
        return bool(self.marks) and not any(m.state == "pending" for m in self.marks)

    def suggest_gate(self, t: float, exclude: Optional[Mark] = None) -> int:
        """The next gate in order after the last real mark before t."""
        prev = [m for m in self.marks if m is not exclude and m.t < t and m.state != "rejected" and m.gate_id >= 1]
        return prev[-1].gate_id % self.n_gates + 1 if prev else 1

    def next_pending(self, direction: int = 1, now: Optional[float] = None) -> Optional[Mark]:
        """The next pending mark after the selected one (or after `now`)."""
        marks = self.marks
        if self.sel in marks:
            k = marks.index(self.sel)
            order = marks[k + 1:] + marks[:k] if direction > 0 else marks[:k][::-1] + marks[k + 1:][::-1]
        else:
            now = now or 0.0
            order = [m for m in marks if m.t >= now - 0.01] + [m for m in marks if m.t < now - 0.01]
            if direction < 0:
                order = order[::-1]
        return next((m for m in order if m.state == "pending"), None)

    def gate_image(self, gate_id: int) -> str:
        """A picture of the gate from the learned memory."""
        g = self.gates[gate_id - 1] if 1 <= gate_id <= len(self.gates) else {}
        for p in [g.get("last_img")] + list(reversed(g.get("embed_imgs") or [])):
            if p and os.path.exists(p):
                return p
        return ""

    def query_image(self, m: Mark) -> str:
        """The system's crop of the gate at this pass."""
        q = (m.pred or {}).get("query_img")
        return str(abs_path(q)) if q else ""

    # ── actions (each saves) ───────────────────────────────────

    def select(self, m: Optional[Mark]):
        self.sel = m

    def set_gate(self, gid: int) -> bool:
        m = self.sel
        if m is None or not 1 <= gid <= self.n_gates:
            return False
        m.gate_id = gid
        if m.state != "pending":
            m.state = "pending"                # a changed ID needs Accept again
        self._changed()
        return True

    def accept(self, playhead: float) -> str:
        """Accept the selected mark at the playhead. Returns "" or why not."""
        m = self.sel
        if m is None:
            return "Select a pass first."
        if m.gate_id < 1:
            return "Type a gate number first."
        if abs(playhead - m.t) > 1.0:
            return f"The video is {playhead - m.t:+.2f} s from this pass — select it again."
        half = 0.5 / max(self.fps, 1.0)
        if abs(playhead - m.t) > half:
            m.t = round(playhead, 4)
        m.time_adjusted = bool(m.pred) and abs(m.t - m.pred["t"]) > half
        if m.kind == "skipped":
            m.state = "skipped"
        elif m.origin == "added" or not m.pred:
            m.state = "added"
        elif m.gate_id != m.pred.get("gate_id"):
            m.state = "fixed_id"
        else:
            m.state = "accepted"
        if m.kind == "pass":
            m.track_at_pass = self.track_at(m.t, m.pred.get("track_id") if m.pred else None)
        self._changed()
        return ""

    def reject(self):
        """No real pass here (one added by hand is just removed)."""
        m = self.sel
        if m is None:
            return
        if m.origin == "added":
            self.marks.remove(m)
            self.sel = None
        else:
            m.state = "rejected"
        self._changed()

    def add(self, kind: str, t: float) -> Mark:
        """A pass the system missed (kind "pass") or a gate the pilot skipped ("skipped")."""
        t = round(t, 4)
        m = Mark(t=t, gate_id=self.suggest_gate(t), kind=kind, origin="added")
        self.marks.append(m)
        self.sel = m
        self._changed()
        return m

    def cycle_tag(self):
        m = self.sel
        if m is None:
            return
        m.tag = TAGS[(TAGS.index(m.tag) + 1) % len(TAGS)]
        self._changed()

    def undo(self):
        """Back to pending (a mark added by hand is removed)."""
        m = self.sel
        if m is None:
            return
        if m.origin == "added" or not m.pred:
            self.marks.remove(m)
            self.sel = None
        else:
            m.t = m.pred["t"]
            m.gate_id = m.pred["gate_id"] if m.pred["gate_id"] >= 1 else self.suggest_gate(m.t, exclude=m)
            m.state = "pending"
            m.time_adjusted = False
            m.track_at_pass = self.track_at(m.t, m.pred.get("track_id"))
        self._changed()

    # ── laps and saving ────────────────────────────────────────

    def _changed(self, save: bool = True):
        self.marks.sort(key=lambda m: m.t)
        lap = 0
        for m in self.marks:
            if m.state not in ("pending", "rejected") and m.gate_id == 1:
                lap += 1                       # G1 is start/finish: each reviewed crossing starts a lap
            m.lap = lap
        if save:
            self.save()

    def _mark_json(self, m: Mark) -> dict:
        g = self.gates[m.gate_id - 1] if 1 <= m.gate_id <= len(self.gates) else {}
        return {"kind": m.kind, "t": round(m.t, 4), "frame": int(round(m.t * self.fps)), "gate_id": m.gate_id,
                "gate_type": g.get("gate_type"), "lap": m.lap, "tag": m.tag, "origin": m.origin, "verdict": m.state,
                "time_adjusted": m.time_adjusted, "pred": m.pred, "track_at_pass": m.track_at_pass}

    def save(self):
        out = {
            "version": 1, "track": self.track_name, "video": rel(self.video),
            **({"crop": self.crop} if self.crop else {}),
            "gate_memory": rel(str(self.mem_path)),
            "race_data": rel(str(self.race_path)) if self.race_path.exists() else "",
            "det_model": self.det_model, "fps": self.fps, "duration": self.duration, "n_gates": len(self.gates),
            "status": "complete" if self.complete else "in_progress",
            "updated": datetime.now().isoformat(timespec="seconds"),
            "marks": [self._mark_json(m) for m in self.marks if m.state != "rejected"],
            "false_passes": [{"t": round(m.t, 4), "pred": m.pred, "tag": m.tag, "track_at_pass": m.track_at_pass}
                             for m in self.marks if m.state == "rejected"],
        }
        self.gt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.gt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, self.gt_path)
