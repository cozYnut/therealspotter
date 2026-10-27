"""
Race day on an FPVTrackside event folder (gate validation for FPVTrackside).

Everything runs between races, by itself:

  new finished heat → each pilot's quadrant analysed in place (workers,
  in parallel) → once the track is learned: gate IDs and laps (workers) →
  lap check against the timing system (marshal assist) → in Auto mode the
  confident corrections are written to the heat's Race.json.

When FPVTrackside starts a race, race_day pauses completely: the event folder
isn't read (except to see the race end) and the worker processes are
stopped (SIGSTOP), so FPVTrackside has the computer to itself.

With FPVTrackside's REST API (the FPVTrackside alpha on, race_day/ftapi.py)
and the event open in FPVTrackside, lap changes go through the API instead
of into Race.json: FPVTrackside shows them at once, and a running race is
seen from FPVTrackside's race control rather than from the files.

State: <event>/lazySpotter/race_day.json — a restart loses nothing.

Live capture (mode "capture"): the same, on a track's own recordings
(trackside.TrackVideos — <track>/videos/, data in <track>/lazySpotter/). Nothing runs by
itself: the user picks the heats to analyse and to learn from. While the app
records a heat, everything pauses (set_recording).
"""

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dataset_paths import TrackPaths
from race_day import labeling, lapcheck, stats as rstats, trackside
from race_day.models import DEFAULT_DET_MODEL

REFRESH_S = 3.0            # look at the event folder (stat only) this often between races
RACE_REFRESH_S = 5.0       # while a race runs: only to see it end
DEFAULT_GATES = 10         # gate count used for analysis before the track is learned


def _now():
    return time.time()


class EventDay:
    def __init__(self, event_dir=None, mode: str = "live", workers: int = 1, det_model: str = str(DEFAULT_DET_MODEL),
                 clip_device: str = "mps", log: Callable[[str], None] = print, event=None, use_coreml: bool = True):
        if event is not None:                             # live capture: trackside.TrackVideos
            self.event = event
        elif (Path(event_dir) / "Event.json").exists():
            self.event = trackside.Event(event_dir)
        else:                                             # Replays: a folder of loose videos
            self.event = trackside.VideoFolder(event_dir)
            mode = "replays"
        self.mode = mode                                  # live | replays | capture
        self.tp = TrackPaths(self.event.data_dir)
        self.tp.dir.mkdir(exist_ok=True)
        self.tp.runs_dir.mkdir(exist_ok=True)
        self.workers, self.det_model, self.clip_device, self.log = workers, det_model, clip_device, log
        self.use_coreml = use_coreml                      # the detector on the Neural Engine (race_day/models.py)
        self.lock = threading.RLock()
        self.cmds: "queue.Queue" = queue.Queue()
        self.state = {"event": self.event.name, "n_gates": None, "learned_from": [], "learned_at": None,
                      "auto": False, "offset": None, "heats": {}, "runs": {}}
        if self.tp.race_day_state.exists():
            try:
                saved = json.loads(self.tp.race_day_state.read_text(encoding="utf-8"))
            except ValueError:
                saved = {}
            if "config" in saved and "event" not in saved:   # the old automatic live mode's state: kept aside
                os.replace(self.tp.race_day_state, self.tp.race_day_state.with_suffix(".old_live.json"))
                saved = {}
            self.state.update({k: v for k, v in saved.items() if k in self.state})
        if mode != "live":
            self.state["auto"] = False
        self.pool = None
        self.paused = False
        self.learning: Optional[object] = None            # Future of the learn job
        self.jobs: Dict[str, tuple] = {}                  # key → (kind, heat id, Future, info)
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stop = threading.Event()
        self._learn_wait: Optional[dict] = None           # learn requested, waiting for analyses
        self.ft = None                                    # ftapi.FPVTrackside (FPVTrackside events)
        self.fril = None                                  # fril.FrilLive: the running order on fril.co.il/live
        self.ft_state = {"level": "none", "serves": False, "race": None, "running": False, "error": None}

    # ── lifecycle ──────────────────────────────────────────────

    def start(self):
        from race_day.analysis import AnalysisPool
        from race_day.models import worker_devices
        self.pool = AnalysisPool(worker_devices(Path(self.det_model), self.workers, self.use_coreml), self.clip_device)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self.pool:
            self.pool.shutdown(wait=False)
        self._save()

    def _save(self):
        with self.lock:
            data = json.dumps(self.state, indent=1)
        tmp = str(self.tp.race_day_state) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, self.tp.race_day_state)

    # ── commands from the app (handled on the loop thread) ─────

    def analyse(self, heat_ids: List[str]):
        self.cmds.put(("analyse", list(heat_ids)))

    def learn(self, n_gates: int):
        """Learn the track from the pilot runs ticked for learning."""
        self.cmds.put(("learn", self.learn_runs(), int(n_gates)))

    def set_auto(self, on: bool):
        self.cmds.put(("auto", bool(on) and self.mode == "live"))

    def set_checked(self, heat_id: str, on: bool):
        """Tick / untick every pilot of a heat for learning."""
        h = self.event.heats.get(heat_id)
        with self.lock:
            self._heat(heat_id)["learn_pilots"] = sorted(h.pilots) if (on and h) else []
        self.cmds.put(("save",))

    def set_pilot_checked(self, heat_id: str, channel: str, on: bool):
        with self.lock:
            sel = set(self._heat(heat_id).get("learn_pilots", []))
            (sel.add if on else sel.discard)(channel)
            self._heat(heat_id)["learn_pilots"] = sorted(sel)
        self.cmds.put(("save",))

    def learn_runs(self) -> List[tuple]:
        """(heat id, channel) of every pilot run ticked for learning."""
        with self.lock:
            return [(hid, ch) for hid, hs in self.state["heats"].items() if hid in self.event.heats
                    for ch in hs.get("learn_pilots", [])]

    def decide(self, heat_id: str, channel: str, key: str, decision: Optional[str]):
        """A finding accepted / rejected (None: undecided again)."""
        with self.lock:
            d = self._heat(heat_id).setdefault("decisions", {})
            if decision:
                d[f"{channel}|{key}"] = decision
            else:
                d.pop(f"{channel}|{key}", None)
        self.cmds.put(("save",))

    def add_lap(self, heat_id: str, channel: str, video_t: float):
        """The marshal marks a lap (or the holeshot) at a moment in the video."""
        h = self.event.heats.get(heat_id)
        q = self._quad(h, channel)
        if not q:
            return
        clock = float(self.event.to_clock(q.video, video_t)) + (self.state.get("offset") or 0.0)
        with self.lock:
            self._heat(heat_id).setdefault("marks", []).append(
                {"op": "add", "channel": channel, "t": round(clock, 3), "video_t": round(video_t, 3), "by": "hand"})
        self.cmds.put(("save",))

    def remove_lap(self, heat_id: str, channel: str, detection_id: Optional[str] = None, mark_index: int = -1):
        """Remove a timing lap (by its detection) or an added mark."""
        with self.lock:
            hs = self._heat(heat_id)
            marks = hs.setdefault("marks", [])
            if detection_id:
                marks.append({"op": "remove", "channel": channel, "detection_id": detection_id, "by": "hand"})
            elif 0 <= mark_index < len(marks):
                marks.pop(mark_index)
        self.cmds.put(("save",))

    def set_recording(self, on: bool):
        """Live capture: a heat is being recorded — pause the workers right
        away (the loop sees it too and stays paused until it ends)."""
        self.event.running = True if on else None
        if self.pool:
            (self.pool.pause if on else self.pool.resume)()
        self.paused = bool(on)

    def review_changed(self, heat_id: str):
        """A pilot of this heat was reviewed: its passes and laps become the
        reviewed ones, and the heat is checked again (stats for the dashboard,
        the lap check) — on the loop thread."""
        self.cmds.put(("review", heat_id))

    def configure_video(self, name: str, info: dict):
        """Replays: what a loose video is (single / 2×2, heat, pilots)."""
        self.event.set_info(name, info)

    def set_fpvtrackside(self, url: str, token: Optional[str] = None):
        """Where FPVTrackside's web server is (its REST API, when it has one)."""
        from race_day import ftapi
        if isinstance(self.event, trackside.Event):
            self.ft = ftapi.FPVTrackside(url or ftapi.DEFAULT_URL, token)
            self.cmds.put(("ft",))

    def set_fril(self, url: Optional[str]):
        """fril's live ingest address (with its token): the race on now and the
        races to come are sent there (race_day/fril.py). Empty: not sent."""
        from race_day import fril
        self.fril = fril.FrilLive(url, log=self.log) if url and isinstance(self.event, trackside.Event) \
            and self.mode == "live" else None

    def write(self, heat_id: str):
        self.cmds.put(("write", heat_id, "hand"))

    def restore(self, heat_id: str):
        self.cmds.put(("restore", heat_id))

    # ── views for the app ──────────────────────────────────────

    def status(self) -> dict:
        with self.lock:
            kinds = [j[0] for j in self.jobs.values()]
            return {"event": self.event.name, "mode": self.mode, "race_running": bool(self.event.running),
                    "paused": self.paused, "learned": labeling.is_learned(self.tp),
                    "n_gates": self.state["n_gates"], "learning": self.learning is not None,
                    "learn_waiting": self._learn_wait is not None,
                    "analysing": kinds.count("analyse"), "labelling": kinds.count("label"),
                    "auto": self.state["auto"], "offset": self.state["offset"], "error": self.error,
                    "heats": len(self.event.heats), "timing": self.event.has_timing,
                    "fpvtrackside": dict(self.ft_state, url=self.ft.url if self.ft else None),
                    "fril": self.fril.status() if self.fril else None,
                    "unconfigured": list(getattr(self.event, "unconfigured", []))}

    def dashboard_status(self) -> dict:
        """/api/status in the dashboard's shape."""
        st = self.status()
        nxt = max(self.event.heats.values(), key=lambda h: h.start, default=None)
        return {"track": st["event"], "mode": self.mode, "phase": "result" if st["learned"] else "learn",
                "learn_progress": [len(self.state["runs"]), 0], "n_gates": st["n_gates"],
                "queue": {"analysing": st["analysing"], "waiting": 0}, "live": {},
                "pilot_info": {"source": "event", "pilots": nxt.pilots if nxt else {},
                               "phase": "running" if st["race_running"] else "finished"},
                "runs_total": len(self.state["runs"]), "heats": st["heats"], "error": st["error"]}

    def race_day(self) -> Optional[dict]:
        """An FPVTrackside event's running order (race_day/schedule.py): the race
        on now and the next ones. None for tracks and folders of videos."""
        if not isinstance(self.event, trackside.Event):
            return None
        if getattr(self, "_schedule", None) is None:
            from race_day.schedule import Schedule
            self._schedule = Schedule(self.event.dir)
        running = self.event.running if self.mode == "live" else None
        if self.ft_state["serves"]:
            running = self.ft_state["race"] if self.ft_state["running"] else None
            self._schedule.web = self.ft.url
        return self._schedule.get(running)

    def heat_list(self) -> List[dict]:
        """Newest first: number, label, pilots with their timing laps, status."""
        out = []
        with self.lock:
            for h in sorted(self.event.heats.values(), key=lambda h: -h.start):
                hs = self.state["heats"].get(h.id, {})
                timing = self.event.timing(h) if self.event.has_timing else {}
                sel = set(hs.get("learn_pilots", []))
                out.append({"id": h.id, "number": h.number, "label": h.label, "pilots": h.pilots,
                            "status": hs.get("status", "new"),
                            "learn": "all" if sel and sel >= set(h.pilots) else ("some" if sel else "none"),
                            "learn_pilots": sorted(sel),
                            "timing_laps": {c: max(0, len(v) - 1) for c, v in timing.items()},
                            "reviewed": [c for c in h.pilots if self.tp.gt(h.stem(c)).exists()],
                            "no_video": [c for c, v in hs.get("check", {}).items() if v.get("no_video")],
                            "open": self._open_findings(h.id), "written": bool(hs.get("written"))})
        return out

    def heat_detail(self, heat_id: str) -> Optional[dict]:
        h = self.event.heats.get(heat_id)
        if not h:
            return None
        with self.lock:
            hs = json.loads(json.dumps(self.state["heats"].get(heat_id, {})))
        timing = self.event.timing(h)
        pilots = {}
        for q in h.quads:
            if q.channel not in h.pilots:
                continue
            ch = q.channel
            dets = timing.get(ch, [])
            check = hs.get("check", {}).get(ch, {})
            race = self._race(h.stem(ch))
            pilots[ch] = {
                "pilot": h.pilots[ch], "quad": {"video": str(q.video), "rect": q.rect, "kind": q.kind},
                "timing": [{"t": d["t"], "video_t": float(self.event.to_video(q.video, d["t"] - (self.state["offset"] or 0))),
                            "id": d["id"], "holeshot": d["holeshot"], "type": d["type"], "lap": d.get("lap"),
                            "race_t": d["t"] - h.start} for d in dets],
                "video_g1": [p["t"] for p in lapcheck.g1_passes(race)] if race else [],
                "passes": [{"t": p["t"], "gate": p.get("gate_id", -1)} for p in (race or {}).get("passes", [])],
                "laps": [{"lap": l.get("lap"), "t0": l["t0"], "t1": l["t1"]} for l in (race or {}).get("laps", [])],
                "stem": h.stem(ch), "race_data": str(self.tp.race_data(h.stem(ch))),
                "gt": str(self.tp.gt(h.stem(ch))), "reviewed": self.tp.gt(h.stem(ch)).exists(),
                "review": (race or {}).get("review"),
                "findings": [dict(f, race_t=f["t"] - h.start) for f in check.get("findings", [])],
                "run": hs.get("runs", {}).get(ch, {}),
                "checked": bool(check), "no_video": bool(check.get("no_video")),
                "no_video_findings": check.get("no_video_findings", 0),
            }
        return {"id": h.id, "number": h.number, "label": h.label, "start": h.start, "end": h.end,
                "status": hs.get("status", "new"), "pilots": pilots, "decisions": hs.get("decisions", {}),
                "marks": hs.get("marks", []), "written": hs.get("written", []),
                "learn_pilots": hs.get("learn_pilots", []),
                "changes": self.changes(heat_id), "todo": self.marshal_todo(heat_id), "n_gates": self._n_gates(),
                "done_in_fpvtrackside": hs.get("done_in_fpvtrackside", 0), "overwritten": hs.get("overwritten"),
                "api_error": hs.get("api_error"), "api": bool(self.ft_state["serves"]),
                "api_writes": sum(1 for w in hs.get("written", []) if w.get("via") == "api" and not w.get("undone")
                                  and w.get("by") != "restore"),
                "videos": sorted({str(q.video) for q in h.quads})}

    def changes(self, heat_id: str) -> List[dict]:
        """The lap changes that 'Update laps in FPVTrackside' would write:
        accepted findings + the marshal's own marks. An accepted finding about
        a timing detection (false, or unsure about a timing lap) removes that
        lap; any other accepted finding is a start/finish crossing seen in the
        video (missed, or unsure about a video lap) and adds a lap there."""
        with self.lock:
            hs = self.state["heats"].get(heat_id, {})
            out = []
            for ch, c in hs.get("check", {}).items():
                for f in c.get("findings", []):
                    if hs.get("decisions", {}).get(f"{ch}|{lapcheck_key(f)}") != "accepted":
                        continue
                    if f.get("detection_id"):
                        out.append({"op": "remove", "channel": ch, "detection_id": f["detection_id"], "by": "check"})
                    else:
                        out.append({"op": "add", "channel": ch, "t": f["t"], "by": "check"})
            out += [dict(m) for m in hs.get("marks", [])]
            return out

    def marshal_todo(self, heat_id: str) -> List[dict]:
        """The same changes as steps for a marshal in FPVTrackside itself (a
        stock FPVTrackside takes no changes from outside while the event is
        open: it shows a written Race.json only after the event is reopened).
        Each step: op (add / remove), channel, pilot, t (clock s), race_t
        (s from the race start: the time FPVTrackside's replay shows), lap
        (remove: the lap number FPVTrackside shows now; add: the lap it
        becomes, 0 = holeshot), lap_time, last (an added lap after the
        pilot's last one). In race-time order: a lap is found by its time,
        which doesn't change when FPVTrackside renumbers the laps."""
        h = self.event.heats.get(heat_id)
        changes = self.changes(heat_id)
        if not h or not changes:
            return []
        timing = self.event.timing(h)
        out = []
        for ch in sorted({c["channel"] for c in changes}):
            dets = timing.get(ch, [])
            removed = {c["detection_id"] for c in changes if c["channel"] == ch and c["op"] == "remove"}
            added = [c["t"] for c in changes if c["channel"] == ch and c["op"] == "add"]
            first = 0 if (dets[0]["holeshot"] if dets else True) else 1      # FPVTrackside's first lap number
            after = sorted([(d["t"], d) for d in dets if d["id"] not in removed] + [(t, None) for t in added],
                           key=lambda x: x[0])
            last_kept = max((d["t"] for d in dets if d["id"] not in removed), default=None)
            for i, (t, d) in enumerate(after):
                if d is None:
                    out.append({"op": "add", "channel": ch, "pilot": h.pilots.get(ch, "?"), "t": t,
                                "race_t": t - h.start, "lap": first + i,
                                "lap_time": t - after[i - 1][0] if i else None,
                                "last": last_kept is None or t > last_kept})
            for k, d in enumerate(dets):
                if d["id"] in removed:
                    out.append({"op": "remove", "channel": ch, "pilot": h.pilots.get(ch, "?"), "t": d["t"],
                                "race_t": d["t"] - h.start,
                                "lap": d.get("lap") if d.get("lap") is not None else first + k,
                                "lap_time": d["t"] - dets[k - 1]["t"] if k else None, "detection_id": d["id"]})
        off = self.state.get("offset") or 0.0
        for st in out:
            q = self._quad(h, st["channel"])
            st["video_t"] = float(self.event.to_video(q.video, st["t"] - off)) if q else None
        return sorted(out, key=lambda x: x["t"])

    # ── helpers ────────────────────────────────────────────────

    def _heat(self, heat_id: str) -> dict:
        return self.state["heats"].setdefault(heat_id, {"status": "new", "runs": {}})

    @staticmethod
    def _quad(h, channel):
        return next((q for q in (h.quads if h else []) if q.channel == channel), None)

    def _race(self, stem: str) -> Optional[dict]:
        p = self.tp.race_data(stem)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except ValueError:
            return None

    def _open_findings(self, heat_id: str) -> int:
        hs = self.state["heats"].get(heat_id, {})
        dec = hs.get("decisions", {})
        return sum(1 for ch, c in hs.get("check", {}).items() for f in c.get("findings", [])
                   if f"{ch}|{lapcheck_key(f)}" not in dec)

    def _n_gates(self) -> int:
        """The track's gate count: as set (setup / learning), else the learned
        gates' (a track learned before it was opened here), else a default."""
        if self.state["n_gates"]:
            return int(self.state["n_gates"])
        try:
            n = len(json.loads(self.tp.gate_memory.read_text(encoding="utf-8")).get("memory", []))
        except (OSError, ValueError):
            n = 0
        return n or DEFAULT_GATES

    # ── the loop ───────────────────────────────────────────────

    def _loop(self):
        last = 0.0
        while not self._stop.is_set():
            try:
                now = _now()
                if now - last >= (RACE_REFRESH_S if self.event.running else REFRESH_S):
                    last = now
                    self._refresh()
                if not self.paused:
                    self._handle_cmds()
                    self._collect()
            except Exception:
                self.error = traceback.format_exc(limit=3)
                self.log(self.error)
            self._stop.wait(0.3)

    def _poll_fpvtrackside(self):
        """The API's level, whether FPVTrackside has this event open, and its
        race control (what's selected, whether it's running)."""
        from race_day import ftapi
        st = {"level": "none", "serves": False, "race": None, "running": False, "error": None}
        if self.ft:
            try:
                st["level"] = self.ft.level()
                st["serves"] = self.ft.serves(self.event.dir.name)
                if st["serves"]:
                    rc = self.ft.race_control()
                    st["race"], st["running"] = rc["race"], rc["state"] in ("starting", "running")
            except ftapi.ApiError as e:
                st["error"] = str(e)
        self.ft_state = st

    def _refresh(self):
        r = self.event.refresh()
        self._poll_fpvtrackside()
        if self.fril:                                 # the running order on fril.co.il/live (sent on a change)
            self.fril.update(self.race_day())
        if self.ft_state["running"]:                  # FPVTrackside's own race control says so
            r["running"] = r["running"] or self.ft_state["race"] or "fpvtrackside"
        what = "Recording" if self.mode == "capture" else "Race running in FPVTrackside"
        if r["running"] and not self.paused:
            self.paused = True
            if self.pool:
                self.pool.pause()
            self.log(f"{what} — paused")
        elif not r["running"] and self.paused:
            self.paused = False
            if self.pool:
                self.pool.resume()
            self.log(f"{what} over — continuing")
        if self.paused:
            return
        for hid in r["new"] + r["changed"]:
            with self.lock:
                hs = self._heat(hid)
            if hid in r["changed"] and hs.get("status") in ("checked", "updated"):
                self._check(hid, external=True)       # Race.json changed (a marshal edit): check again
            elif self.state["auto"] and hs.get("status") == "new":
                self._submit_heat(hid)
        if r["new"]:
            self._save()

    def _handle_cmds(self):
        while True:
            try:
                cmd = self.cmds.get_nowait()
            except queue.Empty:
                return
            kind = cmd[0]
            if kind == "analyse":
                for hid in cmd[1]:
                    self._submit_heat(hid, force=True)
            elif kind == "learn":
                self._start_learn(cmd[1], cmd[2])
            elif kind == "save":
                pass
            elif kind == "auto":
                with self.lock:
                    self.state["auto"] = cmd[1]
                if cmd[1]:
                    for hid, hs in list(self.state["heats"].items()):
                        if hs.get("status") == "new" and hid in self.event.heats:
                            self._submit_heat(hid)
                        elif hs.get("status") == "checked":
                            self._auto_write(hid)
            elif kind == "ft":
                self._poll_fpvtrackside()
            elif kind == "write":
                self._write(cmd[1], cmd[2])
            elif kind == "restore":
                self._restore(cmd[1])
            elif kind == "review":
                h = self.event.heats.get(cmd[1])
                if h:
                    for c in h.pilots:
                        if self.tp.race_data(h.stem(c)).exists():
                            labeling.apply_review(self.tp, h.stem(c))
                    self._check(cmd[1])
            self._save()

    # analysis ------------------------------------------------------

    def _submit_heat(self, hid: str, force: bool = False):
        h = self.event.heats.get(hid)
        if not h:
            return
        with self.lock:
            hs = self._heat(hid)
            if not force and hs.get("status") not in ("new", None):
                return
            if any(j[1] == hid for j in self.jobs.values()):
                return
            hs["status"] = "analysing"
            self._submit_runs(h, [c for c in h.pilots], force)
        self._after_analysis(hid)

    def _submit_runs(self, h, channels: List[str], force: bool = False):
        """Analyse these pilots of heat h: one job per video (a heat is one
        2×2 video, plus one for its DJI pilots), its pilots analysed together
        (race_day/heat_analysis.py). Pilots already analysed or being analysed
        are left out unless force."""
        by_video: Dict[str, list] = {}
        with self.lock:
            hs = self._heat(h.id)
            for q in h.quads:
                if q.channel not in channels or q.channel not in h.pilots:
                    continue
                stem = h.stem(q.channel)
                if f"a:{stem}" in self.jobs or (not force and self.tp.race_data(stem).exists()):
                    continue
                by_video.setdefault(str(q.video), []).append(q)
            for video, quads in by_video.items():
                runs = [{"stem": h.stem(q.channel), "rect": q.rect, "out_json": self.tp.race_data(h.stem(q.channel))}
                        for q in quads]
                fut = self.pool.submit_heat(video, runs, self._n_gates(), self._race_window(h, quads[0].video))
                for q in quads:
                    stem = h.stem(q.channel)
                    self.jobs[f"a:{stem}"] = ("analyse", h.id, fut, {"stem": stem, "channel": q.channel})
                    hs.setdefault("runs", {})[q.channel] = {"stem": stem, "status": "analysing", "kind": q.kind}
                    if hs.get("status") in (None, "new"):
                        hs["status"] = "analysing"

    RACE_BEFORE_S, RACE_AFTER_S = 5.0, 30.0       # analysed around an FPVTrackside race (as the lap check looks)

    def _race_window(self, h, video) -> Optional[tuple]:
        """An FPVTrackside heat: only the race is analysed — from a few seconds
        before the start to the end plus as long as the lap check looks for
        late crossings (lapcheck: end + 30 s) — in video seconds. Other videos
        (loose, recorded here) have no race times: the whole video."""
        if not isinstance(self.event, trackside.Event) or not h.start or not h.end:
            return None
        try:
            a = float(self.event.to_video(video, h.start)) - self.RACE_BEFORE_S
            b = float(self.event.to_video(video, h.end)) + self.RACE_AFTER_S
        except Exception:
            return None
        return (max(0.0, a), b) if b > a else None

    def _collect(self):
        """Finished jobs → next step."""
        done = [k for k, j in list(self.jobs.items()) if j[2].done()]
        for k in done:
            kind, hid, fut, info = self.jobs.pop(k)
            try:
                res = fut.result()
            except Exception as e:
                res, err = None, f"{type(e).__name__}: {e}"
            else:
                err = None
            if kind == "analyse" and not err:          # a heat job: each pilot has its own result
                res = (res or {}).get(info["stem"]) or {"error": "not analysed"}
                err = res.get("error")
            with self.lock:
                hs = self._heat(hid)
                if kind == "analyse":
                    run = hs["runs"].setdefault(info["channel"], {"stem": info["stem"]})
                    run.update(status="failed" if err else "analysed", error=err, analysis=res)
                    if err:
                        self.log(f"{info['stem']}: analysis failed — {err}")
            if kind == "analyse":
                self._after_analysis(hid)
            elif kind == "label":
                if err:
                    self.log(f"heat {hid[:8]}: labelling failed — {err}")
                self._check(hid)
                self._auto_write(hid)
        if self.learning is not None and self.learning.done():
            self._learned()
        if self._learn_wait and not any(j[0] == "analyse" for j in self.jobs.values()):
            w, self._learn_wait = self._learn_wait, None
            self._run_learn(w["runs"], w["n_gates"])
        if done:
            self._save()

    def _after_analysis(self, hid: str):
        """All of a heat's pilots analysed → label it (if the track is learned)."""
        if any(j[1] == hid and j[0] == "analyse" for j in self.jobs.values()):
            return
        with self.lock:
            hs = self._heat(hid)
            if hs.get("status") == "analysing":
                hs["status"] = "analysed"
        if labeling.is_learned(self.tp) and self.learning is None:
            self._submit_label(hid)

    def _submit_label(self, hid: str):
        h = self.event.heats.get(hid)
        if not h:
            return
        stems = [h.stem(c) for c in h.pilots if self.tp.race_data(h.stem(c)).exists()]
        quads = {h.stem(q.channel): (str(q.video), tuple(q.rect)) for q in h.quads if h.stem(q.channel) in stems}
        if stems:
            self.jobs[f"l:{hid}"] = ("label", hid, self.pool.submit_label(self.tp.dir, stems, quads), {})

    # learning ------------------------------------------------------

    def _start_learn(self, runs: List[tuple], n_gates: int):
        """runs: (heat id, channel) chosen by the user. Only those pilots are
        analysed first (if they aren't yet), then learned from."""
        with self.lock:
            self.state["n_gates"] = n_gates
        by_heat: Dict[str, List[str]] = {}
        for hid, ch in runs:
            by_heat.setdefault(hid, []).append(ch)
        for hid, chs in by_heat.items():                 # a heat's chosen pilots analysed together
            if hid in self.event.heats:
                self._submit_runs(self.event.heats[hid], chs)
        self._learn_wait = {"runs": runs, "n_gates": n_gates}
        self.log(f"Learning the track ({n_gates} gates) from {len(runs)} pilot runs…")

    def _run_learn(self, runs: List[tuple], n_gates: int):
        """Learn from the chosen runs. With FPVTrackside's timing, each run's
        lap crossings are passed in as known G1 moments (any run with one
        full lap is enough, and learning is much faster)."""
        stems, g1 = [], {}
        offset = self.state.get("offset") or 0.0
        for hid, ch in runs:
            h = self.event.heats.get(hid)
            if not h or not self.tp.race_data(h.stem(ch)).exists():
                continue
            stems.append(h.stem(ch))
            q = self._quad(h, ch)
            if self.event.has_timing and q:
                dets = self.event.timing(h).get(ch, [])
                g1[h.stem(ch)] = [float(self.event.to_video(q.video, d["t"] - offset)) for d in dets]
        if not stems:
            self.log("Nothing analysed to learn from")
            return
        self._learn_stems = stems
        self.learning = self.pool.submit_learn(self.tp.dir, stems, n_gates, g1_times=g1 or None)

    def _learned(self):
        fut, self.learning = self.learning, None
        try:
            res = fut.result()
        except Exception as e:
            self.error = f"learning failed: {e}"
            self.log(self.error)
            return
        with self.lock:
            self.state.update(learned_from=self._learn_stems, learned_at=_now())
        self.log(f"Track learned from {res['runs']} runs ({res['laps']} laps) in {res.get('seconds')}s")
        for hid, h in list(self.event.heats.items()):        # label every analysed heat (again)
            if any(self.tp.race_data(h.stem(c)).exists() for c in h.pilots) \
                    and not any(j[1] == hid and j[0] == "analyse" for j in self.jobs.values()):
                with self.lock:
                    hs = self._heat(hid)
                    if hs.get("status") in (None, "new"):
                        hs["status"] = "analysed"
                self._submit_label(hid)

    # lap check & writing --------------------------------------------

    def _lap_ref(self) -> Optional[float]:
        try:
            tim = json.loads(self.tp.learned_timing.read_text(encoding="utf-8"))
            return float(sum(tim.get("leg_seconds", {}).values())) or None
        except (OSError, ValueError):
            return None

    def _check(self, hid: str, external: bool = False):
        """The lap check of a heat. external: FPVTrackside saved the heat (a
        marshal's edit there) — the changes it now has are counted as done
        in FPVTrackside, and laps the app wrote earlier are checked to still
        be there (FPVTrackside saves a race from memory, over our write, when
        the event wasn't reopened in between)."""
        h = self.event.heats.get(hid)
        if not h or (self.event.running and self.event.has_timing):   # don't open Race.json during a race
            return
        timing = self.event.timing(h)
        before = {_change_id(c) for c in self.changes(hid)} if external else set()
        if external:
            self._marks_done(hid, timing)
            self._overwritten(hid, timing)
        races = {c: self._race(h.stem(c)) for c in h.pilots}
        q_of = {q.channel: q for q in h.quads}
        if self.state["offset"] is None:                      # the timing antenna vs the visual gate
            diffs = []
            for c, race in races.items():
                if race and c in q_of:
                    vc = [float(self.event.to_clock(q_of[c].video, p["t"])) for p in lapcheck.g1_passes(race)]
                    diffs += lapcheck.pairs_offset([d["t"] for d in timing.get(c, [])], vc)
            off = lapcheck.heat_offset(diffs)
            if off is not None:
                with self.lock:
                    self.state["offset"] = off
        offset = self.state["offset"] or 0.0
        out = {}
        for c, race in races.items():
            if not race or c not in q_of:
                continue
            q = q_of[c]
            if not self.event.has_timing:                     # no timing system: the video's own laps
                out[c] = {"timing_laps": None, "video_laps": len(race.get("laps", [])), "findings": []}
                continue
            pic = self.tp.runs_dir / f"{h.stem(c)}.picture.json"
            gray = json.loads(pic.read_text(encoding="utf-8")).get("gray", []) if pic.exists() else []
            res = lapcheck.check_pilot(race, timing.get(c, []), lambda t, v=q.video: self.event.to_clock(v, t),
                                       self._n_gates(), offset, h.start, h.end, self._lap_ref(),
                                       low_quality=q.kind == "dji", gray=gray,
                                       race_video=(float(self.event.to_video(q.video, h.start)),
                                                   float(self.event.to_video(q.video, h.end))))
            for f in res["findings"]:
                f["key"] = lapcheck_key(f)
            out[c] = res
        for c, race in races.items():                         # stats for the dashboard
            if race and c in q_of:
                self._stats(h, q_of[c], race)
        with self.lock:
            hs = self._heat(hid)
            hs["check"] = out
            if hs.get("status") != "updated":
                hs["status"] = "checked"
        if external:
            done = before - {_change_id(c) for c in self.changes(hid)}
            if done:
                with self.lock:
                    hs["done_in_fpvtrackside"] = hs.get("done_in_fpvtrackside", 0) + len(done)
                self.log(f"heat {h.number}: {len(done)} change(s) done in FPVTrackside")

    def _marks_done(self, hid: str, timing: Dict[str, List[dict]]):
        """The marshal's own marks that FPVTrackside now has are dropped: a lap
        added there near the mark, a removed lap no longer valid there."""
        with self.lock:
            hs = self._heat(hid)
            keep = []
            for m in hs.get("marks", []):
                dets = timing.get(m["channel"], [])
                if m["op"] == "add" and any(abs(d["t"] - m["t"]) <= lapcheck.TOL_S for d in dets):
                    continue
                if m["op"] == "remove" and all(d["id"] != m.get("detection_id") for d in dets):
                    continue
                keep.append(m)
            hs["marks"] = keep
            if hs.get("overwritten") and not any(m.get("by") == "lost" for m in keep):
                hs.pop("overwritten")                 # the lost changes are all back in FPVTrackside

    def _overwritten(self, hid: str, timing: Dict[str, List[dict]]):
        """Laps the app wrote that FPVTrackside no longer has (it saved the race
        from memory over our write): flagged on the heat, until the next write."""
        with self.lock:
            hs = self._heat(hid)
            written = [w for w in hs.get("written", []) if w.get("by") != "restore"]
            if not written or hs.get("status") != "updated":
                return
            lost = []
            for c in written[-1]["changes"]:
                dets = timing.get(c["channel"], [])
                if c["op"] == "remove" and any(d["id"] == c.get("detection_id") for d in dets):
                    lost.append(c)
                elif c["op"] == "add" and not any(abs(d["t"] - c["t"]) <= 0.05 for d in dets):
                    lost.append(c)
            if lost and not hs.get("overwritten"):
                hs["overwritten"] = {"at": _now(), "changes": len(lost)}
                # back on the list: the steps for the marshal, and the next Update laps
                marks = hs.setdefault("marks", [])
                have = {_change_id(m) for m in marks}
                h = self.event.heats.get(hid)
                off = self.state.get("offset") or 0.0
                for c in lost:
                    if _change_id(c) in have:
                        continue
                    m = {k: c[k] for k in ("op", "channel", "t", "detection_id") if k in c}
                    q = self._quad(h, c["channel"])
                    if c["op"] == "add" and q:
                        m["video_t"] = round(float(self.event.to_video(q.video, c["t"] - off)), 3)
                    marks.append(dict(m, by="lost"))
                self.log(f"heat {h.number if h else hid}: FPVTrackside saved this heat over the app's lap changes "
                         f"({len(lost)} lost) — the event wasn't reopened after the update")

    def _stats(self, h, q, race: dict):
        stem = h.stem(q.channel)
        live0 = float(self.event.to_video(q.video, h.start)) if self.event.has_timing else 0.0
        meta = {"live_start_s": live0}
        official = None
        if self.event.has_timing:                     # the official laps: FPVTrackside's now (with any edits)
            off = self.state.get("offset") or 0.0
            official = [float(self.event.to_video(q.video, d["t"] - off))
                        for d in self.event.timing(h).get(q.channel, [])]
        st = rstats.run_stats(race, meta, self._n_gates(), official=official)
        st.update(stem=stem, pilot=h.pilots.get(q.channel, "?"), heat=h.number, channel=q.channel,
                  stage=h.stage, round=h.round, race=h.race, reviewed=(race.get("review") or {}).get("status"))
        labeling._write_json(self.tp.runs_dir / f"{stem}.stats.json", st)
        with self.lock:
            self.state["runs"][stem] = {
                "stem": stem, "status": "labelled", "heat": h.number, "channel": q.channel,
                "quad": [x.channel for x in h.quads].index(q.channel), "pilot": h.pilots.get(q.channel, "?"),
                "recorded_at": h.start, "live_start_s": live0, "window": None, "void": False,
                "stage": h.stage, "round": h.round, "race": h.race, "video_path": str(q.video),
                "crop": list(q.rect), "reviewed": (race.get("review") or {}).get("status"),
                "summary": {"laps": st["laps"], "best_lap": st["best_lap"], "passes": st["passes"]}}

    def _auto_write(self, hid: str):
        if not self.state["auto"]:
            return
        with self.lock:
            hs = self._heat(hid)
            if hs.get("status") != "checked":
                return
            dec = hs.setdefault("decisions", {})
            n = 0
            for ch, c in hs.get("check", {}).items():
                for f in c.get("findings", []):
                    k = f"{ch}|{lapcheck_key(f)}"
                    if f.get("confidence") == "high" and k not in dec:
                        dec[k] = "accepted"
                        n += 1
        if n:
            self._write(hid, "auto")

    def _write(self, hid: str, by: str):
        h = self.event.heats.get(hid)
        if not h or self.event.running or self.ft_state["running"]:
            return
        changes = self.changes(hid)
        if not changes:
            return
        if self.ft_state["serves"]:
            return self._write_api(h, changes, by)
        race = self.event.race_json(h)
        if race is None:
            self.log(f"heat {h.number}: Race.json busy — will retry")
            return
        cids = {v: k for k, v in self.event.channels.items()}
        try:
            new = trackside.apply_changes(race, changes, cids)
            trackside.write_race(h.dir, new, self.tp.dir / "backups")
            self.event.timing(h, new)                 # the cached laps are the new ones right away
        except Exception as e:
            self.log(f"heat {h.number}: writing laps failed — {e}")
            return
        with self.lock:
            hs = self._heat(hid)
            hs.setdefault("written", []).append({"at": _now(), "by": by, "changes": changes})
            hs["status"] = "updated"
            hs.pop("overwritten", None)
            hs["marks"], hs["decisions"] = [], {}
        self.log(f"heat {h.number}: {len(changes)} lap change(s) written to FPVTrackside ({by})")
        self._check(hid)

    def _write_api(self, h, changes: List[dict], by: str):
        """The changes through FPVTrackside's REST API, one lap correction per
        pilot. An added crossing is placed after the pilot's latest earlier
        crossing (both on FPVTrackside's clock); with none, at its time from
        the race start. The crossings it creates and the ones it discounts are
        kept, so Restore can undo exactly those."""
        from race_day import ftapi
        race = self.event.race_json(h) or {}
        pid = {self.event.channel_name(pc["Channel"]): pc["Pilot"] for pc in race.get("PilotChannels") or []
               if pc.get("Channel") and pc.get("Pilot")}
        timing = self.event.timing(h)
        created, invalidated, done, errors = [], [], [], []
        for ch in sorted({c["channel"] for c in changes}):
            mine = [c for c in changes if c["channel"] == ch]
            if ch not in pid:
                errors.append(f"{ch}: no pilot on this channel in FPVTrackside")
                continue
            dets = timing.get(ch, [])
            inval = [c["detection_id"] for c in mine if c["op"] == "remove"]
            anchored, by_time = [], []
            for c in (c for c in mine if c["op"] == "add"):
                before = [d for d in dets if d["t"] < c["t"] - 0.001]
                if before and (c["t"] - before[-1]["t"]) * 1000 <= 600000:
                    anchored.append((c, {"anchor": before[-1]["id"],
                                         "offset_ms": max(1, round((c["t"] - before[-1]["t"]) * 1000))}))
                else:
                    by_time.append(c)
            try:
                known = {d["id"].lower() for d in dets}
                if inval or anchored:
                    laps = self.ft.lap_corrections(h.id, pid[ch], inval, [a for _, a in anchored])
                    for c, _ in anchored:              # the crossing it made for each added lap
                        hit = min((l for l in laps if l["detection"] not in known and l["pilot"] == pid[ch].lower()),
                                  key=lambda l: abs(l["race_time"] - (c["t"] - h.start)), default=None)
                        if hit and abs(hit["race_time"] - (c["t"] - h.start)) < 0.5:
                            created.append(hit["detection"])
                            known.add(hit["detection"])
                    invalidated += inval
                    done += [c for c in mine if c["op"] == "remove"] + [c for c, _ in anchored]
                for c in by_time:
                    created += self.ft.add_lap(h.id, pid[ch], round((c["t"] - h.start) * 1000))
                    done.append(c)
            except ftapi.ApiError as e:
                errors.append(f"{ch}: {e}")
        with self.lock:
            hs = self._heat(h.id)
            if done:
                hs.setdefault("written", []).append({"at": _now(), "by": by, "via": "api", "changes": done,
                                                     "created": created, "invalidated": invalidated})
                hs["status"] = "updated"
                hs.pop("overwritten", None)
                ids = {_change_id(c) for c in done}        # what went through stops being a change to make
                hs["marks"] = [m for m in hs.get("marks", []) if _change_id(m) not in ids]
                for ch, c in hs.get("check", {}).items():
                    for f in c.get("findings", []):
                        if hs.get("decisions", {}).get(f"{ch}|{lapcheck_key(f)}") == "accepted":
                            fc = ({"op": "remove", "detection_id": f["detection_id"]} if f.get("detection_id")
                                  else {"op": "add", "channel": ch, "t": f["t"]})
                            if _change_id(fc) in ids:
                                hs["decisions"].pop(f"{ch}|{lapcheck_key(f)}")
            hs["api_error"] = "; ".join(errors) or None
        if done:
            self.log(f"heat {h.number}: {len(done)} lap change(s) made in FPVTrackside through its API ({by})")
        for e in errors:
            self.log(f"heat {h.number}: FPVTrackside refused a change — {e}")
        self._save()

    def _restore(self, hid: str):
        h = self.event.heats.get(hid)
        if not h or self.event.running:
            return
        with self.lock:
            api_writes = [w for w in self._heat(hid).get("written", []) if w.get("via") == "api" and not w.get("undone")]
        if api_writes:
            if not self.ft_state["serves"]:
                self.log(f"heat {h.number}: changes made through FPVTrackside's API can only be undone through it "
                         "— it isn't reachable (or has another event open)")
                return
            return self._restore_api(h, api_writes)
        if trackside.restore_race(h.dir, self.tp.dir / "backups"):
            restored = self.event.race_json(h)
            if restored is not None:
                self.event.timing(h, restored)            # the cached laps are the restored ones right away
            with self.lock:
                hs = self._heat(hid)
                hs.setdefault("written", []).append({"at": _now(), "by": "restore", "changes": []})
                hs["status"] = "checked"
                hs.pop("overwritten", None)
            self.log(f"heat {h.number}: original laps restored")
            self._check(hid)


    def _restore_api(self, h, writes: List[dict]):
        """Undo the API's changes, newest first: the crossings it added are
        discounted, the ones it discounted count again."""
        from race_day import ftapi
        failed = []
        for w in reversed(writes):
            try:
                for d in w.get("created", []):
                    self.ft.set_valid(h.id, d, False)
                for d in w.get("invalidated", []):
                    self.ft.set_valid(h.id, d, True)
                w["undone"] = _now()
            except ftapi.ApiError as e:
                failed.append(str(e))
        with self.lock:
            hs = self._heat(h.id)
            hs.setdefault("written", []).append({"at": _now(), "by": "restore", "via": "api", "changes": []})
            if not failed:
                hs["status"] = "checked"
            hs["api_error"] = "; ".join(failed) or None
        self.log(f"heat {h.number}: " + (f"undoing in FPVTrackside failed — {failed[0]}" if failed
                                         else "original laps restored in FPVTrackside"))
        self._save()


def lapcheck_key(f: dict) -> str:
    """A finding's identity across re-checks: kind + time (0.1 s)."""
    return f"{'missed' if f['kind'] == 'missed' else 'det'}@{f['t']:.1f}" if not f.get("detection_id") \
        else f"det:{f['detection_id']}"


def _change_id(c: dict) -> tuple:
    """A lap change's identity: the detection it removes, or where it adds a lap (0.1 s)."""
    return ("remove", c["detection_id"]) if c["op"] == "remove" else ("add", c["channel"], round(c["t"], 1))
