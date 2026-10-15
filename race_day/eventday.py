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

State: <event>/race_day/race_day.json — a restart loses nothing.
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
    def __init__(self, event_dir, mode: str = "live", workers: int = 4, det_model: str = str(DEFAULT_DET_MODEL),
                 clip_device: str = "mps", log: Callable[[str], None] = print):
        if (Path(event_dir) / "Event.json").exists():
            self.event = trackside.Event(event_dir)
        else:                                             # Replays: a folder of loose videos
            self.event = trackside.VideoFolder(event_dir)
            mode = "replays"
        self.mode = mode                                  # live | replays
        self.tp = TrackPaths(self.event.data_dir)
        self.tp.dir.mkdir(exist_ok=True)
        self.tp.runs_dir.mkdir(exist_ok=True)
        self.workers, self.det_model, self.clip_device, self.log = workers, det_model, clip_device, log
        self.lock = threading.RLock()
        self.cmds: "queue.Queue" = queue.Queue()
        self.state = {"event": self.event.name, "n_gates": None, "learned_from": [], "learned_at": None,
                      "auto": False, "offset": None, "heats": {}, "runs": {}}
        if self.tp.race_day_state.exists():
            try:
                saved = json.loads(self.tp.race_day_state.read_text(encoding="utf-8"))
                self.state.update({k: v for k, v in saved.items() if k in self.state})
            except ValueError:
                pass
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

    # ── lifecycle ──────────────────────────────────────────────

    def start(self):
        from race_day.analysis import AnalysisPool
        from race_day.models import worker_devices
        self.pool = AnalysisPool(worker_devices(Path(self.det_model), self.workers, False), self.clip_device)
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

    def configure_video(self, name: str, info: dict):
        """Replays: what a loose video is (single / 2×2, heat, pilots)."""
        self.event.set_info(name, info)

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
                            "id": d["id"], "holeshot": d["holeshot"], "type": d["type"]} for d in dets],
                "video_g1": [p["t"] for p in lapcheck.g1_passes(race)] if race else [],
                "passes": [{"t": p["t"], "gate": p.get("gate_id", -1)} for p in (race or {}).get("passes", [])],
                "laps": [{"lap": l.get("lap"), "t0": l["t0"], "t1": l["t1"]} for l in (race or {}).get("laps", [])],
                "stem": h.stem(ch), "race_data": str(self.tp.race_data(h.stem(ch))),
                "gt": str(self.tp.gt(h.stem(ch))), "reviewed": self.tp.gt(h.stem(ch)).exists(),
                "findings": check.get("findings", []), "run": hs.get("runs", {}).get(ch, {}),
                "checked": bool(check), "no_video": bool(check.get("no_video")),
                "no_video_findings": check.get("no_video_findings", 0),
            }
        return {"id": h.id, "number": h.number, "label": h.label, "start": h.start, "end": h.end,
                "status": hs.get("status", "new"), "pilots": pilots, "decisions": hs.get("decisions", {}),
                "marks": hs.get("marks", []), "written": hs.get("written", []),
                "learn_pilots": hs.get("learn_pilots", []),
                "changes": self.changes(heat_id), "videos": sorted({str(q.video) for q in h.quads})}

    def changes(self, heat_id: str) -> List[dict]:
        """The lap changes that 'Update laps in FPVTrackside' would write:
        accepted findings + the marshal's own marks."""
        with self.lock:
            hs = self.state["heats"].get(heat_id, {})
            out = []
            for ch, c in hs.get("check", {}).items():
                for f in c.get("findings", []):
                    if hs.get("decisions", {}).get(f"{ch}|{lapcheck_key(f)}") != "accepted":
                        continue
                    if f["kind"] == "missed":
                        out.append({"op": "add", "channel": ch, "t": f["t"], "by": "check"})
                    elif f.get("detection_id"):
                        out.append({"op": "remove", "channel": ch, "detection_id": f["detection_id"], "by": "check"})
            out += [dict(m) for m in hs.get("marks", [])]
            return out

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
        return int(self.state["n_gates"] or DEFAULT_GATES)

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

    def _refresh(self):
        r = self.event.refresh()
        if r["running"] and not self.paused:
            self.paused = True
            if self.pool:
                self.pool.pause()
            self.log("Race running in FPVTrackside — paused")
        elif not r["running"] and self.paused:
            self.paused = False
            if self.pool:
                self.pool.resume()
            self.log("Race over — continuing")
        if self.paused:
            return
        for hid in r["new"] + r["changed"]:
            with self.lock:
                hs = self._heat(hid)
            if hid in r["changed"] and hs.get("status") in ("checked", "updated"):
                self._check(hid)                      # Race.json changed (a marshal edit): check again
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
            elif kind == "write":
                self._write(cmd[1], cmd[2])
            elif kind == "restore":
                self._restore(cmd[1])
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
            for q in h.quads:
                if q.channel not in h.pilots:
                    continue
                stem = h.stem(q.channel)
                if not force and self.tp.race_data(stem).exists():
                    continue
                fut = self.pool.submit_quad(q.video, q.rect, self.tp.race_data(stem), self._n_gates(), stem)
                self.jobs[f"a:{stem}"] = ("analyse", hid, fut, {"stem": stem, "channel": q.channel})
                hs["runs"][q.channel] = {"stem": stem, "status": "analysing", "kind": q.kind}
        self._after_analysis(hid)

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
        for hid, ch in runs:
            self._submit_quad(hid, ch)
        self._learn_wait = {"runs": runs, "n_gates": n_gates}
        self.log(f"Learning the track ({n_gates} gates) from {len(runs)} pilot runs…")

    def _submit_quad(self, hid: str, channel: str):
        h = self.event.heats.get(hid)
        q = self._quad(h, channel)
        if not q:
            return
        stem = h.stem(channel)
        if self.tp.race_data(stem).exists() or f"a:{stem}" in self.jobs:
            return
        with self.lock:
            hs = self._heat(hid)
            fut = self.pool.submit_quad(q.video, q.rect, self.tp.race_data(stem), self._n_gates(), stem)
            self.jobs[f"a:{stem}"] = ("analyse", hid, fut, {"stem": stem, "channel": channel})
            hs.setdefault("runs", {})[channel] = {"stem": stem, "status": "analysing", "kind": q.kind}
            if hs.get("status") in (None, "new"):
                hs["status"] = "analysing"

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

    def _check(self, hid: str):
        h = self.event.heats.get(hid)
        if not h or self.event.running:
            return
        timing = self.event.timing(h)
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

    def _stats(self, h, q, race: dict):
        stem = h.stem(q.channel)
        live0 = float(self.event.to_video(q.video, h.start)) if self.event.has_timing else 0.0
        meta = {"live_start_s": live0}
        st = rstats.run_stats(race, meta, self._n_gates())
        st.update(stem=stem, pilot=h.pilots.get(q.channel, "?"), heat=h.number, channel=q.channel,
                  stage=h.stage, round=h.round, race=h.race)
        labeling._write_json(self.tp.runs_dir / f"{stem}.stats.json", st)
        with self.lock:
            self.state["runs"][stem] = {
                "stem": stem, "status": "labelled", "heat": h.number, "channel": q.channel,
                "quad": [x.channel for x in h.quads].index(q.channel), "pilot": h.pilots.get(q.channel, "?"),
                "recorded_at": h.start, "live_start_s": live0, "window": None, "void": False,
                "stage": h.stage, "round": h.round, "race": h.race, "video_path": str(q.video),
                "crop": list(q.rect), "summary": {"laps": st["laps"], "best_lap": st["best_lap"],
                                                   "passes": st["passes"]}}

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
        if not h or self.event.running:
            return
        changes = self.changes(hid)
        if not changes:
            return
        race = self.event.race_json(h)
        if race is None:
            self.log(f"heat {h.number}: Race.json busy — will retry")
            return
        cids = {v: k for k, v in self.event.channels.items()}
        try:
            new = trackside.apply_changes(race, changes, cids)
            trackside.write_race(h.dir, new, self.tp.dir / "backups")
        except Exception as e:
            self.log(f"heat {h.number}: writing laps failed — {e}")
            return
        with self.lock:
            hs = self._heat(hid)
            hs.setdefault("written", []).append({"at": _now(), "by": by, "changes": changes})
            hs["status"] = "updated"
            hs["marks"], hs["decisions"] = [], {}
        self.log(f"heat {h.number}: {len(changes)} lap change(s) written to FPVTrackside ({by})")
        self._check(hid)

    def _restore(self, hid: str):
        h = self.event.heats.get(hid)
        if not h or self.event.running:
            return
        if trackside.restore_race(h.dir, self.tp.dir / "backups"):
            with self.lock:
                hs = self._heat(hid)
                hs.setdefault("written", []).append({"at": _now(), "by": "restore", "changes": []})
                hs["status"] = "checked"
            self.log(f"heat {h.number}: original laps restored")
            self._check(hid)


def lapcheck_key(f: dict) -> str:
    """A finding's identity across re-checks: kind + time (0.1 s)."""
    return f"{'missed' if f['kind'] == 'missed' else 'det'}@{f['t']:.1f}" if not f.get("detection_id") \
        else f"det:{f['detection_id']}"
