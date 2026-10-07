"""
The race-day controller: ties capture, analysis workers, learning, labelling
and stats together, and keeps the state in <track>/race_day.json so a
restart loses nothing.

Live:     capture → runs ≥ min_run_s saved → analysed in parallel → after
          learn_runs runs the track is learned (gate count from setup) →
          every run is labelled and gets stats; re-learn every relearn_every.
Replays:  2×2 files → cut into runs → analysed → labelled with the track's
          existing learned data (no learning, no recording).
"""

import json
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dataset_paths import TrackPaths
from race_day import labeling, stats as rstats
from race_day.capture import CaptureConfig, CaptureSession, FileSource, RunInfo
from race_day.models import DEFAULT_DET_MODEL
from race_day.pilots import AttemptTracker, HeatInfoProvider, norm_channel, pilot_name


@dataclass
class DayConfig:
    mode: str = "live"                     # live | replays
    n_gates: int = 0
    layout: List[str] = field(default_factory=lambda: ["R1", "R3", "R6", "R8"])
    min_run_s: float = 40.0
    end_gray_s: float = 10.0
    learn_runs: int = 24
    relearn_every: int = 12
    det_model: str = str(DEFAULT_DET_MODEL)
    workers: int = 4
    use_coreml: bool = False              # Core ML was less accurate on reviewed videos
    fril_live: bool = True                # Live: pilot names, round and race from fril.co.il/api/live/state
    clip_device: str = "mps"


class RaceDay:
    def __init__(self, tp: TrackPaths, cfg: DayConfig, provider: Optional[HeatInfoProvider] = None,
                 log: Callable[[str], None] = print):
        self.tp, self.cfg, self.provider, self.log = tp, cfg, provider or HeatInfoProvider(), log
        tp.dir.mkdir(parents=True, exist_ok=True)
        tp.videos.mkdir(exist_ok=True)
        tp.runs_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.events: "queue.Queue" = queue.Queue()
        self.state = {"track": tp.dir.name, "mode": cfg.mode, "config": asdict(cfg), "phase": "learn",
                      "learned_from": [], "learned_at": None, "runs": {}, "heats": {}, "discarded": 0,
                      "replay_files": {}, "attempts": []}
        if tp.race_day_state.exists():
            saved = json.loads(tp.race_day_state.read_text(encoding="utf-8"))
            for k in ("runs", "heats", "learned_from", "learned_at", "discarded", "replay_files", "phase", "attempts"):
                if k in saved:
                    self.state[k] = saved[k]
        if cfg.mode == "replays" or labeling.is_learned(tp):
            if labeling.is_learned(tp):
                self.state["phase"] = "result"
        if hasattr(self.provider, "tracker") and self.state["attempts"]:
            self.provider.tracker = AttemptTracker(self.state["attempts"])   # resume: keep attempt numbering
        self.live: dict = {}                       # capture status (channels, heat)
        self.pool = None
        self.capture: Optional[CaptureSession] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.running = False
        self.pending: Dict[str, object] = {}      # stem → Future
        self._file_queue: List[str] = []
        self.error: Optional[str] = None

    # ── persistence ────────────────────────────────────────────

    def _save(self):
        with self.lock:
            data = json.dumps(self.state, indent=2)
        tmp = str(self.tp.race_day_state) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, self.tp.race_day_state)

    # ── lifecycle ──────────────────────────────────────────────

    def start(self):
        if self.cfg.mode == "replays" and not labeling.is_learned(self.tp):
            raise RuntimeError(f"Track '{self.tp.name}' isn't learned yet — Replays needs a learned track.")
        from race_day.analysis import AnalysisPool
        from race_day.models import worker_devices
        devices = worker_devices(Path(self.cfg.det_model), self.cfg.workers, self.cfg.use_coreml)
        self.log(f"Analysis workers: {', '.join('coreml' if d[0].endswith('.mlpackage') else d[1] for d in devices)}")
        self.pool = AnalysisPool(devices, self.cfg.clip_device)
        self.running = True
        self.worker.start()
        self._save()
        # resume: runs recorded but not analysed, analysed but not labelled
        for stem, r in list(self.state["runs"].items()):
            if r["status"] in ("recorded", "analysing"):
                self._submit(stem)
            elif r["status"] == "analysed" and self.state["phase"] == "result":
                self.events.put(("label", stem))

    def stop(self):
        if self.capture:
            self.capture.stop()
        if self.capture_thread:
            self.capture_thread.join(timeout=30)
        self.running = False
        self.events.put(("stop",))
        if self.pool:
            self.pool.shutdown(wait=False)
        self.provider.stop()
        self._save()

    def _next_heat(self) -> int:
        return max([int(h) for h in self.state["heats"]] + [0]) + 1

    def _capture_cfg(self) -> CaptureConfig:
        return CaptureConfig(layout=list(self.cfg.layout), min_run_s=self.cfg.min_run_s,
                             end_gray_s=self.cfg.end_gray_s)

    def start_live(self, source):
        self.capture = CaptureSession(self.tp.videos, self._capture_cfg(), self._next_heat(),
                                      on_run=self._on_run, on_status=self._on_status, on_heat=self._on_heat,
                                      on_preview=getattr(self, "on_preview", None))
        self.capture_thread = threading.Thread(target=self._run_capture, args=(source, "live"), daemon=True)
        self.capture_thread.start()

    def add_replay_files(self, paths: List[str]):
        """Queue 2×2 files; one thread cuts them into runs, one file at a time."""
        with self.lock:
            for p in paths:
                self.state["replay_files"].setdefault(str(p), {"status": "queued", "runs": 0})
                self._file_queue.append(str(p))
        self._save()
        if not (self.capture_thread and self.capture_thread.is_alive()):
            self.capture_thread = threading.Thread(target=self._run_files, daemon=True)
            self.capture_thread.start()

    def _run_files(self):
        while True:
            with self.lock:
                if not self._file_queue or not self.running:
                    return
                p = self._file_queue.pop(0)
                self.state["replay_files"][p]["status"] = "splitting"
            self._save()
            self.capture = CaptureSession(self.tp.videos, self._capture_cfg(), self._next_heat(),
                                          on_run=self._on_run, on_status=self._on_status,
                                          on_heat=self._on_heat, on_preview=getattr(self, "on_preview", None))
            self._current_file = p
            try:
                self.capture.run(FileSource(p), Path(p).name)
                status = "split"
            except Exception as e:
                status = f"error: {e}"
            with self.lock:
                self.state["replay_files"][p]["status"] = status
            self._save()

    def _run_capture(self, source, name):
        try:
            self.capture.run(source, name)
        except Exception as e:
            self.error = f"capture stopped: {e}"
            self.log(self.error)

    # ── capture callbacks (capture thread) ─────────────────────

    def _on_status(self, st: dict):
        with self.lock:
            self.live = st

    def _on_heat(self, heat: int, wall: float):
        try:
            info = self.provider.heat_info(wall, list(self.cfg.layout))
        except Exception:
            info = None
        with self.lock:
            self.state["heats"][str(heat)] = {"heat": heat, "wall_start": wall, "info": info,
                                              "round": (info or {}).get("round"), "race": (info or {}).get("race"),
                                              "source": getattr(self, "_current_file", "live")}
        self._save()

    def _on_run(self, info: RunInfo, kept: bool):
        if not kept:
            with self.lock:
                self.state["discarded"] += 1
            self._save()
            self.log(f"{info.stem}: {info.live_s:.0f}s — too short, discarded")
            return
        with self.lock:
            heat = self.state["heats"].setdefault(str(info.heat), {"heat": info.heat, "info": None})
        if not (heat.get("info") or {}).get("pilots"):
            # the timing PC may load the heat only after the drones are live — ask again
            try:
                late = self.provider.heat_info(time.time(), list(self.cfg.layout))
            except Exception:
                late = None
            if late:
                with self.lock:
                    heat.update(info=late, round=late.get("round"), race=late.get("race"))
        with self.lock:
            self.state["runs"][info.stem] = {
                "stem": info.stem, "heat": info.heat, "channel": info.channel, "quad": info.quad,
                "pilot": pilot_name(heat.get("info"), info.heat, info.channel),
                "round": heat.get("round"), "race": heat.get("race"),
                "wall_start": info.wall_start, "live_start_s": info.live_start_s, "live_s": info.live_s,
                "video": Path(info.video).name, "source": info.source, "status": "recorded",
                "recorded_at": time.time()}
            if info.source != "live" and info.source in [Path(p).name for p in self.state["replay_files"]]:
                for p, f in self.state["replay_files"].items():
                    if Path(p).name == info.source:
                        f["runs"] += 1
        self._sync_attempts(force=True)
        self._save()
        r = self.state["runs"][info.stem]
        self.log(f"{info.stem}: {info.live_s:.0f}s saved" + (f" — {r['pilot']}, {r['heat_label']}" if r.get("heat_label") else ""))
        self._submit(info.stem)

    # ── analysis ───────────────────────────────────────────────

    def _submit(self, stem: str):
        with self.lock:
            self.state["runs"][stem]["status"] = "analysing"
        fut = self.pool.submit(self.tp.videos / f"{stem}.mp4", self.tp.race_data(stem), self.cfg.n_gates)
        self.pending[stem] = fut
        fut.add_done_callback(lambda f, s=stem: self.events.put(("analysed", s, f)))

    def _loop(self):
        while True:
            try:
                ev = self.events.get(timeout=2.0)
            except queue.Empty:
                if self.provider.changed:          # a heat started/finished/was restarted
                    try:
                        self._sync_attempts()
                    except Exception:
                        self.log(traceback.format_exc())
                    self._save()
                continue
            if ev[0] == "stop":
                break
            try:
                if ev[0] == "analysed":
                    self._analysed(ev[1], ev[2])
                elif ev[0] == "label":
                    self._label(ev[1])
            except Exception:
                self.log(traceback.format_exc())
            self._save()

    def _analysed(self, stem: str, fut):
        self.pending.pop(stem, None)
        try:
            res = fut.result()
        except Exception as e:
            with self.lock:
                self.state["runs"][stem].update(status="failed", error=str(e))
            self.log(f"{stem}: analysis failed — {e}")
            return
        with self.lock:
            self.state["runs"][stem].update(status="analysed", analysis=res)
        self.log(f"{stem}: analysed in {res['seconds']}s on {res['device']}")
        if self.state["phase"] == "result":
            self._label(stem)
            self._maybe_relearn()
        elif self.cfg.mode == "live":
            self._maybe_learn()

    def _analysed_stems(self) -> List[str]:
        return [s for s, r in sorted(self.state["runs"].items(), key=lambda kv: kv[1]["recorded_at"])
                if r["status"] in ("analysed", "labelled")]

    def _maybe_learn(self):
        stems = self._analysed_stems()
        if len(stems) < self.cfg.learn_runs:
            return
        self.log(f"Learning the track from {len(stems)} runs ({self.cfg.n_gates} gates)…")
        res = labeling.learn(self.tp, stems, self.cfg.n_gates)
        with self.lock:
            self.state.update(phase="result", learned_from=stems, learned_at=time.time(), learn=res)
        self.log(f"Track learned: {res['laps']} laps in {res['runs']} runs")
        for s in stems:
            self._label(s)

    def _maybe_relearn(self):
        if self.cfg.mode != "live":
            return
        stems = self._analysed_stems()
        if len(stems) - len(self.state["learned_from"]) < self.cfg.relearn_every:
            return
        self.log(f"Re-learning the track from {len(stems)} runs…")
        res = labeling.learn(self.tp, stems, self.cfg.n_gates)
        with self.lock:
            self.state.update(learned_from=stems, learned_at=time.time(), learn=res)
        for s in stems:
            self._label(s)

    def _label(self, stem: str):
        labeling.label_run(self.tp, stem)
        self._stats(stem)

    def _stats(self, stem: str):
        """Stats of a labelled run — only its official part when the video
        covers a restarted heat too (window)."""
        r = self.state["runs"][stem]
        race = json.loads(self.tp.race_data(stem).read_text(encoding="utf-8"))
        st = rstats.run_stats(race, r, self._n_gates(), window=r.get("window"))
        st.update(stem=stem, pilot=r["pilot"], heat=r["heat"], channel=r["channel"],
                  stage=r.get("stage"), round=r.get("round"), race=r.get("race"))
        labeling._write_json(self.tp.runs_dir / f"{stem}.stats.json", st)
        with self.lock:
            r.update(status="labelled", summary={"laps": st["laps"], "best_lap": st["best_lap"],
                                                 "passes": st["passes"]})

    # ── heats from fril: attempts, official / void ─────────────

    def _sync_attempts(self, force: bool = False):
        """Copy the provider's attempts into the state and re-match every run."""
        if not (force or self.provider.changed):
            return
        atts = self.provider.attempts()
        with self.lock:
            self.state["attempts"] = atts
            runs = list(self.state["runs"].values())
        restat = []
        for r in runs:
            if self._match(r, atts) and r["status"] == "labelled":
                restat.append(r["stem"])
        for stem in restat:
            self._stats(stem)

    def _match(self, r: dict, atts: List[dict]) -> bool:
        """Tie a run to the fril attempt(s) that were running while its feed
        was live (our own clock). Returns True if its stats window changed."""
        t0 = r["wall_start"] + r.get("live_start_s", 0.0)
        t1 = t0 + r.get("live_s", 0.0)
        now = time.time()
        hits = [a for a in atts if min(t1, a["end"] or now) - max(t0, a["start"]) > 3.0]
        old = r.get("window")
        with self.lock:
            if not hits:
                r.update(fril=None, void=False, window=None)
                return old is not None
            official = [a for a in hits if a.get("official")]
            show = official[-1] if official else max(hits, key=lambda a: a["start"])
            void = not official and all(a["status"] != "running" for a in hits)
            window = None
            if official and len(hits) > 1:      # one video covering a restart: only the official part counts
                window = [round(show["start"] - r["wall_start"] - 2.0, 2),
                          round((show["end"] or now) - r["wall_start"] + 5.0, 2)]
            name = (show.get("pilots") or {}).get(norm_channel(r["channel"]))
            if name:
                r["pilot"] = name
            r.update(fril={"attempt_id": show["id"], "attempt": show["attempt"], "status": show["status"],
                           "official": bool(show.get("official")), "attempts": [a["id"] for a in hits]},
                     stage=show["stage"] or None, round=show["round"], race=show["race"], void=void,
                     window=window,
                     heat_label=" · ".join(x for x in (show["stage"] or None, f"Round {show['round']}",
                                                     f"Race {show['race']}") if x))
        return window != old

    def _n_gates(self) -> int:
        if self.cfg.n_gates:
            return self.cfg.n_gates
        mem = json.loads(self.tp.gate_memory.read_text(encoding="utf-8"))
        return len(mem.get("memory", []))

    # ── status for the UI and dashboard ────────────────────────

    def status(self) -> dict:
        with self.lock:
            runs = list(self.state["runs"].values())
            counts = {}
            for r in runs:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            return {
                "track": self.state["track"], "mode": self.cfg.mode, "phase": self.state["phase"],
                "n_gates": self._n_gates() if self.state["phase"] == "result" else self.cfg.n_gates,
                "learn_progress": [len(self._analysed_stems()), self.cfg.learn_runs],
                "runs_total": len(runs), "runs_by_status": counts, "discarded": self.state["discarded"],
                "queue": {"analysing": sum(1 for f in self.pending.values() if f.running()),
                          "waiting": sum(1 for f in self.pending.values() if not f.running() and not f.done())},
                "live": dict(self.live), "heats": len(self.state["heats"]),
                "replay_files": dict(self.state["replay_files"]), "error": self.error,
                "pilot_info": self.provider.status(),
                "void_runs": sum(1 for r in runs if r.get("void")),
            }
