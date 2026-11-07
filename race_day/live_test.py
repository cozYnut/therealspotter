"""
A race-day rehearsal of the live video and the heat analysis, to see how a Mac
keeps up (made for the M1).

For each setting of the live video's processes (1, then 2), a few heats of a
recorded event are raced again, one every few minutes, the way race day runs:

  every 4 minutes   a heat is replayed: its 2×2 video stands in for the
                    capture card, its FPVTrackside events (race loaded, start,
                    laps, end) go to the announcer at the moments they happened
                    (race_day/heatreplay.py). The live video names the gates at
                    30 fps and keeps its detections (race_day/livedets.py).
  the race ends     the heat analysis starts in the analysis process, as in the
                    app, with the kept detections — and is paused (SIGSTOP)
                    while the next heat races, like "Pause during races".

Nothing is written to the event: the analyses and the kept detections go to
the results folder. Nothing is sent to fril, and nothing is said aloud.

At the end, each heat is also analysed the normal way (YOLO on every frame,
no live detections) to compare the passes and the time (--no-compare skips it).

Results: <results folder>/results.md (and results.json):
  per heat and setting: the live video's fps a pilot, % of frames dropped, how
  far behind it was, gates named; the heat analysis's time, how much of it ran
  during the next race, whether it was done before the next race, and how many
  frames took YOLO's boxes from the live video; passes against the normal
  analysis; the most memory used and the memory pressure.

Run it on the M1 with FPVTrackside and other apps closed:

  venv/bin/python -m race_day.live_test
  venv/bin/python -m race_day.live_test --event "<event folder>" --heats 4 --every 240 --procs 1,2

It takes about (heats × every × settings) plus the comparison: ~45 minutes
with the defaults.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _data_root() -> Path:
    """The app's data folder: FPV_DATA_ROOT, else the one the app was given (~/.lazy_spotter_v2.json)."""
    if os.environ.get("FPV_DATA_ROOT"):
        return Path(os.environ["FPV_DATA_ROOT"])
    try:
        d = json.loads((Path.home() / ".lazy_spotter_v2.json").read_text()).get("data_root")
        if d:
            os.environ["FPV_DATA_ROOT"] = d        # the analysis processes read it too
            return Path(d)
    except (OSError, ValueError):
        pass
    from dataset_paths import data_root
    return data_root()


def _find_event(root: Path, name: str) -> Optional[Path]:
    from race_day.picker import fpvtrackside_events
    want = [w for w in name.lower().split() if w]
    hits = [e for e in fpvtrackside_events(root) if all(w in str(e["name"]).lower() for w in want)]
    for base in (Path.home() / "Documents" / "FPVTrackside" / "events",):
        if not hits and base.is_dir():
            hits = [e for e in fpvtrackside_events(base) if all(w in str(e["name"]).lower() for w in want)]
    return max(hits, key=lambda e: e["time"])["path"] if hits else None


# ── memory ────────────────────────────────────────────────────────

class Memory:
    """The most memory this test's processes used together, and the Mac's memory pressure, every second."""

    def __init__(self):
        import psutil
        self.me = psutil.Process()
        self.peak_mb, self.peak_pressure, self.swap_mb = 0.0, 1, 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        self.peak_mb, self.peak_pressure = 0.0, 1

    def _run(self):
        import psutil
        while not self._stop.wait(1.0):
            try:
                procs = [self.me] + self.me.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs if p.is_running()) / 1e6
                self.peak_mb = max(self.peak_mb, rss)
                self.swap_mb = max(self.swap_mb, psutil.swap_memory().used / 1e6)
                self.peak_pressure = max(self.peak_pressure, _pressure())
            except Exception:                    # noqa: BLE001 — a process ended while being read
                pass

    def stop(self):
        self._stop.set()


def _pressure() -> int:
    """macOS memory pressure: 1 normal, 2 warning, 4 critical."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        return int(out or 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 1


# ── one heat ──────────────────────────────────────────────────────

def _runs(h, quads, folder: Path) -> List[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    return [{"stem": h.stem(q.channel), "rect": tuple(q.rect),
             "out_json": str(folder / f"{h.stem(q.channel)}.race_data.json")} for q in quads]


def _window(ev, h, video):
    a = float(ev.to_video(video, h.start)) - 5.0
    b = float(ev.to_video(video, h.end)) + 30.0
    return max(0.0, a), b


def _heat_parts(ev, h):
    from race_day.eventsetup import EventSetup
    from dataset_paths import TrackPaths
    setup = EventSetup(TrackPaths(ev.data_dir).dir)
    tp = setup.layout_paths(setup.layout_of(h))
    from race_day.heatreplay import track_of
    quads = [q for q in h.quads if q.kind == "analog" and q.channel in h.pilots]
    return tp, track_of(tp)[0], quads


def _pick_heats(ev, n: int) -> list:
    """Heats with a 2×2 video, LapRF timing and a learned track: the fullest first, then in race order."""
    out = []
    for h in sorted(ev.heats.values(), key=lambda h: h.start or 0):
        if not h.start or not h.end:
            continue
        try:
            tp, n_gates, quads = _heat_parts(ev, h)
        except Exception:                        # noqa: BLE001
            continue
        if n_gates and quads and tp.gate_memory.exists():
            out.append((len(quads), h))
    full = max((k for k, _ in out), default=0)
    return [h for k, h in out if k == full][:n] or [h for _, h in out][:n]


def _passes(path: str) -> List[float]:
    try:
        return [float(p["t"]) for p in json.loads(Path(path).read_text()).get("passes", [])]
    except (OSError, ValueError, KeyError):
        return []


def _compare(base: List[float], live: List[float]) -> dict:
    import numpy as np
    if not base:
        return {"base": 0, "live": len(live)}
    lv = np.array(live) if live else np.array([1e9])
    near = [float(np.min(np.abs(lv - t))) for t in base]
    return {"base": len(base), "live": len(live), "same_0_1s": sum(x <= 0.1 for x in near),
            "same_0_5s": sum(x <= 0.5 for x in near)}


# ── the test ──────────────────────────────────────────────────────

def run_setting(ev, heats, procs: int, every: float, fps: float, out: Path, mem: Memory, log) -> List[dict]:
    from race_day.analysis import AnalysisPool
    from race_day.announcer import Announcer
    from race_day.heatreplay import HeatReplay, LEAD_IN_S, _model
    from race_day.livevideo import LiveVideo
    from race_day.models import worker_devices, DEFAULT_DET_MODEL

    pool = AnalysisPool(worker_devices(Path(DEFAULT_DET_MODEL), 1, True), "mps")
    ann = Announcer(enabled=False)
    stats: Dict[object, dict] = {}
    gates = [0]

    def on_video(e):
        if e.get("type") == "stats" and e.get("racing"):
            stats.setdefault(e.get("part"), []).append(e)
        elif e.get("type") == "gate" and e.get("gate"):
            gates[0] += 1
        elif e.get("type") == "error":
            log(f"   live video error: {e.get('error')}")
        ann.on_video(e)

    live = LiveVideo(on_video, log)
    ann.on_race = live.set_race
    rows, jobs = [], []                          # jobs: (row, future, submitted at)
    paused = [False, 0.0, None]                  # paused now, total paused s, since

    def pause_watch(stop: threading.Event):      # "Pause during races": the analysis stops while a heat races
        while not stop.wait(0.2):
            racing = bool(ann.running)
            if racing and not paused[0]:
                pool.pause()
                paused[0], paused[2] = True, time.time()
            elif not racing and paused[0]:
                pool.resume()
                paused[0] = False
                paused[1] += time.time() - paused[2]
                for r, fut, _ in jobs:           # the time each running analysis spent paused
                    if fut.running():
                        r["paused_s"] = round(r.get("paused_s", 0) + time.time() - paused[2], 1)

    stop = threading.Event()
    threading.Thread(target=pause_watch, args=(stop,), daemon=True).start()
    t_first = time.time()
    try:
        for k, h in enumerate(heats):
            due = t_first + k * every
            if due > time.time():
                time.sleep(due - time.time())
            tp, n_gates, quads = _heat_parts(ev, h)
            keep = out / f"p{procs}" / "live" / h.id
            stats.clear()
            gates[0] = 0
            mem.reset()
            log(f"  heat {h.number} ({len(quads)} pilots): racing, {procs} live process(es)")
            rep = HeatReplay(ann, live, str(ev.dir), h.id, fps=fps, log=log, procs=procs, keep=str(keep))
            why = rep.start()
            if why:
                log(f"   can't replay heat {h.number}: {why}")
                continue
            race_start = time.time() + LEAD_IN_S
            rep.done.wait()
            ended = time.time()
            row = {"heat": h.number, "heat_id": h.id, "pilots": len(quads), "procs": procs,
                   "race_s": round(float(h.end - h.start), 1), "gates_named": gates[0],
                   "peak_mb_race": round(mem.peak_mb), "pressure_race": mem.peak_pressure}
            row.update(_live_summary(stats))
            from race_day import livedets
            video = quads[0].video
            secs, _ = ev.frame_times(video)
            paths = {h.stem(q.channel): livedets.find(keep, q.channel, 0, float("inf")) for q in quads}
            paths = {s: p for s, p in paths.items() if p}
            livecfg = {"paths": paths, "secs": [float(x) for x in secs],
                       "start_v": float(ev.to_video(video, h.start))} if paths else None
            runs = _runs(h, quads, out / f"p{procs}" / "analysis" / h.id)
            fut = pool.submit_heat(video, runs, n_gates, _window(ev, h, video), livecfg)
            row.update(next_race_at=round(t_first + (k + 1) * every + LEAD_IN_S - ended, 1)
                       if k + 1 < len(heats) else None, submitted=ended)
            jobs.append((row, fut, ended))
            rows.append(row)
            log(f"   race over: live {row.get('fps')} fps a pilot, {row.get('dropped')} % dropped, "
                f"{row.get('delay_ms')} ms behind; {gates[0]} gates named; the analysis has started")
            fut.add_done_callback(lambda f, row=row, runs=runs: _done(row, f, runs, log))
        for row, fut, _ in jobs:
            fut.result()
    finally:
        stop.set()
        live.stop()
        pool.shutdown(wait=False)
    return rows


def _done(row: dict, fut, runs: List[dict], log):
    t = time.time()
    try:
        res = fut.result()
    except Exception as e:                       # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
        log(f"   heat {row['heat']}: analysis failed — {row['error']}")
        return
    row["analysis_s"] = round(t - row.pop("submitted"), 1)
    ran = max((r.get("seconds") or 0) for r in res.values())       # from its start, paused time included
    row["waited_s"] = round(max(0.0, row["analysis_s"] - ran), 1)  # behind the previous heat's analysis
    row["worked_s"] = round(max(0.0, ran - row.get("paused_s", 0)), 1)
    hits = sum((r.get("live") or {}).get("hits", 0) for r in res.values())
    misses = sum((r.get("live") or {}).get("misses", 0) for r in res.values())
    row.update(yolo_from_live=hits, yolo_run=misses,
               errors=[r["error"] for r in res.values() if r.get("error")] or None,
               passes={r["stem"]: _passes(r["out_json"]) for r in runs})
    nr = row.get("next_race_at")
    row["ready_before_next_race"] = None if nr is None else row["analysis_s"] <= nr
    log(f"   heat {row['heat']}: analysed in {row['analysis_s']} s (paused {row.get('paused_s', 0)} s during the "
        f"next race); YOLO from the live video for {hits} frames, run for {misses}")


def _live_summary(stats: Dict[object, list]) -> dict:
    """The race's live video stats: the slowest part's mean fps, the most dropped, the median delay."""
    import statistics
    if not stats:
        return {"fps": None, "dropped": None, "delay_ms": None, "detector_ms": None}
    fps = [statistics.mean(x["fps"] for x in s) for s in stats.values()]
    drop = [statistics.mean(x["dropped"] for x in s) for s in stats.values()]
    delay = [statistics.median(x["delay_ms"] for x in s if x.get("delay_ms") is not None) for s in stats.values()]
    det = [statistics.mean(x["detector_ms"] for x in s) for s in stats.values()]
    return {"fps": round(min(fps), 1), "dropped": round(max(drop)), "delay_ms": round(max(delay)),
            "detector_ms": round(statistics.mean(det), 1)}


def baseline(ev, heats, out: Path, log) -> Dict[str, dict]:
    """Each heat analysed the normal way (no live detections), one after another."""
    from race_day.analysis import AnalysisPool
    from race_day.models import worker_devices, DEFAULT_DET_MODEL
    pool = AnalysisPool(worker_devices(Path(DEFAULT_DET_MODEL), 1, True), "mps")
    res = {}
    try:
        for h in heats:
            tp, n_gates, quads = _heat_parts(ev, h)
            video = quads[0].video
            runs = _runs(h, quads, out / "normal" / h.id)
            t0 = time.time()
            pool.submit_heat(video, runs, n_gates, _window(ev, h, video)).result()
            res[h.id] = {"analysis_s": round(time.time() - t0, 1),
                         "passes": {r["stem"]: _passes(r["out_json"]) for r in runs}}
            log(f"  heat {h.number}: analysed the normal way in {res[h.id]['analysis_s']} s")
    finally:
        pool.shutdown(wait=False)
    return res


def report(out: Path, info: dict, rows: List[dict], base: Dict[str, dict]):
    for r in rows:
        b = base.get(r["heat_id"])
        if b and r.get("passes"):
            cmp = [_compare(b["passes"].get(s, []), p) for s, p in r["passes"].items()]
            r["passes_cmp"] = {k: sum(c.get(k, 0) for c in cmp) for k in ("base", "live", "same_0_1s", "same_0_5s")}
            r["normal_analysis_s"] = b["analysis_s"]
    (out / "results.json").write_text(json.dumps({"info": info, "heats": [
        {k: v for k, v in r.items() if k != "passes"} for r in rows]}, indent=1))
    L = [f"# Live video test — {info['machine']}", "",
         f"- **Date:** {info['date']}", f"- **Event:** {info['event']}",
         f"- **Heats:** {', '.join(str(h) for h in info['heats'])}, one every {info['every']:.0f} s",
         f"- **Live video:** {info['fps']:.0f} fps a pilot, detections kept; the analysis paused while a heat races",
         "", "## Live video during the race", "",
         "| Processes | Heat | Pilots | fps a pilot | Dropped % | Behind (ms) | YOLO ms | Gates named | Peak memory (MB) | Pressure |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['procs']} | {r['heat']} | {r['pilots']} | {r.get('fps')} | {r.get('dropped')} | "
                 f"{r.get('delay_ms')} | {r.get('detector_ms')} | {r['gates_named']} | {r['peak_mb_race']} | "
                 f"{_pr(r['pressure_race'])} |")
    L += ["", "## Heat analysis after the race", "",
          "| Processes | Heat | Ready after race end (s) | Waited for previous (s) | Paused in next race (s) | "
          "Worked (s) | Normal analysis (s) | Ready before next race | YOLO from live | YOLO run | "
          "Passes normal / with live / same ±0.1 s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pc = r.get("passes_cmp") or {}
        share = (f"{r['yolo_from_live']} ({100 * r['yolo_from_live'] / max(1, r['yolo_from_live'] + r['yolo_run']):.0f}%)"
                 if r.get("yolo_from_live") is not None else "—")
        ready = {True: "yes", False: "no", None: "(last heat)"}[r.get("ready_before_next_race")]
        L.append(f"| {r['procs']} | {r['heat']} | {r.get('analysis_s', r.get('error', '—'))} | {r.get('waited_s', '—')} | "
                 f"{r.get('paused_s', 0)} | {r.get('worked_s', '—')} | {r.get('normal_analysis_s', '—')} | {ready} | "
                 f"{share} | {r.get('yolo_run', '—')} | "
                 f"{pc.get('base', '—')} / {pc.get('live', '—')} / {pc.get('same_0_1s', '—')} |")
    L += ["", "## Notes", "",
          "- *fps a pilot*: the slowest live process's frames analysed a second for each pilot (30 is all of them).",
          "- *Dropped %*: frames the live video skipped to stay on time (a 60 fps card at 30 fps: 50% at the least).",
          "- *Behind*: how late the live video is on the newest frame (the delay of a gate call).",
          "- *YOLO from live*: frames whose boxes came from the live video; *YOLO run*: frames the analysis ran YOLO on.",
          "- *Ready after race end*: from the race's end to its laps being ready = waited (the previous heat's",
          "  analysis was still going: one analysis process) + paused (the next heat racing: Pause during races) + worked.",
          "- *Worked* against *Normal analysis*: the same heat analysed alone the normal way (YOLO on every frame).",
          "- *Pressure*: macOS memory pressure (normal / warning / critical).",
          f"- Swap: {info.get('swap_start_mb', 0):.0f} MB used before the test, {info.get('swap_mb', 0):.0f} MB at most "
          "during it.", ""]
    (out / "results.md").write_text("\n".join(L))


def _pr(p) -> str:
    return {1: "normal", 2: "warning", 4: "critical"}.get(int(p or 1), str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", help="the FPVTrackside event folder (default: the one named --name)")
    ap.add_argument("--name", default="Race 1 2027", help="find the event by these words in its name")
    ap.add_argument("--heats", type=int, default=4)
    ap.add_argument("--every", type=float, default=240.0, help="a heat every this many seconds")
    ap.add_argument("--procs", default="1,2", help="the live video's processes to try, e.g. 1,2")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=None, help="results folder (default: live_test_results/<date>)")
    ap.add_argument("--no-compare", action="store_true", help="skip the normal analysis to compare against")
    a = ap.parse_args()

    root = _data_root()
    from race_day import trackside
    ev_dir = Path(a.event) if a.event else _find_event(root, a.name)
    if not ev_dir or not (ev_dir / "Event.json").exists():
        raise SystemExit(f"No FPVTrackside event found (looked for '{a.name}' in {root}); give it with --event")
    ev = trackside.Event(str(ev_dir))
    for _ in range(3):
        ev.refresh()
        time.sleep(0.3)
    heats = _pick_heats(ev, a.heats)
    if not heats:
        raise SystemExit("No heat of this event has a 2×2 video, timing and a learned track")
    out = Path(a.out or ROOT / "live_test_results" / datetime.now().strftime("%Y-%m-%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(s: str):
        line = f"{datetime.now():%H:%M:%S}  {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    procs = [int(x) for x in a.procs.split(",") if x.strip()]
    total = len(procs) * len(heats) * a.every / 60
    log(f"Event: {ev.name} ({ev_dir}); heats {', '.join(str(h.number) for h in heats)}; live processes {procs}; "
        f"about {total:.0f} min{' + the normal analysis' if not a.no_compare else ''}. Results: {out}")
    import platform
    machine = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    try:
        import psutil
        machine += f", {psutil.virtual_memory().total / 2 ** 30:.0f} GB"
    except Exception:                            # noqa: BLE001
        pass
    info = {"machine": machine or platform.machine(), "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "event": f"{ev.name} ({ev_dir.name})", "heats": [h.number for h in heats], "every": a.every,
            "fps": a.fps, "procs": procs}
    mem = Memory()
    try:
        import psutil
        info["swap_start_mb"] = psutil.swap_memory().used / 1e6
    except Exception:                            # noqa: BLE001
        pass
    rows: List[dict] = []
    base: Dict[str, dict] = {}
    try:
        for n in procs:
            log(f"── {n} live process(es) ──")
            rows += run_setting(ev, heats, n, a.every, a.fps, out, mem, log)
            report(out, info, rows, base)
        if not a.no_compare:
            log("── the normal analysis, to compare ──")
            base = baseline(ev, heats, out, log)
    except KeyboardInterrupt:
        log("stopped — the results so far are written")
    finally:
        info["swap_mb"] = mem.swap_mb
        mem.stop()
        report(out, info, rows, base)
        log(f"Results: {out / 'results.md'}")
        logf.close()


if __name__ == "__main__":
    main()
