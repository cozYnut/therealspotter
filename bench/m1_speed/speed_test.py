"""
One timed analysis of an FPVTrackside heat, as the app does it (race_day.heat_analysis.analyse_heat: Core ML on
the Neural Engine, CLIP on the CPU, only the race), with an idea from gilboash/therealspotter's Live swapped in
from here. The app's code is not changed; nothing is written to the event (the results go to OUT_DIR).

    venv/bin/python bench/m1_speed/speed_test.py EVENT_DIR HEAT_ID VARIANT PILOTS OUT_DIR

VARIANT:
    base        the app as it is
    procN       YOLO in N detector processes (each its own Core ML model), the moment's pictures shared out
    procN_small … each picture shrunk to the detector's 640×360 before it's sent (gilboash 2.11.7)
    splitN      the pilots in N processes, each its own analyse_heat (decodes the video, loads YOLO and CLIP)
PILOTS: 4, or 6 (the heat's 4 pictures + 2 of them again, as a 6-pilot heat would load it)

Prints one JSON line: wall seconds, the video seconds analysed, the detector's busy seconds, peak memory of the
process tree, peak swap used, and every pilot's passes (count and a fingerprint, to compare variants).
"""
import hashlib, json, os, platform, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
os.chdir(APP)


# ── memory ─────────────────────────────────────────────────────
def tree_rss_mb(pid):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.split("\n")
    kids, rss = {}, {}
    for line in out:
        if line.strip():
            p, pp, r = map(int, line.split())
            kids.setdefault(pp, []).append(p)
            rss[p] = r
    tot, todo = 0, [pid]
    while todo:
        p = todo.pop()
        tot += rss.get(p, 0)
        todo += kids.get(p, [])
    return tot / 1024


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout  # total = … used = 12.50M
    try:
        v = s.split("used =")[1].split()[0]
        return float(v[:-1]) * (1024 if v.endswith("G") else 1)
    except Exception:
        return None


def pressure_level():
    """macOS memory pressure: 1 normal, 2 warning, 4 critical."""
    s = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout
    return int(s) if s.strip().isdigit() else None


SWAP_GROWTH_MAX_MB = float(os.environ.get("SWAP_GROWTH_MAX_MB", 1024))    # the memory is full: swap grew this much during the run …
PRESSURE_MAX = 4              # … or macOS says the memory pressure is critical


def kill_tree(pid):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout.split("\n")
    kids = {}
    for line in out:
        if line.strip():
            p, pp = map(int, line.split())
            kids.setdefault(pp, []).append(p)
    todo, found = list(kids.get(pid, [])), []
    while todo:
        p = todo.pop()
        found.append(p)
        todo += kids.get(p, [])
    for p in found:
        try:
            os.kill(p, 9)
        except OSError:
            pass


def watch_memory(stop, peak, swap0, on_full):
    while not stop.is_set():
        peak["rss"] = max(peak["rss"], tree_rss_mb(os.getpid()))
        sw = swap_used_mb()
        if sw is not None:
            peak["swap"] = max(peak["swap"], sw)
        lv = pressure_level()
        if lv is not None:
            peak["pressure"] = max(peak["pressure"], lv)
        if lv is not None and lv >= PRESSURE_MAX:
            return on_full("memory pressure critical")
        if sw is not None and swap0 is not None and sw - swap0 > SWAP_GROWTH_MAX_MB:
            return on_full(f"swap grew {round(sw - swap0)} MB")
        stop.wait(0.5)


# ── YOLO in its own processes ──────────────────────────────────
def _det_proc(model, kw, conn):
    from ultralytics import YOLO
    y = YOLO(model, task="detect")
    y(np.zeros((540, 960, 3), np.uint8), **kw)
    conn.send("ready")
    while True:
        img = conn.recv()
        if img is None:
            return
        b = y(img, **kw)[0].boxes
        conn.send((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()))


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = [xyxy], [conf], [cls]


class _Res:
    def __init__(self, xyxy, conf, cls):
        self.boxes = [_Box(a, c, k) for a, c, k in zip(xyxy, conf, cls)]


def patch_procs(ha, n, small):
    """Detector.detect hands the moment's pictures to n processes (picture i to process i % n), all at once."""
    import multiprocessing as mp
    import cv2
    ctx = mp.get_context("spawn")
    procs = []
    orig_init = ha.Detector.__init__

    def init(self, model_path, device):
        orig_init(self, model_path, device)
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_det_proc, args=(model_path, self.kw, b), daemon=True)
            p.start()
            procs.append(a)
        for a in procs:
            a.recv()
        self.yolo = None                         # the main process's copy isn't used

    def prep(c):
        if not small:
            return c, (1.0, 1.0)
        h, w = c.shape[:2]
        r = min(640 / w, 384 / h)
        sw, sh = round(w * r), round(h * r)
        return cv2.resize(c, (sw, sh), interpolation=cv2.INTER_LINEAR), (w / sw, h / sh)

    def detect(self, crops):
        out, scale = [None] * len(crops), [None] * len(crops)
        queues = [list(range(k, len(crops), n)) for k in range(n)]
        for k, idx in enumerate(queues):          # the first picture of each process, all at once
            if idx:
                img, scale[idx[0]] = prep(crops[idx[0]])
                procs[k].send(np.ascontiguousarray(img))
        for k, idx in enumerate(queues):
            for j, i in enumerate(idx):
                if j > 0:
                    img, scale[i] = prep(crops[i])
                    procs[k].send(np.ascontiguousarray(img))
                xyxy, conf, cls = procs[k].recv()
                sx, sy = scale[i]
                out[i] = _Res(xyxy * [sx, sy, sx, sy], conf, cls)
        return out

    ha.Detector.__init__, ha.Detector.detect = init, detect


def timed_detect(ha, busy):
    d = ha.Detector.detect

    def detect(self, crops):
        t = time.time()
        r = d(self, crops)
        busy[0] += time.time() - t
        busy[1] += len(crops)
        return r
    ha.Detector.detect = detect


def _split_job(video, runs, model, window):
    from race_day import heat_analysis as ha
    from race_day.analysis import _blank_memory
    return ha.analyse_heat(video, runs, model, 14, device=None, clip_device="cpu", blank_memory=_blank_memory,
                           window=window)


def main():
    event, hid, variant, pilots, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=True)
    from race_day import trackside, heat_analysis as ha
    from race_day.analysis import _blank_memory
    from race_day.models import DEFAULT_DET_MODEL, ensure_coreml
    model = ensure_coreml(DEFAULT_DET_MODEL)
    if model is None:
        sys.exit("no Core ML copy of the detector (the app's default) — can't run the test")
    model = str(model)
    e = trackside.Event(event)
    for _ in range(3):
        e.refresh()
        time.sleep(0.3)
    h = next((h for h in e.heats.values() if h.id.startswith(hid)), None)
    if h is None:
        sys.exit(f"heat {hid} not found in {event} (is the event folder complete?)")
    quads = [q for q in h.quads if q.kind != "dji"][:4]
    video = str(quads[0].video)
    a = max(0.0, float(e.to_video(quads[0].video, h.start)) - 5.0)   # the app's race window (eventday._race_window)
    b = float(e.to_video(quads[0].video, h.end)) + 30.0
    rects = [q.rect for q in quads]
    rects += rects[:pilots - len(rects)]
    runs = [{"stem": f"{hid}_{i}", "rect": r, "out_json": str(out / f"{hid}_{i}.race_data.json"),
             "log": str(out / f"{hid}_{i}.log")} for i, r in enumerate(rects)]
    stop, peak, busy = threading.Event(), {"rss": 0.0, "swap": 0.0, "pressure": 0}, [0.0, 0]
    swap0 = swap_used_mb()
    t0 = time.time()

    def on_full(why):                            # the memory is full: this run stops here (exit code 3)
        print(json.dumps({"variant": variant, "pilots": pilots, "stopped": why, "after_s": round(time.time() - t0, 1),
                          "video_s": round(b - a, 1), "peak_rss_mb": round(peak["rss"]),
                          "swap_before_mb": swap0 and round(swap0), "peak_swap_mb": round(peak["swap"]),
                          "peak_pressure": peak["pressure"]}), flush=True)
        kill_tree(os.getpid())
        os._exit(3)
    threading.Thread(target=watch_memory, args=(stop, peak, swap0, on_full), daemon=True).start()
    if variant.startswith("split"):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        k = int(variant[5:])
        groups = [runs[i::k] for i in range(k)]
        res = {}
        with ProcessPoolExecutor(k, mp_context=mp.get_context("spawn")) as ex:
            for f in [ex.submit(_split_job, video, g, model, (a, b)) for g in groups]:
                res.update(f.result())
    else:
        if variant.startswith("proc"):
            patch_procs(ha, int(variant[4:].split("_")[0]), variant.endswith("_small"))
        elif variant != "base":
            sys.exit(f"unknown variant {variant}")
        timed_detect(ha, busy)
        res = ha.analyse_heat(video, runs, model, 14, device=None, clip_device="cpu", blank_memory=_blank_memory,
                              window=(a, b))
    wall = time.time() - t0
    stop.set()
    passes = []
    for r in runs:
        try:
            ps = [(round(p["t"], 3), p.get("gate_type")) for p in json.loads(Path(r["out_json"]).read_text())["passes"]]
            passes.append({"n": len(ps), "hash": hashlib.md5(json.dumps(ps).encode()).hexdigest()[:10]})
        except Exception as ex:
            passes.append({"error": str(ex)[:80]})
    print(json.dumps({"variant": variant, "pilots": pilots, "wall_s": round(wall, 1), "video_s": round(b - a, 1),
                      "x_realtime": round(wall / (b - a), 2), "det_busy_s": round(busy[0], 1),
                      "det_pictures": busy[1], "peak_rss_mb": round(peak["rss"]),
                      "swap_before_mb": swap0 and round(swap0), "peak_swap_mb": round(peak["swap"]),
                      "peak_pressure": peak["pressure"], "passes": passes, "errors": [v.get("error") for v in res.values() if v.get("error")]}),
          flush=True)


if __name__ == "__main__":
    main()
