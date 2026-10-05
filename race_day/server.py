"""
Read-only dashboard server for race day.

Serves the open track's data to any browser on the same Wi-Fi (phones via
the QR code in the desktop app). Only GET requests: nothing can be changed
from the dashboard. Videos are streamed with HTTP range requests, which
phone browsers need to play and seek.

    /                     the dashboard (static/)
    /api/status           mode, phase, learn progress, live channels, queue
    /api/runs             all runs with their summary and speed index
    /api/run/<stem>       one run: stats, passes, laps, comparison with the field
    /api/heat/<n>         the runs of a heat, with passes for the gap chart
    /api/pilot/<name>     a pilot's runs and section strengths
    /api/track            learned gates (images), sections, hardest sections
    /video/<stem>.mp4     a run's video
    /img?p=<path>         an image inside the track folder (gate crops)
"""

import json
import mimetypes
import os
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from race_day import stats as rstats

STATIC = Path(__file__).parent / "static"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class DataView:
    """Everything the dashboard shows, read from the track folder (plus the
    controller's live status). Stats files are cached by modification time."""

    def __init__(self, day):
        self.day = day
        self._cache = {}
        self._lock = threading.Lock()

    @property
    def tp(self):
        return self.day.tp

    def _json(self, path: Path) -> Optional[dict]:
        try:
            mt = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = self._cache.get(path)
            if hit and hit[0] == mt:
                return hit[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._cache[path] = (mt, data)
        return data

    def all_stats(self):
        with self.day.lock:
            runs = list(self.day.state["runs"].values())
        out = []
        for r in runs:
            if r["status"] == "labelled":
                st = self._json(self.tp.runs_dir / f"{r['stem']}.stats.json")
                if st:
                    out.append(st)
        return out

    def field(self):
        return rstats.field_stats(self.all_stats())

    def runs(self):
        field = self.field()
        stats = {s["stem"]: s for s in self.all_stats()}
        with self.day.lock:
            runs = [dict(r) for r in self.day.state["runs"].values()]
        out = []
        for r in sorted(runs, key=lambda r: r.get("recorded_at", 0), reverse=True):
            st = stats.get(r["stem"])
            if st:
                cmp = rstats.compare(st, field)
                r.update(laps=st["laps"], best_lap=st["best_lap"], best3=st["best3"],
                         theoretical_best=st["theoretical_best"], lap_cv=st["lap_cv"],
                         speed_index=cmp["speed_index"], missed=len(st["missed_gates"]),
                         ended_mid_lap=st["ended_mid_lap"], holeshot=st["holeshot"])
            out.append(r)
        return out

    def run(self, stem: str):
        with self.day.lock:
            r = dict(self.day.state["runs"].get(stem) or {})
        if not r:
            return None
        st = self._json(self.tp.runs_dir / f"{stem}.stats.json")
        race = self._json(self.tp.race_data(stem)) or {}
        field = self.field()
        r["stats"] = st
        r["compare"] = rstats.compare(st, field) if st else None
        r["field"] = field
        r["passes"] = [{"t": p["t"], "gate": p.get("gate_id", -1)} for p in race.get("passes", [])]
        r["lap_marks"] = [{"lap": l["lap"], "t0": l["t0"], "t1": l["t1"]} for l in race.get("laps", [])]
        r["video_url"] = f"/video/{stem}.mp4"
        return r

    def heat(self, n: int):
        with self.day.lock:
            runs = [dict(r) for r in self.day.state["runs"].values() if r["heat"] == n]
            heat = dict(self.day.state["heats"].get(str(n)) or {})
        out = []
        for r in sorted(runs, key=lambda r: r["quad"]):
            race = self._json(self.tp.race_data(r["stem"])) or {}
            st = self._json(self.tp.runs_dir / f"{r['stem']}.stats.json")
            r["passes"] = [{"t": p["t"] - r["live_start_s"], "gate": p["gate_id"]}
                           for p in race.get("passes", []) if p.get("gate_id", -1) >= 1]
            r["stats"] = st
            r["video_url"] = f"/video/{r['stem']}.mp4"
            out.append(r)
        return {"heat": n, "info": heat.get("info"), "runs": out}

    def pilot(self, name: str):
        field = self.field()
        mine = [s for s in self.all_stats() if s["pilot"] == name]
        if not mine:
            return None
        n = mine[0]["n_gates"]
        best_sections = []
        for i in range(n):
            vals = [s["section_best"][i] for s in mine if s["section_best"][i] is not None]
            best_sections.append(min(vals) if vals else None)
        vs = [round(b - m, 3) if b is not None and m else None
              for b, m in zip(best_sections, field.get("section_median") or [None] * n)]
        return {"pilot": name, "runs": sorted(mine, key=lambda s: s["stem"]), "sections": field.get("sections"),
                "best_sections": best_sections, "vs_field_median": vs, "field": field}

    def track(self):
        out = {"gates": [], "legs_s": None, "field": self.field()}
        mem = self._json(self.tp.gate_memory)
        root = self.tp.dir.resolve()
        if mem:
            for g in mem.get("memory", []):
                imgs = [p for p in g.get("embed_imgs", []) if p][:4]
                out["gates"].append({"gate": g["gate_id"], "type": g.get("gate_type"),
                                     "images": [f"/img?p={urllib.parse.quote(str(Path(p).resolve().relative_to(root)))}"
                                                for p in imgs if str(Path(p).resolve()).startswith(str(root))]})
        tim = self._json(self.tp.learned_timing)
        if tim:
            out["legs_s"] = list(tim.get("leg_seconds", {}).values())
        return out


def make_handler(view: DataView):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data).encode(), "application/json")

        def _file(self, path: Path):
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            size = path.stat().st_size
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if (a and b) else size - 1
                end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            p = urllib.parse.unquote(u.path)
            try:
                if p in ("/", "/index.html"):
                    return self._file(STATIC / "index.html")
                if p.startswith("/static/"):
                    f = (STATIC / p[len("/static/"):]).resolve()
                    return self._file(f) if str(f).startswith(str(STATIC.resolve())) else self._send(403, b"", "text/plain")
                if p == "/api/status":
                    return self._json(view.day.status())
                if p == "/api/runs":
                    return self._json({"runs": view.runs(), "field": view.field()})
                if p.startswith("/api/run/"):
                    r = view.run(p[len("/api/run/"):])
                    return self._json(r) if r else self._json({"error": "no such run"}, 404)
                if p.startswith("/api/heat/"):
                    return self._json(view.heat(int(p[len("/api/heat/"):])))
                if p.startswith("/api/pilot/"):
                    r = view.pilot(p[len("/api/pilot/"):])
                    return self._json(r) if r else self._json({"error": "no such pilot"}, 404)
                if p == "/api/track":
                    return self._json(view.track())
                if p.startswith("/video/"):
                    name = Path(p).name
                    return self._file(view.tp.videos / name)
                if p == "/img":
                    q = urllib.parse.parse_qs(u.query).get("p", [""])[0]
                    f = (view.tp.dir / q).resolve()
                    if not str(f).startswith(str(view.tp.dir.resolve())):
                        return self._send(403, b"", "text/plain")
                    return self._file(f)
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._json({"error": str(e)}, 500)
                except Exception:
                    pass

        def do_POST(self):            # read-only dashboard
            self._send(405, b"read-only", "text/plain")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return H


class Dashboard:
    def __init__(self, day, port: int = 8765):
        self.view = DataView(day)
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(self.view))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
