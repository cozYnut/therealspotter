"""
The meeting as FPVTrackside runs it, read from an event folder, for the
dashboard's race-day pages (who is racing, who is next, when each pilot
flies again):

  every race in FPVTrackside's running order — round Order, then round
  number, then race number (dd-pits orders them the same way) — with its
  state (pending: no Start; running: a Start and no End; ended: both), its
  line-up with channels and colours, and each pilot's holeshot, laps, total
  and position (Result.json once the race is over);

  per pilot, the next race: "Racing" (in the running race), "Staging" (in
  the next race with pilots in it), else how many races before theirs.

Only races FPVTrackside counts are shown: valid, in a round of the event.
Practice rounds are marked (they don't count towards standings).

Read with the same care as the rest of the app: a race's files are opened
only when they changed and have been still for STABLE_S, and the race
FPVTrackside is running is never opened from disk. Its laps come from
FPVTrackside's own web server when that is switched on (Settings → Web →
Auto start HTTP Server), which serves the same files without touching them;
otherwise they appear when the race ends.
"""

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from race_day.trackside import OLD_SUBDIR, STABLE_S, SUBDIR, _stat, parse_time, read_json

WEB = "http://localhost:8080"    # FPVTrackside's web server (stock: Settings → Web → Auto start HTTP Server)
WEB_TIMEOUT = 0.5
REFRESH_S = 2.0                  # the schedule is rebuilt at most this often
WEB_RETRY_S = 30.0               # after the web server didn't answer, ask again this much later
STALE_S = 240.0                  # a race started, never ended and quiet this long is over (left selected)
ZERO = "00000000-0000-0000-0000-000000000000"


def _practice(event_type: Optional[str]) -> bool:
    t = (event_type or "").lower()
    return "practic" in t or "practis" in t


class Schedule:
    def __init__(self, event_dir, web: Optional[str] = WEB):
        self.dir = Path(event_dir)
        self.web = web
        self._files: Dict[str, tuple] = {}         # race folder → (sig, Race.json, Result.json)
        self._built: Optional[dict] = None
        self._built_at = 0.0
        self._running_key = None
        self._web_ok: Optional[bool] = None
        self._web_failed_at = 0.0

    # ── reading ────────────────────────────────────────────────

    def _race_files(self, name: str, running: bool) -> Optional[tuple]:
        d = self.dir / name
        sig = (_stat(d / "Race.json"), _stat(d / "Result.json"))
        old = self._files.get(name)
        if old and old[0] == sig:
            return old[1:]
        if running or sig[0] is None:
            return old[1:] if old else None          # FPVTrackside is writing it: not opened
        if time.time() - max(s[1] for s in sig if s) < STABLE_S:
            return old[1:] if old else None          # still being saved: next time
        race = read_json(d / "Race.json")
        if not isinstance(race, dict):
            return old[1:] if old else None
        result = read_json(d / "Result.json") if sig[1] else None
        self._files[name] = (sig, race, result if isinstance(result, list) else None)
        return self._files[name][1:]

    def _from_web(self, event_id: str, race_id: str) -> Optional[dict]:
        """The running race from FPVTrackside's web server (the same Race.json,
        served by FPVTrackside itself)."""
        if not self.web or (self._web_ok is False and time.time() - self._web_failed_at < WEB_RETRY_S):
            return None
        try:
            with urllib.request.urlopen(f"{self.web}/events/{event_id}/{race_id}/Race.json",
                                        timeout=WEB_TIMEOUT) as r:
                d = json.loads(r.read().decode("utf-8-sig"))
            self._web_ok = True
            d = d[0] if isinstance(d, list) and d else d
            return d if isinstance(d, dict) else None
        except Exception:
            self._web_ok, self._web_failed_at = False, time.time()
            return None

    # ── the schedule ───────────────────────────────────────────

    def get(self, running: Optional[str] = None) -> dict:
        """running: the heat folder FPVTrackside is recording now, when the app
        knows it (trackside.Event.running) — never opened from disk."""
        now = time.time()
        if self._built and now - self._built_at < REFRESH_S and running == self._running_key:
            return self._built
        self._built, self._built_at, self._running_key = self._build(running), now, running
        return self._built

    def _build(self, running: Optional[str]) -> dict:
        ev = read_json(self.dir / "Event.json")
        if not isinstance(ev, dict):
            return {"has": False, "races": [], "current": None, "upcoming": [], "next": {}}
        pilots = read_json(self.dir / "Pilots.json") or []
        rounds = {r["ID"]: r for r in (read_json(self.dir / "Rounds.json") or []) if isinstance(r, dict) and "ID" in r}
        stages = {s["ID"]: s.get("Name") or "" for s in (read_json(self.dir / "Stages.json") or [])
                  if isinstance(s, dict) and "ID" in s}
        names = {p["ID"]: p.get("Name") or "?" for p in pilots if isinstance(p, dict) and "ID" in p}
        ids, disp, cols = ev.get("Channels") or [], ev.get("ChannelDisplayNames") or [], ev.get("ChannelColors") or []
        chan = {i: (disp[k] if k < len(disp) and disp[k] else i[:8]) for k, i in enumerate(ids)}
        color = {i: (cols[k] if k < len(cols) else "") for k, i in enumerate(ids)}
        in_event = set(ev.get("Rounds") or [])
        holeshot_first = (ev.get("PrimaryTimingSystemLocation") or "Holeshot") == "Holeshot"
        try:
            folders = [e.name for e in os.scandir(self.dir) if e.is_dir() and e.name not in (SUBDIR, OLD_SUBDIR)]
        except OSError:
            folders = []
        races = []
        for name in folders:
            got = self._race_files(name, running == name)
            race, result = got if got else (None, None)
            if running == name:
                race = self._from_web(ev.get("ID", self.dir.name), name) or race
            if not race or not race.get("Valid", False):
                continue
            rnd = rounds.get(race.get("Round"))
            if not rnd or not rnd.get("Valid", True) or (in_event and rnd["ID"] not in in_event):
                continue
            races.append(self._race(name, race, result, rnd, stages, names, chan, color, holeshot_first,
                                    running == name, running is None))
        races.sort(key=lambda r: r["order"])
        for i, r in enumerate(races):
            r["index"] = i
        cur = next((r for r in races if r["state"] == "running"), None)
        upcoming = [r for r in races if r["state"] == "pending" and r["pilots"]]
        last = max((r for r in races if r["state"] == "ended"), key=lambda r: r["end"] or 0, default=None)
        return {"has": True, "event": ev.get("Name") or self.dir.name, "races": races,
                "current": cur, "last": last, "upcoming": upcoming, "next": next_status(races),
                "laps": ev.get("Laps"), "web": bool(self._web_ok)}

    @staticmethod
    def _race(name, race, result, rnd, stages, names, chan, color, holeshot_first, running, unknown) -> dict:
        """running: FPVTrackside is recording this race now (the app saw it);
        unknown: the app can't tell (the event isn't open in it) — then a
        started race without an end is running unless it has gone quiet."""
        start, end = parse_time(race.get("Start")), parse_time(race.get("End"))
        stage = stages.get(rnd.get("Stage") or ZERO, "")
        label = " · ".join(b for b in (stage or None, f"Round {rnd.get('RoundNumber')}",
                                       f"Race {race.get('RaceNumber')}") if b)
        res = {x.get("Pilot"): x for x in (result or []) if x.get("Valid", True)}
        dets: Dict[str, List[float]] = {}
        for d in race.get("Detections") or []:
            if d.get("Valid") and d.get("IsLapEnd", True) and not d.get("TimingSystemIndex"):
                t = parse_time(d.get("Time"))
                if t is not None:
                    dets.setdefault(d.get("Pilot"), []).append(t)
        if start is None:
            state = "pending"
        elif running:
            state = "running"
        elif end is None and unknown:
            quiet = time.time() - max([start] + [t for ts in dets.values() for t in ts])
            state = "running" if quiet < STALE_S else "ended"
        else:
            state = "ended"
        pilots = []
        for pc in race.get("PilotChannels") or []:
            pid, cid = pc.get("Pilot"), pc.get("Channel")
            if not pid or pid == ZERO:
                continue
            ts = sorted(dets.get(pid, []))
            hs = (ts[0] - start) if ts and start is not None and holeshot_first else None
            marks = ([start] if start is not None and not holeshot_first else []) + ts
            laps = [round(b - a, 3) for a, b in zip(marks, marks[1:])]
            r = res.get(pid) or {}
            pilots.append({"name": names.get(pid, "?"), "channel": chan.get(cid, (cid or "")[:8]),
                           "color": color.get(cid, ""), "holeshot": round(hs, 3) if hs is not None else None,
                           "laps": laps, "total": round(ts[-1] - start, 3) if ts and start is not None else None,
                           "last": ts[-1] if ts else None, "position": r.get("Position"), "dnf": bool(r.get("DNF"))})
        if not any(p["position"] for p in pilots):              # no results yet: by laps, then who crossed first
            for i, p in enumerate(sorted(pilots, key=lambda p: (-len(p["laps"]), p["last"] or 1e18))):
                p["position"] = i + 1 if p["last"] else None
        pilots.sort(key=lambda p: (p["position"] or 99, p["channel"]))
        return {"id": name, "label": label, "stage": stage, "round": rnd.get("RoundNumber"),
                "race": race.get("RaceNumber"), "type": rnd.get("EventType") or "Race",
                "practice": _practice(rnd.get("EventType")), "state": state, "start": start, "end": end,
                "target_laps": race.get("TargetLaps"), "pilots": pilots,
                "order": (rnd.get("Order") or 0, rnd.get("RoundNumber") or 0, race.get("RaceNumber") or 0)}


def next_status(races: List[dict]) -> Dict[str, str]:
    """Each pilot's next race, as dd-pits' boardNextStatus: "Racing" (in the
    running race), "Staging" (in the next race with pilots), else the number
    of pending races before theirs; pilots with no race left aren't listed."""
    running = next((r["index"] for r in races if r["state"] == "running"), None)
    staging = next((r["index"] for r in races if r["state"] == "pending" and r["pilots"]), None)
    out: Dict[str, str] = {}
    for r in races:
        if r["state"] == "ended":
            continue
        for p in r["pilots"]:
            if p["name"] in out or p["name"] in ("", "?"):
                continue
            if r["index"] == running:
                out[p["name"]] = "Racing"
            elif r["index"] == staging:
                out[p["name"]] = "Staging"
            else:
                n = sum(1 for x in races[:r["index"]] if x["state"] == "pending" and x["pilots"])
                out[p["name"]] = str(n)
    return out
