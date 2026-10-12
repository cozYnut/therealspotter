"""
Heat and pilot names for a race-day run.

A provider is asked when a heat starts. FrilLiveProvider reads the public
fril.co.il live API (GET https://fril.co.il/api/live/state — the same
endpoint the fril.co.il/live page polls) for the heat flying now: stage,
round, race, and each pilot's name and channel. Without it — or without
internet, or before the timing PC is connected — pilots get a sequential name
from the heat number and channel, e.g. H012-R3.

Attempts: a heat (stage + round + race) can be flown more than once — a
restart. Each time its phase becomes "running" an attempt starts; it ends
"completed" at "finished", or "abandoned" if another heat replaces it (or it
goes back to scheduled/prestart) first. Per heat, the latest completed
attempt is the official one; earlier attempts are void — the same rule as
fril, which keeps only the last result of a heat.
"""

import json
import threading
import time
import urllib.request
from typing import Dict, List, Optional

FRIL_LIVE_STATE = "https://fril.co.il/api/live/state"
# Cloudflare refuses Python's default User-Agent (403); a named client is fine
USER_AGENT = "race-day/1.0 (therealspotter)"

# Band names as FPVTrackside sends them → the letter used in channel names (R1, F4, …)
BAND_LETTER = {"raceband": "R", "fatshark": "F", "boscama": "A", "boscamb": "B", "boscame": "E",
               "lowband": "L", "a": "A", "b": "B", "e": "E", "f": "F", "r": "R", "l": "L"}


class HeatInfoProvider:
    def heat_info(self, wall_time: float, channels: List[str]) -> Optional[dict]:
        """{"round": …, "race": …, "pilots": {channel: name}} or None."""
        return None

    def status(self) -> dict:
        return {"source": "none"}

    def attempts(self) -> List[dict]:
        return []

    changed = False

    def stop(self):
        pass


def fallback_name(heat: int, channel: str) -> str:
    return f"H{heat:03d}-{channel}"


def pilot_name(info: Optional[dict], heat: int, channel: str) -> str:
    pilots = (info or {}).get("pilots") or {}
    name = pilots.get(norm_channel(channel))
    return name or fallback_name(heat, channel)


def norm_channel(ch) -> str:
    """'R1', 'r 1', {'band': 'Raceband', 'number': 1} → 'R1'."""
    if isinstance(ch, dict):
        band = str(ch.get("band") or ch.get("shortBand") or "").replace(" ", "").lower()
        num = ch.get("number")
        if num is None:
            return ""
        return f"{BAND_LETTER.get(band, band[:1].upper())}{int(num)}"
    return str(ch or "").replace(" ", "").upper()


def parse_state(state: dict) -> dict:
    """The parts of /api/live/state race day uses."""
    heat = state.get("currentHeat") or {}
    stage = (state.get("activeStage") or {}).get("name")
    pilots = {}
    for p in heat.get("pilots") or []:
        ch = norm_channel(p.get("channel"))
        if ch and p.get("pilotName"):
            pilots[ch] = p["pilotName"]
    return {"connected": bool(state.get("connected")), "stage": stage, "round": heat.get("round"),
            "race": heat.get("race"), "phase": heat.get("phase"), "pilots": pilots}


def heat_key(snap: dict):
    if not snap or snap.get("round") is None or snap.get("race") is None:
        return None
    return (snap.get("stage") or "", int(snap["round"]), int(snap["race"]))


def heat_label(stage, rnd, race) -> str:
    return " · ".join(x for x in (stage or None, f"Round {rnd}" if rnd is not None else None,
                                  f"Race {race}" if race is not None else None) if x)


class AttemptTracker:
    """Turns the sequence of polled heat states into attempts (see module doc)."""

    def __init__(self, attempts: Optional[List[dict]] = None):
        self.attempts: List[dict] = [dict(a) for a in (attempts or [])]
        self.open: Optional[dict] = None
        for a in self.attempts:                    # resumed: a still-running attempt can't be trusted
            if a["status"] == "running":
                a["status"] = "abandoned"
                a["end"] = a.get("end") or a["start"]

    def _close(self, now: float, status: str):
        if self.open:
            self.open.update(status=status, end=now)
            self.open = None

    def update(self, snap: dict, now: float) -> bool:
        """Feed one polled state; True if the attempts changed."""
        key = heat_key(snap)
        phase = (snap or {}).get("phase")
        before = [(a["id"], a["status"]) for a in self.attempts]
        cur = self.open
        if cur and (key is None or tuple(cur["key"]) != key):
            self._close(now, "abandoned")          # another heat replaced it before it finished
        elif cur and phase in ("scheduled", "prestart"):
            self._close(now, "abandoned")          # the same heat was reset to be flown again
        elif cur and phase == "finished":
            self._close(now, "completed")
        if phase == "running" and key is not None:
            if self.open is None:
                n = 1 + sum(1 for a in self.attempts if tuple(a["key"]) == key)
                self.open = {"id": f"{key[0]}|{key[1]}|{key[2]}|{n}", "key": list(key), "stage": key[0],
                             "round": key[1], "race": key[2], "attempt": n, "start": now, "end": None,
                             "status": "running", "pilots": dict(snap.get("pilots") or {})}
                self.attempts.append(self.open)
            elif snap.get("pilots"):
                self.open["pilots"].update(snap["pilots"])
        self._mark_official()
        return before != [(a["id"], a["status"]) for a in self.attempts]

    def _mark_official(self):
        latest = {}
        for a in self.attempts:
            if a["status"] == "completed":
                k = tuple(a["key"])
                if k not in latest or a["end"] >= latest[k]["end"]:
                    latest[k] = a
        for a in self.attempts:
            a["official"] = latest.get(tuple(a["key"])) is a


class FrilLiveProvider(HeatInfoProvider):
    """Polls the fril.co.il live API in the background (every `interval` s)
    and answers with the heat flying now."""

    def __init__(self, url: str = FRIL_LIVE_STATE, interval: float = 2.0, max_age: float = 30.0,
                 attempts: Optional[List[dict]] = None, start: bool = True):
        self.url, self.interval, self.max_age = url, interval, max_age
        self.tracker = AttemptTracker(attempts)
        self.changed = False                       # attempts changed since attempts() was last read
        self.snap: Optional[dict] = None
        self.updated = 0.0
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    def fetch(self) -> dict:
        req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode("utf-8"))

    def _loop(self):
        while not self._stop.is_set():
            try:
                snap = parse_state(self.fetch())
                self.feed(snap, time.time())
            except Exception as e:
                with self._lock:
                    self.error = f"{type(e).__name__}: {e}"
            self._stop.wait(self.interval)

    def feed(self, snap: dict, now: float):
        """One polled state (also used to replay a recorded log in tests)."""
        with self._lock:
            self.snap, self.updated, self.error = snap, now, None
            if self.tracker.update(snap, now):
                self.changed = True

    def attempts(self) -> List[dict]:
        with self._lock:
            self.changed = False
            return [dict(a, pilots=dict(a["pilots"])) for a in self.tracker.attempts]

    def heat_info(self, wall_time: float, channels: List[str]) -> Optional[dict]:
        """The heat that's on now — only while it's scheduled, starting or
        running, so a finished heat's pilots aren't given to the next one."""
        with self._lock:
            snap, fresh = self.snap, time.time() - self.updated <= self.max_age
        if not snap or not fresh or not snap["pilots"] or snap.get("phase") not in ("scheduled", "prestart", "running"):
            return None
        return {"source": "fril", "stage": snap.get("stage"), "round": snap["round"], "race": snap["race"],
                "pilots": dict(snap["pilots"])}

    def status(self) -> dict:
        with self._lock:
            s = dict(self.snap or {})
            age = time.time() - self.updated if self.updated else None
            return {"source": "fril", "reachable": self.error is None and age is not None and age <= self.max_age,
                    "connected": s.get("connected", False), "stage": s.get("stage"), "round": s.get("round"),
                    "race": s.get("race"), "phase": s.get("phase"), "pilots": s.get("pilots", {}),
                    "error": self.error, "age_s": round(age, 1) if age else None}

    def stop(self):
        self._stop.set()
