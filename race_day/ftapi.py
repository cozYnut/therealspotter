"""
FPVTrackside's REST API (/api/v1, from the FPVTrackside alpha on): lap
corrections made inside FPVTrackside itself, so they show at once and are
never saved over — unlike a Race.json written while the event is open (see
race_day/trackside.py: FPVTrackside keeps the event in memory).

What the app uses:

  GET   /                                    the open event's ID (in the index page)
  GET   /api/v1/event                        is the REST API there (the level)
  GET   /api/v1/race-control                 the race selected / running now
  GET   /api/v1/races/{race}                 a race with all its laps
  POST  /api/v1/races/{race}/lap-corrections {pilotId, invalidate: [detection IDs],
                                              add: [{anchorDetectionId, offsetMs}]}
        one pilot's fix in one step, checked in full before anything changes;
        an added crossing is placed after another crossing of the same pilot,
        so the clocks of the two machines don't matter; repeating it is harmless
  POST  /api/v1/races/{race}/laps            {pilot, raceTimeMs}: a crossing with no
                                              earlier one of the pilot to measure from
  PATCH /api/v1/races/{race}/detections/{id} {valid, validityType}: undo (restore)

Reads are open to anyone; changes are allowed from the FPVTrackside machine
itself, or with a token (a web user's API key or FPV_API_TOKEN) from another
one. The API sends camelCase keys and ISO times with an offset.

Levels, worked out on connect:
  rest   the REST API answers: corrections go through it
  web    only the web server answers (stock FPVTrackside, Auto start HTTP Server)
  none   nothing answers: the app works from the event folder alone
"""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

DEFAULT_URL = "http://localhost:8080"
TIMEOUT = 5.0
LEVEL_TTL = 15.0                  # the level is asked again after this long
EVENT_RE = re.compile(r'new EventManager\("events/([0-9a-fA-F-]+)"')


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _key(d: dict, name: str, default=None):
    """A field by its camelCase name, or PascalCase (older builds)."""
    if name in d:
        return d[name]
    return d.get(name[:1].upper() + name[1:], default)


class FPVTrackside:
    def __init__(self, url: str = DEFAULT_URL, token: Optional[str] = None):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.token = token or None
        self._level = ("none", None, 0.0)          # (level, open event id, asked at)

    # ── plumbing ───────────────────────────────────────────────

    def _call(self, method: str, path: str, body=None, raw: bool = False, timeout: float = TIMEOUT):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8-sig")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8", "ignore")).get("error") or ""
            except ValueError:
                msg = ""
            raise ApiError(msg or f"HTTP {e.code}", e.code) from None
        except (urllib.error.URLError, OSError) as e:
            raise ApiError(f"FPVTrackside doesn't answer at {self.url} ({getattr(e, 'reason', e)})") from None
        if raw:
            return text
        return json.loads(text) if text.strip() else None

    # ── what's there ───────────────────────────────────────────

    def level(self, force: bool = False) -> str:
        """rest / web / none (cached for LEVEL_TTL)."""
        lvl, _, at = self._level
        if force or time.time() - at > LEVEL_TTL:
            self._level = self._ask_level()
        return self._level[0]

    def open_event(self) -> Optional[str]:
        """The ID of the event open in FPVTrackside (lower case), if it answers."""
        self.level()
        return self._level[1]

    def _ask_level(self) -> tuple:
        try:
            page = self._call("GET", "/", raw=True, timeout=2.0)
        except ApiError:
            return ("none", None, time.time())
        m = EVENT_RE.search(page or "")
        event = m.group(1).lower() if m else None
        try:
            self._call("GET", "/api/v1/event", timeout=2.0)
            return ("rest", event, time.time())
        except ApiError:
            return ("web", event, time.time())

    def serves(self, event_id: str) -> bool:
        """The REST API is there and FPVTrackside has this event open."""
        return self.level() == "rest" and (self.open_event() or "") == (event_id or "").lower()

    # ── reading ────────────────────────────────────────────────

    def race_control(self) -> dict:
        """{state: none|ready|starting|running|finished, race: id or None,
        next: id or None, elapsed: s}."""
        d = self._call("GET", "/api/v1/race-control") or {}
        cur, nxt = _key(d, "currentRace") or {}, _key(d, "nextRace") or {}
        return {"state": _key(d, "state", "none"), "race": (_key(cur, "id") or "").lower() or None,
                "next": (_key(nxt, "id") or "").lower() or None, "elapsed": _key(d, "elapsedSeconds", 0.0)}

    def laps(self, race_id: str) -> List[dict]:
        """Every lap of a race: [{detection, pilot, number, valid, race_time}]."""
        d = self._call("GET", f"/api/v1/races/{race_id}") or {}
        return [_lap(l) for l in _key(d, "laps") or []]

    # ── changing ───────────────────────────────────────────────

    def lap_corrections(self, race_id: str, pilot_id: str, invalidate: List[str],
                        add: List[Dict[str, object]]) -> List[dict]:
        """One pilot's fix. add: [{"anchor": detection ID, "offset_ms": int}].
        Returns the pilot's laps after it."""
        body = {"pilotId": pilot_id, "invalidate": list(invalidate),
                "add": [{"anchorDetectionId": a["anchor"], "offsetMs": int(a["offset_ms"])} for a in add]}
        d = self._call("POST", f"/api/v1/races/{race_id}/lap-corrections", body) or {}
        if not _key(d, "ok", True):
            raise ApiError(_key(d, "error") or "the correction was refused")
        return [_lap(l) for l in _key(d, "laps") or []]

    def add_lap(self, race_id: str, pilot_id: str, race_time_ms: int) -> List[dict]:
        """A crossing at a time from the race start. Returns the created
        detection IDs (none when one was already within 250 ms)."""
        d = self._call("POST", f"/api/v1/races/{race_id}/laps",
                       {"pilot": pilot_id, "raceTimeMs": int(race_time_ms)}) or {}
        return [(_key(c, "id") or "").lower() for c in _key(d, "created") or []]

    def set_valid(self, race_id: str, detection_id: str, valid: bool):
        """Count (or discount) a crossing, as a marshal would."""
        self._call("PATCH", f"/api/v1/races/{race_id}/detections/{detection_id}",
                   {"valid": bool(valid), "validityType": "Marshall"})


def _lap(l: dict) -> dict:
    return {"detection": (_key(l, "detectionId") or "").lower(), "pilot": (_key(l, "pilotId") or "").lower(),
            "number": _key(l, "number"), "valid": bool(_key(l, "valid", True)),
            "race_time": float(_key(l, "raceTimeSeconds", 0.0) or 0.0)}
