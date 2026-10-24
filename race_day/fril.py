"""
The race day's running order on fril.co.il/live.

The same way FPVTrackside sends fril its live events — an HTTP PUT of one
JSON object to fril's ingest address, with its token
(https://fril.co.il/api/live/ingest?token=…) — the app sends the race on now
and every race to come (race_day/schedule.py), as one UpcomingRaces event:

  {"type": "UpcomingRaces", "ts": "…Z", "source": "lazy-spotter", "event": "…",
   "current": <race or null>,
   "races": [<race>, …]}             the races to come, next first
  race = {"label", "stage", "round", "race", "raceType",
          "pilots": [{"name", "channel": {"label", "band", "number",
                                          "frequency", "colorR", "colorG", "colorB"}}]}

It is sent whenever that changes (a race starts or ends, a race or a line-up
changes in FPVTrackside) and again every RESEND_S, because fril keeps its live
board in memory only and shows the list only while it is fresh. It carries
no "seq": that is FPVTrackside's, and fril uses it to notice FPVTrackside
restarting. Sending runs on its own thread: a slow or unreachable fril never
holds up the app.
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

RESEND_S = 60.0          # fril shows the list while it's fresh (3 min): re-sent this often
RETRY_S = 10.0           # after a failed send, the next try
TIMEOUT = 5.0
MAX_RACES = 30           # what fril keeps


def _rgb(hexcolor: str):
    h = (hexcolor or "").lstrip("#")
    if len(h) != 6:
        return None, None, None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None, None, None


def _race(r: dict) -> dict:
    pilots = []
    for p in r["pilots"]:
        red, green, blue = _rgb(p.get("color"))
        pilots.append({"name": p["name"], "channel": {
            "label": p.get("channel"), "band": p.get("band"), "number": p.get("number"),
            "frequency": p.get("frequency"), "colorR": red, "colorG": green, "colorB": blue}})
    return {"label": r["label"], "stage": r.get("stage") or None, "round": r.get("round"), "race": r.get("race"),
            "raceType": r.get("type"), "pilots": pilots}


def payload(schedule: dict) -> dict:
    """The UpcomingRaces event for a schedule (race_day/schedule.py)."""
    cur = schedule.get("current")
    return {"type": "UpcomingRaces", "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source": "lazy-spotter", "event": schedule.get("event"),
            "current": _race(cur) if cur else None,
            "races": [_race(r) for r in schedule.get("upcoming", [])[:MAX_RACES]]}


def _key(p: dict) -> str:
    """What fril shows: everything but the send time."""
    return json.dumps({k: v for k, v in p.items() if k != "ts"}, sort_keys=True)


class FrilLive:
    def __init__(self, url: str, log=print):
        self.url, self.log = url.strip(), log
        self.last_ok: Optional[float] = None
        self.error: Optional[str] = None
        self._key, self._sent_at = None, 0.0
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        threading.Thread(target=self._worker, daemon=True).start()

    def update(self, schedule: Optional[dict]):
        """Called with the current schedule (often): sends when it changed, or
        when the last send is RESEND_S old."""
        if not schedule or not schedule.get("has"):
            return
        p = payload(schedule)
        key = _key(p)
        if key == self._key and time.time() - self._sent_at < RESEND_S:
            return
        self._key, self._sent_at = key, time.time()
        try:                                     # only the newest list matters
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put_nowait(p)

    def _worker(self):
        while True:
            p = self._q.get()
            req = urllib.request.Request(self.url, data=json.dumps(p).encode(), method="PUT",
                                         headers={"Content-Type": "application/json; charset=utf-8"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    r.read()
                if self.error:
                    self.log("fril live: sending again")
                self.last_ok, self.error = time.time(), None
            except urllib.error.HTTPError as e:
                self._failed("the token was refused (HTTP 401)" if e.code == 401 else f"HTTP {e.code}")
            except (urllib.error.URLError, OSError) as e:
                self._failed(str(getattr(e, "reason", e)))

    def _failed(self, why: str):
        if why != self.error:
            self.log(f"fril live: couldn't send the next races — {why}")
        self.error = why
        self._sent_at = time.time() - RESEND_S + RETRY_S      # try again in RETRY_S

    def status(self) -> dict:
        return {"url": self.url, "last_ok": self.last_ok, "error": self.error}
