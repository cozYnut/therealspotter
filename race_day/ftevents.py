"""
FPVTrackside's live events, taken in and passed on to fril.

FPVTrackside's notifier (Application Settings → Gate / LED POST notifications:
Notifier = ExtensionMode, NotificationURL) PUTs every event to ONE address as
it happens: Hello, RaceLoaded, RaceStartAnnouncement, RaceStart, DetectionExt
(every crossing, with the live positions), RaceTimesUp, RaceEnd, RaceResult,
PilotCrashedOut, … — one JSON object per PUT, camelCase, in order. Pointing
that address at the app lets the announcer (race_day/announcer.py) hear the
race, and the app passes every event on to fril (fril.co.il/live's ingest),
unchanged, so fril's live board works as before.

FPVTrackside's rules for whoever receives (fril/HANDOFF_live-results-push.md):
answer 200 at once, before doing anything with the event — it sends one at a
time, waits at most 1.5 s, and drops an event it got no answer for, never
retrying. So the handler only queues the body; passing it on to fril and
handing it to the listeners run on their own threads, each with its own queue,
so a slow fril never delays a call and a slow listener never delays fril.

    relay = EventRelay(fril_url, log).start()      # FPVTrackside → http://<this Mac>:8766/
    relay.subscribe(fn)                            # fn(event dict), on the relay's thread
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional

from race_day.fril import USER_AGENT

PORT = 8766
QUEUE_MAX = 200          # what FPVTrackside keeps itself: past this, events are dropped
FORWARD_TIMEOUT = 5.0
REPLAY_HEADER = "X-Lazy-Spotter-Replay"   # an old heat played back (race_day/ftreplay.py): heard, never passed on


class EventRelay:
    def __init__(self, forward_url: Optional[str] = None, log: Callable[[str], None] = print, port: int = PORT):
        self.log = log
        self.forward_url = forward_url or None
        self.listeners: List[Callable[[dict], None]] = []
        self._fwd: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAX)
        self._evt: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAX)
        self.stats = {"received": 0, "last_type": None, "last_at": None, "seq": None, "forwarded": 0,
                      "forward_ok": None, "forward_error": None, "dropped": 0, "fpvtrackside": None}
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    def start(self) -> "EventRelay":
        for target in (self.httpd.serve_forever, self._forwarder, self._dispatcher):
            threading.Thread(target=target, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for q in (self._fwd, self._evt):
            q.put(None)

    def set_forward(self, url: Optional[str]):
        """fril's ingest address (with its token); None: events aren't passed on."""
        self.forward_url = url or None
        self.stats["forward_error"] = None

    def subscribe(self, fn: Callable[[dict], None]):
        self.listeners.append(fn)

    def status(self) -> dict:
        return dict(self.stats, port=self.port, forwarding=bool(self.forward_url))

    # ── taking events in ───────────────────────────────────────

    def _handler(self):
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)                  # first: FPVTrackside waits on this
                self.send_header("Content-Length", "0")
                self.end_headers()
                relay._took(body, bool(self.headers.get(REPLAY_HEADER)))

            do_POST = do_PUT

            def do_GET(self):                            # a check from a browser: is the app listening?
                data = json.dumps(relay.status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        return Handler

    def _took(self, body: bytes, replay: bool = False):
        self.stats["received"] += 1
        self.stats["last_at"] = time.time()
        for q in ((self._evt,) if replay else (self._fwd, self._evt)):   # a replay is never passed on to fril
            try:
                q.put_nowait(body)
            except queue.Full:
                self.stats["dropped"] += 1

    # ── passing them on ────────────────────────────────────────

    def _forwarder(self):
        while True:
            body = self._fwd.get()
            if body is None:
                return
            url = self.forward_url
            if not url:
                continue
            req = urllib.request.Request(url, data=body, method="PUT", headers={
                "Content-Type": "application/json; charset=utf-8", "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=FORWARD_TIMEOUT) as r:
                    r.read()
                self.stats["forwarded"] += 1
                self.stats["forward_ok"] = time.time()
                if self.stats["forward_error"]:
                    self.log("FPVTrackside events: passing them on to fril again")
                self.stats["forward_error"] = None
            except (urllib.error.URLError, OSError, ValueError) as e:
                why = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(getattr(e, "reason", e))
                if why != self.stats["forward_error"]:      # once per kind of failure, not per event
                    self.log(f"FPVTrackside events: fril didn't take one ({why})")
                self.stats["forward_error"] = why

    def _dispatcher(self):
        while True:
            body = self._evt.get()
            if body is None:
                return
            try:
                e = json.loads(body.decode("utf-8"))
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            self.stats["last_type"], self.stats["seq"] = e.get("type"), e.get("seq")
            if e.get("type") == "Hello":
                self.stats["fpvtrackside"] = e.get("fpvtVersion")
            for fn in list(self.listeners):
                try:
                    fn(e)
                except Exception as ex:                  # noqa: BLE001 — one listener's bug never stops the relay
                    self.log(f"FPVTrackside events: {type(ex).__name__}: {ex}")
