"""
The announcer's live side, with no race: what it starts on race day, and an
old heat replayed through it.

  start_live(...)   race day: the live video on FPVTrackside's capture card,
                    the pilots' pictures cut as the event's last heat recorded
                    them, gates named on the event's learned track.
  HeatReplay        a heat of an FPVTrackside event played back at its real
                    pace: its video stands in for the capture card, its
                    crossings for FPVTrackside's events (race_day/ftreplay.py),
                    both on one clock — the announcer can't tell it from a
                    race. Nothing goes to fril.

    venv/bin/python -m race_day.heatreplay EVENT_DIR HEAT_ID [--club fril --track wingate] [--speak]
"""

import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from dataset_paths import TrackPaths
from race_day import trackside
from race_day.livevideo import FPS, LiveVideo

LEAD_IN_S = 30.0          # the replay's first event this long before its race starts (time to load the models)


def track_of(tp: TrackPaths):
    """(gate count, usual lap in seconds, seconds from each gate to the next) of the event's learned track;
    (0, 0, []) if it isn't learned."""
    try:
        d = json.loads(tp.learned_timing.read_text(encoding="utf-8"))
        n, legs = int(d.get("n_gates") or 0), d.get("leg_seconds") or {}
        seq = [float(legs.get(f"G{i}->G{i % n + 1}", 0.0)) for i in range(1, n + 1)]
        return n, float(sum(seq)), seq
    except (OSError, ValueError):
        return 0, 0.0, []


def layout(event, heat=None) -> Dict[str, tuple]:
    """Each analog pilot's picture in the capture card's 2×2 (channel → rect), as a heat recorded it (the last
    one by default)."""
    if heat is None:
        heat = max((h for h in event.heats.values() if any(q.kind == "analog" for q in h.quads)),
                   key=lambda h: h.start or 0, default=None)
    return {q.channel: tuple(q.rect) for q in heat.quads if q.kind == "analog"} if heat else {}


def _model() -> str:
    from race_day.models import DEFAULT_DET_MODEL, ensure_coreml
    m = ensure_coreml(Path(DEFAULT_DET_MODEL))
    return str(m or DEFAULT_DET_MODEL)


def start_live(live: LiveVideo, announcer, event, tp: TrackPaths, device: int, fps: float = FPS) -> Optional[str]:
    """The live video on capture device `device` for this event. None, or why it can't start."""
    n, lap, legs = track_of(tp)
    if not n or not tp.gate_memory.exists():
        return "the event's track isn't learned yet (Learn track): gates can't be named"
    rects = layout(event)
    if not rects:
        return "no heat of this event has recorded the 2×2 layout yet"
    announcer.set_track(n, lap, legs)
    live.start(device, rects, _model(), str(tp.gate_memory), str(tp.learned_timing), fps=fps,
               racing=announcer.running)
    return None


class HeatReplay:
    def __init__(self, announcer, live: LiveVideo, event_dir: str, heat_id: str, fps: float = FPS,
                 log: Callable[[str], None] = print):
        from race_day import ftreplay
        self.announcer, self.live, self.log = announcer, live, log
        self.event = trackside.Event(event_dir)
        for _ in range(3):
            self.event.refresh()
            time.sleep(0.3)
        self.heat, self.events = ftreplay.events(event_dir, heat_id)
        from race_day.eventsetup import EventSetup
        setup = EventSetup(TrackPaths(self.event.data_dir).dir)
        self.tp = setup.layout_paths(setup.layout_of(self.heat))   # the heat's class's track layout
        self.fps = fps
        self.video, self.clock = None, None       # the heat's 2×2 video and the replay's clock, once started
        self._stop = threading.Event()
        self.done = threading.Event()

    def start(self) -> Optional[str]:
        """None, or why it can't play."""
        h = self.heat
        quads = [q for q in h.quads if q.kind == "analog"]
        n, lap, legs = track_of(self.tp)
        if not quads:
            return "this heat has no 2×2 analog video"
        if not n:
            return "the event's track isn't learned yet: gates can't be named"
        video = quads[0].video
        t0 = time.time() + LEAD_IN_S              # the race starts then, on the clock
        v0 = float(self.event.to_video(video, h.start))
        self.announcer.set_track(n, lap, legs)
        self.video, self.clock = str(video), {"from": max(0.0, v0 - 5.0), "video": v0, "wall": t0}
        self.live.start(self.video, layout(self.event, h), _model(), str(self.tp.gate_memory), str(self.tp.learned_timing),
                        fps=self.fps, clock=self.clock)
        threading.Thread(target=self._play, args=(t0,), daemon=True).start()
        return None

    def _play(self, t0: float):
        from datetime import datetime, timezone
        for seq, (t, e) in enumerate(self.events, 1):
            wait = t0 + t - time.time()
            if wait > 0 and self._stop.wait(wait):
                break
            e = {"type": e["type"], "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                 .replace("+00:00", "Z"), "seq": seq, **e}
            if e["type"] == "RaceLoaded":
                e["heatId"] = self.heat.id                  # the announcer's heat_info finds the heat by it
            for p in e.get("pilots") or []:          # the replay's roster gets its channels from the heat
                if isinstance(p, dict) and "channel" not in p:
                    ch = next((c for c, name in self.heat.pilots.items() if name == p.get("name")), None)
                    p["channel"] = {"displayName": ch} if ch else None
            if e["type"] == "RaceStart":
                self.announcer.play_sound("RaceStart")     # FPVTrackside's start tone: on race day it plays it itself
            self.announcer.on_event(e)
        self._stop.wait(5)
        self.live.stop()
        self.done.set()

    def stop(self):
        self._stop.set()
        self.live.stop()


def main():
    import argparse
    from race_day.announcer import Announcer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event_dir")
    ap.add_argument("heat_id")
    ap.add_argument("--club", default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--speak", action="store_true", help="say the calls (default: print them only)")
    a = ap.parse_args()
    ann = Announcer(enabled=a.speak, club=a.club, track=a.track)
    printed = [0]
    stats = {}

    def on_video(e):
        if e.get("type") == "stats":
            stats.update(e)
        elif e.get("type") == "gate" and e.get("gate"):
            print(f"   video  {e['channel']:4} G{e['gate']:<2} lap {e['laps']}  ({time.time() - e['t']:.2f} s late)")
        elif e.get("type") in ("error", "picture", "ready"):
            print("   video ", e)
        ann.on_video(e)

    live = LiveVideo(on_video)
    ann.on_race = live.set_race
    r = HeatReplay(ann, live, a.event_dir, a.heat_id, fps=a.fps)
    why = r.start()
    if why:
        raise SystemExit(why)
    print(f"heat {r.heat.number}: the race starts in {LEAD_IN_S:.0f} s")
    def show():
        for c in ann.calls[printed[0]:]:
            rt = f"{time.time() - ann.started_at:6.1f} s" if ann.started_at else "  start "
            print(f"{rt}  🗣 {c}")
        printed[0] = len(ann.calls)
    while not r.done.wait(0.5):
        show()
    show()
    print("live video:", stats)


if __name__ == "__main__":
    main()
