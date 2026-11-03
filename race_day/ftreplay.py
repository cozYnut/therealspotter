"""
An old heat played to the app as FPVTrackside would send it live (its
notifier's ExtensionMode events, race_day/ftevents.py), to try the announcer
and the relay without a race.

    venv/bin/python -m race_day.ftreplay EVENT_DIR HEAT_ID [--url http://localhost:8766/] [--speed 1]

The heat's valid crossings (from its Race.json; as FPVTrackside saved them,
marshalled or not) are sent at the moments they happened: Hello, RaceLoaded,
RaceStartAnnouncement, RaceStart, a DetectionExt per crossing (with the
places as FPVTrackside works them out: laps done, then who got there first),
RaceResult and RaceEnd. They're marked as a replay: the app hears them, but never passes them on to fril.
"""

import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from race_day import trackside
from race_day.ftevents import REPLAY_HEADER
from race_day.trackside import parse_time


def events(event_dir: str, heat_id: str):
    """(seconds after the race start, event) of one heat."""
    ev = trackside.Event(event_dir)
    for _ in range(3):
        ev.refresh()
        time.sleep(0.3)
    h = next((h for h in ev.heats.values() if h.id.startswith(heat_id)), None)
    if h is None:
        raise SystemExit(f"heat {heat_id} isn't in {event_dir}")
    d = Path(event_dir)
    race = json.loads((d / h.id / "Race.json").read_text(encoding="utf-8"))
    race = race[0] if isinstance(race, list) else race
    pilots = {p["ID"]: p for p in json.loads((d / "Pilots.json").read_text(encoding="utf-8"))}
    rounds = {r["ID"]: r for r in json.loads((d / "Rounds.json").read_text(encoding="utf-8"))}
    rnd = rounds.get(race.get("Round"), {})
    start, end = parse_time(race["Start"]), parse_time(race["End"])
    who = lambda pid: {"name": pilots.get(pid, {}).get("Name", "?"), "phonetic": pilots.get(pid, {}).get("Phonetic")}
    roster = [who(pc["Pilot"]) for pc in race.get("PilotChannels", [])]
    target = race.get("TargetLaps") or 0
    kind = rnd.get("EventType") or "Race"
    out = [(-30.0, {"type": "Hello", "fpvtVersion": "replay", "decimalPlaces": 2}),
           (-25.0, {"type": "RaceLoaded", "round": rnd.get("RoundNumber"), "race": race.get("RaceNumber"),
                    "raceType": kind, "targetLaps": target, "pilots": roster}),
           (-8.0, {"type": "RaceStartAnnouncement", "delaySeconds": 5}),
           (0.0, {"type": "RaceStart"})]
    dets = sorted((parse_time(x["Time"]), x) for x in race.get("Detections", []) if x.get("Valid"))
    laps, last, prev = {}, {}, {}
    for t, x in dets:
        pid = x["Pilot"]
        n = x.get("LapNumber", 0)
        laps[pid], last[pid] = n, t
        order = sorted(laps, key=lambda p: (-laps[p], last[p]))
        lap = t - prev.get(pid, start)
        prev[pid] = t
        out.append((t - start, {
            "type": "DetectionExt", "detectionId": str(uuid.uuid4()), "round": rnd.get("RoundNumber"),
            "race": race.get("RaceNumber"), "pilotName": who(pid)["name"], "isLapEnd": True, "lapNumber": n,
            "raceTime": round(t - start, 3), "sectorTime": round(lap, 3), "lapTimeSoFar": round(lap, 3),
            "position": order.index(pid) + 1, "valid": True,
            "positionSnapshot": [{"pilotName": who(p)["name"], "position": i + 1, "raceSector": laps[p] * 100,
                                  "lastDetectionTime": round(last[p] - start, 3)} for i, p in enumerate(order)],
            "raceFinishedForPilot": kind == "Race" and bool(target) and n >= target}))
    out.append((end - start - 0.5, {"type": "RaceResult", "raceType": kind, "pilots": []}))
    out.append((end - start, {"type": "RaceEnd"}))
    return h, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event_dir")
    ap.add_argument("heat_id", help="the heat's folder name (its start is enough)")
    ap.add_argument("--url", default="http://localhost:8766/")
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    a = ap.parse_args()
    h, evs = events(a.event_dir, a.heat_id)
    print(f"heat {h.number}: {len(evs)} events → {a.url}")
    t0 = time.time() - evs[0][0] / a.speed
    for seq, (t, e) in enumerate(evs, 1):
        wait = t0 + t / a.speed - time.time()
        if wait > 0:
            time.sleep(wait)
        e = {"type": e["type"], "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
             .replace("+00:00", "Z"), "seq": seq, **e}
        req = urllib.request.Request(a.url, data=json.dumps(e).encode(), method="PUT",
                                     headers={"Content-Type": "application/json; charset=utf-8", REPLAY_HEADER: "1"})
        with urllib.request.urlopen(req, timeout=2) as r:
            r.read()
        print(f"{t:7.2f}  {e['type']}" + (f"  {e['pilotName']} lap {e['lapNumber']}" if "pilotName" in e else ""))


if __name__ == "__main__":
    main()
