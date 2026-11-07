"""
What the announcer knows when it talks between races (race_day/talk.py): only
facts, from the app's own data — never made up.

  the event        its name, date, venue, classes, qualifying and finals
                   (event_setup.json), every race flown (each race's Race.json)
  the standings    each class's qualifying order (best `laps` in a row,
                   race_day/eventsetup.py: qualifying_bests)
  the pilots       the club's cards (clubs/<club>/pilots.yaml): real name,
                   role, "known for", new
  the season       clubs/<club>/seasons.yaml: last season's table and this
                   season's events so far (their points); this season's
                   races, wins and best lap from those events' files
  the track        gates, nicknames and crash spots (the club's track file),
                   the section times of every run the app analysed
                   (<layout>/runs/*.stats.json): where a pilot loses time
                   against the fastest, and where pilots go down
  the finals       the Seeder's stages (FPVTrackside's API)
"""

import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from race_day.eventsetup import MAIN, fril_points, qualifying_bests, section_legs, standing
from race_day.trackside import parse_time, read_json

SEASON_FILE = "seasons.yaml"
CACHE_S = 10.0
MIN_SAMPLES = 4          # a section's median counts with this many legs at least
MIN_GAIN = 0.2           # a section tip only when the pilot loses this much a lap there
TOP_REF = 3              # the field reference for a section: the median of the fastest this many pilots


def _span(legs: List[Optional[float]], idx: List[int]) -> Optional[float]:
    """A part of the lap's usual time: the sum of its legs' (None if one is unknown)."""
    vals = [legs[i] for i in idx if i < len(legs)]
    return sum(vals) if vals and len(vals) == len(idx) and all(v is not None for v in vals) else None


def season_of(t: float) -> str:
    """FRIL's season by its end year: 1 September to 31 August (9 Oct 2026 → "2027")."""
    d = datetime.fromtimestamp(t)
    return str(d.year + 1 if d.month >= 9 else d.year)


def save_event_points(club: str, season: str, event_id: str, name: str, start: float,
                      points: Dict[str, int]) -> str:
    """An event's points into the club's season file (replacing them if it's there)."""
    p = Path(__file__).resolve().parent.parent / "clubs" / club / SEASON_FILE
    try:
        data = (yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None) or {}
    except (OSError, yaml.YAMLError) as e:
        return f"Can't read {p}: {e}"
    s = data.setdefault("seasons", {}).setdefault(str(season), {})
    events = [e for e in s.get("events") or [] if e.get("id") != event_id]
    events.append({"id": event_id, "name": name, "date": datetime.fromtimestamp(start).strftime("%Y-%m-%d"),
                   "points": dict(points)})
    s["events"] = sorted(events, key=lambda e: e.get("date") or "")
    head = "".join(l for l in (p.read_text(encoding="utf-8").splitlines(True) if p.exists() else [])
                   if l.startswith("#"))
    p.write_text(head + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return f"Saved: {name}, {len(points)} pilots, in the {season} season."


class Facts:
    def __init__(self, day, style, log=print):
        self.day, self.style, self.log = day, style, log
        self._races: Dict[str, tuple] = {}       # event dir → (read at, races)
        self._stats: Dict[str, tuple] = {}       # runs dir → (read at, [stats])

    # ── the event ──────────────────────────────────────────────

    @property
    def setup(self):
        return self.day.setup

    def event(self) -> dict:
        ev = self.day.event
        evj = read_json(Path(ev.dir) / "Event.json") or {}
        evj = evj[0] if isinstance(evj, list) and evj else evj
        start = parse_time((evj or {}).get("Start")) or time.time()
        classes = [c for c in self.setup.classes if c["pilots"]]
        return {"name": ev.name, "date": datetime.fromtimestamp(start).strftime("%-d %B"),
                "start": start, "season": season_of(start),
                "venue": "",
                "classes": classes, "n_pilots": sum(len(c["pilots"]) for c in classes),
                "race_length": (getattr(ev, "rules", {}) or {}).get("race_length"),
                "min_lap": (getattr(ev, "rules", {}) or {}).get("min_lap")}

    def class_of(self, pilot: str) -> Optional[dict]:
        return self.setup.class_of_pilot(pilot)

    def races(self, event_dir: Optional[Path] = None) -> List[dict]:
        """Every race flown in an event (this one by default), in order: {id, start, end, type, round, race,
        target, pilots, laps: {pilot: [lap times]}, order: finishing order, best: (time, pilot)}."""
        d = Path(event_dir or self.day.event.dir)
        hit = self._races.get(str(d))
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
        rounds = {r.get("ID"): r for r in (read_json(d / "Rounds.json") or []) if isinstance(r, dict)}
        names = {p.get("ID"): p.get("Name") for p in (read_json(d / "Pilots.json") or []) if isinstance(p, dict)}
        out = []
        for sub in d.iterdir():
            f = sub / "Race.json"
            if not f.is_file():
                continue
            r = read_json(f)
            r = r[0] if isinstance(r, list) and r else r
            if not isinstance(r, dict) or not r.get("Valid", True):
                continue
            start, end = parse_time(r.get("Start")), parse_time(r.get("End"))
            if not start or not end or end <= start:
                continue
            rnd = rounds.get(r.get("Round")) or {}
            dets = {x.get("ID"): x for x in r.get("Detections") or []}
            laps: Dict[str, List[Tuple[int, float, float]]] = {}
            for lap in r.get("Laps") or []:
                det = dets.get(lap.get("Detection")) or {}
                if not det.get("Valid", True) or (lap.get("LapNumber") or 0) < 1:
                    continue
                n = names.get(det.get("Pilot"))
                if n:
                    laps.setdefault(n, []).append((lap["LapNumber"], float(lap.get("LengthSeconds") or 0),
                                                   parse_time(lap.get("EndTime")) or 0.0))
            pilots = [names.get(pc.get("Pilot")) for pc in r.get("PilotChannels") or [] if names.get(pc.get("Pilot"))]
            target = int(r.get("TargetLaps") or 0)
            fin = {}
            for p in pilots:
                ls = sorted(laps.get(p, []))
                done = ls[:target] if target else ls
                fin[p] = (-len(done), done[-1][2] if done else 1e18)
            floor = self.min_lap()
            best = min(((t, p) for p, ls in laps.items() for _, t, _ in ls if t >= floor), default=None)
            out.append({"id": sub.name, "start": start, "end": end, "type": rnd.get("EventType") or "",
                        "round": rnd.get("RoundNumber"), "round_name": rnd.get("Name") or "",
                        "race": r.get("RaceNumber"), "target": target, "pilots": pilots,
                        "laps": {p: [t for _, t, _ in sorted(ls)] for p, ls in laps.items()},
                        "order": sorted(pilots, key=lambda p: fin[p]), "best": best})
        out.sort(key=lambda x: x["start"])
        if out:                                            # the event's day only (a test flown weeks later isn't it)
            evj = read_json(d / "Event.json") or {}
            evj = evj[0] if isinstance(evj, list) and evj else evj
            t0 = parse_time((evj or {}).get("Start")) if isinstance(evj, dict) else None
            day = datetime.fromtimestamp(t0 or out[0]["start"]).date()
            out = [r for r in out if datetime.fromtimestamp(r["start"]).date() == day]
        self._races[str(d)] = (time.time(), out)
        return out

    def min_lap(self) -> float:
        """The shortest lap that counts (the event's minimum lap; a lap under it is a false crossing)."""
        return float((getattr(self.day.event, "rules", {}) or {}).get("min_lap") or 5.0)

    def fastest_today(self, cls: Optional[dict] = None) -> Optional[Tuple[float, str]]:
        """The fastest lap of the day (a class's: the classes may fly different layouts; the first class's
        by default)."""
        cls = cls or next((c for c in self.setup.classes if c["pilots"]), None)
        mine = {p.lower() for p in (cls or {}).get("pilots") or []}
        floor = self.min_lap()
        return min(((t, p) for r in self.races() for p, ls in r["laps"].items()
                    if not mine or p.lower() in mine for t in ls if t >= floor), default=None)

    def race_counts(self) -> dict:
        rs = self.races()
        return {"flown": len(rs), "qualifying": sum(1 for r in rs if r["type"] == "TimeTrial"),
                "finals": sum(1 for r in rs if r["type"] == "Race")}

    # ── the standings ──────────────────────────────────────────

    def seed_laps(self, cls: Optional[dict]) -> int:
        return int(((cls or {}).get("qualifying") or {}).get("laps") or 2)

    def standings(self, cls: dict, before: Optional[float] = None, exclude: str = "") -> List[Tuple[str, float]]:
        b = qualifying_bests(self.day.event, self.seed_laps(cls), before=before, exclude=exclude,
                             pilots=cls.get("pilots") or None)
        return sorted(b.items(), key=lambda kv: kv[1])

    def seed(self, pilot: str) -> Optional[int]:
        cls = self.class_of(pilot)
        if not cls:
            return None
        return standing(dict(self.standings(cls)), pilot)

    def qualifying_done(self) -> bool:
        """Every qualifying race the app made (or FPVTrackside has) is flown."""
        ft = getattr(self.day, "ft", None)
        if ft and self.day.ft_state.get("serves"):
            try:
                rounds = {r["id"]: r for r in ft.rounds() if r.get("eventType") == "TimeTrial"}
                left = [r for r in ft.get("/races?state=pending") or [] if r.get("roundId") in rounds]
                return bool(rounds) and not left
            except Exception:                              # noqa: BLE001
                return False
        return False

    # ── the pilots ─────────────────────────────────────────────

    def card(self, pilot: str) -> dict:
        return self.style.pilots.get(str(pilot).lower()) or {}

    def say(self, pilot: str) -> str:
        return self.style.pilot(pilot)

    def hook(self, pilot: str) -> Optional[str]:
        """A short fact about the pilot, for an intro: their card, last season, this season."""
        c = self.card(pilot)
        bits = []
        if c.get("known_for"):
            bits.append(str(c["known_for"]))
        last = self.season_rank(pilot, self.last_season())
        if last and last[0] <= 3:
            bits.append(f"{self.style.place(last[0])} in the {self.last_season()} season")
        if c.get("role") and not bits:
            bits.append(f"one of our {c['role']}s")
        if c.get("new") and not bits:
            bits.append("new to the league")
        return bits[0] if bits else None

    def new_pilots(self, names: List[str]) -> List[str]:
        """Pilots with no points in the season file at all (first time with us, as far as it knows)."""
        known = set()
        for s in (self.seasons() or {}).values():
            for row in s.get("standings") or []:
                known.add(str(row.get("pilot", "")).lower())
            for e in s.get("events") or []:
                known |= {k.lower() for k in (e.get("points") or {})}
        return [n for n in names if n.lower() not in known]

    # ── the season ─────────────────────────────────────────────

    def seasons(self) -> dict:
        d = Path(__file__).resolve().parent.parent / "clubs" / (self.style.club or "") / SEASON_FILE
        try:
            return (yaml.safe_load(d.read_text(encoding="utf-8")) or {}).get("seasons") or {}
        except (OSError, yaml.YAMLError):
            return {}

    def this_season(self) -> str:
        return self.event()["season"]

    def last_season(self) -> str:
        return str(int(self.this_season()) - 1)

    def season_table(self, season: Optional[str] = None, today: Optional[Dict[str, int]] = None) -> List[dict]:
        """[{pilot, total, events}] best first: a season's standings — a finished season's table as given; this
        season's from its events' points, plus `today` (this event's points, when it's over) if this event isn't
        in the file yet."""
        season = season or self.this_season()
        s = self.seasons().get(season) or {}
        if s.get("standings"):
            return [{"pilot": r["pilot"], "total": int(r.get("total") or 0), "events": r.get("rounds")}
                    for r in s["standings"]]
        tot: Dict[str, int] = {}
        n = {}
        events = list(s.get("events") or [])
        if today and not any(e.get("id") == self.day.event.dir.name for e in events):
            events.append({"id": self.day.event.dir.name, "points": today})
        for e in events:
            for p, pts in (e.get("points") or {}).items():
                tot[p] = tot.get(p, 0) + int(pts or 0)
                n[p] = n.get(p, 0) + 1
        return [{"pilot": p, "total": t, "events": n[p]} for p, t in sorted(tot.items(), key=lambda kv: -kv[1])]

    def season_rank(self, pilot: str, season: Optional[str] = None) -> Optional[Tuple[int, int]]:
        """(rank, points) of a pilot in a season's standings."""
        for i, r in enumerate(self.season_table(season), 1):
            if str(r["pilot"]).lower() == pilot.lower():
                return i, r["total"]
        return None

    def season_events(self) -> List[Path]:
        """This season's event folders on this Mac: the season file's events, then this one."""
        root = Path(self.day.event.dir).parent
        ids = [e.get("id") for e in (self.seasons().get(self.this_season()) or {}).get("events") or []
               if e.get("id")]
        dirs = [root / i for i in ids if (root / i).is_dir()]
        if Path(self.day.event.dir) not in dirs:
            dirs.append(Path(self.day.event.dir))
        return dirs

    def season_stats(self, pilot: str) -> dict:
        """This season so far: races flown, races won (finals), best lap, from the season's events' files."""
        flown = wins = 0
        best = None
        for d in self.season_events():
            for r in self.races(d):
                if pilot not in r["pilots"]:
                    continue
                flown += 1
                if r["type"] == "Race" and r["order"] and r["order"][0] == pilot and len(r["pilots"]) > 1:
                    wins += 1
                b = min([t for t in r["laps"].get(pilot) or [] if t >= self.min_lap()] or [1e9])
                if b < 1e9 and (best is None or b < best):
                    best = b
        return {"races": flown, "wins": wins, "best_lap": best}

    # ── the track ──────────────────────────────────────────────

    def layout_of(self, cls: Optional[dict]) -> str:
        return ((cls or {}).get("layout")) or MAIN

    def n_gates(self, layout: str = MAIN) -> int:
        try:
            return int(self.day._n_gates(layout))
        except Exception:                                  # noqa: BLE001
            return 0

    def gate_name(self, gate: int, layout: str = MAIN) -> str:
        """A gate as the voice calls it: the track file's name, else the named section it's in, else "gate 5"."""
        g = ((self.style.track or {}).get("gates") or {}).get(gate) or {}
        if g.get("say"):
            return g["say"]
        n = self.n_gates(layout)
        for x in self.setup.sections(layout):
            if gate - 1 in section_legs(x["from"], x["to"], n) or gate == x["to"]:
                return x["name"]
        return f"gate {gate}"

    def units(self, layout: str = MAIN) -> List[dict]:
        """The parts of a lap the talk compares: the layout's named sections (Event setup → Sections) when it
        has them, else every leg gate to gate. [{name, legs}]."""
        n = self.n_gates(layout)
        named = [{"name": x["name"], "legs": section_legs(x["from"], x["to"], n)} for x in self.setup.sections(layout)]
        named = [u for u in named if u["legs"]]
        return named or [{"name": self.section_name(i, n), "legs": [i]} for i in range(n)]

    def section_name(self, i: int, n: int) -> str:
        """Section i (0-based: gate i+1 → gate i+2): "the dive" if the track file names the gate it ends at,
        else "from gate 4 to gate 5"."""
        a, b = i + 1, (i + 1) % n + 1
        gates = (self.style.track or {}).get("gates") or {}
        for g in (b, a):
            if (gates.get(g) or {}).get("say") and g != 1:
                return gates[g]["say"]
        return f"from gate {a} to gate {b}"

    def run_stats(self, layout: str = MAIN) -> List[dict]:
        runs = self.setup.layout_paths(layout).runs_dir
        hit = self._stats.get(str(runs))
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        out = []
        for f in runs.glob("*.stats.json"):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        self._stats[str(runs)] = (time.time(), out)
        return out

    def section_medians(self, runs: List[dict], n: int) -> Dict[str, List[Optional[float]]]:
        legs: Dict[str, List[List[float]]] = {}
        for s in runs:
            if s.get("n_gates") != n:
                continue
            p = s.get("pilot") or "?"
            legs.setdefault(p, [[] for _ in range(n)])
            for row in s.get("section_grid") or []:
                for i, v in enumerate(row[:n]):
                    if v:
                        legs[p][i].append(v)
        return {p: [statistics.median(v) if len(v) >= MIN_SAMPLES else None for v in ls] for p, ls in legs.items()}

    def tip(self, pilot: str) -> Optional[dict]:
        """Where the pilot loses the most time a lap against the fastest pilots (their usual time through a part
        of the lap — a named section, or a leg — against the fastest pilots' usual), and what their best legs
        add up to: {section, gain, ref, theo, best}."""
        cls = self.class_of(pilot)
        lay = self.layout_of(cls)
        n = self.n_gates(lay)
        runs = self.run_stats(lay)
        if not n or not runs:
            return None
        med = self.section_medians(runs, n)
        if not med.get(pilot):
            return None
        best_lap = {}
        for s in runs:
            if s.get("best_lap"):
                p = s.get("pilot")
                best_lap[p] = min(best_lap.get(p, 1e9), s["best_lap"])
        top = [p for p, _ in sorted(best_lap.items(), key=lambda kv: kv[1]) if p != pilot and p in med][:TOP_REF]
        best = None
        for u in self.units(lay):
            mine = _span(med[pilot], u["legs"])
            ref = [(v, p) for p in top for v in [_span(med[p], u["legs"])] if v is not None]
            if mine is None or len(ref) < 2:
                continue
            gain = mine - statistics.median([v for v, _ in ref])
            if gain >= MIN_GAIN and (best is None or gain > best["gain"]):
                best = {"section": u["name"], "gain": gain, "ref": min(ref)[1]}
        legs = [min((v for s in runs if s.get("pilot") == pilot and s.get("n_gates") == n
                     for v in [s["section_best"][i]] if v), default=None) for i in range(n)]
        out = best or {}
        if all(legs) and best_lap.get(pilot) and sum(legs) >= 0.85 * best_lap[pilot]:   # else: video noise
            out.update(theo=sum(legs), best=best_lap[pilot])
        return out or None

    def crash_spots(self, layout: str = MAIN) -> List[Tuple[int, int]]:
        """[(gate, runs)] where pilots went down most: the gate after the last one seen, in runs that didn't fly
        as many laps as the others in their heat (landing after the finish isn't a crash)."""
        runs_dir = self.setup.layout_paths(layout).runs_dir
        n = self.n_gates(layout)
        stats = self.run_stats(layout)
        most: Dict[object, int] = {}
        for s in stats:
            most[s.get("heat")] = max(most.get(s.get("heat"), 0), int(s.get("laps") or 0))
        count: Dict[int, int] = {}
        for s in stats:
            if not s.get("stem") or int(s.get("laps") or 0) >= most.get(s.get("heat"), 0):
                continue
            try:
                rd = json.loads((runs_dir / f"{s['stem']}.race_data.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            passes = [p for p in rd.get("passes") or [] if p.get("gate_id", 0) >= 1]
            if passes and n:
                g = passes[-1]["gate_id"] % n + 1               # down before the next gate
                count[g] = count.get(g, 0) + 1
        return sorted(count.items(), key=lambda kv: -kv[1])

    def sections(self, layout: str = MAIN) -> List[dict]:
        """Every part of the lap (units): {i, name, typical (the pilots' usual, median), spread (the fastest
        pilot's usual to the slowest's)} — the long ones, and the tricky ones (where pilots differ most)."""
        n = self.n_gates(layout)
        med = self.section_medians(self.run_stats(layout), n) if n else {}
        out = []
        for k, u in enumerate(self.units(layout)):
            vals = [v for v in (_span(m, u["legs"]) for m in med.values()) if v is not None]
            if len(vals) >= 3:
                out.append({"i": k, "name": u["name"], "typical": statistics.median(vals),
                            "spread": max(vals) - min(vals)})
        return out

    def lap_guide(self, layout: str = MAIN) -> dict:
        """The usual lap and the fastest lap the app has seen on a layout (its analysed runs)."""
        laps = [t for s in self.run_stats(layout) for t in s.get("lap_times") or [] if t]
        best = min(((s["best_lap"], s.get("pilot")) for s in self.run_stats(layout) if s.get("best_lap")),
                   default=None)
        return {"median": statistics.median(laps) if laps else (self.style.track or {}).get("expected_lap"),
                "best": best}

    # ── the finals ─────────────────────────────────────────────

    def stage(self, cls_name: str, main: str) -> Optional[dict]:
        sid = ((self.setup.data["made"]["stages"].get(cls_name)) or {}).get(main)
        ft = getattr(self.day, "ft", None)
        if sid is None or not ft:
            return None
        try:
            return ft.stage(sid)
        except Exception:                                  # noqa: BLE001
            return None

    # ── what's next: FPVTrackside's running order ──────────────

    def _ft(self):
        ft = getattr(self.day, "ft", None)
        return ft if ft and self.day.ft_state.get("serves") else None

    def pending(self) -> List[dict]:
        """The races not flown yet, in FPVTrackside's running order (the one loaded first)."""
        ft = self._ft()
        if not ft:
            return []
        try:
            return ft.get("/races?state=pending") or []
        except Exception:                                  # noqa: BLE001
            return []

    def up_next(self, n: int = 2) -> List[dict]:
        """The next `n` races with pilots in them: [{id, name, pilots, cls, main}] (cls/main: a finals race)."""
        fin = getattr(self.day, "finals", None)
        out = []
        for r in self.pending():
            pilots = [(p.get("pilot", p) if isinstance(p, dict) else {"name": p}).get("name")
                      for p in r.get("pilots") or []]
            pilots = [p for p in pilots if p]
            if not pilots:
                continue                                   # a finals race nobody is known for yet
            rec = fin.by_fpv(r.get("id")) if fin else None
            out.append({"id": (r.get("id") or "").lower(), "name": r.get("name"), "pilots": pilots,
                        "cls": (rec or {}).get("class"), "main": (rec or {}).get("main")})
            if len(out) >= n:
                break
        return out

    def results(self, fpv_id: Optional[str]) -> List[str]:
        """A flown race's pilots by FPVTrackside's result, first first."""
        ft = self._ft()
        if not ft or not fpv_id:
            return []
        try:
            res = sorted(ft.get(f"/races/{fpv_id}/results") or [], key=lambda x: x.get("position") or 99)
        except Exception:                                  # noqa: BLE001
            return []
        return [x.get("name") for x in res if x.get("name")]

    def bracket_next(self, cls_name: str, main: str, rid: Optional[str], order: List[str]) -> Dict[str, dict]:
        """Where each pilot of a bracket race goes, by their place in it (the Seeder's diagram):
        pilot → {to: stage race, kind: winners|losers|final|out, round: its Finals round, ahead: races before
        it in FPVTrackside (None: not made yet)}."""
        st = self.stage(cls_name, main) if rid else None
        if not st:
            return {}
        from race_day.finals import Finals
        edges = [e for g in st.get("diagrams") or [] for e in g.get("edges") or [] if e.get("from") == rid]
        group = {r.get("id"): (r.get("group") or "").lower() for r in st.get("races") or []}
        fr = Finals.finals_round(st)
        fin = getattr(self.day, "finals", None)
        recs = {r.get("rid"): r for r in (fin.races.values() if fin else [])
                if r.get("class") == cls_name and r.get("main") == main and not r.get("void")}
        pend = [(r.get("id") or "").lower() for r in self.pending()]
        out = {}
        for k, pilot in enumerate(order, 1):
            e = next((e for e in edges if k in (e.get("ranks") or [])), None)   # a chase: its first race
            if not e:
                out[pilot] = {"kind": "out"}
                continue
            to, g = e.get("to"), group.get(e.get("to"), "")
            kind = "final" if any(w in g for w in ("chase", "final", "grand")) else \
                "losers" if "loser" in g else "winners"
            rec = recs.get(to) or {}
            fpv = (rec.get("fpv") or "").lower()
            out[pilot] = {"to": to, "kind": kind, "round": rec.get("round") or fr.get(to),
                          "ahead": pend.index(fpv) if fpv in pend else None}
        return out

    def mains(self) -> List[Tuple[str, str]]:
        made = self.setup.data["made"]["stages"]
        return [(c["name"], m) for c in self.setup.classes for m in sorted(made.get(c["name"]) or {})]

    def chase_wins(self, st: dict) -> Dict[str, int]:
        w: Dict[str, int] = {}
        for r in st.get("races") or []:
            if "chase" in (r.get("group") or "").lower() and r.get("state") == "finished":
                win = (r.get("winners") or [None])[0] or ((r.get("pilots") or [{}])[0].get("name"))
                if win:
                    w[win] = w.get(win, 0) + 1
        return w

    def target_laps(self) -> Optional[int]:
        evj = read_json(Path(self.day.event.dir) / "Event.json") or {}
        evj = evj[0] if isinstance(evj, list) and evj else evj
        return int(evj.get("Laps") or 0) or None if isinstance(evj, dict) else None

    # ── the calendar ───────────────────────────────────────────

    def calendar(self) -> List[dict]:
        from race_day import calendar as cal
        if getattr(self, "_cal", None) is None or time.time() - self._cal[0] > 600:
            folder = Path(__file__).resolve().parent.parent / "clubs" / (self.style.club or "")
            text = cal.fetch(self.style.info.get("calendar_ics") or "", folder, self.log) if self.style.club else None
            self._cal = (time.time(), cal.parse(text or ""))
        return self._cal[1]

    def next_event(self) -> Optional[dict]:
        """The club's next race after this event: {title, date, place}, as the voice says them."""
        from race_day import calendar as cal
        ev = cal.next_event(self.calendar(), self.event()["start"])
        if not ev or not cal.say_title(ev):
            return None
        return {"title": cal.say_title(ev), "date": cal.say_date(ev["start"]), "place": cal.say_place(ev)}

    def venue(self) -> Optional[str]:
        """Where today's event is: the calendar's entry for the day, else the venue's track file."""
        from race_day import calendar as cal
        ev = cal.on_day(self.calendar(), self.event()["start"])
        return (cal.say_place(ev) if ev else None) or (self.style.track or {}).get("name") or None

    def points_example(self, total: int) -> List[int]:
        return [fril_points(i, total) for i in (1, 2, 3)]
