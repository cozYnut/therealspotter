"""
An event's setup: its classes (Open, Beginners…), the track layout each class
flies, its qualifying, how its finals are made, and its points. Set at the
start of an event; FPVTrackside then gets the rounds and races through its
REST API (race_day/ftapi.py).

    <event data folder>/event_setup.json

    {"layouts":  {"main": {"gates": 14}, "beginners": {"gates": 10}},
     "channels": ["R1", "R3", "R6", "R8"],          seats, in order (empty: FPVTrackside picks)
     "classes": [{"name": "Open", "layout": "main", "pilots": ["Pennz", …],
                  "qualifying": {"rounds": 4, "per_race": 4, "laps": 3},
                  "finals": {"main_size": 8, "per_race": 4, "advance": 2,
                             "chase_main": "A", "ace_wins": 2, "final": "single",
                             "leftover": "series", "series_races": 5, "min_bracket": 6},
                  "pins": {"Slow": "B"},              a pilot put in a main by hand (level)
                  "points": "fril",
                  "style": {"qualifying": "times", "finals": "places"}}],
     "made": {"qualifying": {"Open": [round externalIds]},
              "stages": {"Open": {"A": stage id, "B": stage id}}}}

Layouts. "main" is the event's own track folder (what the app always used);
any other layout is a track folder of its own, <data>/layouts/<name>/, learned
from its own runs. A run belongs to its pilot's class's layout.

The finals ("Make the brackets"). Each class's qualifying order (FPVTrackside's
best consecutive laps over that class's qualifying rounds) is cut into mains:
A = the fastest `main_size` pilots, then B, C… A remainder too small for a
bracket of its own (under `min_bracket`) becomes a series main (leftover
"series": the same pilots race `series_races` times, for points; under 3
pilots they join the last main instead), joins the
last main (leftover "join"), or the pilots are spread so every main is about
the same size (leftover "spread"). A pilot pinned to a main ("level") takes a
seat there and the others fill up around them, in qualifying order. Each main
is one FPVTrackside stage (its Seeder): a bracket main is a double
elimination, `per_race` a race and `advance` going through; the chase main
ends in a chase the ace (first to `ace_wins` wins); the others end in `final`:
"single" (the club's: the same bracket as A, its final one race), "chase-the-ace" (to `ace_wins` too) or
"grand-final" (the Seeder's classic double elimination: a 2-up winners final, more rounds).

Places and points. In a bracket main the pilots are placed by how far they
got: the champion first, then by the last race they flew (a later race is
better), and within that race by finishing place; the chase the ace by wins,
then best place. Mains follow each other (A's pilots, then B's…). FRIL points:
place N of TOTAL pilots in the class earns TOTAL+1−N, plus ceil(TOTAL/5),
ceil(TOTAL/8), ceil(TOTAL/13) for 1st, 2nd and 3rd (never under 1).
"""

import copy
import json
import math
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataset_paths import TrackPaths

FILE = "event_setup.json"
MAIN = "main"
LAYOUTS_DIR = "layouts"

DEFAULT_QUALIFYING = {"rounds": 4, "per_race": 4, "laps": 3}
DEFAULT_FINALS = {"main_size": 8, "per_race": 4, "advance": 2, "chase_main": "A", "ace_wins": 2,
                  "final": "single", "leftover": "series", "series_races": 5, "min_bracket": 6}
DEFAULT_STYLE = {"qualifying": "times", "finals": "places"}
LEFTOVERS = ("series", "join", "spread")
FINALS = ("single", "chase-the-ace", "grand-final")   # the other mains' final: one race (the club's), a chase the
                                                     # ace to ace_wins too, or the Seeder's classic grand final
POINTS = ("fril", "none")
STYLES = ("times", "places", "both")
MIN_SERIES = 3                    # a smaller remainder joins the last main instead of racing a series of its own
EXT_BASE = 710000                 # FPVTrackside externalIds of the rounds and races the app makes
FIN_BASE = 720000                 # … of the finals rounds (FIN_BASE + n) and races (FIN_BASE + 1000 + i)


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return s or "layout"


def new_class(name: str, layout: str = MAIN, pilots: Optional[List[str]] = None) -> dict:
    return {"name": name, "layout": layout, "pilots": list(pilots or []),
            "qualifying": dict(DEFAULT_QUALIFYING), "finals": dict(DEFAULT_FINALS),
            "pins": {}, "points": "fril", "style": dict(DEFAULT_STYLE)}


class EventSetup:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.path = self.dir / FILE
        self.data = {"layouts": {MAIN: {"gates": None}}, "channels": [], "classes": [],
                     "made": {"qualifying": {}, "stages": {}, "finals": {"races": {}, "next": 1}}}
        self._sig = None
        self.reload()

    # ── the file ───────────────────────────────────────────────

    def reload(self):
        try:
            st = self.path.stat()
        except OSError:
            return
        sig = (st.st_mtime, st.st_size)
        if sig == self._sig:
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self._sig = sig
        self.data.update({k: v for k, v in d.items() if k in self.data})
        self.data["layouts"].setdefault(MAIN, {"gates": None})
        for c in self.data["classes"]:                    # fields added later get their defaults
            for k, v in new_class(c.get("name", "")).items():
                c.setdefault(k, copy.deepcopy(v))
            for k, v in DEFAULT_FINALS.items():
                c["finals"].setdefault(k, v)
            for k, v in DEFAULT_QUALIFYING.items():
                c["qualifying"].setdefault(k, v)
        self.data["made"].setdefault("qualifying", {})
        self.data["made"].setdefault("stages", {})
        self.data["made"].setdefault("finals", {"races": {}, "next": 1})

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)
        st = self.path.stat()
        self._sig = (st.st_mtime, st.st_size)

    # ── classes and layouts ────────────────────────────────────

    @property
    def classes(self) -> List[dict]:
        return self.data["classes"]

    @property
    def layouts(self) -> Dict[str, dict]:
        return self.data["layouts"]

    def cls(self, name: str) -> Optional[dict]:
        return next((c for c in self.classes if c["name"] == name), None)

    def class_of_pilot(self, pilot: str) -> Optional[dict]:
        low = (pilot or "").lower()
        return next((c for c in self.classes if low in {p.lower() for p in c["pilots"]}), None)

    def class_of_heat(self, heat) -> Optional[dict]:
        """The class a heat belongs to: by its round's or stage's name ("Open Qualifying 2", "Open A main"),
        else the class most of its pilots are in."""
        for text in (getattr(heat, "round_name", "") or "", getattr(heat, "stage", "") or ""):
            for c in sorted(self.classes, key=lambda c: -len(c["name"])):
                if c["name"] and text.lower().startswith(c["name"].lower() + " "):
                    return c
        votes: Dict[str, int] = {}
        for p in (getattr(heat, "pilots", {}) or {}).values():
            c = self.class_of_pilot(p)
            if c:
                votes[c["name"]] = votes.get(c["name"], 0) + 1
        return self.cls(max(votes, key=votes.get)) if votes else None

    def layout_of(self, heat, channel: Optional[str] = None) -> str:
        """The track layout a run is flown on: its pilot's class's (a heat's class, without a pilot)."""
        c = None
        if channel is not None:
            c = self.class_of_pilot((getattr(heat, "pilots", {}) or {}).get(channel, ""))
        c = c or self.class_of_heat(heat)
        lay = (c or {}).get("layout") or MAIN
        return lay if lay in self.layouts else MAIN

    def layout_paths(self, name: str) -> TrackPaths:
        if not name or name == MAIN:
            return TrackPaths(self.dir)
        return TrackPaths(self.dir / LAYOUTS_DIR / _slug(name))

    def sections(self, layout: str = MAIN) -> List[dict]:
        """A layout's named parts of the track: [{name, from, to}] (gates; "the dive" from gate 5 to gate 8)."""
        return list(((self.layouts.get(layout or MAIN) or {}).get("sections")) or [])

    def set_sections(self, layout: str, sections: List[dict]):
        self.layouts.setdefault(layout or MAIN, {"gates": None})["sections"] = [
            {"name": str(x["name"]).strip(), "from": int(x["from"]), "to": int(x["to"])}
            for x in sections if str(x.get("name") or "").strip()]

    def add_layout(self, name: str, gates: Optional[int] = None) -> str:
        name = name.strip()
        if name and name not in self.layouts:
            self.layouts[name] = {"gates": gates}
        return name

    def rename_class(self, old: str, new: str):
        c = self.cls(old)
        if not c or not new or self.cls(new):
            return
        c["name"] = new
        for k in ("qualifying", "stages"):
            if old in self.data["made"][k]:
                self.data["made"][k][new] = self.data["made"][k].pop(old)
        for r in self.data["made"]["finals"].get("races", {}).values():
            if r.get("class") == old:
                r["class"] = new

    # ── qualifying ─────────────────────────────────────────────

    def qualifying_plan(self, seat=None) -> dict:
        """FPVTrackside's meeting/sync plan for the whole event's qualifying: time-trial rounds "Qualifying 1"…,
        each holding every class, in class order (Open's races, then Beginners'…). A class's pilots fly in races
        of up to its `per_race` (as even as possible, the same races every round); a class with fewer `rounds`
        is left out of the later ones. seat(pilots) → [{pilot, channel?}] (default: the seat channels in
        order). Sent again, it changes nothing that's there (externalIds); a race already flown is never
        touched."""
        seat = seat or self.seat_default
        groups = [(c, split_even(list(c["pilots"]), int(c["qualifying"].get("per_race") or 4)))
                  for c in self.classes if c["pilots"]]
        n = max([int(c["qualifying"].get("rounds") or 1) for c, _ in groups] or [0])
        rounds = []
        for r in range(1, n + 1):
            races = []
            for c, gs in groups:
                if r > int(c["qualifying"].get("rounds") or 1):
                    continue
                for g in gs:
                    races.append({"externalId": EXT_BASE + r * 100 + len(races) + 1, "pilots": seat(g)})
            rounds.append({"externalId": EXT_BASE + r, "eventType": "TimeTrial", "name": f"Qualifying {r}",
                           "races": races})
        return {"rounds": rounds, "removeMissing": False}      # pilots by name: their own records are left alone

    def seat_default(self, pilots: List[str]) -> List[dict]:
        chans = list(self.data.get("channels") or [])
        return [{"pilot": p, **({"channel": chans[j]} if j < len(chans) else {})} for j, p in enumerate(pilots)]

    def qualifying_round_ids(self, cls: dict, rounds: List[dict]) -> List[str]:
        """FPVTrackside's IDs of the class's qualifying rounds (from GET /rounds): those the app made, else time
        trials named after the class, else every time trial (an event set up by hand)."""
        made = set(self.data["made"]["qualifying"].get(cls["name"], []))
        mine = [r["id"] for r in rounds if r.get("externalId") in made]
        if mine:
            return mine
        named = [r["id"] for r in rounds if r.get("eventType") == "TimeTrial"
                 and ((r.get("name") or "").lower().startswith(cls["name"].lower())
                      or (r.get("stage") or "").lower().startswith(cls["name"].lower()))]
        return named or [r["id"] for r in rounds if r.get("eventType") == "TimeTrial"]

    # ── the finals ─────────────────────────────────────────────

    def seeded(self, cls: dict, seeding: List[dict]) -> List[dict]:
        """The class's pilots in qualifying order: [{name, best}] — FPVTrackside's seeding, filtered to the class;
        pilots with no time last, in the class's own order."""
        mine = {p.lower(): p for p in cls["pilots"]}
        out, seen = [], set()
        for s in seeding:
            n = (s.get("name") or "").lower()
            if n in mine and n not in seen and s.get("bestSeconds") is not None:
                out.append({"name": mine[n], "best": float(s["bestSeconds"])})
                seen.add(n)
        out += [{"name": p, "best": None} for p in cls["pilots"] if p.lower() not in seen]
        return out

    def mains_plan(self, cls: dict, seeded: List[dict]) -> List[dict]:
        """The class cut into mains: [{main: "A", kind: "bracket"|"series", pilots: [{name, best, seed, pinned}]}]."""
        f = cls["finals"]
        sizes = main_sizes(len(seeded), int(f["main_size"]), int(f["min_bracket"]), f["leftover"])
        kinds = ["bracket"] * len(sizes)
        if f["leftover"] == "series" and len(sizes) > 1 and MIN_SERIES <= sizes[-1] < int(f["min_bracket"]):
            kinds[-1] = "series"
        letters = list(string.ascii_uppercase[:len(sizes)])
        pins = {p.lower(): m for p, m in (cls.get("pins") or {}).items() if m}
        seats: Dict[str, List[dict]] = {m: [] for m in letters}
        for i, s in enumerate(seeded, 1):
            s = dict(s, seed=i, pinned=False)
            m = pins.get(s["name"].lower())
            if m in seats:
                seats[m].append(dict(s, pinned=True))
        free = [dict(s, seed=i, pinned=False) for i, s in enumerate(seeded, 1)
                if pins.get(s["name"].lower()) not in seats]
        for m, size in zip(letters, sizes):                 # the others fill up around them, fastest first
            while len(seats[m]) < size and free:
                seats[m].append(free.pop(0))
        if free:                                            # more pinned than room: the rest go last
            seats[letters[-1]] += free
        for m in letters:
            seats[m].sort(key=lambda s: s["seed"])
        return [{"main": m, "kind": k, "pilots": seats[m]} for m, k in zip(letters, kinds) if seats[m]]

    def stage_request(self, cls: dict, main: dict, channels: Optional[List[str]] = None) -> dict:
        """POST /api/v1/stages for one main."""
        f = cls["finals"]
        chans = list(channels if channels is not None else self.data.get("channels") or [])
        pilots = [p["name"] for p in main["pilots"]]
        name = f"{cls['name']} {main['main']} main"
        if main["kind"] == "series":
            return {"name": name, "format": "mains", "pilots": pilots, "seatChannels": chans, "autoSend": False,
                    "setup": {"mains": [main["main"]], "rounds": int(f["series_races"]),
                              "pilotsPerRace": max(len(pilots), 1)}}
        final = "chase-the-ace" if main["main"] == (f.get("chase_main") or "") else (f.get("final") or "single")
        setup = {"eliminationType": "double", "pilotsPerRace": int(f["per_race"]),
                 "pilotsToAdvance": int(f["advance"])}
        if final == "grand-final":                     # the Seeder's classic: a 2-up winners final, more rounds
            setup["finalsFormat"] = "grand-final"
        else:                                          # the same bracket as A; "single" = a chase of one race
            setup["finalsFormat"] = "chase-the-ace"
            setup["aceWinsRequired"] = 1 if final == "single" else int(f["ace_wins"])
        return {"name": name, "format": "elimination", "pilots": pilots, "seatChannels": chans,
                "autoSend": False, "setup": setup}

    # ── places and points ──────────────────────────────────────

    def places(self, cls: dict, stages: Dict[str, dict]) -> List[dict]:
        """The class's places from its stages (main letter → GET /stages/{id}): [{place, name, main, points,
        detail, final}], A's pilots first. final: the main is over."""
        out = []
        for m in sorted(stages):
            st = stages[m]
            ranked = (series_order if st.get("format") == "mains" else bracket_order)(st)
            for name, detail in ranked:
                out.append({"name": name, "main": m, "detail": detail, "final": bool(st.get("isComplete"))})
        total = len(cls["pilots"]) or len(out)
        for i, r in enumerate(out, 1):
            r["place"] = i
            r["points"] = fril_points(i, total) if cls.get("points") == "fril" else None
        return out

    # ── what the announcer needs ───────────────────────────────

    def heat_kind(self, heat=None, race_type: str = "", round_name: str = "", stage: str = "") -> str:
        """qualifying | finals | practice, for the announcer's style."""
        rt = (race_type or getattr(heat, "race_type", "") or "").lower()
        if "practice" in rt:
            return "practice"
        if rt == "timetrial" or "qualif" in (round_name or getattr(heat, "round_name", "") or "").lower():
            return "qualifying"
        return "finals"


# ── helpers ────────────────────────────────────────────────────

def split_even(items: list, per: int) -> List[list]:
    """Items in groups of at most `per`, as even as possible, in order (10 by 4 → 4, 3, 3)."""
    if not items:
        return []
    per = max(1, per)
    n = math.ceil(len(items) / per)
    base, extra = divmod(len(items), n)
    out, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        out.append(items[i:i + size])
        i += size
    return out


def main_sizes(n: int, size: int, min_bracket: int, leftover: str) -> List[int]:
    """How many pilots each main gets, A first."""
    if n <= 0:
        return []
    size = max(2, size)
    full, rem = divmod(n, size)
    if full == 0:
        return [n]
    if rem == 0:
        return [size] * full
    if leftover == "spread":
        k = full + (1 if rem >= min_bracket else 0)
        base, extra = divmod(n, k)
        return [base + (1 if i < extra else 0) for i in range(k)]
    if rem >= min_bracket or (leftover == "series" and rem >= MIN_SERIES):
        return [size] * full + [rem]
    return [size] * (full - 1) + [size + rem]           # "join": the last main takes them


def seat_by_channel(pilots: List[str], own: Dict[str, str], channels: List[dict],
                    seats: Optional[List[str]] = None, taken=()) -> List[dict]:
    """A race's seats: each pilot on their own channel in the event (a DJI pilot keeps DR8) unless its frequency
    is taken in the race already, else the first free seat channel (`seats`, else the event's channels).
    channels: the event's [{label, frequency}]; own: pilot (lower case) → channel label."""
    freq = {c.get("label"): c.get("frequency") for c in channels}
    order = list(seats or []) or [c.get("label") for c in channels if not str(c.get("label", "")).upper().startswith("D")]
    used, out = {freq.get(c, c) for c in taken}, []     # taken: channels already given in the race
    for p in pilots:
        ch = own.get(p.lower())
        if not ch or ch not in freq or freq[ch] in used:
            ch = next((c for c in order if freq.get(c, c) not in used), None)
        if ch:
            used.add(freq.get(ch, ch))
        out.append({"pilot": p, **({"channel": ch} if ch else {})})
    return out


def section_legs(a: int, b: int, n: int) -> List[int]:
    """The legs (0-based: leg i = gate i+1 → gate i+2) from gate a to gate b, round the lap if b ≤ a."""
    if not n or not (1 <= a <= n) or not (1 <= b <= n):
        return []
    out, g = [], a
    while True:
        out.append(g - 1)
        g = g % n + 1
        if g == b or len(out) >= n:
            return out


def best_consecutive(laps: List[float], n: int) -> Optional[float]:
    """The best `n` laps in a row (their total), or None with fewer laps."""
    n = max(1, int(n or 1))
    if len(laps) < n:
        return None
    return min(sum(laps[i:i + n]) for i in range(len(laps) - n + 1))


def qualifying_bests(event, n: int, before: Optional[float] = None, exclude: str = "",
                     pilots: Optional[List[str]] = None) -> Dict[str, float]:
    """Each pilot's best `n` laps in a row over the event's time trials, the way FPVTrackside seeds (within one
    race), from every race's Race.json (its valid laps; races without video too). Only races started before
    `before`, without `exclude` (the race being flown or replayed); only `pilots` if given (a class)."""
    from race_day.trackside import parse_time, read_json
    mine = {p.lower(): p for p in pilots} if pilots else None
    rounds = getattr(event, "rounds", {}) or {}
    names = getattr(event, "pilot_names", {}) or {}
    out: Dict[str, float] = {}
    for d in Path(event.dir).iterdir():
        if d.name == exclude or not (d / "Race.json").is_file():
            continue
        race = read_json(d / "Race.json")
        race = race[0] if isinstance(race, list) and race else race
        if not isinstance(race, dict) or not race.get("Valid", True):
            continue
        if (rounds.get(race.get("Round")) or {}).get("EventType") != "TimeTrial":
            continue
        start = parse_time(race.get("Start"))
        if start is None or (before is not None and start >= before):
            continue
        dets = {x.get("ID"): x for x in race.get("Detections") or []}
        laps: Dict[str, List[Tuple[int, float]]] = {}
        for lap in race.get("Laps") or []:
            det = dets.get(lap.get("Detection")) or {}
            if not det.get("Valid", True) or (lap.get("LapNumber") or 0) < 1:
                continue
            laps.setdefault(det.get("Pilot"), []).append((lap["LapNumber"], float(lap.get("LengthSeconds") or 0)))
        for pid, ls in laps.items():
            name = names.get(pid)
            if not name or (mine is not None and name.lower() not in mine):
                continue
            name = mine[name.lower()] if mine else name
            best = best_consecutive([t for _, t in sorted(ls)], n)
            if best is not None and best < out.get(name, 1e9):
                out[name] = best
    return out


def standing(bests: Dict[str, float], pilot: str) -> Optional[int]:
    """A pilot's place in the qualifying order (1 = fastest), None without a time."""
    if pilot not in bests:
        return None
    return 1 + sum(1 for p, t in bests.items() if p != pilot and t < bests[pilot])


def fril_points(place: int, total: int) -> int:
    pts = total + 1 - place
    if place == 1:
        pts += math.ceil(total / 5)
    elif place == 2:
        pts += math.ceil(total / 8)
    elif place == 3:
        pts += math.ceil(total / 13)
    return max(1, pts)


def _names(lst) -> List[str]:
    return [p if isinstance(p, str) else (p.get("name") or "") for p in (lst or [])]


def bracket_order(stage: dict) -> List[Tuple[str, str]]:
    """A double elimination's pilots by how far they got: [(name, detail)]."""
    races = [r for r in stage.get("races") or [] if r.get("state") != "void"]
    last: Dict[str, Tuple[int, int]] = {}                 # pilot → (index of the last race flown, place in it)
    cta_wins: Dict[str, int] = {}
    cta_best: Dict[str, int] = {}
    for i, r in enumerate(races):
        if r.get("state") != "finished":
            continue
        order = _names(r.get("winners")) + [p for p in _names(r.get("losers")) if p not in _names(r.get("winners"))]
        order += [p for p in _names(r.get("pilots")) if p not in order]
        for k, p in enumerate(order):
            last[p] = (i, k)
            if "chase" in (r.get("group") or "").lower():
                cta_wins[p] = cta_wins.get(p, 0) + (1 if k == 0 else 0)
                cta_best[p] = min(cta_best.get(p, 99), k + 1)
    seeds = _names(stage.get("pilots"))
    for p in seeds:                                       # not flown yet: by seed, at the end
        last.setdefault(p, (-1, seeds.index(p)))
    champ = stage.get("champion")
    champ = champ if isinstance(champ, str) else (champ or {}).get("name") if champ else None

    def key(p):
        i, k = last[p]
        in_cta = p in cta_wins
        return (0 if p == champ else 1, 0 if in_cta else 1, -cta_wins.get(p, 0), cta_best.get(p, 99), -i, k)

    out = []
    for p in sorted(last, key=key):
        if p in cta_wins:
            detail = f"chase the ace: {cta_wins[p]} win{'s' if cta_wins[p] != 1 else ''}"
        elif last[p][0] >= 0:
            detail = f"{_ordinal(last[p][1] + 1)} in {races[last[p][0]].get('title') or races[last[p][0]].get('id')}"
        else:
            detail = "not flown yet"
        out.append((p, ("champion · " if p == champ else "") + detail))
    return out


def series_order(stage: dict) -> List[Tuple[str, str]]:
    """A series (points) main: FPVTrackside's own standings."""
    rows = sorted(stage.get("standings") or [], key=lambda s: s.get("rank") or 99)
    out = [(s.get("name") or "", s.get("detail") or "") for s in rows]
    seen = {n for n, _ in out}
    return out + [(p, "not flown yet") for p in _names(stage.get("pilots")) if p not in seen]


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
