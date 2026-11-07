"""
A club's announcer style: what the announcer says, in the club's own words.

Every call the announcer makes is a line from the club's folder, never written
by a model — so it's instant, and it never says anything the club didn't
write. A club folder (clubs/<club>/, see clubs/example/):

  club.yaml          the club's name, the announcer's name, how lap times are
                     read, the safety line
  lines.yaml         the calls: for each kind, several ways to say it (one is
                     picked at random, never the one said last time)
  pilots.yaml        pilot cards: how to say a callsign, and whether a pilot is
                     new (kinder lines)
  tracks/<t>.yaml    a track: what to call its gates ("the dive gate"), the
                     usual crash spots, the expected lap time

A line names what it needs in braces: {pilot}, {lap}, {time}, {place},
{gate}, {margin}, {ahead}, {behind}, {next}. A line whose words aren't all
known for a call is skipped for another of that kind (so "{pilot} is down at
{gate}" is only said when the gate is known). Anything missing from the
club's files falls back to DEFAULT_LINES.
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CLUBS = Path(__file__).resolve().parent.parent / "clubs"

DEFAULT_LINES: Dict[str, List[str]] = {
    "start": ["Arm your quads. Starting on the tone in less than {time}"],
    "holeshot": ["{pilot} takes the holeshot"],
    "lap": ["{pilot}, lap {lap}, {time}, in {place}"],
    "lap_lead": ["{pilot}, lap {lap}, {time}, takes the lead"],
    "lap_leader": ["{pilot}, lap {lap}, {time}, still out front"],
    # a heat called by places only (brackets: the club's "finals" style), and by lap times only (qualifying)
    "lap_place": ["{pilot}, lap {lap}, in {place}", "{pilot} in {place}"],
    "lap_lead_place": ["{pilot} takes the lead on lap {lap}", "{pilot} takes the lead"],
    "lap_leader_place": ["{pilot}, lap {lap}, still out front", "{pilot} still leads"],
    "lap_time": ["{pilot}, lap {lap}, {time}", "{pilot}, {time}"],
    # brackets (Event setup: FPVTrackside's Seeder)
    "through": ["{pilot} goes through", "{pilot} is through to the next round"],
    "to_losers": ["{pilot} drops to the losers bracket", "{pilot} goes to the losers bracket, still in it"],
    "eliminated": ["That's the end of the day for {pilot}", "{pilot} is out of the bracket"],
    "ace_win": ["{pilot} takes a chase the ace win", "That's a win in the chase for {pilot}"],
    "final_win": ["{pilot} wins the main", "And {pilot} takes the final"],
    # the finish and what it means in one line (a 4-pilot race: 4 lines after the flag, not 8)
    "finish_win_through": ["{pilot} wins it and goes through"],
    "finish_win_ace_win": ["{pilot} wins it, a chase the ace win"],
    "finish_win_final_win": ["{pilot} wins the final"],
    "finish_through": ["{pilot} in {place}, goes through"],
    "finish_close_through": ["{pilot} in {place}, {margin} behind {ahead}, and goes through"],
    "finish_to_losers": ["{pilot} in {place}, drops to the losers bracket"],
    "finish_close_to_losers": ["{pilot} in {place}, {margin} behind {ahead}, drops to the losers bracket"],
    "finish_out": ["{pilot} in {place}, and that's the end of the day. Lots to build on"],
    "finish_close_out": ["{pilot} in {place}, just {margin} short, and that's the end of the day"],
    # qualifying: a new best of the class's seeding laps, and what it's worth in the standings
    "qual_first": ["{pilot} sets {time} for {laps} laps, {place} for now"],
    "qual_top": ["{pilot} goes top of the standings, {time} for {laps} laps"],
    "qual_up": ["{pilot}, new best {time}, up {moved} to {place}"],
    "qual_same": ["{pilot}, new best {time}, still {place}"],
    # the Auto Runner's countdown to the next race (FPVTrackside's "Announce next race in")
    "start_in": ["{time} to the next race", "Next race in {time}: {pilots}"],
    "start_in_first": ["{time} to the next race. {pilots}, to the start line please, video on"],
    "start_in_last": ["{time}! {pilots}, quads on the line"],
    "hurry_video": ["Hurry up {pilot}, we can't see your video"],
    "video_ok": ["{time} to go. Video looks good"],
    "last_lap": ["Last lap for {pilot}"],
    "fastest": ["{pilot}, fastest lap of the race, {time}"],
    "finish_win": ["{pilot} wins it"],
    "finish": ["{pilot} finishes {place}"],
    "finish_close": ["{pilot} finishes {place}, {margin} behind {ahead}"],
    "kind": ["{pilot} brings it home"],
    "kind_new": ["{pilot} brings it home, well flown"],
    "practice_lap": ["{pilot}, {time}"],
    "practice_fastest": ["{pilot}, {time}, fastest lap"],
    "race_over": ["Race over"],
    "video_lead": ["{pilot} takes the lead at {gate}", "{pilot} takes the lead"],
    "video_place": ["{pilot} moves up to {place}"],
    "gap": ["{margin} between {ahead} and {behind} at {gate}", "{margin} between {ahead} and {behind}"],
    "down": ["{pilot} is down at {gate}, so {next} moves up to {place}", "{pilot} is down, so {next} moves up to {place}",
             "{pilot} is down"],
    "back": ["{pilot} is back in it"],
    "out": ["{pilot} is out"],
    "safety": ["Stop. Pilots, land now."],
}
ORDINAL = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
           "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
           "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
           "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth",
           "thirtieth"]


def speakable(callsign: str) -> str:
    """A callsign the voice can say, when nobody wrote how: "Shauli_fpv" → "Shauli F P V", "DeanFpv" → "Dean F P V"."""
    s = re.sub(r"(?i)[_.\-]?fpv$", " F P V", callsign)
    s = re.sub(r"[_]+", " ", s)
    return s.strip() or callsign


def clubs() -> List[str]:
    """The club folders there are (example first)."""
    if not CLUBS.exists():
        return []
    names = sorted(p.name for p in CLUBS.iterdir() if p.is_dir() and (p / "club.yaml").exists())
    return sorted(names, key=lambda n: n != "example")


def _yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    except (yaml.YAMLError, OSError):
        return None



GO = re.compile(r"\b(here we go|let'?s go(?!\s+(?:through|over|to)\b)|let'?s do this|let'?s start|let'?s race|"
                r"go go go)\b", re.I)


def _go_ok(lines: List[str], go: bool) -> List[str]:
    """Without a start in a moment: the lines with no "here we go" — or, if every line has one, the lines with
    that sentence taken out."""
    if go:
        return lines
    plain = [t for t in lines if not GO.search(t)]
    if plain:
        return plain
    cut = [re.sub(r"\s+", " ", re.sub(r"[^.!?]*" + GO.pattern + r"[^.!?]*[.!?]*", "", t, flags=re.I)).strip()
           for t in lines]
    return [t for t in cut if re.search(r"\w", t)]

class ClubStyle:
    def __init__(self, club: Optional[str] = None, track: Optional[str] = None):
        self.club = club
        d = CLUBS / club if club else None
        self.info = (_yaml(d / "club.yaml") if d else None) or {}
        lines = (_yaml(d / "lines.yaml") if d else None) or {}
        self.lines = {k: list(v if isinstance(v, list) else [v]) for k, v in {**DEFAULT_LINES, **lines}.items() if v}
        if self.info.get("safety_line"):
            self.lines["safety"] = [self.info["safety_line"]]
        self.pilots = {}
        for p in (_yaml(d / "pilots.yaml") if d else None) or []:
            if isinstance(p, dict) and p.get("callsign"):
                self.pilots[str(p["callsign"]).lower()] = p
        talk = (_yaml(d / "talk.yaml") if d else None) or {}     # between races (race_day/talk.py)
        self.talk = {k: list(v if isinstance(v, list) else [v]) for k, v in talk.items() if v}
        self.tracks = sorted(p.stem for p in (d / "tracks").glob("*.yaml")) if d and (d / "tracks").exists() else []
        self.track_name = track if track in self.tracks else None
        self.track = (_yaml(d / "tracks" / f"{self.track_name}.yaml") if self.track_name else None) or {}
        self._last: Dict[str, str] = {}
        self.sections: List[dict] = []            # the event's named sections (lazy_spotter_v2 sets them per heat)

    # ── what to call things ──
    @property
    def name(self) -> str:
        return self.info.get("announcer") or "the announcer"

    def pilot(self, callsign: str, phonetic: Optional[str] = None) -> str:
        card = self.pilots.get(str(callsign).lower()) or {}
        return card.get("say_it_like") or phonetic or speakable(str(callsign))

    def is_new(self, callsign: str) -> bool:
        return bool((self.pilots.get(str(callsign).lower()) or {}).get("new"))

    def gate(self, n: Optional[int]) -> Optional[str]:
        """What the track calls gate n (1 = the start/finish gate): the venue's track file, else the event's
        named section it's in (Event setup → Sections: set_sections); None when nothing names it."""
        g = (self.track.get("gates") or {}).get(n) if n else None
        name = (g.get("say") if isinstance(g, dict) else g) or None
        if name or not n:
            return name
        for x in self.sections:
            a, b = int(x["from"]), int(x["to"])
            inside = a < n <= b if a < b else (n > a or n <= b)
            if inside:
                return x["name"]
        return None

    def set_sections(self, sections: Optional[List[dict]]):
        self.sections = list(sections or [])

    def crash_spot(self, n: Optional[int]) -> bool:
        g = (self.track.get("gates") or {}).get(n) if n else None
        return bool(isinstance(g, dict) and g.get("crash_spot"))

    def expected_lap(self) -> Optional[float]:
        v = self.track.get("expected_lap")
        return float(v) if isinstance(v, (int, float)) else None

    def lap_time(self, s: float, decimals: int = 2) -> str:
        """21.96 → "21.96" (natural: "twenty-one point nine six") or "21 96" (digits: "twenty-one ninety-six")."""
        text = f"{s:.{decimals}f}"
        if self.info.get("lap_times") == "digits" and "." in text:
            whole, frac = text.split(".")
            return f"{whole} {frac}"
        return text

    @staticmethod
    def countdown(s: float) -> str:
        """Seconds to the start, said: 120 → "2 minutes", 90 → "a minute and a half", 45 → "45 seconds"."""
        s = int(round(s / 5.0) * 5) if s > 15 else int(round(s))
        if s >= 60:
            m, r = divmod(s, 60)
            if r == 30:
                return "a minute and a half" if m == 1 else f"{m} and a half minutes"
            if r < 10:
                return "one minute" if m == 1 else f"{m} minutes"
            return f"{m} minute{'s' if m > 1 else ''} {r} seconds"
        return f"{s} seconds"

    @staticmethod
    def margin(s: float) -> str:
        if s < 0.95:
            n = max(1, round(s * 10))
            return "a tenth of a second" if n == 1 else f"{n} tenths of a second"
        return f"{s:.1f} seconds"

    @staticmethod
    def place(n) -> str:
        return ORDINAL[n] if isinstance(n, int) and 0 < n < len(ORDINAL) else f"number {n}"

    # ── a call ──
    def talk_line(self, kind: str, go: bool = True, **words) -> Optional[str]:
        """One of talk.yaml's ways to say `kind` that these words fill, or None. go: something starts now (else
        no "here we go")."""
        known = {k: v for k, v in words.items() if v not in (None, "")}
        fits = _go_ok([t for t in self.talk.get(kind, []) if set(re.findall(r"\{(\w+)\}", t)) <= set(known)], go)
        if not fits:
            return None
        most = max(len(set(re.findall(r"\{(\w+)\}", t))) for t in fits)   # the lines that say the most we know
        fits = [t for t in fits if len(set(re.findall(r"\{(\w+)\}", t))) == most]
        key = "talk:" + kind
        if len(fits) > 1 and self._last.get(key) in fits:
            fits.remove(self._last[key])
        t = random.choice(fits)
        self._last[key] = t
        return t.format(**known)

    # ── a call ──
    def line(self, kind: str, go: bool = True, **words) -> Optional[str]:
        """One way to say `kind` with these words, or None (no line of that kind fits them). go: a race starts
        in a moment (else no "here we go")."""
        known = {k: v for k, v in words.items() if v not in (None, "")}
        fits = _go_ok([t for t in self.lines.get(kind, []) if set(re.findall(r"\{(\w+)\}", t)) <= set(known)], go)
        if not fits:
            return None
        if len(fits) > 1 and self._last.get(kind) in fits:
            fits.remove(self._last[kind])
        t = random.choice(fits)
        self._last[kind] = t
        return t.format(**known)
