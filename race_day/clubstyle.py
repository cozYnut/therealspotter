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
ORDINAL = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


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
        self.tracks = sorted(p.stem for p in (d / "tracks").glob("*.yaml")) if d and (d / "tracks").exists() else []
        self.track_name = track if track in self.tracks else None
        self.track = (_yaml(d / "tracks" / f"{self.track_name}.yaml") if self.track_name else None) or {}
        self._last: Dict[str, str] = {}

    # ── what to call things ──
    @property
    def name(self) -> str:
        return self.info.get("announcer") or "the announcer"

    def pilot(self, callsign: str, phonetic: Optional[str] = None) -> str:
        card = self.pilots.get(str(callsign).lower()) or {}
        return card.get("say_it_like") or phonetic or str(callsign)

    def is_new(self, callsign: str) -> bool:
        return bool((self.pilots.get(str(callsign).lower()) or {}).get("new"))

    def gate(self, n: Optional[int]) -> Optional[str]:
        """What the track calls gate n (1 = the start/finish gate); None when it has no name for it."""
        g = (self.track.get("gates") or {}).get(n) if n else None
        return (g.get("say") if isinstance(g, dict) else g) or None

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
    def margin(s: float) -> str:
        if s < 0.95:
            n = max(1, round(s * 10))
            return "a tenth of a second" if n == 1 else f"{n} tenths of a second"
        return f"{s:.1f} seconds"

    @staticmethod
    def place(n) -> str:
        return ORDINAL[n] if isinstance(n, int) and 0 < n < len(ORDINAL) else f"number {n}"

    # ── a call ──
    def line(self, kind: str, **words) -> Optional[str]:
        """One way to say `kind` with these words, or None (no line of that kind fits them)."""
        known = {k: v for k, v in words.items() if v not in (None, "")}
        fits = [t for t in self.lines.get(kind, []) if set(re.findall(r"\{(\w+)\}", t)) <= set(known)]
        if not fits:
            return None
        if len(fits) > 1 and self._last.get(kind) in fits:
            fits.remove(self._last[kind])
        t = random.choice(fits)
        self._last[kind] = t
        return t.format(**known)
