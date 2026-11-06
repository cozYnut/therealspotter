"""
The club's calendar (club.yaml: calendar_ics, an iCalendar feed — FRIL's is
the public Google Calendar behind fril.co.il/calendar): the next event, for
the announcer ("The next round: Race 2, 5 inch Open Spec, on Friday the 13th
of November, in Modi'in"), and where today's event is.

A venue often has no internet, so the feed is kept: clubs/<club>/calendar.ics
is written each time it's read, and used when it can't be.
"""

import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

CACHE = "calendar.ics"
USER_AGENT = "LazySpotter/2 (race-day app)"
TTL_S = 6 * 3600          # read again after this long
HEBREW = re.compile(r"[֐-׿]")
PRACTICE_WORDS = ("אימון", "practice", "training")
RACE_WORDS = ("race", "round", "סבב", "gq")


def fetch(url: str, folder: Path, log=print) -> Optional[str]:
    """The feed's text: fresh if it can be read (and kept), else the kept copy."""
    path = Path(folder) / CACHE
    if url and (not path.exists() or time.time() - path.stat().st_mtime > TTL_S):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as r:
                text = r.read().decode("utf-8", "replace")
            if "BEGIN:VCALENDAR" in text:
                path.write_text(text, encoding="utf-8")
                return text
        except (OSError, ValueError) as e:
            log(f"calendar: can't read it now ({e}); using the copy kept")
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def parse(text: str) -> List[dict]:
    """[{start (epoch), title, location}] in date order."""
    if not text:
        return []
    lines = []
    for raw in text.splitlines():                    # unfold continued lines
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    out, ev = [], None
    for l in lines:
        if l == "BEGIN:VEVENT":
            ev = {}
        elif l == "END:VEVENT" and ev is not None:
            if ev.get("start"):
                out.append(ev)
            ev = None
        elif ev is not None and ":" in l:
            key, val = l.split(":", 1)
            name = key.split(";")[0]
            val = val.replace("\\n", "\n").replace("\\,", ",").replace("\\;", ";").strip()
            if name == "DTSTART":
                ev["start"] = _time(val, "VALUE=DATE" in key)
            elif name == "SUMMARY":
                ev["title"] = val
            elif name == "LOCATION":
                ev["location"] = val
    return sorted(out, key=lambda e: e["start"])


def _time(v: str, date_only: bool) -> Optional[float]:
    try:
        if date_only or len(v) == 8:
            return datetime.strptime(v[:8], "%Y%m%d").timestamp()
        if v.endswith("Z"):
            return datetime.strptime(v, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        return datetime.strptime(v[:15], "%Y%m%dT%H%M%S").timestamp()
    except ValueError:
        return None


def is_race(ev: dict) -> bool:
    t = (ev.get("title") or "").lower()
    return any(w in t for w in RACE_WORDS) and not any(w in t for w in PRACTICE_WORDS)


def series(ev: Optional[dict]) -> str:
    """What a race belongs to, from its title: 'Race 2 - 5" Open Spec' → '5" open spec'."""
    t = (ev or {}).get("title") or ""
    return re.sub(r"(?i)^\s*(race|round)\s*\d+\s*[-–:]?\s*", "", t).strip().lower()


def next_event(events: List[dict], after: float, races_only: bool = True) -> Optional[dict]:
    """The first event after `after` (the end of that day): a race round (not a spare day), in the same series
    as the event on that day when the calendar has it; any event, with races_only off."""
    day_end = datetime.fromtimestamp(after).replace(hour=23, minute=59).timestamp()
    later = [e for e in events if e["start"] > day_end]
    if not races_only:
        return later[0] if later else None
    races = [e for e in later if is_race(e) and "spare" not in (e.get("title") or "").lower()]
    mine = series(on_day(events, after))
    same = [e for e in races if mine and series(e) == mine]
    return (same or races or [None])[0]


def on_day(events: List[dict], t: float) -> Optional[dict]:
    d = datetime.fromtimestamp(t).date()
    return next((e for e in events if datetime.fromtimestamp(e["start"]).date() == d), None)


def say_title(ev: dict) -> Optional[str]:
    """The title as the English voice can say it: 'Race 2 - 5" Open Spec' → 'Race 2, 5 inch Open Spec'; a
    Hebrew title only if it's a practice day."""
    t = (ev.get("title") or "").strip()
    if HEBREW.search(t):
        return "an open practice day" if any(w in t for w in PRACTICE_WORDS) else None
    t = re.sub(r'(\d)\s*(?:"|״|inch\b)', r"\1 inch", t).replace('"', "").replace("״", "")
    t = re.sub(r"\s+-\s+", ", ", t)
    return re.sub(r"\s+", " ", t).strip(" ,")


def say_place(ev: dict) -> Optional[str]:
    """Where, said simply: the first part of the location that isn't a street number or Hebrew
    ('Wingate Institute', "Modi'in")."""
    for part in re.split(r"[\n,]", ev.get("location") or ""):
        p = HEBREW.sub("", part).strip(" -")
        p = re.sub(r"\b\d{5,7}\b", "", p).strip()
        if not p or re.match(r"^\d", p) or p.lower() in ("israel",):
            continue
        return p.split("-")[0].strip() if "-" in p and "'" in p else p
    return None


def say_date(t: float) -> str:
    d = datetime.fromtimestamp(t)
    day = d.day
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{d.strftime('%A')} the {day}{suffix} of {d.strftime('%B')}"
