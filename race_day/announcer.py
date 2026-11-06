"""
The race announcer: one English voice for the whole race, in place of
FPVTrackside's, in the club's own words (race_day/clubstyle.py: clubs/<club>/).

Three sources feed it:

  FPVTrackside's events   (race_day/ftevents.py) the race calls: the start ("Arm
                          your quads…", the moment FPVTrackside announces it), the
                          holeshot, every lap with its time and place, who takes
                          the lead, the fastest lap, the last lap, who finishes
                          where and by how much, a pilot out, "Race over".
  the live video          (race_day/livevideo.py: FPVTrackside's capture card) where
                          each pilot is mid-lap: who takes the lead at which gate,
                          the gap between the first two at a gate, who is down (so
                          who moves up), who is back.
  FPVTrackside's log      everything else FPVTrackside decides to say — "Next up…",
                          "In the hole…", "Hurry up …", results — where it writes
                          every spoken line ("SpeechManager: TTS (…)"); said as it is.

A logged line that is one of the calls made here is not said twice: it is
recognised by WHICH of FPVTrackside's sounds it is (matched against the wording
in its Sounds.xml), never by its text, which differs from ours. The venue's
choices in Sounds.xml count for both: a sound switched off there isn't said.

Safety first: FPVTrackside's emergency stop or stand-down is said as it is, at
once, everything waiting is dropped, and the race calls stop until the next race.

FPVTrackside's own voice is switched off by its TextToSpeechVolume = 0
(ProfileSettings.xml; read when it starts, and written over when it exits, so
only changed while it isn't running): its start tone and horn still play, and
it still writes what it would have said to its log. set_ft_speech() does that.

Speaking is macOS's `say`, one line at a time: the start first, then lap
calls, then places, then the rest; a call that waited too long is dropped
rather than said late. No call is written by a model.
"""

import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from xml.etree import ElementTree as ET

from race_day.clubstyle import ClubStyle

FT_HOME = Path.home() / "Documents" / "FPVTrackside"     # FPVTrackside's working folder on a Mac
P_START, P_LAP, P_PLACE, P_INFO = 0, 1, 2, 3              # speaking order
MAX_WAIT = {P_START: 6.0, P_LAP: 4.0, P_PLACE: 3.0, P_INFO: 10.0}   # said this late at most
STALE_S = 5.0            # an event older than this (FPVTrackside's own clock) is history, not news
LEAD_MARGIN_S = 0.5      # a new leader through the same gate: this far ahead at least (or a gate ahead)
HOLD_S = 1.5             # a new place from the video: held this long before it's called …
LEAD_EVERY_S = 5.0       # … a lead call at most this often, …
PLACE_EVERY_S = 3.0      # … any place call at most this often
GAP_CALL_S = 1.0         # the first two within this at a gate: the margin is called …
GAP_EVERY_S = 10.0       # … at most this often
DOWN_MIN_S = 15.0        # no gate seen this long (at least; ¾ of a lap) and the pilot is down
SETTLE_S = 6.0           # no place calls from the video this soon after the start (the pack sorts itself out)
MAX_STEP = 4             # a gate from the video counts at most this many gates on from the last (missed ones) …
STEADY = 2               # a pilot's video gates count for a gap or a place only after this many in order
MIN_PACE = 0.6           # … and only if the pilot had time to fly there: this share of the learned time at least
RACE_TYPES = ("Race", "Endurance", "AggregateLaps")      # places count; otherwise lap times do

# FPVTrackside's sounds that are called here from the events; the rest are said from its log (a time trial's
# "Pilot 2 laps in 45.3" and "… finished", a casual practice's laps: as FPVTrackside words them)
OWNED = {"StartRaceIn", "RaceOver", "Holeshot", "RaceLap", "RaceDone", "Sector", "PracticeLap", "TimeTrialEveryLap"}
START_SOUNDS = {"StartRaceIn8", "StartRaceIn5", "StartRaceIn4", "StartRaceIn3", "StartRaceIn2", "StartRaceIn1",
                "StaggeredStart", "StaggeredPilot"}
SAFETY_SOUNDS = {"EmergencyStop", "StandDownCancelled", "StandDownTimingSystem"}


def _ts(e: dict) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError):
        return None


def channel_label(ch) -> Optional[str]:
    """FPVTrackside's channel of a pilot ({"band": "Raceband", "number": 1, …}) as the event's channels are
    named (R1) — how the live video's pictures are labelled."""
    if not isinstance(ch, dict):
        return str(ch) if ch else None
    if ch.get("displayName"):
        return ch["displayName"]
    band, num = str(ch.get("band") or ""), ch.get("number")
    return f"{band[:1].upper()}{ch.get('channelPrefix') or ''}{num}" if band and num is not None else None


# ── speaking ───────────────────────────────────────────────────

def english_voices() -> List[str]:
    """The Mac's English voices, as `say -v ?` lists them."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [m.group(1).strip() for m in re.finditer(r"^(.+?)\s+(en_[A-Z]{2})\s+#", out, re.M)]


class Speaker:
    """One line at a time, in priority order (the earlier of two the same)."""

    def __init__(self, voice: Optional[str] = None, rate: Optional[int] = None):
        self.voice, self.rate = voice, rate
        self.q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._n = 0
        self.speaking = False
        self.last: Optional[str] = None
        threading.Thread(target=self._run, daemon=True).start()

    def clear(self):
        """Nothing waiting is said (a safety stop)."""
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def say(self, text: str, prio: int = P_INFO):
        self._n += 1
        self.q.put((prio, self._n, text, time.time() + MAX_WAIT.get(prio, 10.0)))

    def _run(self):
        while True:
            prio, _, text, until = self.q.get()
            if time.time() > until:
                continue
            cmd = ["say"] + (["-v", self.voice] if self.voice else []) + (["-r", str(self.rate)] if self.rate else [])
            self.speaking, self.last = True, text
            try:
                subprocess.run(cmd + [text], timeout=20)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                self.speaking = False


# ── FPVTrackside's sounds and its log ──────────────────────────

# what each placeholder of FPVTrackside's wording can be (anything else: some words)
_SLOT = {"laptime": r"[\d:.]+", "lapstime": r"[\d:.]+", "count": r"\d+", "lapnumber": r"\d+", "points": r"\d+",
         "position": r"\S+", "s": r"s?"}
_MAYBE_EMPTY = {"bracket", "type"}       # "Next up Round 9 Race 3 ." has no bracket: the word and its space go


def _pattern(text: str) -> str:
    parts = re.split(r"\{(\w+)\}", text)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out.append(re.escape(p))
        elif p in _MAYBE_EMPTY and out and out[-1].endswith(r"\ "):
            out[-1] = out[-1][:-2]
            out.append(r"(?:\s+.+?)?")
        else:
            out.append(f"(?:{_SLOT.get(p, '.+?')})")
    return "".join(out)


class Sounds:
    """FPVTrackside's Sounds.xml: which sounds are on, and which sound a spoken line is."""

    def __init__(self, path: Optional[Path]):
        self.enabled: Dict[str, bool] = {}
        self.files: Dict[str, str] = {}          # key → its sound file, relative to FPVTrackside's folder
        self.matchers = []                       # (literal characters, key, regex)
        if not path or not path.exists():
            return
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return
        for s in root:
            key, text = s.findtext("Key") or "", (s.findtext("TextToSpeech") or "").strip().rstrip(". ")
            self.enabled[key] = (s.findtext("Enabled") or "").lower() == "true"
            if (s.findtext("Filename") or "").strip():
                self.files[key] = s.findtext("Filename").strip()
            if not text:
                continue
            literal = sum(len(p) for p in re.split(r"\{\w+\}", text))
            self.matchers.append((literal, key, re.compile(rf"^{_pattern(text)}$", re.I)))
        self.matchers.sort(key=lambda m: -m[0])  # the most specific wording first

    def on(self, key: str) -> bool:
        return self.enabled.get(key, True)

    def which(self, line: str) -> Optional[str]:
        line = line.strip().rstrip(". ")
        return next((key for _, key, rx in self.matchers if rx.match(line)), None)


_TTS = re.compile(r"SpeechManager: TTS \((.*)\)\s*$")
_STAMP = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d(?:\.\d+)?)")


class LogFollower:
    """Every line FPVTrackside speaks, from the end of its log on: fn(text)."""

    def __init__(self, path: Path, fn: Callable[[str], None]):
        self.path, self.fn = path, fn
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        pos = self.path.stat().st_size if self.path.exists() else 0
        while not self._stop.wait(0.2):
            try:
                size = self.path.stat().st_size
            except OSError:
                continue
            if size < pos:                           # a new log (FPVTrackside restarted)
                pos = 0
            if size == pos:
                continue
            with open(self.path, "rb") as f:
                f.seek(pos)
                chunk = f.read(size - pos)
            end = chunk.rfind(b"\n") + 1             # a half-written line waits for the next read
            pos += end
            for raw in chunk[:end].decode("utf-8", "replace").splitlines():
                m = _TTS.search(raw)
                if not m:
                    continue
                st = _STAMP.match(raw)
                if st:                               # its own clock: a line from long ago is no news
                    try:
                        t = datetime.strptime(st.group(1)[:23], "%Y/%m/%d %H:%M:%S.%f" if "." in st.group(1)
                                              else "%Y/%m/%d %H:%M:%S").timestamp()
                        if time.time() - t > 15:
                            continue
                    except ValueError:
                        pass
                self.fn(m.group(1).strip())


# ── FPVTrackside's own voice ───────────────────────────────────

_VOL = re.compile(r"(<TextToSpeechVolume>)(\d+)(</TextToSpeechVolume>)")


def ft_running() -> bool:
    return subprocess.run(["pgrep", "-f", "FPVTrackside.app/Contents/MacOS"], capture_output=True).returncode == 0


def ft_speech(profile_dir: Path) -> Optional[bool]:
    """Does FPVTrackside speak (its TextToSpeechVolume > 0)? None: no settings file."""
    p = profile_dir / "ProfileSettings.xml"
    m = _VOL.search(p.read_text(encoding="utf-8")) if p.exists() else None
    return int(m.group(2)) > 0 if m else None


def set_ft_speech(profile_dir: Path, on: bool) -> str:
    """FPVTrackside's voice on (100) or off (0). Only while FPVTrackside is closed: it reads the file when it
    starts and writes its own copy back when it exits."""
    p = profile_dir / "ProfileSettings.xml"
    if not p.exists():
        return f"FPVTrackside's settings aren't at {p}"
    if ft_running():
        return "Close FPVTrackside first: it writes its settings back when it exits."
    text = p.read_text(encoding="utf-8")
    if not _VOL.search(text):
        return "FPVTrackside's settings have no speech volume — has it run once?"
    p.write_text(_VOL.sub(lambda m: m.group(1) + ("100" if on else "0") + m.group(3), text), encoding="utf-8")
    return "FPVTrackside speaks again." if on else "FPVTrackside's voice is off: the app speaks the calls."


# ── the announcer ──────────────────────────────────────────────

class Announcer:
    """Listens to FPVTrackside's events (EventRelay.subscribe(announcer.on_event)) and the live video
    (LiveVideo(announcer.on_video)) and, when on, speaks. on_race(running) is told when a race starts and
    ends (the live video analyses only then)."""

    def __init__(self, log: Callable[[str], None] = print, voice: Optional[str] = None, rate: Optional[int] = None,
                 enabled: bool = False, club: Optional[str] = None, track: Optional[str] = None):
        self.log = log
        self.speaker = Speaker(voice, rate)
        self.style = ClubStyle(club, track)
        self.enabled = False
        self.ft_dir = FT_HOME
        self.profile_dir = FT_HOME / "data" / "Profile 1"
        self.sounds = Sounds(self.profile_dir / "Sounds.xml")
        self.decimals = 2
        self.n_gates = 0                          # the track's gates (the live video's learned track)
        self.leg_s = 0.0                          # its learned lap, in seconds
        self.legs: List[float] = []               # its learned seconds from each gate to the next
        self.follower: Optional[LogFollower] = None
        self.on_race: Optional[Callable[[bool], None]] = None
        self.heat_info: Optional[Callable[[dict], dict]] = None   # RaceLoaded → the heat's setup (lazy_spotter_v2)
        self.lock = threading.RLock()
        self.calls: List[str] = []               # the last calls made (for the screen)
        self._new_race({})
        threading.Thread(target=self._tick, daemon=True).start()
        self.set_enabled(enabled)

    # ── settings ──
    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        if self.follower:
            self.follower.stop()
            self.follower = None
        if self.enabled:
            self.follower = LogFollower(self.ft_dir / "log" / "Log.txt", self._logged)

    def set_voice(self, voice: Optional[str], rate: Optional[int] = None):
        self.speaker.voice, self.speaker.rate = voice or None, rate or None

    def set_style(self, club: Optional[str], track: Optional[str] = None):
        with self.lock:
            self.style = ClubStyle(club, track)

    def set_track(self, n_gates: int, lap_s: float = 0.0, legs: Optional[List[float]] = None):
        """The learned track the live video names gates on: its gate count, its usual lap, and the learned
        seconds from each gate to the next (G1→G2 first)."""
        self.n_gates, self.leg_s = int(n_gates or 0), float(lap_s or 0.0)
        self.legs = list(legs or [])

    def _flight(self, g0: int, d: int) -> float:
        """The learned seconds from gate g0 to d gates on (0 when the track's times aren't known)."""
        if len(self.legs) != self.n_gates:
            return 0.0
        return sum(self.legs[(g0 - 1 + k) % self.n_gates] for k in range(d))

    def play_sound(self, key: str):
        """Play one of FPVTrackside's own sound files (RaceStart: its start tone), as FPVTrackside would.
        For a replayed heat only: on race day FPVTrackside plays them itself, also with its voice off."""
        f = self.sounds.files.get(key)
        if not (self.enabled and f and self.sounds.on(key)):
            return
        path = Path(f) if os.path.isabs(f) else FT_HOME / f
        if path.exists():
            subprocess.Popen(["afplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test(self):
        self.speaker.say(self.style.line("lap_lead", pilot="Pennz", lap=2, time=self.style.lap_time(21.96))
                         or "Pennz, lap 2, 21.96, takes the lead", P_LAP)

    def status(self) -> dict:
        with self.lock:
            return {"enabled": self.enabled, "race": self.race.get("label"), "running": self.running,
                    "speaking": self.speaker.speaking, "last": self.calls[-1] if self.calls else None,
                    "voice": self.speaker.voice, "ft_speech": ft_speech(self.profile_dir),
                    "club": self.style.club, "track": self.style.track_name, "silenced": self.silenced,
                    "log": str(self.ft_dir / "log" / "Log.txt")}

    # ── calls ──
    def _say(self, kind: str, prio: int, **words) -> bool:
        text = self.style.line(kind, **words)
        if not text:
            return False
        self.calls = (self.calls + [text])[-30:]
        if self.enabled:
            self.speaker.say(text, prio)
        return True

    def _name(self, pilot) -> str:
        if isinstance(pilot, dict):
            callsign = pilot.get("name") or "a pilot"
            return self.style.pilot(callsign, (pilot.get("phonetic") or "").strip(" '\"") or None)
        return self.style.pilot(pilot, self.race["say"].get(pilot))

    def _new_race(self, e: dict, info: Optional[dict] = None):
        """info (heat_info): the heat's class, kind (qualifying / finals), call style (times / places / both),
        and in a bracket its group (Winners / Losers / Chase the Ace) and how many go through."""
        info = info or {}
        pilots, chans = {}, {}
        for p in e.get("pilots") or []:
            p = p.get("pilot", p) if isinstance(p, dict) else {"name": p}
            if p.get("name"):
                pilots[p["name"]] = (p.get("phonetic") or "").strip(" '\"") or None
                lab = channel_label(p.get("channel"))
                if lab:
                    chans[lab] = p["name"]
        self.race = {"label": f"Round {e.get('round')} race {e.get('race')}" if e else None,
                     "type": e.get("raceType") or "Race", "target": e.get("targetLaps") or 0, "say": pilots,
                     "channels": chans, "style": info.get("style") or "both", "kind": info.get("kind"),
                     "group": info.get("group"), "advance": info.get("advance"), "class": info.get("class"),
                     "ace_wins": info.get("ace_wins")}
        self.running = False
        self.silenced = False
        self.leader: Optional[str] = None         # who leads, as last called
        self.best: Dict[str, float] = {}          # each pilot's best lap so far this race
        self.finished: List[tuple] = []           # (pilot, race time) in finishing order
        self.holeshot = False
        self.out: set = set()
        self.laps: Dict[str, int] = {}            # laps done (LapRF)
        self.vid: Dict[str, dict] = {}            # pilot → where the video and LapRF put them: lap, gate, prog, …
        self.started_at: Optional[float] = None
        self.places: List[str] = []               # the order as last called (video)
        self.pending: Optional[tuple] = None      # (order, since) not yet called
        self.called = {"lead": 0.0, "place": 0.0, "gap": 0.0}

    def on_event(self, e: dict):
        kind = e.get("type")
        info = None
        if kind == "RaceLoaded" and self.heat_info:          # asked outside the lock: it may ask FPVTrackside
            try:
                info = self.heat_info(e)
            except Exception as ex:              # noqa: BLE001
                self.log(f"announcer: the heat's setup: {ex}")
        with self.lock:
            if kind == "Hello":
                self._hello(e)
            elif kind == "RaceLoaded":
                self._new_race(e, info)
            elif kind == "RaceStartAnnouncement":
                if self.sounds.on("StartRaceIn") and self._fresh(e):
                    secs = e.get("delaySeconds")
                    self._say("start", P_START, time=round(secs) if isinstance(secs, (int, float)) else 5)
            elif kind == "RaceStart":
                self.running, self.started_at = True, time.time()
                self._race_hook(True)
            elif kind in ("RaceEnd", "RaceCancelled"):
                if kind == "RaceEnd" and self.running and not self.silenced and self.sounds.on("RaceOver"):
                    self._say("race_over", P_INFO)
                self.running = False
                self._race_hook(False)
            elif kind == "PilotCrashedOut" and self.running and not self.silenced:
                p = e.get("pilot") or e.get("name")
                name = p.get("name") if isinstance(p, dict) else p
                if name and name not in self.out:
                    self.out.add(name)
                    self._say("out", P_LAP, pilot=self._name(p))
                    self._say("kind_new" if self.style.is_new(name) else "kind", P_INFO, pilot=self._name(p))
            elif kind == "DetectionExt" and self.running and not self.silenced and e.get("valid", True) \
                    and self._fresh(e):
                self._detection(e)

    def _race_hook(self, on: bool):
        if self.on_race:
            try:
                self.on_race(on)
            except Exception as ex:              # noqa: BLE001
                self.log(f"announcer: {ex}")

    def _fresh(self, e: dict) -> bool:
        t = _ts(e)
        return t is None or time.time() - t < STALE_S

    def _hello(self, e: dict):
        paths = e.get("paths") or {}
        work = Path(paths["workingDirectory"]) if paths.get("workingDirectory") else FT_HOME
        prof = paths.get("profileDirectory")
        self.ft_dir = work
        self.profile_dir = (work / prof if prof and not os.path.isabs(prof) else Path(prof)) if prof \
            else work / "data" / ((e.get("profile") or {}).get("name") or "Profile 1")
        self.sounds = Sounds(self.profile_dir / "Sounds.xml")
        if isinstance(e.get("decimalPlaces"), int):
            self.decimals = e["decimalPlaces"]
        self.set_enabled(self.enabled)           # follow the log where this FPVTrackside writes it

    def _detection(self, e: dict):
        pilot, n = e.get("pilotName"), e.get("lapNumber")
        if not pilot or not isinstance(n, int) or not e.get("isLapEnd", True):
            return                               # a split gate: no call
        name = self._name(pilot)
        if n == 0:                               # the holeshot
            self._anchor(pilot, 0, _ts(e))
            if not self.holeshot and self.sounds.on("Holeshot"):
                self._say("holeshot", P_LAP, pilot=name)
            self.holeshot = True
            return
        self.laps[pilot] = n
        self._anchor(pilot, n, _ts(e))
        lap = e.get("lapTimeSoFar") if e.get("lapTimeSoFar") is not None else e.get("sectorTime")
        t = self.style.lap_time(lap, self.decimals) if isinstance(lap, (int, float)) else None
        # the fastest lap of the race so far (not a pilot's own best: lap 2 nearly always beats lap 1)
        fastest = isinstance(lap, (int, float)) and bool(self.best) and lap < min(self.best.values())
        if isinstance(lap, (int, float)):
            self.best[pilot] = min(lap, self.best.get(pilot, 1e9))
        if self.race["type"] not in RACE_TYPES:  # practice, time trial: the lap time is the news
            key = "TimeTrialEveryLap" if self.race["type"] == "TimeTrial" else "PracticeLap"
            if t and self.sounds.on(key):
                self._say("practice_fastest" if fastest else "practice_lap", P_LAP, pilot=name, time=t) or \
                    self._say("practice_lap", P_LAP, pilot=name, time=t)
            return
        pos = e.get("position")
        leader = next((s.get("pilotName") for s in e.get("positionSnapshot") or [] if s.get("position") == 1), None)
        if e.get("raceFinishedForPilot"):
            self._finish(pilot, name, e.get("raceTime"))
            self.leader = self.leader or leader
            return
        if not self.sounds.on("RaceLap"):
            self.leader = self.leader or leader
            return
        style = self.race.get("style") or "both"
        words = dict(pilot=name, lap=n, time=t, place=self.style.place(pos) if isinstance(pos, int) else None)
        if style == "times":                     # the club calls this heat by lap times: no places
            self._say("lap_time", P_LAP, **dict(words, place=None))
            if fastest:
                self._say("fastest", P_PLACE, pilot=name, time=t)
            if pos == 1 and self.race["target"] and n == self.race["target"] - 1:
                self._say("last_lap", P_LAP, pilot=name)
            self.leader = self.leader or leader
            return
        alt = "_place" if style == "places" else ""   # places only: the lines without a lap time
        if alt:
            words["time"] = None
        if pos == 1 and leader == pilot and self.leader not in (None, pilot):
            self._say("lap_lead" + alt, P_LAP, **words) or self._say("lap_lead", P_LAP, **words) or \
                self._say("lap" + alt, P_LAP, **words)
            self.called["lead"], self.leader = time.time(), pilot
        elif pos == 1:
            self._say("lap_leader" + alt, P_LAP, **words) or self._say("lap_leader", P_LAP, **words) or \
                self._say("lap" + alt, P_LAP, **words)
        else:
            self._say("lap" + alt, P_LAP, **words) or self._say("lap", P_LAP, **words)
        if fastest and style != "places":
            self._say("fastest", P_PLACE, pilot=name, time=t)
        if pos == 1 and self.race["target"] and n == self.race["target"] - 1:
            self._say("last_lap", P_LAP, pilot=name)
        self.leader = self.leader or leader

    def _finish(self, pilot: str, name: str, race_t):
        if pilot in (p for p, _ in self.finished):
            return
        self.finished.append((pilot, race_t))
        if not self.sounds.on("RaceDone"):
            return
        k = len(self.finished)
        prev = self.finished[-2] if k > 1 else None
        margin = race_t - prev[1] if prev and isinstance(race_t, (int, float)) and isinstance(prev[1], (int, float)) \
            else None
        if k == 1:
            self._say("finish_win", P_LAP, pilot=name)
        elif margin is not None and margin < GAP_CALL_S * 1.5:
            self._say("finish_close", P_LAP, pilot=name, place=self.style.place(k), margin=self.style.margin(margin),
                      ahead=self._name(prev[0])) or self._say("finish", P_LAP, pilot=name, place=self.style.place(k))
        else:
            self._say("finish", P_LAP, pilot=name, place=self.style.place(k))
        self._bracket_call(pilot, name, k)
        if k == len(self.race["say"]) and k > 1:   # the last one home gets a kind word
            self._say("kind_new" if self.style.is_new(pilot) else "kind", P_INFO, pilot=name) or \
                self._say("kind", P_INFO, pilot=name)

    def _bracket_call(self, pilot: str, name: str, k: int):
        """In a bracket (Event setup): what this finishing place means — through, to the losers bracket, out, or
        a chase the ace win."""
        group = (self.race.get("group") or "").lower()
        adv = self.race.get("advance")
        if not group or self.race.get("kind") != "finals":
            return
        if "chase" in group:
            if k == 1:                             # a final of one race (the other mains) or a chase the ace win
                if self.race.get("ace_wins") == 1:
                    self._say("final_win", P_INFO, pilot=name) or self._say("ace_win", P_INFO, pilot=name)
                else:
                    self._say("ace_win", P_INFO, pilot=name)
            return
        if not isinstance(adv, int) or adv <= 0:
            return
        if k <= adv:
            self._say("through", P_INFO, pilot=name)
        elif "winner" in group:
            self._say("to_losers", P_INFO, pilot=name)
        elif "loser" in group:
            self._say("eliminated", P_INFO, pilot=name)

    def _logged(self, text: str):
        """A line FPVTrackside spoke (into its log): said here unless it's a call made from the events."""
        key = self.sounds.which(text)
        if key in OWNED:
            return
        if key in SAFETY_SOUNDS:                 # safety first: said at once and alone, the race calls stop
            with self.lock:
                self.silenced = True
            self.speaker.clear()
            self.calls = (self.calls + [text])[-30:]
            if self.enabled:
                self.speaker.say(text, P_START)
            return
        self.calls = (self.calls + [text])[-30:]
        if self.enabled:
            self.speaker.say(text, P_START if key in START_SOUNDS else P_INFO)

    # ── the live video: where each pilot is mid-lap ──
    def _v(self, pilot: str) -> dict:
        return self.vid.setdefault(pilot, {"gate": None, "prog": -1, "t": None, "last": time.time(), "down": False,
                                           "gray": False, "anchored": False, "cand": None, "lap": 0, "steady": 0})

    def _anchor(self, pilot: str, lap: int, t: Optional[float]):
        """LapRF saw the pilot cross the start/finish: lap `lap` starts at G1. The video's place follows it,
        unless the video already has the pilot a gate or two on."""
        v = self._v(pilot)
        v["anchored"], v["last"], v["lap"] = True, time.time(), lap
        at = lap * max(1, self.n_gates)
        if not at <= v["prog"] <= at + 2:
            v.update(gate=1, prog=at, t=t or time.time(), cand=None, steady=STEADY)

    def on_video(self, e: dict):
        """An event of the live video (race_day/livevideo.py): a gate named, a picture lost or back.

        A pilot's place on the track (prog: gates since the start) moves on by a video gate only when it's a
        few gates on (MAX_STEP) and there was time to fly there (MIN_PACE of the learned time): the live
        names wobble. Two passes in a row that agree with each other count even if they don't fit the last
        one, so a pilot is never stuck; LapRF's crossings put it right each lap."""
        with self.lock:
            pilot = self.race["channels"].get(e.get("channel"))
            if not pilot or not self.running:
                return
            v = self._v(pilot)
            if e.get("type") == "picture":
                v["gray"] = bool(e.get("gray"))
                return
            if e.get("type") != "gate" or not e.get("gate") or not self.n_gates:
                return                               # judged not a real pass
            v["last"] = time.time()                  # flying, whatever it counts for
            if v["down"]:
                v["down"] = False
                self._say("back", P_PLACE, pilot=self._name(pilot))
            if not v["anchored"] and time.time() - (self.started_at or 0) < 10:
                return                               # no start crossing yet: on the pad, or the holeshot to come
            g, t = int(e["gate"]), float(e["t"])

            def fits(g0, t0, most=MAX_STEP):
                d = (g - g0) % self.n_gates
                return d if 1 <= d <= most and (t0 is None or t - t0 >= MIN_PACE * self._flight(g0, d)) else 0

            g0 = v["gate"] or 1
            d = fits(g0, v["t"])
            if not d and v["cand"] and fits(*v["cand"]):          # two passes agree: take them, if not too far
                d = fits(g0, v["t"], 2 * MAX_STEP)
            prog = (v["prog"] if v["prog"] >= 0 else -1) + d
            # never more than a gate into the lap after the one LapRF has the pilot on
            if not d or prog > (v["lap"] + 1) * self.n_gates:
                v["cand"] = (g, t)
                return
            v.update(gate=g, prog=prog, t=t, cand=None, steady=v["steady"] + 1 if d == 1 else 0)
            self._gap(pilot)

    def _order(self) -> List[str]:
        """The pilots still flying, first first: further round the track, then there sooner."""
        done = {p for p, _ in self.finished}
        live = [p for p, v in self.vid.items() if not v["down"] and p not in done and p not in self.out]
        return sorted(live, key=lambda p: (-self.vid[p]["prog"], self.vid[p]["t"] or 0))

    def _gap(self, pilot: str):
        """The second pilot through the gate the leader went through: the margin, when it's tight."""
        if self.finished or time.time() - (self.started_at or 0) < SETTLE_S:
            return
        order = self._order()
        if len(order) < 2 or order[1] != pilot or time.time() - self.called["gap"] < GAP_EVERY_S:
            return
        a, b = self.vid[order[0]], self.vid[pilot]
        if a["prog"] != b["prog"] or a["t"] is None or min(a["steady"], b["steady"]) < STEADY:
            return
        margin = b["t"] - a["t"]
        if 0 < margin < GAP_CALL_S:
            if self._say("gap", P_PLACE, margin=self.style.margin(margin), ahead=self._name(order[0]),
                         behind=self._name(pilot), gate=self.style.gate(b["gate"])):
                self.called["gap"] = time.time()

    def _tick(self):
        while True:
            time.sleep(0.25)
            with self.lock:
                if self.running and not self.silenced and self.vid:
                    self._downs()
                    self._places()

    def _downs(self):
        now = time.time()
        lap = self.style.expected_lap() or self.leg_s or 40.0
        for pilot, v in self.vid.items():
            if v["down"] or pilot in self.out or pilot in (p for p, _ in self.finished):
                continue
            if v["gray"] or now - v["last"] > max(DOWN_MIN_S, 0.75 * lap):
                before = self._order()
                place = before.index(pilot) + 1 if pilot in before else None
                v["down"] = True
                after = self._order()
                nxt = after[place - 1] if place and place - 1 < len(after) else None
                nxt_gate = (v["gate"] % self.n_gates) + 1 if v["gate"] and self.n_gates else None
                gate = self.style.gate(nxt_gate) if self.style.crash_spot(nxt_gate) else None
                self._say("down", P_PLACE, pilot=self._name(pilot), gate=gate,
                          next=self._name(nxt) if nxt and place and place <= 3 else None,
                          place=self.style.place(place) if nxt and place and place <= 3 else None)
                self.places = after

    def _clearly_ahead(self, pilot: str, others: List[str]) -> bool:
        """`pilot` ahead of each of `others` by a gate, or by LEAD_MARGIN_S through the same gate."""
        me = self.vid[pilot]
        for o in others:
            v = self.vid.get(o)
            if v and (me["prog"] < v["prog"] or me["prog"] == v["prog"] and (v["t"] or 0) - (me["t"] or 0) < LEAD_MARGIN_S):
                return False
        return True

    def _places(self):
        """A new order from the video, once it has held HOLD_S: who takes the lead, who moves up to second —
        not in the scramble after the start, and not once the winner is home."""
        if self.finished or time.time() - (self.started_at or 0) < SETTLE_S:
            return
        order = self._order()
        if len(order) < 2:
            return
        now = time.time()
        if order[:3] == self.places[:3]:
            self.pending = None
            return
        if not self.pending or self.pending[0] != order[:3]:
            self.pending = (order[:3], now, {p: self.vid[p]["prog"] for p in order[:3]})
            return
        if now - self.pending[1] < HOLD_S or now - self.called["place"] < PLACE_EVERY_S:
            return
        first = order[0]
        if self.leader is None:                      # the first leader seen is the baseline: the holeshot
            if self.vid[first]["steady"] >= STEADY:  # call covered the start; changes from it are the news
                self.leader, self.places, self.pending = first, order, None
            return
        # a wrongly named gate can put a pilot ahead for a moment: whoever moved up must also pass the next
        # gate while still there before it's called
        moved_on = lambda p: self.vid[p]["prog"] > self.pending[2].get(p, -1) and self.vid[p]["steady"] >= STEADY
        if first != self.leader:
            if not moved_on(first):
                return
            if not self._clearly_ahead(first, [self.leader] if self.leader in order else []):
                return                               # not clearly ahead yet
            if now - self.called["lead"] >= LEAD_EVERY_S:
                self._say("video_lead", P_PLACE, pilot=self._name(first), gate=self.style.gate(self.vid[first]["gate"]))
                self.called["lead"] = self.called["place"] = now
                self.leader = first
        else:
            # who moved up into second — only if the club asks for it (club.yaml: video_places): in the tests
            # the video's leads and gaps were right every time, its other places only about half the time
            if not self.style.info.get("video_places"):
                self.places, self.pending = order, None
                return
            moved = next((p for i, p in enumerate(order[1:2], 1)
                          if p not in self.places[:i + 1] or self.places.index(p) > i), None)
            if moved and not moved_on(moved):
                return
            if moved and not self._clearly_ahead(moved, order[order.index(moved) + 1:order.index(moved) + 2]):
                return
            if moved:
                self._say("video_place", P_PLACE, pilot=self._name(moved), place=self.style.place(order.index(moved) + 1))
                self.called["place"] = now
        self.places, self.pending = order, None
