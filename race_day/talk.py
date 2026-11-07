"""
The announcer between races: it fills the gaps the way a race-day commentator
does, from ready-written lines (clubs/<club>/talk.yaml: many ways to say each
thing, with the names, times and places filled in) and the app's facts
(race_day/talkfacts.py). No line is written by a model; a fact the app
doesn't have is left out, never guessed.

When it talks (and never during a race: everything stops at the start call,
FPVTrackside's lines and the race calls always go first):

  ▶ Opening (a button)   about 10 minutes, one part after another: welcome,
                         the day's plan, safety, the track gate by gate, the
                         rules of qualifying and the brackets, the points,
                         the season so far, who to watch
  a race is loaded       its preview: who flies, their standing, where they
                         can find time, what's at stake (qualifying / an early
                         round / an elimination race / the last race before
                         the final / the final walk-out / the chase)
  a race is over         a short wrap: who moved up the seeding, or where each
                         pilot goes in the bracket and when they fly again; the
                         verdict on the pick; then who's next (and the race
                         after), with the time to the start when the Auto
                         Runner is on
  qualifying is over     the seeds, one line per pilot, and the brackets
  the first finals race  the bracket explained (shorter before each losers race)
  a main is decided      its champion, with their day and season
  a quiet gap (~40 s)    filler: the format, a pilot, the season, the track,
                         the crew and sponsors, a joke at the bot's expense
  Places & points        the closing: results, points, the season standings,
                         thanks

Energy builds through the day (talk.yaml's own words do most of it; the voice
is a little quicker for the big moments): lower mains get the person, the A
main gets ceremony.
"""

import math
import queue
import re
import random
import statistics
import threading
import time
from typing import Callable, Dict, List, Optional

from race_day.eventsetup import standing

IDLE_S = 40.0            # this quiet (no line said, no race) and a filler is said …
FILL_EVERY_S = 30.0      # … then at most this often
ACTIVE_S = 25 * 60       # filler only while the event is going (a race loaded or ended this recently)
PREVIEW_DELAY_S = 3.0    # after a race is loaded: FPVTrackside's "Next up" first
WRAP_DELAY_S = 4.0       # after a race ends: the finish calls first
TALK_QUIET_S = 35.0      # no talk in the Auto Runner's last this many seconds (the countdown and hurry-ups)
PAUSE_S = 4.0            # between two parts of the opening
RATE = {1: 0.92, 2: 0.96, 3: 1.0, 4: 1.05, 5: 1.1}   # the voice by energy (the base: the Announcer window's speed)


class Talk:
    def __init__(self, announcer, facts: Callable[[], object], log: Callable[[str], None] = print):
        self.a, self.facts_fn, self.log = announcer, facts, log
        self.enabled = True
        self.jobs: "queue.Queue" = queue.Queue()
        self.gen = 0                              # bumped to stop whatever is being said (a race starts)
        self.busy = False
        self.last_said = 0.0
        self.last_fill = 0.0
        self.active_at = 0.0
        self.did: Dict[str, bool] = {}            # once a day: the bracket explained, the qualifying results …
        self.pick: Optional[dict] = None          # the preview's favourite: the wrap gives the verdict
        self.racing = False
        self.what = ""                            # what it's saying now (the Announcer window)
        self.what_job = ""
        self.next_said: Dict[str, float] = {}     # FPVTrackside race → when the wrap named its pilots as next up
        threading.Thread(target=self._loop, daemon=True).start()

    # ── the outside ────────────────────────────────────────────

    def opening(self):
        self.stop()
        self.active_at = time.time()
        self.jobs.put(("opening", {}))

    def closing(self, results: Dict[str, List[dict]]):
        """results: class → its places rows (race_day/eventsetup.py: places)."""
        self.stop()
        self.active_at = time.time()
        self.jobs.put(("closing", {"results": results}))

    def stop(self):
        """Everything waiting and being said is dropped."""
        self.gen += 1
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.a.speaker.cut_talk()

    def on_event(self, kind: str, e: dict, race: dict):
        """From the announcer, for every FPVTrackside event (outside its lock)."""
        if kind in ("RaceStartAnnouncement", "RaceStart"):
            self.racing = True
            self.stop()
        elif kind in ("RaceEnd", "RaceCancelled"):
            self.racing = False
            self.active_at = time.time()
            if kind == "RaceEnd":
                self.jobs.put(("wrap", {"race": dict(race), "at": time.time() + WRAP_DELAY_S}))
        elif kind == "RaceLoaded":
            self.active_at = time.time()
            self._drop("filler")
            self.jobs.put(("preview", {"race": dict(race), "e": dict(e), "at": time.time() + PREVIEW_DELAY_S}))

    def safety(self):
        self.stop()

    # ── speaking ───────────────────────────────────────────────

    def line(self, kind: str, **words) -> Optional[str]:
        go = self.what_job in ("opening", "closing") or (hasattr(self.a, "go_now") and self.a.go_now())
        return self.a.style.talk_line(kind, go=go, **words)

    def _say(self, gen: int, lines: List[Optional[str]], energy: int = 3) -> bool:
        """Lines one after another, waiting for each (so a race call or FPVTrackside's line can come between
        two). False once stopped."""
        for text in lines:
            if not text:
                continue
            if gen != self.gen or not self._may_talk():
                return False
            text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text[:1].upper() + text[1:])
            self.what = text
            self.a.talk_say(text, RATE.get(energy, 1.0))
            while self.a.speaker.talk_pending():
                if gen != self.gen:
                    return False
                time.sleep(0.1)
            self.last_said = time.time()
        return gen == self.gen

    def _may_talk(self) -> bool:
        left = self.a.start_in() if hasattr(self.a, "start_in") else None   # the Auto Runner's last half minute
        return self.enabled and self.a.enabled and not self.racing and not self.a.silenced and \
            (left is None or left > TALK_QUIET_S)

    def _drop(self, what: str):
        keep = []
        while True:
            try:
                j = self.jobs.get_nowait()
            except queue.Empty:
                break
            if j[0] != what:
                keep.append(j)
        for j in keep:
            self.jobs.put(j)
        if self.busy and self.what_job == what:
            self.gen += 1
            self.a.speaker.cut_talk()

    # ── the loop ───────────────────────────────────────────────

    def _loop(self):
        while True:
            try:
                job = self.jobs.get(timeout=1.0)
            except queue.Empty:
                job = None
            if job is None:
                if self._filler_due():
                    job = ("filler", {})
                else:
                    continue
            kind, args = job
            wait = args.get("at", 0) - time.time()
            if wait > 0:
                time.sleep(wait)
            if kind not in ("opening", "closing") and not self._may_talk():
                continue
            gen = self.gen
            self.busy, self.what_job = True, kind
            try:
                facts = self.facts_fn()
                if facts is None:
                    continue
                getattr(self, f"_{kind}")(gen, facts, **{k: v for k, v in args.items() if k != "at"})
            except Exception as ex:                        # noqa: BLE001 — never stop talking for a bad fact
                self.log(f"talk ({kind}): {ex}")
            finally:
                self.busy, self.what_job, self.what = False, "", ""
                if kind == "filler":
                    self.last_fill = time.time()

    def _filler_due(self) -> bool:
        now = time.time()
        return (self._may_talk() and now - self.active_at < ACTIVE_S and now - self.last_said > IDLE_S
                and now - self.last_fill > FILL_EVERY_S and not self.a.speaker.busy())

    # ── helpers ────────────────────────────────────────────────

    def _t(self, f, s: Optional[float]) -> Optional[str]:
        return f.style.lap_time(s, 2) if isinstance(s, (int, float)) else None

    def _list(self, names: List[str]) -> str:
        names = [n for n in names if n]
        if len(names) <= 1:
            return "".join(names)
        return ", ".join(names[:-1]) + " and " + names[-1]

    def _p(self, f, pilot: str) -> str:
        return f.say(pilot)

    # ── the opening: about 10 minutes ──────────────────────────

    def _opening(self, gen, f):
        ev = f.event()
        info = f.style.info
        classes = ev["classes"]
        first = classes[0] if classes else {}
        q = (first or {}).get("qualifying") or {}
        fin = (first or {}).get("finals") or {}
        parts = []
        words = dict(event=ev["name"], date=ev["date"], venue=f.venue(), bot=f.style.name,
                     club=info.get("name"), n_pilots=ev["n_pilots"], hour=time.strftime("%-I %M %p").replace(" 00 ", " o'clock "))
        parts.append((4, [self.line("open_welcome", **words), self.line("open_where", **words),
                          self.line("open_me", **words)]))
        crew = info.get("crew") or {}
        parts.append((4, [self.line("open_thanks", crew=self._list(crew.get("league_managers") or []) or None,
                                    roles=self._list(crew.get("volunteers") or []) or None),
                          self.line("open_sponsors", sponsors=self._list(info.get("sponsors") or []) or None)]))
        # the day's plan
        cl = [f"{len(c['pilots'])} in {c['name']}" for c in classes]
        plan = [self.line("plan_intro"), self.line("plan_classes", classes=self._list(cl), n=len(classes))]
        rounds = max([int((c.get("qualifying") or {}).get("rounds") or 0) for c in classes] or [0])
        plan.append(self.line("plan_qualifying", rounds=rounds, laps=f.seed_laps(first),
                              order=self._list([c["name"] for c in classes])))
        mains = []
        for c in classes:
            n = len(c["pilots"])
            size = int((c.get("finals") or {}).get("main_size") or 8)
            k = max(1, math.ceil(n / size))
            mains.append(f"{c['name']}: {self._list([chr(65 + i) for i in range(k)])} main{'s' if k > 1 else ''}")
        plan.append(self.line("plan_finals", mains=self._list(mains)))
        plan.append(self.line("plan_order", order=self._list([chr(65 + i) for i in range(4)] +
                                                              [c["name"] for c in classes[1:]])))
        parts.append((4, plan))
        # safety
        parts.append((3, [self.line("safety_intro")] + [self.line("safety_rule", rule=r)
                                                         for r in (info.get("spectators") or [])][:4]))
        # the track
        parts.append((3, self._track_walk(f, classes)))
        # the rules
        parts.append((2, self._rules_qualifying(f, first, ev)))
        parts.append((2, self._rules_brackets(f, first, long=True)))
        parts.append((2, self._rules_points(f, classes)))
        # the season, who to watch, and the whole field
        parts.append((4, self._season_story(f, classes)))
        parts.append((4, self._watch(f, classes)))
        parts.append((3, self._meet(f, classes)))
        parts.append((4, [self.line("open_close", **words)]))
        for energy, lines in parts:
            if not self._say(gen, lines, energy):
                return
            time.sleep(PAUSE_S)

    def _track_walk(self, f, classes) -> List[Optional[str]]:
        out = []
        n = f.n_gates()
        guide = f.lap_guide()
        out.append(self.line("track_intro", venue=f.venue(), gates=n or None,
                             lap=self._t(f, guide.get("median"))))
        gates = (f.style.track or {}).get("gates") or {}
        named = f.setup.sections()
        if named:                                  # the event's own sections (Event setup → Sections), in order
            typ = {x["name"]: x["typical"] for x in f.sections()}
            for k, x in enumerate(named, 1):
                out.append(self.line("track_section", k=k, name=x["name"], a=x["from"], b=x["to"],
                                     time=self._t(f, typ.get(x["name"]))))
        else:
            for g in sorted(gates):
                if (gates.get(g) or {}).get("say"):
                    out.append(self.line("track_gate", gate=g, name=gates[g]["say"], note=gates[g].get("note") or None))
        spots = f.crash_spots()
        crash = [g for g, k in spots[:2] if k >= 2] or [g for g, v in gates.items() if (v or {}).get("crash_spot")]
        if crash:
            out.append(self.line("track_crash", where=self._list([f.gate_name(g) for g in crash])))
        # the longest and quickest sections, from the analysed runs
        secs = f.sections() if n else []
        if secs:
            for s_ in sorted(secs, key=lambda x: -x["typical"])[:2]:
                out.append(self.line("track_longest", section=s_["name"], time=self._t(f, s_["typical"])))
            long2 = {x["i"] for x in sorted(secs, key=lambda x: -x["typical"])[:2]}
            for s_ in [x for x in sorted(secs, key=lambda x: -x["spread"]) if x["i"] not in long2][:2]:
                out.append(self.line("track_tricky", section=s_["name"], spread=f.style.margin(s_["spread"])))
        if guide.get("best"):
            out.append(self.line("track_best_lap", time=self._t(f, guide["best"][0]),
                                 pilot=self._p(f, guide["best"][1])))
        for c in classes:
            lay = c.get("layout") or "main"
            if lay != "main":
                out.append(self.line("track_layout", cls=c["name"], gates=f.n_gates(lay) or None))
        return out

    def _rules_qualifying(self, f, cls, ev) -> List[Optional[str]]:
        q = (cls or {}).get("qualifying") or {}
        rl = ev.get("race_length")
        return [self.line("rules_qual_intro"),
                self.line("rules_qual_how", per=q.get("per_race") or 4,
                          length=(f"{int(rl // 60)} minutes" if rl and rl >= 60 else None)),
                self.line("rules_qual_best", laps=f.seed_laps(cls)),
                self._qual_example(f, cls),
                self.line("rules_qual_rounds", rounds=q.get("rounds") or 4),
                self.line("rules_qual_cut", size=(cls or {}).get("finals", {}).get("main_size") or 8)]

    def _qual_example(self, f, cls) -> Optional[str]:
        lap = f.lap_guide().get("median")
        n = f.seed_laps(cls)
        if not lap or n < 2:
            return None
        a, b = round(lap - 0.6, 1), round(lap + 0.4, 1)
        laps = [a, b] + [round(lap, 1)] * (n - 2)
        return self.line("rules_qual_example", laps=self._list([f"{x:.1f}" for x in laps]),
                         total=f"{sum(laps):.1f}", n=n)

    def _rules_brackets(self, f, cls, long: bool = False) -> List[Optional[str]]:
        fi = (cls or {}).get("finals") or {}
        out = [self.line("rules_bracket_lives"), self.line("rules_bracket_through", adv=fi.get("advance") or 2,
                                                            per=fi.get("per_race") or 4)]
        if long:
            out += [self.line("rules_race_laps", laps=f.target_laps()),
                    self.line("rules_bracket_seeds"),
                    self.line("rules_bracket_rounds"), self.line("rules_bracket_semi")]
            if fi.get("chase_main"):
                out.append(self.line("rules_chase", main=fi["chase_main"], wins=fi.get("ace_wins") or 2))
            if (fi.get("final") or "single") == "single":
                out.append(self.line("rules_final_single"))
            if fi.get("leftover") == "series":
                out.append(self.line("rules_series", races=fi.get("series_races") or 5))
        return out

    def _rules_points(self, f, classes) -> List[Optional[str]]:
        c = next((c for c in classes if c.get("points") == "fril"), None)
        if not c:
            return []
        n = len(c["pilots"])
        p1, p2, p3 = f.points_example(n)
        import math as _m
        return [self.line("rules_points", n=n, cls=c["name"], p1=p1, p2=p2, p3=p3, last=1),
                self.line("rules_points_bonus", b1=_m.ceil(n / 5), b2=_m.ceil(n / 8), b3=_m.ceil(n / 13), n=n)]

    def _season_story(self, f, classes) -> List[Optional[str]]:
        out = []
        last = f.last_season()
        table = f.season_table(last)
        if table:
            top = table[:3]
            out.append(self.line("season_last", season=last, champ=self._p(f, top[0]["pilot"]), pts=top[0]["total"],
                                 second=self._p(f, top[1]["pilot"]) if len(top) > 1 else None,
                                 third=self._p(f, top[2]["pilot"]) if len(top) > 2 else None))
        now = f.season_table()
        if now:
            done = len(((f.seasons().get(f.this_season()) or {}).get("events")) or [])
            out.append(self.line("season_now", season=f.this_season(), events=done,
                                 leader=self._p(f, now[0]["pilot"]), pts=now[0]["total"]))
            for i, r in enumerate(now[1:5], 2):
                out.append(self.line("season_row", place=f.style.place(i), pilot=self._p(f, r["pilot"]), pts=r["total"]))
        return out

    def _meet(self, f, classes) -> List[Optional[str]]:
        """Meet the field: one line per pilot, each class, the season's order first."""
        out = []
        for c in classes:
            out.append(self.line("meet_intro", cls=c["name"], n=len(c["pilots"])))
            ranked = sorted(c["pilots"], key=lambda p: (f.season_rank(p) or (999, 0))[0])
            new = set(f.new_pilots(c["pilots"]))
            for p in ranked:
                rk = f.season_rank(p)
                last = f.season_rank(p, f.last_season())
                out.append(self.line("meet_new" if p in new else "meet_pilot", pilot=self._p(f, p), hook=f.hook(p),
                                     rank=f.style.place(rk[0]) if rk else None, pts=rk[1] if rk else None,
                                     last=f.style.place(last[0]) if last and last[0] > 3 else None))
        return out

    def _watch(self, f, classes) -> List[Optional[str]]:
        here = {p.lower() for c in classes for p in c["pilots"]}
        out = []
        now = [r for r in f.season_table() if r["pilot"].lower() in here][:3]
        for r in now:
            st = f.season_stats(r["pilot"])
            out.append(self.line("watch_pilot", pilot=self._p(f, r["pilot"]), hook=f.hook(r["pilot"]),
                                 rank=f.style.place(f.season_rank(r["pilot"])[0]), pts=r["total"]))
        new = f.new_pilots([p for c in classes for p in c["pilots"]])
        if new:
            out.append(self.line("watch_new", pilots=self._list([self._p(f, p) for p in new[:4]]), n=len(new)))
        return out

    # ── a race is loaded: its preview ──────────────────────────

    def _preview(self, gen, f, race: dict, e: dict):
        names = list(race.get("say") or {})
        if not names:
            return
        left = self.a.start_in() if hasattr(self.a, "start_in") else None
        named = time.time() - self.next_said.get(race.get("fpv") or "", 0) < 300   # the wrap said who's next
        if left and left > TALK_QUIET_S + 10 and not named:   # the Auto Runner's clock: when this one goes
            if not self._say(gen, [self.line("pre_start_in", time=f.style.countdown(left))], 3):
                return
        kind = race.get("kind")
        if kind == "qualifying" or (race.get("type") == "TimeTrial"):
            return self._say(gen, self._preview_qual(f, race, names)[1 if named else 0:], 3)
        if kind == "finals" and race.get("group"):
            return self._preview_finals(gen, f, race, names)
        if not named:
            self._say(gen, [self.line("pre_plain", pilots=self._list([self._p(f, n) for n in names]))], 3)

    def _preview_qual(self, f, race, names) -> List[Optional[str]]:
        cls = f.setup.cls(race.get("class") or "") or f.class_of(names[0])
        out = [self.line("pre_qual_intro", pilots=self._list([self._p(f, n) for n in names]),
                         round=race.get("round") or None)]
        bests = dict(f.standings(cls)) if cls else {}
        for n in names:
            pl = standing(bests, n) if bests else None
            if pl:
                out.append(self.line("pre_qual_pilot", pilot=self._p(f, n), place=f.style.place(pl),
                                     time=self._t(f, bests[n]), laps=f.seed_laps(cls)))
            else:
                out.append(self.line("pre_qual_first", pilot=self._p(f, n), hook=f.hook(n)))
        tipped = [n for n in names if f.tip(n) and f.tip(n).get("section")]
        if tipped:
            n = random.choice(tipped)
            t = f.tip(n)
            out.append(self.line("tip_section", pilot=self._p(f, n), section=t["section"],
                                 gain=f.style.margin(t["gain"]), ref=self._p(f, t["ref"])))
        top = [p for p, _ in sorted(bests.items(), key=lambda kv: kv[1])[:1]]
        if top and top[0] not in names and bests:
            out.append(self.line("pre_qual_target", leader=self._p(f, top[0]), time=self._t(f, bests[top[0]])))
        return out

    def _preview_finals(self, gen, f, race, names):
        group = (race.get("group") or "").lower()
        main, cls_name = race.get("main") or "", race.get("class") or ""
        cls = f.setup.cls(cls_name)
        top_main = bool(cls) and main == "A" and f.setup.classes.index(cls) == 0
        bests = dict(f.standings(cls)) if cls else {}
        seeds = {n: standing(bests, n) for n in names}
        energy = 4 if top_main else 3
        words = dict(cls=cls_name, main=main)
        st = f.stage(cls_name, main) if main else None
        out = []
        if "chase" in group:
            wins = f.chase_wins(st) if st else {}
            if race.get("ace_wins") == 1 or not wins:          # the final: the walk-out
                if not self._say(gen, [self.line("walkout_intro" if top_main else "final_intro", **words,
                                                 wins=race.get("ace_wins") or None)], 5 if top_main else 4):
                    return
                for n in sorted(names, key=lambda n: -(seeds.get(n) or 0)):
                    lines = [self.line("walkout_pilot", hook=f.hook(n), seed=f.style.place(seeds[n]) if seeds.get(n) else None,
                                       pilot=self._p(f, n))]
                    if not self._say(gen, lines, 5 if top_main else 4):
                        return
                rule = (self.line("rules_chase_short", wins=race.get("ace_wins")) if (race.get("ace_wins") or 1) > 1
                        else self.line("rules_final_short"))
                return self._say(gen, [rule, self.line("pre_quiet")], 5 if top_main else 4)
            lead = max(wins.items(), key=lambda kv: kv[1])
            return self._say(gen, [self.line("chase_status", leader=self._p(f, lead[0]), wins=lead[1],
                                             need=(race.get("ace_wins") or 2) - lead[1]),
                                   self.line("pre_quiet")], 5)
        stage_txt = self.line("pre_main", **words)
        if race.get("semi"):
            out += [stage_txt, self.line("pre_semi", **words)]
            energy = 5
        elif "loser" in group:
            if not self.did.get(f"losers:{cls_name}"):
                out += self._rules_brackets(f, cls)[:1]
                self.did[f"losers:{cls_name}"] = True
            out += [stage_txt, self.line("pre_elimination", adv=race.get("advance") or 2)]
            energy = max(energy, 4)
        elif race.get("stage_round") == 1:
            if not self.did.get("bracket_explained"):
                out += [self.line("bracket_explain_intro")] + self._rules_brackets(f, cls, long=True)
                self.did["bracket_explained"] = True
            out += [stage_txt, self.line("pre_early", adv=race.get("advance") or 2)]
        else:
            out += [stage_txt, self.line("pre_winners", adv=race.get("advance") or 2)]
        for n in names:
            card = f.card(n)
            hook = f.hook(n) if (top_main or not card.get("known_for")) else card.get("known_for")
            out.append(self.line("pre_pilot", pilot=self._p(f, n),
                                 seed=f.style.place(seeds[n]) if seeds.get(n) else None, hook=hook))
        fav = min((n for n in names if seeds.get(n)), key=lambda n: seeds[n], default=None)
        if fav:
            threat = min((n for n in names if seeds.get(n) and n != fav), key=lambda n: seeds[n], default=None)
            self.pick = {"fav": fav, "race": race.get("label")}
            out.append(self.line("pre_pick", fav=self._p(f, fav), threat=self._p(f, threat) if threat else None))
        out.append(self.line("pre_quiet"))
        self._say(gen, out, energy)

    # ── a race is over: the wrap ───────────────────────────────

    def _wrap(self, gen, f, race: dict):
        names = list(race.get("say") or {})
        if not names:
            return
        out = []
        if race.get("kind") == "qualifying" or race.get("type") == "TimeTrial":
            cls = f.setup.cls(race.get("class") or "") or f.class_of(names[0])
            out += self._wrap_seeding(f, race, names)
            st = f.standings(cls) if cls else []
            if len(st) >= 3:
                out.append(self.line("wrap_qual_top", p1=self._p(f, st[0][0]), t1=self._t(f, st[0][1]),
                                     p2=self._p(f, st[1][0]), p3=self._p(f, st[2][0])))
            if f.qualifying_done() and not self.did.get("qual_results"):
                self.did["qual_results"] = True
                self._say(gen, out, 3)
                return self._qualifying_results(gen, f)
        elif race.get("kind") == "finals":
            order = f.results(race.get("fpv")) or [p for p, _ in race.get("finished") or []]
            if self.pick and order:
                won = order[0]
                if won == self.pick["fav"]:
                    out.append(self.line("wrap_pick_right", pilot=self._p(f, won)))
                else:
                    out.append(self.line("wrap_pick_wrong", pilot=self._p(f, won), fav=self._p(f, self.pick["fav"])))
            self.pick = None
            if "chase" in (race.get("group") or "").lower():
                if not (self._say(gen, out, 4) and self._crowning(gen, f, race)):
                    return
                out = []
            elif race.get("group"):
                out += self._wrap_bracket(f, race, order)
        if self._say(gen, out, 3):
            self._say(gen, self._next_up(f), 3)

    def _wrap_seeding(self, f, race, names) -> List[Optional[str]]:
        """Qualifying: who moved up the seeding in this race (the live calls' standings, before and after)."""
        before, after = race.get("qual_before") or {}, (race.get("qual") or {}).get("best") or {}
        out = []
        for n in names:
            a, b = standing(after, n), standing(before, n)
            if a is None or (b is not None and a >= b):
                continue
            if b is None:
                out.append(self.line("wrap_qual_in", pilot=self._p(f, n), place=f.style.place(a)))
            else:
                d = b - a
                out.append(self.line("wrap_qual_move", pilot=self._p(f, n), place=f.style.place(a),
                                     moved="one place" if d == 1 else f"{d} places"))
        return out[:4]

    def _when(self, nxt: dict) -> Optional[str]:
        """When a pilot flies again: "the next race", "3 races", or "Finals 4" (not made in FPVTrackside yet)."""
        if nxt.get("ahead") == 0:
            return "the next race"
        if isinstance(nxt.get("ahead"), int):
            return f"{nxt['ahead'] + 1} races"
        return f"Finals {nxt['round']}" if nxt.get("round") else None

    def _wrap_bracket(self, f, race, order) -> List[Optional[str]]:
        """A bracket race: where its pilots go (winners side, losers bracket, the final) and when they fly again;
        those knocked out heard it in the finish call."""
        nxt = f.bracket_next(race.get("class") or "", race.get("main") or "", race.get("rid"), order)
        groups: Dict[tuple, List[str]] = {}
        for p in order:
            x = nxt.get(p)
            if x and x["kind"] != "out":
                groups.setdefault((x["kind"], x.get("to")), []).append(p)
        loser = "loser" in (race.get("group") or "").lower()
        out = []
        for (kind, _), ps in groups.items():
            k = "survive" if kind == "losers" and loser else kind
            out.append(self.line(f"wrap_bracket_{k}", pilots=self._list([self._p(f, p) for p in ps]),
                                 when=self._when(nxt[ps[0]])))
        return out

    def _next_up(self, f) -> List[Optional[str]]:
        """Who flies next, and the race after: the pilots get ready in time (with the Auto Runner's clock)."""
        ups = f.up_next(2)
        if not ups:
            return []
        left = self.a.start_in() if hasattr(self.a, "start_in") else None
        first = ups[0]
        self.next_said[first["id"]] = time.time()
        if hasattr(self.a, "up_next"):
            self.a.up_next = list(first["pilots"])
        what = f"the {first['cls']} {first['main']} main" if first.get("cls") and first.get("main") else None
        out = [self.line("next_up", pilots=self._list([self._p(f, p) for p in first["pilots"]]), what=what,
                         time=f.style.countdown(left) if left and left > TALK_QUIET_S else None)]
        if len(ups) > 1:
            out.append(self.line("on_deck", pilots=self._list([self._p(f, p) for p in ups[1]["pilots"]])))
        return out

    def _qualifying_results(self, gen, f):
        for c in [c for c in f.setup.classes if c["pilots"]]:
            st = f.standings(c)
            if not st:
                continue
            lines = [self.line("qres_intro", cls=c["name"], n=len(st))]
            for i, (p, t) in list(enumerate(st, 1))[::-1]:            # building up to the top qualifier
                lines.append(self.line("qres_pilot", seed=f.style.place(i), pilot=self._p(f, p), time=self._t(f, t),
                                       hook=f.hook(p)))
            lines.append(self.line("qres_top", pilot=self._p(f, st[0][0]), time=self._t(f, st[0][1]),
                                   laps=f.seed_laps(c)))
            fast = f.fastest_today(c)
            if fast:
                lines.append(self.line("qres_fastest_lap", pilot=self._p(f, fast[1]), time=self._t(f, fast[0])))
            size = int((c.get("finals") or {}).get("main_size") or 8)
            lines.append(self.line("qres_cut", size=size, cls=c["name"]))
            if not self._say(gen, lines, 4):
                return

    def _crowning(self, gen, f, race):
        st = f.stage(race.get("class") or "", race.get("main") or "")
        if not st or not st.get("isComplete") and not st.get("champion"):
            return True
        champ = st.get("champion")
        champ = champ if isinstance(champ, str) else (champ or {}).get("name")
        if not champ:
            return True
        cls = f.setup.cls(race.get("class") or "")
        top_main = bool(cls) and race.get("main") == "A" and f.setup.classes.index(cls) == 0
        stats = f.season_stats(champ)
        wins = f.chase_wins(st)
        runner = sorted((p for p in wins if p != champ), key=lambda p: -wins[p])
        lines = [self.line("crown_champ" if top_main else "crown_main", pilot=self._p(f, champ),
                         main=race.get("main"), cls=race.get("class")),
                 self.line("crown_stats", pilot=self._p(f, champ), races=stats["races"], wins=stats["wins"],
                           best=self._t(f, stats["best_lap"]))]
        if runner:
            lines.append(self.line("crown_runner", pilot=self._p(f, runner[0])))
        return self._say(gen, lines, 5 if top_main else 4)

    # ── quiet: filler ──────────────────────────────────────────

    def _filler(self, gen, f):
        topics = ["format", "pilot", "pilot", "season", "track", "tip", "tip", "crew", "joke", "safety", "today", "next"]
        random.shuffle(topics)
        for t in topics:
            lines = getattr(self, f"_fill_{t}")(f)
            if lines and any(lines):
                return self._say(gen, lines, 3)

    def _here(self, f) -> List[str]:
        return [p for c in f.setup.classes for p in c["pilots"]]

    def _fill_format(self, f):
        cls = next((c for c in f.setup.classes if c["pilots"]), None)
        if f.race_counts()["finals"] == 0:
            return self._rules_qualifying(f, cls, f.event())[1:3]
        return [self.line("fill_format_intro")] + self._rules_brackets(f, cls)

    def _fill_pilot(self, f):
        here = [p for p in self._here(f) if f.card(p)]
        if not here:
            return None
        p = random.choice(here)
        c = f.card(p)
        st = f.season_stats(p)
        rk = f.season_rank(p)
        return [self.line("fill_pilot", pilot=self._p(f, p), name=None, known_for=c.get("known_for"),
                          role=c.get("role"), rank=f.style.place(rk[0]) if rk else None, pts=rk[1] if rk else None,
                          races=st["races"] or None, wins=st["wins"] or None)]

    def _fill_season(self, f):
        now = f.season_table()
        if not now:
            return None
        return [self.line("fill_season", leader=self._p(f, now[0]["pilot"]), pts=now[0]["total"],
                          second=self._p(f, now[1]["pilot"]) if len(now) > 1 else None,
                          gap=now[0]["total"] - now[1]["total"] if len(now) > 1 else None),
                self.line("fill_points", n=len(self._here(f)))]

    def _fill_track(self, f):
        return self._track_walk(f, [])[:3]

    def _fill_tip(self, f):
        here = [p for p in self._here(f) if (f.tip(p) or {}).get("section")]
        if not here:
            return None
        p = random.choice(here)
        t = f.tip(p)
        out = [self.line("tip_section", pilot=self._p(f, p), section=t["section"], gain=f.style.margin(t["gain"]),
                         ref=self._p(f, t["ref"]))]
        if t.get("theo") and t.get("best") and t["best"] - t["theo"] > 0.5:
            out.append(self.line("tip_theo", pilot=self._p(f, p), theo=self._t(f, t["theo"]), best=self._t(f, t["best"])))
        return out

    def _fill_crew(self, f):
        info = f.style.info
        crew = info.get("crew") or {}
        return [self.line("fill_crew", roles=self._list(crew.get("volunteers") or []) or None,
                          managers=self._list(crew.get("league_managers") or []) or None),
                self.line("fill_sponsors", sponsors=self._list(info.get("sponsors") or []) or None)]

    def _fill_joke(self, f):
        return [self.line("fill_joke")]

    def _fill_safety(self, f):
        rules = f.style.info.get("spectators") or []
        return [self.line("fill_safety", rule=random.choice(rules) if rules else None)]

    def _fill_today(self, f):
        fast = f.fastest_today()
        n = f.race_counts()
        return [self.line("fill_today", races=n["flown"] or None, pilot=self._p(f, fast[1]) if fast else None,
                          time=self._t(f, fast[0]) if fast else None)]

    # ── Places & points: the closing ───────────────────────────

    def _next_line(self, f, kind: str) -> Optional[str]:
        nxt = f.next_event()
        if nxt:
            return self.line(kind, title=nxt["title"], date=nxt["date"], place=nxt["place"])
        return self.line(kind, next=f.style.info.get("next_event") or None)

    def _fill_next(self, f):
        return [self._next_line(f, "fill_next")]

    def _closing(self, gen, f, results: Dict[str, List[dict]]):
        ev = f.event()
        lines = [self.line("close_intro", event=ev["name"])]
        today = {}
        for cls_name, rows in results.items():
            if not rows:
                continue
            lines.append(self.line("close_class", cls=cls_name, n=len(rows)))
            for r in rows[:3][::-1]:
                lines.append(self.line("close_podium", place=f.style.place(r["place"]), pilot=self._p(f, r["name"]),
                                       pts=r.get("points")))
            for r in rows[3:8]:
                lines.append(self.line("close_row", place=f.style.place(r["place"]), pilot=self._p(f, r["name"]),
                                       pts=r.get("points")))
            if (f.setup.cls(cls_name) or {}).get("points") == "fril" and not today:
                today = {r["name"]: r["points"] for r in rows if r.get("points")}
        if not self._say(gen, lines, 4):
            return
        if today:
            table = f.season_table(today=today)
            lines = [self.line("close_season", season=f.this_season())]
            for i, r in enumerate(table[:5], 1):
                lines.append(self.line("season_row", place=f.style.place(i), pilot=self._p(f, r["pilot"]), pts=r["total"]))
            if not self._say(gen, lines, 4):
                return
        info = f.style.info
        crew = info.get("crew") or {}
        self._say(gen, [self.line("close_thanks", managers=self._list(crew.get("league_managers") or []) or None,
                                  roles=self._list(crew.get("volunteers") or []) or None),
                        self.line("close_sponsors", sponsors=self._list(info.get("sponsors") or []) or None),
                        self.line("close_lower"),
                        self._next_line(f, "close_next"),
                        self.line("close_signoff", bot=f.style.name)], 3)

