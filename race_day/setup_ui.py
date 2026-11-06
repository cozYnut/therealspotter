"""
The event setup window (the FPVTrackside screen's "Event setup…"): the
event's classes, the track layout each flies, its qualifying, its finals and
its points (race_day/eventsetup.py), and the three steps that put it into
FPVTrackside through its REST API:

    Create qualifying…   the time-trial rounds, every class in each (Open's
                         races, then Beginners'); a dry run first
    Make the brackets…   a class's qualifying order cut into mains (A, B, C…),
                         each made a Seeder stage; once every class has its
                         brackets, the races go into shared "Finals n" rounds,
                         one race per main in turn (race_day/finals.py)
    Places & points…     where each pilot finished, and their points

    ┌ Event setup ────────────────────────────────────────────────────────┐
    │ Layouts  main 14 gates ✓ learned · beginners 10 gates   [+] [−]     │
    │ Seats    R1 R3 R6 R8                                                │
    │ ┌ Open ┬ Beginners ┬ + ┐                                            │
    │ │ Name [Open]  Layout [main ▾]                                      │
    │ │ ☑ Pilot      Main      │ Qualifying  4 rounds · 4 a race · 3 laps │
    │ │ ☑ Pennz      auto      │ [Create qualifying…]                     │
    │ │ ☐ Moses      —         │ Finals  mains of 8 · 4 a race · 2 go on  │
    │ │ ☑ Slow       B         │ chase the ace in A, 2 wins · leftover    │
    │ │                        │ [Make the brackets…]                     │
    │ │                        │ Points FRIL  · Announcer: times / places │
    │ │                        │ [Places & points…]                       │
    └─────────────────────────────────────────────────────────── [Close] ┘
"""

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from race_day import eventsetup as es
from race_day.ftapi import ApiError

AUTO = "auto"
LEFTOVER_TEXT = {"series": "a series main of their own (points)", "join": "join the last main",
                 "spread": "spread over the mains (about equal)"}
STYLE_TEXT = {"times": "lap times", "places": "places", "both": "both"}


def _spin(lo, hi, val, suffix=""):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(int(val))
    if suffix:
        s.setSuffix(suffix)
    return s


def _combo(items: Dict[str, str], current: str):
    c = QComboBox()
    for k, t in items.items():
        c.addItem(t, k)
    i = c.findData(current)
    c.setCurrentIndex(max(0, i))
    return c


class SetupDialog(QDialog):
    def __init__(self, win, day):
        super().__init__(win)
        self.setWindowTitle(f"Event setup — {day.event.name}")
        self.resize(1050, 720)
        self.win, self.day = win, day
        self.setup: es.EventSetup = day.setup
        self.setup.reload()
        v = QVBoxLayout(self)

        top = QFormLayout()
        row = QHBoxLayout()
        self.layout_lbl = QLabel("")
        row.addWidget(self.layout_lbl, 1)
        add_l, del_l = QPushButton("+ Layout"), QPushButton("− Layout")
        add_l.setToolTip("A track layout of its own: a class that flies other gates, or in another order\n"
                         "(its own track folder, learned from its own runs)")
        add_l.clicked.connect(self._add_layout)
        del_l.clicked.connect(self._del_layout)
        gates = QPushButton("Gates…")
        gates.setToolTip("How many gates each layout has")
        gates.clicked.connect(self._gates)
        for b in (gates, add_l, del_l):
            row.addWidget(b)
        top.addRow("Track layouts", row)
        self.seats = QLineEdit(" ".join(self.setup.data.get("channels") or []))
        self.seats.setPlaceholderText(" ".join(self._event_channels()[:4]) or "R1 R3 R6 R8")
        self.seats.setToolTip("The channels of a race's seats, in order (empty: FPVTrackside picks free ones).\n"
                              "The event's channels: " + " ".join(self._event_channels()))
        self.seats.editingFinished.connect(self._seats)
        top.addRow("Seats (channels)", self.seats)
        row = QHBoxLayout()
        qb = QPushButton("Create qualifying in FPVTrackside…")
        qb.setToolTip("Every class in the same rounds: Qualifying 1 = Open's races, then Beginners'… (a dry run\n"
                      "first: what FPVTrackside would create). Again later: only what's missing or changed\n"
                      "(a race already flown is never touched).")
        qb.clicked.connect(self._qualifying)
        row.addWidget(qb)
        self.q_lbl = QLabel("")
        self.q_lbl.setWordWrap(True)
        row.addWidget(self.q_lbl, 1)
        top.addRow("Qualifying", row)
        self.fin_lbl = QLabel("")
        self.fin_lbl.setWordWrap(True)
        top.addRow("Finals", self.fin_lbl)
        v.addLayout(top)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        add_c, del_c = QPushButton("+ Class"), QPushButton("− Class")
        add_c.clicked.connect(self._add_class)
        del_c.clicked.connect(self._del_class)
        row.addWidget(add_c)
        row.addWidget(del_c)
        sync_p = QPushButton("⟳ Pilots from FPVTrackside")
        sync_p.setToolTip("Read the event's pilots from FPVTrackside again (pilots added or removed there since\n"
                          "this window opened). New pilots show unticked in every class; nothing changes in FPVTrackside.")
        sync_p.clicked.connect(self._sync_pilots)
        row.addWidget(sync_p)
        self.api_lbl = QLabel("")
        self.api_lbl.setStyleSheet("color:#888")
        row.addWidget(self.api_lbl, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

        self.pilots = self._event_pilots()
        self._fill()

    # ── the event as it is ─────────────────────────────────────

    def _event_channels(self) -> List[str]:
        ch = getattr(self.day.event, "channels", {}) or {}
        return [c for c in ch.values() if c and not c.upper().startswith("D")] + \
               [c for c in ch.values() if c and c.upper().startswith("D")]

    def _event_pilots(self) -> List[str]:
        names = set((getattr(self.day.event, "pilot_names", {}) or {}).values())
        api = self._api(quiet=True)
        if api:
            try:
                names |= {p.get("name") for p in api.pilots() if p.get("name")}
            except ApiError:
                pass
        for h in list(self.day.event.heats.values()):
            names |= set(h.pilots.values())
        for c in self.setup.classes:
            names |= set(c["pilots"])
        names.discard("?")
        return sorted((n for n in names if n), key=str.lower)

    def _api(self, quiet: bool = False):
        """FPVTrackside's API, if it has this event open (else None, and why in the window)."""
        ft = getattr(self.day, "ft", None)
        why = None
        if ft is None:
            why = "not an FPVTrackside event"
        elif ft.level(force=True) != "rest":
            why = f"FPVTrackside's REST API doesn't answer at {ft.url}"
        elif not ft.serves(self.day.event.dir.name):
            why = "FPVTrackside has another event open"
        self.api_lbl.setText(f"FPVTrackside: {why}" if why else f"FPVTrackside: this event is open ({ft.url})")
        if why and not quiet:
            QMessageBox.information(self, "FPVTrackside", f"Can't reach this event in FPVTrackside: {why}.\n\n"
                                    "Open the event in FPVTrackside (its web server on), then try again.")
        return None if why else ft

    # ── layouts and seats ──────────────────────────────────────

    def _layouts_text(self):
        bits = []
        for name, info in self.setup.layouts.items():
            n = info.get("gates") or (self.day._n_gates(name) if hasattr(self.day, "_n_gates") else None)
            learned = self.day.learned(name) if hasattr(self.day, "learned") else False
            users = [c["name"] for c in self.setup.classes if (c.get("layout") or es.MAIN) == name]
            bits.append(f"<b>{name}</b> {n or '?'} gates {'✓ learned' if learned else '· not learned yet'}"
                        + (f" ({', '.join(users)})" if users else ""))
        self.layout_lbl.setText(" &nbsp;·&nbsp; ".join(bits))

    def _add_layout(self):
        name, ok = QInputDialog.getText(self, "New track layout",
                                        "Name of the layout (for example: beginners):")
        if not ok or not name.strip():
            return
        n, ok = QInputDialog.getInt(self, "New track layout", f"Gates in '{name.strip()}':", 10, 1, 60)
        if not ok:
            return
        self.setup.add_layout(name, n)
        self._save()
        self._fill()

    def _del_layout(self):
        names = [n for n in self.setup.layouts if n != es.MAIN]
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Remove a layout", "Layout:", names, 0, False)
        if not ok:
            return
        self.setup.layouts.pop(name, None)
        for c in self.setup.classes:
            if c.get("layout") == name:
                c["layout"] = es.MAIN
        self._save()
        self._fill()

    def _gates(self):
        for name in list(self.setup.layouts):
            cur = self.setup.layouts[name].get("gates") or self.day._n_gates(name)
            n, ok = QInputDialog.getInt(self, "Gates", f"Gates in layout '{name}':", int(cur or 10), 1, 60)
            if not ok:
                return
            self.setup.layouts[name]["gates"] = n
        self._save()
        self._layouts_text()

    def _seats(self):
        self.setup.data["channels"] = self.seats.text().replace(",", " ").split()
        self._save()

    def _save(self):
        self.setup.save()
        self.day._lay_at = 0                       # the top bar's layouts: again

    # ── classes ────────────────────────────────────────────────

    def _fill(self):
        cur = self.tabs.currentIndex()
        self.tabs.clear()
        for c in self.setup.classes:
            self.tabs.addTab(ClassPage(self, c), c["name"])
        if not self.setup.classes:
            hint = QLabel("No classes yet. <b>+ Class</b> adds one (for example Open, then Beginners).<br>"
                          "Each class has its pilots, the track layout it flies, its qualifying and its finals.")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(hint, "—")
        self.tabs.setCurrentIndex(max(0, min(cur, self.tabs.count() - 1)))
        self._layouts_text()
        self._api(quiet=True)
        self._order_text()

    def _order_text(self):
        """How the rounds will run: every class in each round."""
        cs = [c for c in self.setup.classes if c["pilots"]]
        self.q_lbl.setText(("Each round: " + " → ".join(f"{c['name']} ({len(c['pilots'])})" for c in cs)
                            + " (the class tabs' order)") if cs else "")
        day_fin = getattr(self.day, "finals", None)
        waiting = day_fin.waiting_for() if day_fin else [c["name"] for c in cs
                                                         if not self.setup.data["made"]["stages"].get(c["name"])]
        if waiting:
            self.fin_lbl.setText("Each round: one race per main in turn (A, B, C, D, then the next class's), "
                                 "then back to A. The races go into FPVTrackside by themselves once every class's "
                                 f"brackets are made — still to make: <b>{', '.join(waiting)}</b>.")
        else:
            self.fin_lbl.setText("Brackets made. " + ((day_fin.status if day_fin else "") or
                                 "The finals rounds fill by themselves after each race."))

    def _seat(self, api):
        """seat(pilots) for a plan: each pilot on their own channel when it's free (a DJI pilot keeps DR8)."""
        try:
            channels = (api.get("/event") or {}).get("channels") or []
            own = {(p.get("name") or "").lower(): (p.get("channel") or {}).get("label") for p in api.pilots()}
        except ApiError:
            channels, own = [], {}
        seats = list(self.setup.data.get("channels") or [])
        return lambda pilots: es.seat_by_channel(pilots, own, channels, seats)

    def _qualifying(self):
        cs = [c for c in self.setup.classes if c["pilots"]]
        if not cs:
            return QMessageBox.information(self, "Qualifying", "Make the classes and tick their pilots first.")
        api = self._api()
        if not api:
            return
        plan = self.setup.qualifying_plan(self._seat(api))
        try:
            dry = api.meeting_sync(plan, dry_run=True)
        except ApiError as e:
            return QMessageBox.warning(self, "Qualifying", f"FPVTrackside refused the plan: {e}")
        text = _sync_text(dry)
        if not dry.get("changed"):
            return QMessageBox.information(self, "Qualifying", f"FPVTrackside has all of it already.\n\n{text}")
        lines = []
        for r in plan["rounds"]:
            lines.append(f"<b>{r['name']}</b>")
            for i, race in enumerate(r["races"], 1):
                lines.append(f"&nbsp;&nbsp;race {i}: " + ", ".join(
                    s["pilot"] + (f" ({s['channel']})" if s.get("channel") else "") for s in race["pilots"]))
        ok = _confirm(self, "Create qualifying in FPVTrackside",
                      f"{len(plan['rounds'])} time-trial rounds, every class in each: "
                      + " → ".join(f"{c['name']} ({len(c['pilots'])})" for c in cs)
                      + ". FPVTrackside says it would:<br><br>" + text.replace("\n", "<br>"),
                      "<br>".join(lines), "Create it")
        if not ok:
            return
        try:
            res = api.meeting_sync(plan, dry_run=False)
        except ApiError as e:
            return QMessageBox.warning(self, "Qualifying", f"FPVTrackside refused it: {e}")
        for c in cs:
            self.setup.data["made"]["qualifying"][c["name"]] = [r["externalId"] for r in plan["rounds"]]
        self._save()
        QMessageBox.information(self, "Qualifying", "Done in FPVTrackside:\n\n" + _sync_text(res))

    def _sync_pilots(self):
        api = self._api()
        if not api:
            return
        try:
            there = {p.get("name") for p in api.pilots() if p.get("name")}
        except ApiError as e:
            return QMessageBox.warning(self, "Pilots", f"FPVTrackside's pilots: {e}")
        before = {n.lower() for n in self.pilots}
        self.pilots = self._event_pilots()
        new = sorted((n for n in there if n.lower() not in before), key=str.lower)
        low = {n.lower() for n in there}
        gone = [(c, p) for c in self.setup.classes for p in c["pilots"] if p.lower() not in low]
        taken = {p.lower() for c in self.setup.classes for p in c["pilots"]}
        free = sorted((n for n in there if n.lower() not in taken), key=str.lower)
        msg = [f"{len(there)} pilots in the event in FPVTrackside."]
        msg.append(f"New since the window opened: {', '.join(new)}" if new else "No new pilots.")
        msg.append(f"In no class yet: {', '.join(free)} — tick them in their class." if free
                   else "Every pilot is in a class.")
        if gone:
            msg.append("In a class here but not in the event in FPVTrackside: "
                       + ", ".join(f"{p} ({c['name']})" for c, p in gone))
            if QMessageBox.question(self, "Pilots", "\n\n".join(msg) + "\n\nTake those out of their classes?") \
                    == QMessageBox.StandardButton.Yes:
                for c, p in gone:
                    c["pilots"] = [x for x in c["pilots"] if x != p]
                    (c.get("pins") or {}).pop(p, None)
                self._save()
                self.pilots = [n for n in self.pilots if n.lower() in low or
                               any(n in c["pilots"] for c in self.setup.classes)]
        else:
            QMessageBox.information(self, "Pilots", "\n\n".join(msg))
        self._fill()

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "New class", "Name of the class (for example: Open, Beginners):")
        name = name.strip()
        if not ok or not name:
            return
        if self.setup.cls(name):
            return QMessageBox.information(self, "New class", f"There's a class '{name}' already.")
        taken = {p for c in self.setup.classes for p in c["pilots"]}
        pilots = [] if self.setup.classes else [p for p in self.pilots if p not in taken]
        self.setup.classes.append(es.new_class(name, es.MAIN, pilots))
        self._save()
        self._fill()
        self.tabs.setCurrentIndex(len(self.setup.classes) - 1)

    def _del_class(self):
        i = self.tabs.currentIndex()
        if not (0 <= i < len(self.setup.classes)):
            return
        c = self.setup.classes[i]
        if QMessageBox.question(self, "Remove the class", f"Remove the class '{c['name']}' from the setup?\n"
                                "(Nothing changes in FPVTrackside.)") != QMessageBox.StandardButton.Yes:
            return
        self.setup.classes.pop(i)
        self._save()
        self._fill()


class ClassPage(QWidget):
    def __init__(self, dlg: SetupDialog, cls: dict):
        super().__init__()
        self.dlg, self.cls, self.setup = dlg, cls, dlg.setup
        h = QHBoxLayout(self)

        left = QVBoxLayout()
        form = QFormLayout()
        self.name = QLineEdit(cls["name"])
        self.name.editingFinished.connect(self._rename)
        form.addRow("Name", self.name)
        self.layout_c = QComboBox()
        self.layout_c.addItems(list(self.setup.layouts))
        self.layout_c.setCurrentText(cls.get("layout") or es.MAIN)
        self.layout_c.setToolTip("The track layout this class flies: its runs are analysed and checked on it")
        self.layout_c.currentTextChanged.connect(lambda t: self._set("layout", t))
        form.addRow("Track layout", self.layout_c)
        left.addLayout(form)
        self.count = QLabel("")
        left.addWidget(self.count)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Pilot", "Main", "Also in"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setToolTip("☑ = in this class. Main: 'auto' places the pilot by qualifying; a letter puts them in\n"
                              "that main whatever their time (their level).")
        left.addWidget(self.table, 1)
        row = QHBoxLayout()
        allb, noneb, addp = QPushButton("All"), QPushButton("None"), QPushButton("+ Pilot…")
        allb.setToolTip("Every pilot of the event who isn't in another class")
        allb.clicked.connect(lambda: self._tick_all(True))
        noneb.clicked.connect(lambda: self._tick_all(False))
        addp.setToolTip("A pilot who isn't in the event yet (FPVTrackside adds them when the races are made)")
        addp.clicked.connect(self._add_pilot)
        for b in (allb, noneb, addp):
            row.addWidget(b)
        left.addLayout(row)
        h.addLayout(left, 1)

        right = QVBoxLayout()
        q = cls["qualifying"]
        g = QGroupBox("Qualifying (time trial)")
        f = QFormLayout(g)
        self.q_rounds = _spin(1, 20, q["rounds"], " rounds")
        self.q_per = _spin(1, 8, q["per_race"], " pilots a race")
        self.q_laps = _spin(1, 10, q["laps"], " consecutive laps")
        self.q_laps.setToolTip("The qualifying time: a pilot's best this many consecutive laps")
        for w, k in ((self.q_rounds, "rounds"), (self.q_per, "per_race"), (self.q_laps, "laps")):
            w.valueChanged.connect(lambda val, k=k: self._set_in("qualifying", k, val))
        f.addRow("Rounds", self.q_rounds)
        f.addRow("Races", self.q_per)
        f.addRow("Seeded by", self.q_laps)
        right.addWidget(g)

        fi = cls["finals"]
        g = QGroupBox("Finals (FPVTrackside's Seeder: double elimination per main)")
        f = QFormLayout(g)
        self.f_size = _spin(2, 32, fi["main_size"], " pilots a main")
        self.f_per = _spin(2, 8, fi["per_race"], " a race")
        self.f_adv = _spin(1, 7, fi["advance"], " go on from a race")
        self.f_min = _spin(2, 32, fi["min_bracket"], " pilots")
        self.f_min.setToolTip("A remainder this big gets a bracket main of its own; smaller, the 'Leftover' rule")
        self.f_left = _combo(LEFTOVER_TEXT, fi["leftover"])
        self.f_series = _spin(1, 20, fi["series_races"], " races")
        self.f_chase = QComboBox()
        self.f_chase.addItem("none", "")
        for L in "ABCD":
            self.f_chase.addItem(L, L)
        self.f_chase.setCurrentIndex(max(0, self.f_chase.findData(fi.get("chase_main") or "")))
        self.f_wins = _spin(1, 5, fi["ace_wins"], " wins")
        self.f_final = _combo({"single": "one final race (the same bracket as A)",
                               "chase-the-ace": "a chase the ace too",
                               "grand-final": "the Seeder's classic grand final (more rounds)"},
                              fi.get("final") or "single")
        for w, k in ((self.f_size, "main_size"), (self.f_per, "per_race"), (self.f_adv, "advance"),
                     (self.f_min, "min_bracket"), (self.f_series, "series_races"), (self.f_wins, "ace_wins")):
            w.valueChanged.connect(lambda val, k=k: self._set_in("finals", k, val))
        self.f_left.currentIndexChanged.connect(lambda _: self._set_in("finals", "leftover", self.f_left.currentData()))
        self.f_chase.currentIndexChanged.connect(lambda _: self._set_in("finals", "chase_main", self.f_chase.currentData()))
        self.f_final.currentIndexChanged.connect(lambda _: self._set_in("finals", "final", self.f_final.currentData()))
        f.addRow("Mains", self.f_size)
        f.addRow("Races", self.f_per)
        f.addRow("Through", self.f_adv)
        f.addRow("Own bracket from", self.f_min)
        f.addRow("Leftover", self.f_left)
        f.addRow("Series main", self.f_series)
        f.addRow("Chase the ace in", self.f_chase)
        f.addRow("Chase the ace to", self.f_wins)
        f.addRow("Other mains end in", self.f_final)
        b = QPushButton("Make the brackets…")
        b.setToolTip("The qualifying order cut into mains (A = fastest), shown first; then each main is made a\n"
                     "Seeder stage. Its races go into the shared Finals rounds once every class's brackets are made")
        b.clicked.connect(self._brackets)
        f.addRow(b)
        right.addWidget(g)

        g = QGroupBox("Points and the announcer")
        f = QFormLayout(g)
        self.points = _combo({"fril": "FRIL (TOTAL+1−place, top 3 bonus)", "none": "none"}, cls.get("points") or "fril")
        self.points.currentIndexChanged.connect(lambda _: self._set("points", self.points.currentData()))
        st = cls.get("style") or {}
        self.s_q = _combo(STYLE_TEXT, st.get("qualifying") or "times")
        self.s_f = _combo(STYLE_TEXT, st.get("finals") or "places")
        self.s_q.setToolTip("What the announcer calls in this class's qualifying")
        self.s_f.setToolTip("What the announcer calls in this class's finals")
        self.s_q.currentIndexChanged.connect(lambda _: self._set_in("style", "qualifying", self.s_q.currentData()))
        self.s_f.currentIndexChanged.connect(lambda _: self._set_in("style", "finals", self.s_f.currentData()))
        f.addRow("Points", self.points)
        f.addRow("Qualifying calls", self.s_q)
        f.addRow("Finals calls", self.s_f)
        b = QPushButton("Places && points…")
        b.clicked.connect(self._places)
        f.addRow(b)
        right.addWidget(g)
        right.addStretch()
        h.addLayout(right, 1)
        self._fill_pilots()

    # ── editing ────────────────────────────────────────────────

    def _set(self, key, val):
        self.cls[key] = val
        self.dlg._save()
        if key == "layout":
            self.dlg._layouts_text()

    def _set_in(self, group, key, val):
        self.cls.setdefault(group, {})[key] = val
        self.dlg._save()

    def _rename(self):
        new = self.name.text().strip()
        if new and new != self.cls["name"]:
            if self.setup.cls(new):
                self.name.setText(self.cls["name"])
                return
            self.setup.rename_class(self.cls["name"], new)
            self.dlg._save()
            self.dlg.tabs.setTabText(self.dlg.tabs.indexOf(self), new)

    def _fill_pilots(self):
        others = {p.lower(): c["name"] for c in self.setup.classes if c is not self.cls for p in c["pilots"]}
        mine = {p.lower() for p in self.cls["pilots"]}
        names = list(self.dlg.pilots) + [p for p in self.cls["pilots"] if p not in self.dlg.pilots]
        names.sort(key=lambda n: (n.lower() not in mine, n.lower()))
        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        pins = {k.lower(): v for k, v in (self.cls.get("pins") or {}).items()}
        for r, n in enumerate(names):
            it = QTableWidgetItem(n)
            it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(Qt.CheckState.Checked if n.lower() in mine else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, it)
            pin = QComboBox()
            pin.addItem(AUTO, "")
            for L in "ABCDE":
                pin.addItem(L, L)
            pin.setCurrentIndex(max(0, pin.findData(pins.get(n.lower(), ""))))
            pin.setEnabled(n.lower() in mine)
            pin.currentIndexChanged.connect(lambda _, n=n, pin=pin: self._pin(n, pin.currentData()))
            self.table.setCellWidget(r, 1, pin)
            o = QTableWidgetItem(others.get(n.lower(), ""))
            o.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 2, o)
        self.table.blockSignals(False)
        try:
            self.table.itemChanged.disconnect()
        except TypeError:
            pass
        self.table.itemChanged.connect(self._ticked)
        self.count.setText(f"{len(self.cls['pilots'])} pilots in {self.cls['name']}")

    def _ticked(self, it: QTableWidgetItem):
        if it.column() != 0:
            return
        n = it.text()
        on = it.checkState() == Qt.CheckState.Checked
        lst = [p for p in self.cls["pilots"] if p.lower() != n.lower()]
        if on:
            lst.append(n)
        self.cls["pilots"] = lst
        if not on:
            (self.cls.get("pins") or {}).pop(n, None)
        self.dlg._save()
        w = self.table.cellWidget(it.row(), 1)
        if w:
            w.setEnabled(on)
        self.count.setText(f"{len(self.cls['pilots'])} pilots in {self.cls['name']}")

    def _tick_all(self, on: bool):
        others = {p.lower() for c in self.setup.classes if c is not self.cls for p in c["pilots"]}
        self.cls["pilots"] = [n for n in self.dlg.pilots if n.lower() not in others] if on else []
        if not on:
            self.cls["pins"] = {}
        self.dlg._save()
        self._fill_pilots()

    def _add_pilot(self):
        n, ok = QInputDialog.getText(self, "Add a pilot", "Callsign, as in FPVTrackside:")
        n = n.strip()
        if ok and n:
            if n not in self.dlg.pilots:
                self.dlg.pilots.append(n)
            if n.lower() not in {p.lower() for p in self.cls["pilots"]}:
                self.cls["pilots"].append(n)
            self.dlg._save()
            self._fill_pilots()

    def _pin(self, name: str, main: str):
        pins = self.cls.setdefault("pins", {})
        if main:
            pins[name] = main
        else:
            pins.pop(name, None)
        self.dlg._save()

    # ── into FPVTrackside ──────────────────────────────────────

    def _brackets(self):
        if not self.cls["pilots"]:
            return QMessageBox.information(self, "Brackets", "Tick the class's pilots first.")
        api = self.dlg._api()
        if not api:
            return
        try:
            ids = self.setup.qualifying_round_ids(self.cls, api.rounds())
            seeding = api.seeding(ids, int(self.cls["qualifying"]["laps"])) if ids else []
        except ApiError as e:
            return QMessageBox.warning(self, "Brackets", f"FPVTrackside's qualifying order: {e}")
        BracketDialog(self, api, self.setup, self.cls, self.setup.seeded(self.cls, seeding), len(ids)).exec()
        self._fill_pilots()                       # pins changed there
        self.dlg._order_text()

    def _places(self):
        made = self.setup.data["made"]["stages"].get(self.cls["name"]) or {}
        if not made:
            return QMessageBox.information(self, "Places & points", "Make the brackets first.")
        api = self.dlg._api()
        if not api:
            return
        try:
            stages = {m: api.stage(sid) for m, sid in made.items()}
        except ApiError as e:
            return QMessageBox.warning(self, "Places & points", f"FPVTrackside's stages: {e}")
        PlacesDialog(self, self.cls, self.setup.places(self.cls, stages)).exec()


class BracketDialog(QDialog):
    """The mains as the qualifying order makes them; a pilot can be moved to another main (pinned); then each
    main is made a Seeder stage in FPVTrackside."""

    def __init__(self, page: ClassPage, api, setup: es.EventSetup, cls: dict, seeded: List[dict], n_rounds: int):
        super().__init__(page)
        self.setWindowTitle(f"Make the brackets — {cls['name']}")
        self.resize(820, 680)
        self.page, self.api, self.setup, self.cls, self.seeded = page, api, setup, cls, seeded
        v = QVBoxLayout(self)
        timed = sum(1 for s in seeded if s["best"] is not None)
        f = cls["finals"]
        info = QLabel(
            f"<b>{len(seeded)} pilots</b>, seeded by their best {cls['qualifying']['laps']} consecutive laps over "
            f"{n_rounds} qualifying round{'s' if n_rounds != 1 else ''}"
            + (f" — <b>{len(seeded) - timed} without a time</b> (seeded last)" if timed < len(seeded) else "")
            + f".<br>Mains of {f['main_size']}, {f['per_race']} a race, {f['advance']} through; "
              f"leftover: {LEFTOVER_TEXT[f['leftover']]}. Change a pilot's main to move them (it's kept as their "
              f"level in the setup).")
        info.setWordWrap(True)
        v.addWidget(info)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Main", "Seed", "Pilot", "Qualifying", "Move to"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, 1)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        bb = QDialogButtonBox()
        self.make = bb.addButton("Create in FPVTrackside", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.make.clicked.connect(self._create)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._fill()

    def _fill(self):
        self.plan = self.setup.mains_plan(self.cls, self.seeded)
        rows = [(m, p) for m in self.plan for p in m["pilots"]]
        self.table.setRowCount(len(rows))
        letters = [m["main"] for m in self.plan]
        for r, (m, p) in enumerate(rows):
            vals = [m["main"] + (" (series)" if m["kind"] == "series" else ""), str(p["seed"]),
                    p["name"] + ("  📌" if p["pinned"] else ""),
                    f"{p['best']:.2f} s" if p["best"] is not None else "no time"]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(r, c, it)
            mv = QComboBox()
            mv.addItem(AUTO, "")
            for L in sorted(set(letters) | set("ABCD")):
                mv.addItem(L, L)
            mv.setCurrentIndex(max(0, mv.findData(m["main"] if p["pinned"] else "")))
            mv.currentIndexChanged.connect(lambda _, n=p["name"], mv=mv: self._move(n, mv.currentData()))
            self.table.setCellWidget(r, 4, mv)
        made = self.setup.data["made"]["stages"].get(self.cls["name"]) or {}
        lines = []
        for m in self.plan:
            req = self.setup.stage_request(self.cls, m)
            s = req["setup"]
            fin = ("a grand final" if s["finalsFormat"] == "grand-final" else "one final race"
                   if s.get("aceWinsRequired") == 1 else f"chase the ace to {s.get('aceWinsRequired')} wins")
            what = (f"a series: {s['rounds']} races for points" if m["kind"] == "series" else
                    f"double elimination, {s['pilotsPerRace']} a race, {s['pilotsToAdvance']} through, {fin}")
            lines.append(f"<b>{req['name']}</b> — {len(m['pilots'])} pilots, {what}"
                         + (" <i>(already made: skipped)</i>" if m["main"] in made else ""))
        self.summary.setText("<br>".join(lines) or "No pilots.")
        self.make.setEnabled(any(m["main"] not in made for m in self.plan))

    def _move(self, name: str, main: str):
        self.page._pin(name, main)
        self._fill()

    def _create(self):
        made = self.setup.data["made"]["stages"].setdefault(self.cls["name"], {})
        todo = [m for m in self.plan if m["main"] not in made]
        others = [c["name"] for c in self.setup.classes if c is not self.cls and c["pilots"]
                  and not self.setup.data["made"]["stages"].get(c["name"])]
        then = (f"The finals races go into FPVTrackside once {', '.join(others)} "
                f"{'has' if len(others) == 1 else 'have'} its brackets too." if others else
                "Then the first Finals round goes into FPVTrackside: one race per main in turn, every class.")
        if not _confirm(self, "Create in FPVTrackside",
                        f"Make {len(todo)} Seeder stage{'s' if len(todo) != 1 else ''} in FPVTrackside? {then}",
                        "<br>".join(f"{self.cls['name']} {m['main']} main: "
                                    + ", ".join(p['name'] for p in m['pilots']) for m in todo),
                        "Create"):
            return
        out = []
        for m in todo:
            try:
                st = self.api.create_stage(self.setup.stage_request(self.cls, m))
                made[m["main"]] = st.get("id")
                self.setup.save()
                out.append(f"{st.get('name')}: {st.get('raceCount')} races in its bracket")
            except ApiError as e:
                out.append(f"{self.cls['name']} {m['main']} main: ⚠ {e}")
                break
        fin = getattr(self.page.dlg.day, "finals", None)
        if fin and not others:
            try:
                fin.tick()
                out.append(fin.status)
            except (ApiError, ValueError, KeyError, TypeError) as e:
                out.append(f"⚠ the first Finals round: {e}")
        QMessageBox.information(self, "Brackets", "\n".join(out))
        self.accept()


class PlacesDialog(QDialog):
    def __init__(self, parent, cls: dict, rows: List[dict]):
        super().__init__(parent)
        self.setWindowTitle(f"Places & points — {cls['name']}")
        self.resize(720, 620)
        v = QVBoxLayout(self)
        done = all(r["final"] for r in rows) if rows else False
        v.addWidget(QLabel(("Final." if done else "<b>Not over yet</b>: places so far, from the races flown.")
                           + f" {len(cls['pilots'])} pilots in {cls['name']}."))
        t = QTableWidget(len(rows), 5)
        t.setHorizontalHeaderLabels(["Place", "Pilot", "Main", "Points", "How"])
        t.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        for i, r in enumerate(rows):
            for c, val in enumerate([r["place"], r["name"], r["main"], "" if r["points"] is None else r["points"],
                                     r["detail"]]):
                it = QTableWidgetItem(str(val))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                t.setItem(i, c, it)
        v.addWidget(t, 1)
        self.text = "\n".join("\t".join(str(x) for x in (r["place"], r["name"], r["main"],
                                                         "" if r["points"] is None else r["points"])) for r in rows)
        row = QHBoxLayout()
        cp = QPushButton("Copy (for a spreadsheet)")
        cp.clicked.connect(lambda: QApplication.clipboard().setText("Place\tPilot\tMain\tPoints\n" + self.text))
        row.addWidget(cp)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)


def _sync_text(res: dict) -> str:
    def n(kind, what):
        return sum(1 for x in res.get(kind) or [] if (x.get("action") or x.get("result") or "") == what)
    bits = []
    for kind in ("pilots", "rounds", "races"):
        items = res.get(kind) or []
        acts: Dict[str, int] = {}
        for x in items:
            a = x.get("action") or x.get("result") or x.get("status") or "?"
            acts[a] = acts.get(a, 0) + 1
        if acts:
            bits.append(f"{kind}: " + ", ".join(f"{v} {k}" for k, v in sorted(acts.items())))
    for w in res.get("warnings") or []:
        bits.append(f"⚠ {w}")
    return "\n".join(bits) or "nothing to change"


def _confirm(parent, title: str, head: str, body: str, yes: str) -> bool:
    d = QDialog(parent)
    d.setWindowTitle(title)
    d.resize(620, 420)
    v = QVBoxLayout(d)
    lbl = QLabel(head)
    lbl.setWordWrap(True)
    v.addWidget(lbl)
    t = QTextEdit()
    t.setReadOnly(True)
    t.setHtml(body.replace("\n", "<br>"))
    v.addWidget(t, 1)
    bb = QDialogButtonBox()
    bb.addButton(yes, QDialogButtonBox.ButtonRole.AcceptRole)
    bb.addButton(QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(d.accept)
    bb.rejected.connect(d.reject)
    v.addWidget(bb)
    return d.exec() == QDialog.DialogCode.Accepted
