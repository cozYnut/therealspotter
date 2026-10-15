"""
The FPVTrackside event screen of race_day_app (gate validation for FPVTrackside).

    ┌ top bar: event · Learn / Relearn track · Auto · status ──────────────┐
    │ heats (newest first, ☑ = learn) │ heat: 2×2 video with pilot names  │
    │                                 │ timeline (timing / video / checks) │
    │                                 │ per pilot: laps, disagreements     │
    │                                 │ Accept · Reject · Add lap · Remove │
    │                                 │ Update laps in FPVTrackside        │
    └─────────────────────────────────┴────────────────────────────────────┘
"""

import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

JUMP_FRAMES = 6         # clicking a lap jumps this many frames before its pass
STATUS_ICON = {"new": "•", "analysing": "⏳", "analysed": "◦", "checked": "✓", "updated": "↻"}
LANE_COLORS = {"timing": QColor(235, 235, 235), "video": QColor(80, 200, 120), "flag": QColor(255, 160, 40),
               "mark": QColor(90, 160, 255), "gate": QColor(60, 130, 90)}


def _mmss(t: Optional[float]) -> str:
    if t is None:
        return "–"
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 10)}"


class Timeline(QWidget):
    """One lane per pilot: timing laps (white), G1 passes in the video (green),
    things to check (orange), the marshal's marks (blue); click to seek."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(70)
        self.duration, self.pos = 1.0, 0.0
        self.lanes: List[dict] = []

    def set_data(self, duration: float, lanes: List[dict]):
        self.duration, self.lanes = max(duration, 1.0), lanes
        self.setMinimumHeight(14 + 16 * max(1, len(lanes)))
        self.update()

    def set_pos(self, t: float):
        self.pos = t
        self.update()

    def _x(self, t):
        return 40 + (self.width() - 50) * t / self.duration

    def mousePressEvent(self, e):
        t = (e.position().x() - 40) / max(1, self.width() - 50) * self.duration
        self.seek.emit(float(min(max(t, 0.0), self.duration)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setFont(QFont("Helvetica", 9))
        for i, lane in enumerate(self.lanes):
            y = 6 + 16 * i
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(2, y, 36, 14), Qt.AlignmentFlag.AlignVCenter, lane["name"])
            for kind, ts in lane["marks"].items():
                pen = QPen(LANE_COLORS[kind])
                pen.setWidthF(2.0 if kind in ("flag", "mark") else 1.5)
                p.setPen(pen)
                for t in ts:
                    if t is None:
                        continue
                    x = self._x(t)
                    top = y + (0 if kind in ("timing", "flag") else 7)
                    p.drawLine(int(x), top + (3 if kind == "gate" else 0), int(x), top + 7)
        pen = QPen(QColor(255, 60, 60))
        pen.setWidth(2)
        p.setPen(pen)
        x = self._x(self.pos)
        p.drawLine(int(x), 0, int(x), self.height())


class HeatPlayer(QWidget):
    """Plays a heat video in place with each quadrant's channel and pilot."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = QLabel("Choose a heat")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(480, 270)
        self.view.setStyleSheet("background:#111;color:#666")
        v.addWidget(self.view, 1)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        v.addWidget(self.timeline)
        row = QHBoxLayout()
        self.which = QComboBox()
        self.which.currentIndexChanged.connect(self._switch)
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle)
        back, fwd = QPushButton("◀ 1s"), QPushButton("1s ▶")
        back.clicked.connect(lambda: self.seek(self.t - 1.0))
        fwd.clicked.connect(lambda: self.seek(self.t + 1.0))
        fb, ff = QPushButton("◀ frame"), QPushButton("frame ▶")
        fb.clicked.connect(lambda: self.seek(self.t - 1 / max(self.fps, 1)))
        ff.clicked.connect(lambda: self.step())
        self.time_lbl = QLabel("")
        for w in (self.which, self.play_btn, back, fb, ff, fwd, self.time_lbl):
            row.addWidget(w)
        row.addStretch()
        v.addLayout(row)
        self.cap = None
        self.videos: List[str] = []
        self.quads: Dict[str, List[dict]] = {}    # video → [{"rect", "channel", "pilot"}]
        self.laps_at = {}                        # channel → sorted video times of timing laps
        self.t, self.fps, self.duration = 0.0, 30.0, 1.0
        self.playing, self._t0, self._p0 = False, 0.0, 0.0
        self.frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def load(self, videos: List[str], quads: Dict[str, List[dict]], laps_at: Dict[str, List[float]]):
        self.stop()
        self.videos, self.quads, self.laps_at = videos, quads, laps_at
        self.which.blockSignals(True)
        self.which.clear()
        for vpath in videos:
            self.which.addItem("DJI feeds" if vpath.endswith("3.mp4") else "Analog 2×2", vpath)
        self.which.setVisible(len(videos) > 1)
        self.which.blockSignals(False)
        self._open(videos[0] if videos else None)

    def _switch(self, i):
        t = self.t
        self._open(self.which.itemData(i))
        self.seek(t)

    def _open(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path) if path else None
        self.path = path
        if not self.cap or not self.cap.isOpened():
            self.view.setText("Can't open the video")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.seek(0.0)

    def seek(self, t: float):
        if not self.cap:
            return
        t = min(max(t, 0.0), self.duration)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        self.step()
        if self.playing:
            self._t0, self._p0 = time.time(), self.t

    def seek_frames_before(self, t: float, frames: int):
        """Show the frame `frames` frames before time t (counted in real frames:
        heat videos don't have evenly spaced frames)."""
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_MSEC, min(max(t, 0.0), self.duration) * 1000.0)
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - frames))
        self.step()

    def step(self):
        if not self.cap:
            return
        ok, f = self.cap.read()
        if ok:
            self.frame, self.t = f, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self._show()

    def toggle(self):
        if self.playing:
            self.stop()
        elif self.cap:
            self.playing = True
            self._t0, self._p0 = time.time(), self.t
            self.play_btn.setText("❚❚")
            self.timer.start(15)

    def stop(self):
        self.playing = False
        self.play_btn.setText("▶")
        self.timer.stop()

    def _tick(self):
        target = self._p0 + (time.time() - self._t0)
        if target >= self.duration:
            return self.stop()
        n = 0
        while self.t < target - 0.5 / self.fps and n < 8:      # catch up: skip frames
            if not self.cap.grab():
                return self.stop()
            self.t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            n += 1
        if self.t < target:
            self.step()

    def _show(self):
        img = self.frame
        W, H = self.view.width(), self.view.height()
        h0, w0 = img.shape[:2]
        s = min(W / w0, H / h0)
        img = cv2.resize(img, (max(1, int(w0 * s)), max(1, int(h0 * s))), interpolation=cv2.INTER_AREA)
        h1, w1 = img.shape[:2]
        for q in self.quads.get(self.path, []):
            x, y = int(q["rect"][0] * w1), int(q["rect"][1] * h1)
            laps = sum(1 for t in self.laps_at.get(q["channel"], []) if t <= self.t)
            text = f"{q['channel']}  {q['pilot']}" + (f"   lap {laps - 1}" if laps > 1 else (" HS" if laps == 1 else ""))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x + 4, y + 4), (x + 12 + tw, y + 12 + th), (0, 0, 0), -1)
            cv2.putText(img, text, (x + 8, y + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qi.copy()))
        self.time_lbl.setText(f"{_mmss(self.t)} / {_mmss(self.duration)}")
        self.timeline.set_pos(self.t)

    def close_video(self):
        self.stop()
        if self.cap:
            self.cap.release()
        self.cap = None


class EventPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win, self.day = win, None
        self.dash_url = ""
        self.heat_id: Optional[str] = None
        self._detail_sig = None
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self.title.setFont(f)
        self.learn_btn = QPushButton("Learn track")
        self.learn_btn.clicked.connect(self._learn)
        self.auto = QCheckBox("Auto")
        self.auto.setToolTip("New heats are analysed and checked by themselves, and confident lap corrections\n"
                             "are written to FPVTrackside. Uncertain ones wait for you.")
        self.auto.toggled.connect(lambda on: self.day and self.day.set_auto(on))
        self.status = QLabel("")
        self.status.setStyleSheet("color:#aaa")
        stop = QPushButton("Close")
        stop.clicked.connect(lambda: win.stop_event())
        for w in (self.title, self.learn_btn, self.auto):
            top.addWidget(w)
        top.addWidget(self.status, 1)
        top.addWidget(stop)
        v.addLayout(top)
        self.banner = QLabel("")
        self.banner.setStyleSheet("background:#7a4b00;color:white;padding:4px")
        self.banner.setVisible(False)
        v.addWidget(self.banner)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.heats = QListWidget()
        self.heats.setMinimumWidth(180)
        self.heats.currentItemChanged.connect(self._select)
        self.heats.itemChanged.connect(self._checked)
        split.addWidget(self.heats)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.heat_title = QLabel("")
        self.heat_title.setStyleSheet("font-weight:bold")
        rv.addWidget(self.heat_title)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.player = HeatPlayer()
        vsplit.addWidget(self.player)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Video", "Clock", "What", "Lap time", "Check"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemDoubleClicked.connect(self._jump)
        self.tree.itemClicked.connect(self._jump)
        self.tree.itemChanged.connect(self._pilot_checked)
        vsplit.addWidget(self.tree)
        vsplit.setSizes([520, 300])
        rv.addWidget(vsplit, 1)
        btns = QHBoxLayout()
        self.b_analyse = QPushButton("Analyse")
        self.b_accept = QPushButton("✓ Accept")
        self.b_reject = QPushButton("✗ Reject")
        self.b_add = QPushButton("+ Lap / HS here")
        self.b_remove = QPushButton("− Remove lap")
        self.b_write = QPushButton("Update laps in FPVTrackside")
        self.b_write.setStyleSheet("QPushButton{background:#2b6cb0;color:white;font-weight:bold;padding:6px 12px}"
                                   "QPushButton:disabled{background:#444;color:#888}")
        self.b_restore = QPushButton("Restore original laps")
        self.b_review = QPushButton("Review gates")
        self.b_review.setToolTip("Check every gate pass of the selected pilot in learn_ui (opens a new window)")
        self.b_review.clicked.connect(self._review)
        self.b_analyse.clicked.connect(lambda: self.heat_id and self.day.analyse([self.heat_id]))
        self.b_accept.clicked.connect(lambda: self._decide("accepted"))
        self.b_reject.clicked.connect(lambda: self._decide("rejected"))
        self.b_add.clicked.connect(self._add)
        self.b_remove.clicked.connect(self._remove)
        self.b_write.clicked.connect(self._write)
        self.b_restore.clicked.connect(self._restore)
        for b in (self.b_analyse, self.b_review, self.b_accept, self.b_reject, self.b_add, self.b_remove):
            btns.addWidget(b)
        btns.addStretch()
        btns.addWidget(self.b_restore)
        btns.addWidget(self.b_write)
        rv.addLayout(btns)
        split.addWidget(right)
        split.setSizes([300, 980])
        v.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

    # ── setup ──────────────────────────────────────────────────

    def open(self, day):
        self.day = day
        self.heat_id, self._detail_sig = None, None
        self.heats.clear()
        self.tree.clear()
        self.auto.blockSignals(True)
        self.auto.setChecked(bool(day.state.get("auto")))
        self.auto.blockSignals(False)
        self.auto.setVisible(day.mode == "live")
        timing = day.event.has_timing
        for b in (self.b_accept, self.b_reject, self.b_add, self.b_remove, self.b_write, self.b_restore):
            b.setVisible(timing)
        self.b_review.setVisible(day.mode == "replays")
        self._asked = set()
        self.timer.start(1000)
        self.tick()

    def close_day(self):
        self.timer.stop()
        self.player.close_video()

    # ── periodic refresh (state only — cheap) ──────────────────

    def tick(self):
        if not self.day:
            return
        st = self.day.status()
        self.title.setText(f"{'LIVE' if st['mode'] == 'live' else 'REPLAYS'} — {st['event']}")
        self.learn_btn.setText("Relearn track" if st["learned"] else "Learn track")
        bits = [f"learned · {st['n_gates']} gates" if st["learned"] else "not learned yet"]
        if st["learning"] or st["learn_waiting"]:
            bits.append("learning…")
        if st["analysing"]:
            bits.append(f"analysing {st['analysing']} pilots")
        if st["labelling"]:
            bits.append(f"labelling {st['labelling']} heats")
        bits.append(f"{st['heats']} heats")
        if self.dash_url:
            bits.append(f"dashboard {self.dash_url}")
        self.status.setText("   ·   ".join(bits))
        self.banner.setVisible(st["paused"])
        if st["paused"]:
            self.banner.setText("⏸  Race running in FPVTrackside — race_day is paused until it ends")
            self.player.stop()
        self._fill_heats()
        if self.heat_id:
            self._show_heat()
        for name in st.get("unconfigured", []):
            if name not in self._asked:
                self._asked.add(name)
                self._configure_video(name)
                break

    def _fill_heats(self):
        rows = self.day.heat_list()
        self.heats.blockSignals(True)
        cur = self.heat_id
        if self.heats.count() != len(rows):
            self.heats.clear()
            for _ in rows:
                it = QListWidgetItem()
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                self.heats.addItem(it)
        for i, r in enumerate(rows):
            it = self.heats.item(i)
            laps = ", ".join(f"{c} {r['pilots'][c]}" + (f" ({r['timing_laps'][c]})" if r["timing_laps"].get(c) is not None
                                                          else "") + (" 📵" if c in r.get("no_video", []) else "")
                             for c in sorted(r["pilots"]))
            flag = f"  ⚠ {r['open']}" if r["open"] else ""
            it.setText(f"{STATUS_ICON.get(r['status'], '•')} {r['number']}  {r['label']}{flag}\n    {laps}")
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            it.setCheckState({"all": Qt.CheckState.Checked, "some": Qt.CheckState.PartiallyChecked}
                             .get(r["learn"], Qt.CheckState.Unchecked))
            it.setToolTip("☑ = learn the track from this heat's pilots (or tick single pilots in the heat view)")
            if r["id"] == cur and self.heats.currentRow() != i:
                self.heats.setCurrentRow(i)
        self.heats.blockSignals(False)

    def _checked(self, item):
        hid = item.data(Qt.ItemDataRole.UserRole)
        if hid and self.day:
            self.day.set_checked(hid, item.checkState() == Qt.CheckState.Checked)

    def _select(self, item, _prev=None):
        if not item:
            return
        hid = item.data(Qt.ItemDataRole.UserRole)
        if hid == self.heat_id:
            return
        self.heat_id, self._detail_sig = hid, None
        self._show_heat(load_video=True)

    # ── heat view ──────────────────────────────────────────────

    def _show_heat(self, load_video: bool = False):
        d = self.day.heat_detail(self.heat_id)
        if not d:
            return
        self.detail = d
        sig = (d["status"], len(d["marks"]), len(d["written"]), str(d["decisions"]), str(d["learn_pilots"]),
               sum(len(p["findings"]) for p in d["pilots"].values()), sum(len(p["timing"]) for p in d["pilots"].values()))
        if load_video:
            quads = {}
            for ch, p in d["pilots"].items():
                quads.setdefault(p["quad"]["video"], []).append({"rect": p["quad"]["rect"], "channel": ch,
                                                                 "pilot": p["pilot"]})
            self.player.load(d["videos"], quads, {ch: [x["video_t"] for x in p["timing"]] for ch, p in d["pilots"].items()})
        if sig == self._detail_sig and not load_video:
            return
        self._detail_sig = sig
        self.heat_title.setText(f"Heat {d['number']} — {d['label']}   ·   {_clock(d['start'])}–{_clock(d['end'])}"
                                f"   ·   {d['status']}" + (f"   ·   updated {len(d['written'])}×" if d["written"] else ""))
        self.player.laps_at = {ch: [x["video_t"] for x in p["timing"]] for ch, p in d["pilots"].items()}
        lanes = []
        for ch, p in sorted(d["pilots"].items()):
            lanes.append({"name": ch, "marks": {
                "timing": [x["video_t"] for x in p["timing"]],
                "video": p["video_g1"],
                "gate": [x["t"] for x in p["passes"] if x["gate"] != 1] if self.day.mode == "replays" else [],
                "flag": [f.get("video_t") for f in p["findings"]
                         if f"{ch}|{f['key']}" not in d["decisions"]],
                "mark": [m.get("video_t") for m in d["marks"] if m["channel"] == ch and m["op"] == "add"]}})
        self.player.timeline.set_data(self.player.duration, lanes)
        self._fill_tree(d)
        has_changes = bool(d["changes"])
        self.b_write.setEnabled(has_changes)
        self.b_write.setText(f"Update laps in FPVTrackside ({len(d['changes'])})" if has_changes
                             else "Update laps in FPVTrackside")
        self.b_restore.setEnabled(bool(d["written"]))

    def _fill_tree(self, d):
        sel = self._selected()
        sel_key = (sel or {}).get("key")
        self.tree.blockSignals(True)
        self.tree.clear()
        for ch, p in sorted(d["pilots"].items()):
            run = p.get("run", {})
            n_t = max(0, len(p["timing"]) - 1)
            n_v = max(0, len(p["video_g1"]) - 1)
            open_n = sum(1 for f in p["findings"] if f"{ch}|{f['key']}" not in d["decisions"])
            if not self.day.event.has_timing:
                n_t = len(p.get("laps", []))
            head = QTreeWidgetItem([ch, "", f"{p['pilot']}  —  {'timing' if self.day.event.has_timing else 'video'} "
                                    f"{n_t} laps" + (f" · video {n_v} laps" if self.day.event.has_timing else "")
                                    + (f" · ⚠ {open_n} to check" if open_n else "")
                                    + (" · 📵 no video — no picture recorded for this pilot" if p.get("no_video") else "")
                                    + (f" · {p['no_video_findings']} not checked (no picture then)"
                                       if p.get("no_video_findings") else "")
                                    + ("" if run.get("status") in (None, "analysed") else f" · {run.get('status')}"),
                                    "", ""])
            head.setData(0, Qt.ItemDataRole.UserRole, {"channel": ch, "pilot_row": True})
            head.setFlags(head.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            head.setCheckState(0, Qt.CheckState.Checked if ch in d["learn_pilots"] else Qt.CheckState.Unchecked)
            head.setToolTip(0, "☑ = learn the track from this pilot's run")
            f0 = head.font(2)
            f0.setBold(True)
            head.setFont(2, f0)
            self.tree.addTopLevelItem(head)
            flagged = {f.get("detection_id"): f for f in p["findings"] if f.get("detection_id")}
            removed = {m.get("detection_id") for m in d["marks"] if m["op"] == "remove"}
            prev = None
            rows = []
            for i, x in enumerate(p["timing"]):
                f = flagged.get(x["id"])
                lap = x["t"] - prev if prev is not None else None
                prev = x["t"]
                what = "Holeshot" if x["holeshot"] else f"Lap {i}"
                what += "  (manual)" if x["type"] == "Manual" else ""
                if x["id"] in removed:
                    chk = "removed by hand"
                elif f:
                    dec = d["decisions"].get(f"{ch}|{f['key']}")
                    chk = (f"⚠ {f['reason']}" if not dec else f"{dec}: {f['reason']}")
                elif p.get("no_video"):
                    chk = "no video"
                elif p.get("checked"):
                    chk = "✓ seen in the video"
                else:
                    chk = ""
                rows.append((x["video_t"], x["t"], what, lap, chk,
                             {"channel": ch, "detection_id": x["id"], "video_t": x["video_t"],
                              "key": f["key"] if f else None}))
            for f in p["findings"]:
                if f["kind"] == "missed" or (f["kind"] == "unsure" and not f.get("detection_id")):
                    dec = d["decisions"].get(f"{ch}|{f['key']}")
                    rows.append((f.get("video_t"), f["t"], "Lap in the video, not in timing", f.get("lap_time"),
                                 (f"⚠ {f['reason']}" if not dec else f"{dec}: {f['reason']}"),
                                 {"channel": ch, "key": f["key"], "video_t": f.get("video_t")}))
            if not self.day.event.has_timing:                 # Replays without timing: the video's laps
                for l in p.get("laps", []):
                    rows.append((l["t1"], l["t1"], f"Lap {l['lap']}", l["t1"] - l["t0"],
                                 "reviewed ✓" if p.get("reviewed") else "",
                                 {"channel": ch, "video_t": l["t1"]}))
            for mi, m in enumerate(d["marks"]):
                if m["channel"] == ch and m["op"] == "add":
                    rows.append((m.get("video_t"), m["t"], "Lap added by hand", None, "will be written",
                                 {"channel": ch, "mark_index": mi, "video_t": m.get("video_t")}))
            for vt, ct, what, lap, chk, data in sorted(rows, key=lambda r: r[1]):
                it = QTreeWidgetItem([_mmss(vt), _clock(ct) if self.day.event.has_timing else "", what,
                                      f"{lap:.2f}s" if lap else "", chk])
                it.setData(0, Qt.ItemDataRole.UserRole, data)
                if chk.startswith("⚠"):
                    for c in range(5):
                        it.setForeground(c, QColor(255, 170, 60))
                head.addChild(it)
                if sel_key and data.get("key") == sel_key:
                    self.tree.setCurrentItem(it)
            head.setExpanded(True)
        for c in range(4):
            self.tree.resizeColumnToContents(c)
        self.tree.blockSignals(False)

    def _pilot_checked(self, item, col):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        if col == 0 and data.get("pilot_row") and self.heat_id:
            self.day.set_pilot_checked(self.heat_id, data["channel"], item.checkState(0) == Qt.CheckState.Checked)
            self._detail_sig = None

    def _selected(self) -> Optional[dict]:
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def _jump(self, item, _col=0):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        if data.get("video_t") is not None:
            self.player.stop()                       # 6 frames before the pass, to watch it happen
            self.player.seek_frames_before(data["video_t"], JUMP_FRAMES)

    # ── actions ────────────────────────────────────────────────

    def _decide(self, decision):
        s = self._selected()
        if not s or not s.get("key"):
            return QMessageBox.information(self, "Check", "Select a ⚠ line first.")
        self.day.decide(self.heat_id, s["channel"], s["key"], decision)
        self._detail_sig = None

    def _add(self):
        s = self._selected()
        if not s:
            return QMessageBox.information(self, "Add lap", "Select the pilot (or one of their laps) first.")
        self.day.add_lap(self.heat_id, s["channel"], self.player.t)
        self._detail_sig = None

    def _remove(self):
        s = self._selected()
        if not s or not (s.get("detection_id") or "mark_index" in s):
            return QMessageBox.information(self, "Remove lap", "Select a timing lap (or a lap you added) first.")
        self.day.remove_lap(self.heat_id, s["channel"], detection_id=s.get("detection_id"),
                            mark_index=s.get("mark_index", -1))
        self._detail_sig = None

    def _write(self):
        d = self.day.heat_detail(self.heat_id)
        if not d or not d["changes"]:
            return
        lines = []
        for c in d["changes"]:
            name = d["pilots"].get(c["channel"], {}).get("pilot", "")
            if c["op"] == "add":
                lines.append(f"+ add a lap for {c['channel']} {name} at {_clock(c['t'])}")
            else:
                t = next((x["t"] for x in d["pilots"].get(c["channel"], {}).get("timing", [])
                          if x["id"] == c["detection_id"]), None)
                lines.append(f"− remove {c['channel']} {name}'s lap at {_clock(t) if t else '?'}")
        if QMessageBox.question(self, "Update laps in FPVTrackside",
                                f"Heat {d['number']} — {d['label']}\n\n" + "\n".join(lines) +
                                "\n\nWritten like a marshal's correction. The original Race.json is kept "
                                "(Restore original laps)."
                                "\nIf FPVTrackside has this heat open, reopen it to see the change.") \
                == QMessageBox.StandardButton.Yes:
            self.day.write(self.heat_id)
            self._detail_sig = None

    def _restore(self):
        if QMessageBox.question(self, "Restore", "Put this heat's original laps back in FPVTrackside?") \
                == QMessageBox.StandardButton.Yes:
            self.day.restore(self.heat_id)
            self._detail_sig = None

    def _review(self):
        import subprocess
        import sys
        from pathlib import Path
        s = self._selected()
        d = self.day.heat_detail(self.heat_id) if self.heat_id else None
        if not s or not d:
            return QMessageBox.information(self, "Review gates", "Select a pilot first.")
        p = d["pilots"][s["channel"]]
        if not Path(p["race_data"]).exists():
            return QMessageBox.information(self, "Review gates", "Analyse this heat first.")
        if not self.day.tp.gate_memory.exists():
            return QMessageBox.information(self, "Review gates", "Learn the track first.")
        q = p["quad"]
        self.player.stop()
        subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent.parent / "learn_ui.py"),
                          "--review", q["video"], "--crop", ",".join(str(x) for x in q["rect"]),
                          "--race-data", p["race_data"], "--gt", p["gt"],
                          "--gate-memory", str(self.day.tp.gate_memory),
                          "--title", f"heat {d['number']} · {s['channel']} {p['pilot']}"])

    def _configure_video(self, name: str):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"What is {name}?")
        f = QFormLayout(dlg)
        kind = QComboBox()
        kind.addItems(["Single pilot", "2×2 (four pilots)"])
        heat = QSpinBox()
        heat.setRange(1, 999)
        heat.setValue(1 + max([h["number"] for h in self.day.heat_list()] + [0]))
        f.addRow("Video", QLabel(name))
        f.addRow("Kind", kind)
        f.addRow("Heat number", heat)
        single = QLineEdit()
        f.addRow("Pilot", single)
        quads = []
        for pos, ch in zip(["top left", "top right", "bottom left", "bottom right"], ["R1", "R3", "R6", "R8"]):
            c, n = QLineEdit(ch), QLineEdit()
            c.setMaximumWidth(60)
            row = QHBoxLayout()
            row.addWidget(c)
            row.addWidget(n)
            f.addRow(f"{pos}: channel, pilot", row)
            quads.append((c, n, row))

        def show(i):
            for c, n, _ in quads:
                c.setEnabled(i == 1)
                n.setEnabled(i == 1)
            single.setEnabled(i == 0)
        kind.currentIndexChanged.connect(show)
        show(0)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if kind.currentIndex() == 0:
            info = {"kind": "single", "heat": heat.value(), "pilots": {"P1": single.text().strip() or "?"}}
        else:
            chans = [c.text().strip() or f"Q{i + 1}" for i, (c, _, _) in enumerate(quads)]
            info = {"kind": "2x2", "heat": heat.value(), "channels": chans,
                    "pilots": {ch: n.text().strip() for ch, (_, n, _) in zip(chans, quads) if n.text().strip()}}
        self.day.configure_video(name, info)

    def _learn(self):
        runs = self.day.learn_runs()
        if not runs:
            return QMessageBox.information(self, "Learn", "Tick ☑ the pilot runs to learn the track from first — "
                                           "in the heat list (all pilots of a heat) or per pilot in the heat view.\n\n"
                                           "About 5 runs with at least one full lap each are enough.")
        heats = {r["id"]: r for r in self.day.heat_list()}
        lines = [f"heat {heats[h]['number']} · {ch} {heats[h]['pilots'].get(ch, '?')} — "
                 f"{heats[h]['timing_laps'].get(ch, '?')} laps" for h, ch in runs if h in heats]
        few = [l for l, (h, ch) in zip(lines, runs) if (heats.get(h, {}).get("timing_laps", {}).get(ch) or 0) < 1]
        n, ok = QInputDialog.getInt(self, "Learn the track",
                                    f"Learning from {len(runs)} pilot runs:\n  " + "\n  ".join(lines) +
                                    (f"\n\n⚠ {len(few)} of them have no full lap — they add little." if few else "") +
                                    "\n\nNumber of gates on the track:",
                                    int(self.day.state.get("n_gates") or 10), 3, 40)
        if ok:
            self.day.learn(n)
