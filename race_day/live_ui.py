"""
The live-capture screen of race_day_app: a track's heats, recorded from the
capture card when the operator presses Record.

    ┌ top bar: track · Learn / Relearn track · status · dashboard ─────────────┐
    │ heats (newest first,  │ LIVE FEED (channel names on the quadrants)        │
    │ ☑ = learn from)       │ ● REC 1:12 / 3:00              [● Record] [■ Stop]│
    │                       ├───────────────────────────────────────────────────┤
    │                       │ chosen heat: video · gates timeline · laps        │
    │ [Analyse selected]    │ [Analyse] [Review gates]                          │
    └───────────────────────┴───────────────────────────────────────────────────┘

Each recording is <track>/videos/heat_NNN.mp4 (the whole feed), at most
max_s long; everything else the app keeps is in <track>/lazySpotter/.

Replays is the same screen without the live feed (live=False): a track
folder's videos, analysed and learned from on demand. While it records, analysis and learning wait (EventDay.set_recording).
Nothing is analysed by itself: the operator picks heats to analyse and to learn
from. Videos in videos/ that weren't recorded here are asked about once.
"""

import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from race_day.event_ui import EventPage, _mmss
from race_day.trackside import QUAD_RECTS

REC_STYLE = ("QPushButton{background:#c53030;color:white;font-weight:bold;padding:8px 18px;border-radius:6px}"
             "QPushButton:disabled{background:#444;color:#888}")


def feed_pixmap(frame: np.ndarray, w: int, h: int, channels: List[str], recording: bool) -> QPixmap:
    """The feed scaled to the label, with each quadrant's channel (2×2) and a REC mark."""
    H, W = frame.shape[:2]
    s = min(w / W, h / H)
    img = cv2.resize(frame, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    h1, w1 = img.shape[:2]
    rects = QUAD_RECTS if len(channels) == 4 else [(0.0, 0.0, 1.0, 1.0)]
    for ch, (x, y, _, _) in zip(channels, rects):
        px, py = int(x * w1), int(y * h1)
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (px + 4, py + 4), (px + 14 + tw, py + 14 + th), (0, 0, 0), -1)
        cv2.putText(img, ch, (px + 9, py + 9 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if len(channels) == 4:
        cv2.line(img, (w1 // 2, 0), (w1 // 2, h1), (40, 40, 40), 1)
        cv2.line(img, (0, h1 // 2), (w1, h1 // 2), (40, 40, 40), 1)
    if recording:
        cv2.circle(img, (w1 - 22, 20), 9, (40, 40, 230), -1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


class LivePage(EventPage):
    def __init__(self, win, live: bool = True):
        super().__init__(win)
        self.live = live
        self.rec = None                          # race_day.capture.HeatRecorder
        self.setup: dict = {}
        self._was_recording = False
        if live:
            self._build_feed()
        self._build_heat_list()

    def _build_feed(self):
        """The live feed with Record / Stop, above the chosen heat."""
        live = QWidget()
        lv = QVBoxLayout(live)
        lv.setContentsMargins(0, 0, 0, 0)
        self.feed = QLabel("No feed")
        self.feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed.setMinimumSize(320, 180)
        self.feed.setStyleSheet("background:#111;color:#666")
        lv.addWidget(self.feed, 1)
        row = QHBoxLayout()
        self.rec_lbl = QLabel("")
        self.rec_lbl.setStyleSheet("font-size:15px;font-weight:bold")
        self.b_rec = QPushButton("●  Record")
        self.b_rec.setStyleSheet(REC_STYLE)
        self.b_rec.setToolTip("Record a heat: the whole feed goes to <track>/videos/heat_NNN.mp4")
        self.b_rec.clicked.connect(self._record)
        self.b_stop = QPushButton("■  Stop")
        self.b_stop.setMinimumHeight(34)
        self.b_stop.clicked.connect(self._stop)
        row.addWidget(self.rec_lbl, 1)
        row.addWidget(self.b_rec)
        row.addWidget(self.b_stop)
        lv.addLayout(row)
        idx = self.split.indexOf(self.right)
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(live)
        vs.addWidget(self.right)
        vs.setSizes([380, 560])
        self.split.insertWidget(idx, vs)

    def _build_heat_list(self):
        """Several heats can be chosen for analysis."""
        left = QWidget()
        hv = QVBoxLayout(left)
        hv.setContentsMargins(0, 0, 0, 0)
        self.heats.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        hv.addWidget(self.heats, 1)
        self.b_analyse_sel = QPushButton("Analyse selected heats")
        self.b_analyse_sel.setToolTip("Select heats in the list (⌘/Shift-click for several), then analyse them")
        self.b_analyse_sel.clicked.connect(self._analyse_selected)
        hv.addWidget(self.b_analyse_sel)
        self.split.insertWidget(0, left)
        self.split.setSizes([280, 1000])
        self.b_analyse.setText("Analyse this heat")

    # ── setup ──────────────────────────────────────────────────

    def open_track(self, day, recorder=None, setup: Optional[dict] = None):
        """Live capture (with the recorder) or Replays (without)."""
        self.rec, self.setup = recorder, setup or {}
        self._was_recording = False
        self._n = 0
        self.open(day)
        self.timer.start(100)                    # the feed; the heat list refreshes every 10th tick

    def close_day(self):
        super().close_day()
        if self.rec:
            self.rec.close()
        self.rec = None

    # ── periodic ───────────────────────────────────────────────

    def tick(self):
        if not self.day:
            return
        self._n = getattr(self, "_n", 0) + 1
        if self.rec:
            self._tick_feed()
        if self._n % 10 == 1:
            super().tick()

    def _tick_feed(self):
        st = self.rec.status()
        frame = self.rec.latest()
        if frame is not None:
            self.feed.setPixmap(feed_pixmap(frame, self.feed.width(), self.feed.height(),
                                            self.setup.get("channels", []), st["recording"]))
        elif st["ended"]:
            self.feed.setText("The feed stopped" + (f" — {st['error']}" if st["error"] else ""))
        if st["recording"]:
            self.rec_lbl.setText(f"<span style='color:#e53e3e'>● REC</span>  {st['name']}   "
                                 f"{_mmss(st['seconds'])[:-3]} / {_mmss(st['max_s'])[:-3]}")
        elif st["saving"]:
            self.rec_lbl.setText("saving…")
        else:
            self.rec_lbl.setText("Ready to record" if not st["ended"] else "")
        self.b_rec.setEnabled(not st["recording"] and not st["ended"])
        self.b_stop.setEnabled(st["recording"])
        # analysis and learning wait while a heat is recorded (and saved)
        busy = st["recording"] or st["saving"] > 0
        if busy != self._was_recording:
            self._was_recording = busy
            self.day.set_recording(busy)

    # ── actions ────────────────────────────────────────────────

    def _record(self):
        if not self.rec:
            return
        n = self.day.event.next_heat()
        name = f"heat_{n:03d}"
        chans = self.setup.get("channels") or ["R1"]
        info = {"kind": "2x2" if len(chans) == 4 else "single", "heat": n, "channels": chans,
                "pilots": {c: c for c in chans}, "recorded_by": "race_day_app"}
        self.day.configure_video(f"{name}.mp4", info)       # described before the file appears
        self.day.set_recording(True)
        self._was_recording = True
        self.rec.record(name)

    def _stop(self):
        if self.rec:
            self.rec.stop_recording()

    def _analyse_selected(self):
        ids = [it.data(Qt.ItemDataRole.UserRole) for it in self.heats.selectedItems()]
        if not ids:
            return QMessageBox.information(self, "Analyse", "Select one or more heats in the list first.")
        self.day.analyse([i for i in ids if i])

    # ── videos the app didn't record ───────────────────────────

    def _ask_videos(self, names: List[str]):
        """Videos found in videos/ that weren't recorded here: describe them
        all at once (most come from one place), or one by one."""
        names = [n for n in names if n not in self._asked]
        if not names:
            return
        self._asked.update(names)
        if len(names) == 1:
            return self._configure_video(names[0])
        chans = self.setup.get("channels") or ["R1", "R3", "R6", "R8"]
        box = QMessageBox(self)
        box.setInformativeText("Pilots are named by their channel; to give names, choose One by one.")
        box.setWindowTitle("Videos in the track's videos folder")
        box.setText(f"{len(names)} videos in videos/ weren't recorded here:\n  "
                    + "\n  ".join(names[:8]) + ("\n  …" if len(names) > 8 else "")
                    + "\n\nWhat are they?")
        b4 = box.addButton(f"All 2×2 ({' '.join(chans if len(chans) == 4 else ['R1', 'R3', 'R6', 'R8'])})",
                           QMessageBox.ButtonRole.AcceptRole)
        b1 = box.addButton("All single pilot", QMessageBox.ButtonRole.AcceptRole)
        each = box.addButton("One by one…", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked in (b4, b1):
            quad = chans if len(chans) == 4 else ["R1", "R3", "R6", "R8"]
            n = self.day.event.next_heat()
            for i, name in enumerate(sorted(names)):
                c = quad if clicked is b4 else [chans[0] if len(chans) == 1 else "P1"]
                self.day.configure_video(name, {"kind": "2x2" if clicked is b4 else "single", "heat": n + i,
                                                "channels": c, "pilots": {x: x for x in c}})
        elif clicked is each:
            for name in sorted(names):
                self._configure_video(name)
