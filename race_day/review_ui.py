"""
Reviewing one pilot's run inside the heat view (race_day/review.py does the
work): the video shows only that pilot's quadrant, the timeline the review
marks, and this panel the list of marks, the system's crop next to the
learned gate's picture, and the actions. Saved after every change as
<data>/gt/<run>.gt.json.

  N / Shift+N   next / previous pass to check      Enter  accept (at the video's time)
  0–9           gate number (1 then 2 → G12)        X      false pass (no real pass)
  A             add a pass the system missed        M      a gate the pilot skipped
  T             tag: crash/clipped → bad video → reverse → unsure → none
  Backspace     undo (back to "to check"; one you added is removed)
  Space         play / pause      ← / →  one frame      Shift+← / →  one second
"""

import time
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from race_day.event_ui import REVIEW_COLORS
from race_day.review import TAG_LABELS, Review

ICONS = {"pending": "…", "accepted": "✓", "fixed_id": "✎", "added": "+", "skipped": "⤼"}
WORDS = {"pending": "to check", "accepted": "accepted", "fixed_id": "gate fixed", "added": "added",
         "skipped": "skipped gate"}


class ReviewPanel(QWidget):
    finished = pyqtSignal()              # the user closed the review

    def __init__(self, page: QWidget, player):
        super().__init__()
        self.player = player
        self.review: Optional[Review] = None
        self._digits, self._digit_t = "", 0.0
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight:bold")
        v.addWidget(self.title)
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.setStyleSheet("font-family:Menlo,monospace;font-size:12px")
        self.list.itemClicked.connect(lambda it: self._select(self.review.marks[self.list.row(it)]))
        row.addWidget(self.list, 3)
        crops = QVBoxLayout()
        self.det_img, self.det_cap = QLabel(), QLabel("System")
        self.mem_img, self.mem_cap = QLabel(), QLabel("Gate")
        for img, cap in ((self.det_img, self.det_cap), (self.mem_img, self.mem_cap)):
            img.setFixedSize(220, 124)
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setStyleSheet("background:#111;color:#666")
            cap.setStyleSheet("color:#aaa;font-size:11px")
            crops.addWidget(cap)
            crops.addWidget(img)
        crops.addStretch()
        row.addLayout(crops, 1)
        v.addLayout(row, 1)
        btns = QHBoxLayout()
        actions = [("✓ Accept  ⏎", self.accept), ("✗ False pass  X", self.reject), ("+ Missed pass  A", lambda: self.add("pass")),
                   ("⤼ Skipped gate  M", lambda: self.add("skipped")), ("Tag  T", self.tag), ("Undo  ⌫", self.undo),
                   ("◀ Prev  ⇧N", lambda: self.step(-1)), ("Next ▶  N", lambda: self.step(+1))]
        for text, fn in actions:
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch()
        done = QPushButton("Done reviewing")
        done.setStyleSheet("QPushButton{background:#2b6cb0;color:white;font-weight:bold;padding:6px 12px}")
        done.clicked.connect(self.finished.emit)
        btns.addWidget(done)
        v.addLayout(btns)
        self.msg = QLabel("")
        self.msg.setStyleSheet("font-weight:600")
        v.addWidget(self.msg)
        help_ = QLabel("Type the gate number (1 then 2 → G12) · Space play · ← → one frame · ⇧← → one second")
        help_.setStyleSheet("color:#888;font-size:11px")
        v.addWidget(help_)
        # keys: only while a review is open
        keys = [("Return", self.accept), ("Enter", self.accept), ("N", lambda: self.step(+1)),
                ("Shift+N", lambda: self.step(-1)), ("X", self.reject), ("A", lambda: self.add("pass")),
                ("M", lambda: self.add("skipped")), ("T", self.tag), ("Backspace", self.undo),
                ("Space", self.player.toggle), ("Left", lambda: self._frame(-1)), ("Right", lambda: self._frame(+1)),
                ("Shift+Left", lambda: self.player.seek(self.player.t - 1.0)),
                ("Shift+Right", lambda: self.player.seek(self.player.t + 1.0))]
        keys += [(str(d), lambda d=d: self.digit(d)) for d in range(10)]
        self.shortcuts = []
        for key, fn in keys:
            sc = QShortcut(QKeySequence(key), page)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
            sc.setEnabled(False)
            self.shortcuts.append(sc)
        self.player.timeline.mark_clicked.connect(lambda m: self.review and self._select(m))

    def say(self, text: str):
        self.msg.setText(text)

    # ── open / close ───────────────────────────────────────────

    def start(self, review: Review, channel: str, pilot: str, rect):
        self.review = review
        self._digits = ""
        for sc in self.shortcuts:
            sc.setEnabled(True)
        self.player.stop()
        self.player.set_focus(rect, channel)
        self.title_text = f"Review — {channel}{'' if pilot in ('', '?', channel) else ' ' + pilot}"
        self._refresh()
        self._select(review.next_pending(now=0.0) or (review.marks[0] if review.marks else None))
        if not review.marks:
            self.say("No passes in this run — play the video and press A at each pass.")

    def stop(self):
        self.review = None
        for sc in self.shortcuts:
            sc.setEnabled(False)
        self.player.set_focus(None)
        self.player.timeline.set_review([], None)

    # ── actions ────────────────────────────────────────────────

    def _select(self, m):
        r = self.review
        if r is None:
            return
        r.select(m)
        self._digits = ""
        if m is not None:
            self.player.stop()
            self.player.seek(m.t)
            p = m.pred or {}
            if p:
                sim = f"{p['sim']:.2f}" if isinstance(p.get("sim"), (int, float)) else "-"
                det = f"G{p['gate_id']}" if p.get("gate_id", -1) >= 1 else "no gate"
                self.say(f"{m.t:.2f} s — the system: {det} ({p.get('source')}, look {sim})")
            else:
                self.say(f"{m.t:.2f} s — added by you ({'skipped gate' if m.kind == 'skipped' else 'pass'})")
        self._refresh()

    def step(self, direction: int):
        if self.review is None:
            return
        m = self.review.next_pending(direction, now=self.player.t)
        if m is None:
            self.say(f"All {len(self.review.marks)} marks checked ✓ — the review is complete." if self.review.marks
                     else "No marks.")
            return
        self._select(m)

    def accept(self):
        r = self.review
        if r is None:
            return
        if r.sel is None:
            return self.step(+1)
        why = r.accept(self.player.t)
        if why:
            return self.say(why)
        self._refresh()
        self.step(+1)

    def reject(self):
        if self.review and self.review.sel is not None:
            self.review.reject()
            self._refresh()
            self.step(+1)

    def add(self, kind: str):
        if self.review is None:
            return
        m = self.review.add(kind, self.player.t)
        self.say(f"{'Missed pass' if kind == 'pass' else 'Skipped gate'} added at {m.t:.2f} s as G{m.gate_id} — "
                 "type a gate number to change it, Enter to confirm.")
        self._refresh()

    def tag(self):
        if self.review and self.review.sel is not None:
            self.review.cycle_tag()
            self.say(f"Tag: {TAG_LABELS[self.review.sel.tag] or 'none'}")
            self._refresh()

    def undo(self):
        if self.review and self.review.sel is not None:
            m = self.review.sel
            self.review.undo()
            if self.review.sel is m:
                self.player.seek(m.t)
            self._refresh()

    def digit(self, d: int):
        r = self.review
        if r is None or r.sel is None:
            return self.say("Select a pass first (N, or click it).")
        now = time.monotonic()
        buf = self._digits + str(d) if self._digits and now - self._digit_t < 0.8 else str(d)
        self._digit_t = now
        if not 1 <= int(buf) <= r.n_gates:
            buf = str(d)                         # too big → a new number
        if not 1 <= int(buf) <= r.n_gates:
            self._digits = ""
            return self.say(f"No gate G{buf} — this track has G1…G{r.n_gates}.")
        self._digits = buf
        r.set_gate(int(buf))
        self._refresh()

    def _frame(self, k: int):
        self.player.stop()
        if k > 0:
            self.player.step()
        else:
            self.player.seek(self.player.t - 1.0 / max(self.player.fps, 1.0))

    # ── view ───────────────────────────────────────────────────

    def _refresh(self):
        r = self.review
        if r is None:
            return
        self.title.setText(f"{self.title_text}   ·   {r.done}/{len(r.marks)} checked"
                           + ("   ·   complete ✓" if r.complete else "") + "   ·   saved")
        self.list.blockSignals(True)
        self.list.clear()
        for m in r.marks:
            if m.state == "rejected":
                text = f"{m.t:7.2f}s      ✕ false pass"
            else:
                gid = f"G{m.gate_id}" if m.gate_id >= 1 else "G?"
                skip = "SKIP " if m.kind == "skipped" else ""
                text = f"{m.t:7.2f}s L{m.lap:<2} {skip}{gid:<4}{ICONS[m.state]} {WORDS[m.state]}"
                if m.state == "pending" and m.pred and m.pred.get("source") not in (None, "RACE"):
                    text += f" ({m.pred.get('source')})"
            if m.tag:
                text += f"  [{TAG_LABELS[m.tag]}]"
            it = QListWidgetItem(text)
            it.setForeground(REVIEW_COLORS.get(m.state, REVIEW_COLORS["pending"]))
            self.list.addItem(it)
        if r.sel in r.marks:
            row = r.marks.index(r.sel)
            self.list.setCurrentRow(row)
            self.list.scrollToItem(self.list.item(row))
        self.list.blockSignals(False)
        self.player.timeline.set_review(r.marks, r.sel)
        m = r.sel
        self._image(self.det_img, r.query_image(m) if m else "", "no crop")
        self.det_cap.setText("System" if m and m.pred else "System (none)")
        self._image(self.mem_img, r.gate_image(m.gate_id) if m else "", "no picture")
        self.mem_cap.setText(f"Gate G{m.gate_id} (learned)" if m and m.gate_id >= 1 else "Gate")

    @staticmethod
    def _image(label: QLabel, path: str, missing: str):
        pm = QPixmap(path) if path else QPixmap()
        if pm.isNull():
            label.clear()
            label.setText(missing)
        else:
            label.setPixmap(pm.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
