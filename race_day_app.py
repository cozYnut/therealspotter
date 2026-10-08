#!/usr/bin/env python3
"""
Race-day desktop app (operator) — see docs/RACE_DAY_PLAN.md.

    python race_day_app.py

Start screen: Live track or Replays.
  Live:     track name + number of gates + channel layout + capture device
            → records every run, learns the track from the first runs,
            then shows results.
  Replays:  an already-learned track + 2×2 video files → cut into runs,
            analysed and shown. No learning, no recording.

The browser dashboard (read-only) is served on the local network; its
address and QR code are shown on the running screen.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from dataset_paths import TrackPaths, data_root, list_tracks
from race_day import labeling
from race_day.capture import QUADS, DeviceSource, FileSource, list_devices, open_device
from race_day.controller import DayConfig, RaceDay
from race_day.pilots import FrilLiveProvider
from race_day.server import Dashboard

CHANNELS = [f"{b}{i}" for b in "RFEAL" for i in range(1, 9)]
VIDEO_FILTER = "2×2 videos (*.mp4 *.mov *.mkv *.avi *.MP4 *.MOV)"


def _big(text: str, size: int = 22, bold: bool = True) -> QLabel:
    l = QLabel(text)
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    l.setFont(f)
    return l


def _btn(text: str, primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(44)
    if primary:
        b.setStyleSheet("QPushButton{background:#2b6cb0;color:white;font-weight:bold;border-radius:6px;padding:8px 18px}")
    return b


def _saved_config(name: str) -> dict:
    p = TrackPaths(data_root() / name).race_day_state
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("config", {})
        except Exception:
            pass
    return {}


class Bridge(QObject):
    """Capture/controller threads → GUI thread."""
    preview = pyqtSignal(object)
    log = pyqtSignal(str)


# ──────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────

class StartPage(QWidget):
    def __init__(self, win):
        super().__init__()
        v = QVBoxLayout(self)
        v.addStretch()
        t = _big("Race day", 30)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(t)
        sub = QLabel(f"Data folder: {data_root()}")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#888")
        v.addWidget(sub)
        row = QHBoxLayout()
        live = _btn("●  Live track\nrecord, learn and show results", True)
        rep = _btn("▶  Replays\nanalyse 2×2 videos of a learned track")
        for b in (live, rep):
            b.setMinimumSize(320, 120)
        live.clicked.connect(lambda: win.show_page(win.live_setup))
        rep.clicked.connect(lambda: win.show_page(win.replay_setup))
        row.addStretch()
        row.addWidget(live)
        row.addWidget(rep)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()


class LayoutEditor(QGroupBox):
    """Which channel is in each quadrant of the 2×2."""

    def __init__(self):
        super().__init__("Channel layout (as on the screen)")
        g = QGridLayout(self)
        self.boxes = []
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems(CHANNELS)
            cb.setMinimumWidth(110)
            g.addWidget(QLabel(["top left", "top right", "bottom left", "bottom right"][i]), r * 2, c)
            g.addWidget(cb, r * 2 + 1, c)
            self.boxes.append(cb)
        self.set(["R1", "R3", "R6", "R8"])

    def set(self, layout):
        for cb, ch in zip(self.boxes, layout):
            cb.setCurrentText(ch)

    def get(self):
        return [cb.currentText().strip() or f"Q{i + 1}" for i, cb in enumerate(self.boxes)]


class LiveSetupPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.addWidget(_big("Live track — setup"))
        form = QFormLayout()
        self.track = QComboBox()
        self.track.setEditable(True)
        self.track.addItems([t.dir.name for t in list_tracks()])
        self.track.setCurrentText("")
        self.track.lineEdit().setPlaceholderText("new or existing track name")
        self.track.currentTextChanged.connect(self._load_saved)
        self.gates = QSpinBox()
        self.gates.setRange(2, 40)
        self.gates.setValue(10)
        self.device = QComboBox()
        self._fill_devices()
        form.addRow("Track", self.track)
        form.addRow("Number of gates", self.gates)
        dev_row = QHBoxLayout()
        dev_row.addWidget(self.device, 1)
        refresh = _btn("Refresh")
        refresh.setToolTip("List the devices again (after plugging in a capture card)")
        refresh.clicked.connect(self._fill_devices)
        dev_row.addWidget(refresh)
        form.addRow("Capture device", dev_row)
        self.fril = QCheckBox("Pilot names, round and race from fril.co.il live")
        self.fril.setChecked(True)
        form.addRow("", self.fril)
        v.addLayout(form)
        self.layout_ed = LayoutEditor()
        v.addWidget(self.layout_ed)
        adv = QGroupBox("Advanced (usually untouched)")
        af = QFormLayout(adv)
        self.min_run = QDoubleSpinBox(); self.min_run.setRange(5, 600); self.min_run.setValue(40); self.min_run.setSuffix(" s")
        self.end_gray = QDoubleSpinBox(); self.end_gray.setRange(1, 60); self.end_gray.setValue(10); self.end_gray.setSuffix(" s")
        self.learn_runs = QSpinBox(); self.learn_runs.setRange(2, 200); self.learn_runs.setValue(24)
        self.relearn = QSpinBox(); self.relearn.setRange(1, 500); self.relearn.setValue(12)
        af.addRow("Minimum run length", self.min_run)
        af.addRow("Run ends after gray for", self.end_gray)
        af.addRow("Runs to learn the track from", self.learn_runs)
        af.addRow("Re-learn every … runs", self.relearn)
        v.addWidget(adv)
        self.preview = QLabel("Preview")
        self.preview.setMinimumHeight(220)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#666")
        v.addWidget(self.preview)
        self.preview_info = QLabel("")
        self.preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.preview_info)
        row = QHBoxLayout()
        back = _btn("← Back")
        back.clicked.connect(lambda: win.show_page(win.start))
        pv = _btn("Preview device")
        pv.clicked.connect(self._preview)
        go = _btn("Start", True)
        go.clicked.connect(self._start)
        row.addWidget(back)
        row.addStretch()
        row.addWidget(pv)
        row.addWidget(go)
        v.addLayout(row)

    def _fill_devices(self):
        current = self.device.currentText().split(": ", 1)[-1]
        self.device.clear()
        try:
            for i, name in list_devices():           # OpenCV's index for each name
                self.device.addItem(f"{i}: {name}", i)
        except Exception:
            for i in range(4):
                self.device.addItem(f"device {i}", i)
        self.device.addItem("Test: play a 2×2 video file as if live…", "file")
        for k in range(self.device.count()):         # keep the chosen device after a refresh
            if self.device.itemText(k).split(": ", 1)[-1] == current:
                self.device.setCurrentIndex(k)

    def _load_saved(self, name):
        cfg = _saved_config(name.strip()) if name.strip() else {}
        if cfg:
            self.gates.setValue(int(cfg.get("n_gates") or self.gates.value()))
            self.layout_ed.set(cfg.get("layout", ["R1", "R3", "R6", "R8"]))
            self.min_run.setValue(cfg.get("min_run_s", 40))
            self.end_gray.setValue(cfg.get("end_gray_s", 10))
            self.learn_runs.setValue(cfg.get("learn_runs", 24))
            self.relearn.setValue(cfg.get("relearn_every", 12))
            self.fril.setChecked(cfg.get("fril_live", True))

    def _preview(self):
        d = self.device.currentData()
        if d == "file":
            return
        cap = open_device(int(d))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = False, None
        for _ in range(10):
            ok, frame = cap.read()
        cap.release()
        if ok:
            self.preview.setPixmap(_pixmap(frame, self.preview.width(), self.preview.height(), self.layout_ed.get()))
            self.preview_info.setText(_size_note(frame.shape[1], frame.shape[0]))
        else:
            self.preview.setText("No picture from this device")
            self.preview_info.setText("")

    def _start(self):
        name = self.track.currentText().strip()
        if not name:
            return QMessageBox.warning(self, "Track", "Type a track name.")
        source = None
        d = self.device.currentData()
        if d == "file":
            p, _ = QFileDialog.getOpenFileName(self, "2×2 test video (played in real time)", str(data_root()), VIDEO_FILTER)
            if not p:
                return
            source = FileSource(p, realtime=True, wall_start=time.time())
        else:
            try:
                source = DeviceSource(int(d))
            except Exception as e:
                return QMessageBox.critical(self, "Capture device", str(e))
            if source.size != (1920, 1080):
                ans = QMessageBox.question(
                    self, "Capture device",
                    f"{self.device.currentText()} gives {source.size[0]}×{source.size[1]}, not 1920×1080 — "
                    "it may not be the capture card. Start anyway?")
                if ans != QMessageBox.StandardButton.Yes:
                    source.cap.release()
                    return
        cfg = DayConfig(mode="live", n_gates=self.gates.value(), layout=self.layout_ed.get(),
                        min_run_s=self.min_run.value(), end_gray_s=self.end_gray.value(),
                        learn_runs=self.learn_runs.value(), relearn_every=self.relearn.value(),
                        fril_live=self.fril.isChecked())
        self.win.start_day(TrackPaths(data_root() / name), cfg, live_source=source)


class ReplaySetupPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.addWidget(_big("Replays — setup"))
        form = QFormLayout()
        self.track = QComboBox()
        self.info = QLabel("")
        self.info.setStyleSheet("color:#888")
        self.track.currentTextChanged.connect(self._track_changed)
        form.addRow("Learned track", self.track)
        form.addRow("", self.info)
        v.addLayout(form)
        self.layout_ed = LayoutEditor()
        v.addWidget(self.layout_ed)
        v.addWidget(QLabel("2×2 video files to analyse:"))
        self.files = QListWidget()
        self.files.setAcceptDrops(True)
        v.addWidget(self.files, 1)
        row = QHBoxLayout()
        back = _btn("← Back")
        back.clicked.connect(lambda: win.show_page(win.start))
        add = _btn("Add videos…")
        add.clicked.connect(self._add)
        go = _btn("Analyse", True)
        go.clicked.connect(self._start)
        row.addWidget(back)
        row.addStretch()
        row.addWidget(add)
        row.addWidget(go)
        v.addLayout(row)
        self.setAcceptDrops(True)

    def refresh(self):
        self.track.clear()
        self.track.addItems([t.dir.name for t in list_tracks() if labeling.is_learned(t)])

    def _track_changed(self, name):
        if not name:
            return
        tp = TrackPaths(data_root() / name)
        n = len(json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", []))
        self.info.setText(f"{n} gates learned")
        cfg = _saved_config(name)
        if cfg.get("layout"):
            self.layout_ed.set(cfg["layout"])

    def _add(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "2×2 video files", str(data_root()), VIDEO_FILTER)
        for p in ps:
            self.files.addItem(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.files.addItem(u.toLocalFile())

    def _start(self):
        name = self.track.currentText()
        if not name:
            return QMessageBox.warning(self, "Track", "No learned track — learn one in Live mode or with learn_track.py.")
        files = [self.files.item(i).text() for i in range(self.files.count())]
        if not files:
            return QMessageBox.warning(self, "Videos", "Add at least one 2×2 video.")
        tp = TrackPaths(data_root() / name)
        n = len(json.loads(tp.gate_memory.read_text(encoding="utf-8")).get("memory", []))
        cfg = DayConfig(mode="replays", n_gates=n, layout=self.layout_ed.get())
        self.win.start_day(tp, cfg, replay_files=files)


def _size_note(w: int, h: int) -> str:
    if (w, h) == (1920, 1080):
        return f"<span style='color:#4caf50'>{w}×{h} ✓</span>"
    return f"<span style='color:#e0a030'>{w}×{h} — the capture card should give 1920×1080; is this the right device?</span>"


def _pixmap(frame: np.ndarray, w: int, h: int, layout=None, status=None, names=None) -> QPixmap:
    img = frame.copy()
    H, W = img.shape[:2]
    if layout:
        for i, ch in enumerate(layout):
            x, y = (i % 2) * W // 2, (i // 2) * H // 2
            st = (status or {}).get(ch, {})
            state = st.get("state", "waiting")
            text = f"{ch}  " + ({"live": f"LIVE {st.get('seconds', 0):.0f}s", "dropout": "signal lost…"}.get(state, "waiting"))
            col = {"live": (60, 60, 255), "dropout": (0, 190, 255)}.get(state, (180, 180, 180))
            cv2.rectangle(img, (x + 10, y + 10), (x + 30 + 24 * len(text), y + 70), (0, 0, 0), -1)
            cv2.putText(img, text, (x + 20, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3, cv2.LINE_AA)
            name = (names or {}).get(ch.replace(" ", "").upper())
            if name:
                cv2.rectangle(img, (x + 10, y + 74), (x + 30 + 24 * len(name), y + 130), (0, 0, 0), -1)
                cv2.putText(img, name, (x + 20, y + 118), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(img, (W // 2, 0), (W // 2, H), (40, 40, 40), 2)
        cv2.line(img, (0, H // 2), (W, H // 2), (40, 40, 40), 2)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy()).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)


class RunningPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        self.title = _big("")
        left.addWidget(self.title)
        self.preview = QLabel("Waiting for video…")
        self.preview.setMinimumSize(640, 360)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#666")
        left.addWidget(self.preview, 1)
        self.log = QListWidget()
        self.log.setMaximumHeight(160)
        left.addWidget(self.log)
        h.addLayout(left, 3)
        right = QVBoxLayout()
        self.phase = _big("", 20)
        self.phase.setWordWrap(True)
        right.addWidget(self.phase)
        self.counts = QLabel("")
        self.counts.setWordWrap(True)
        self.counts.setStyleSheet("font-size:14px")
        right.addWidget(self.counts)
        right.addStretch()
        right.addWidget(QLabel("Dashboard (phones on this Wi-Fi):"))
        self.url = QLabel("")
        self.url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url.setStyleSheet("font-size:16px;font-weight:bold")
        right.addWidget(self.url)
        self.qr = QLabel()
        right.addWidget(self.qr)
        self.add_more = _btn("Add more videos…")
        self.add_more.clicked.connect(self._add_more)
        right.addWidget(self.add_more)
        stop = _btn("Stop")
        stop.clicked.connect(win.stop_day)
        right.addWidget(stop)
        h.addLayout(right, 1)
        self.frame = None

    def set_dashboard(self, url: str):
        self.url.setText(url)
        try:
            import io
            import segno
            buf = io.BytesIO()
            segno.make(url).save(buf, kind="png", scale=5, border=2)
            pm = QPixmap()
            pm.loadFromData(buf.getvalue())
            self.qr.setPixmap(pm)
        except Exception:
            self.qr.setText("(install segno for a QR code)")

    def on_preview(self, frame):
        self.frame = frame

    def _add_more(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "2×2 video files", str(data_root()), VIDEO_FILTER)
        if ps and self.win.day:
            self.win.day.add_replay_files(ps)

    def tick(self):
        day = self.win.day
        if not day:
            return
        st = day.status()
        live = st.get("live", {})
        pi = st.get("pilot_info") or {}
        if self.frame is not None:
            self.preview.setPixmap(_pixmap(self.frame, self.preview.width(), self.preview.height(),
                                           day.cfg.layout, live.get("channels"), pi.get("pilots")))
        if st["phase"] == "learn":
            n, need = st["learn_progress"]
            self.phase.setText(f"Learning the track\n{n} / {need} runs")
        else:
            self.phase.setText("Results" if st["mode"] == "live" else "Replays — results")
        by = st["runs_by_status"]
        per_ch = {}
        for r in day.state["runs"].values():
            per_ch[r["channel"]] = per_ch.get(r["channel"], 0) + 1
        lines = [f"Runs saved: <b>{st['runs_total']}</b> ({', '.join(f'{c}: {per_ch.get(c, 0)}' for c in day.cfg.layout)})",
                 f"Too short (discarded): {st['discarded']}",
                 f"Analysing: {st['queue']['analysing']} · waiting: {st['queue']['waiting']} · done: {by.get('labelled', 0) + by.get('analysed', 0)}"
                 + (f" · failed: {by['failed']}" if by.get("failed") else "")]
        if live.get("heat"):
            lines.append(f"Heat {live['heat']} in the air")
        if pi.get("source") == "fril":
            if not pi.get("reachable"):
                lines.append(f"fril live: <span style='color:#e55'>not reachable</span> — pilots named by heat and channel"
                             + (f" ({pi['error']})" if pi.get("error") else ""))
            elif pi.get("round") is None:
                lines.append("fril live: reachable, no heat loaded yet")
            else:
                heat_txt = " · ".join(x for x in (pi.get("stage"), f"Round {pi['round']}", f"Race {pi.get('race', '–')}") if x)
                names = ", ".join(f"{c} {n}" for c, n in sorted(pi.get("pilots", {}).items())) or "no pilots"
                lines.append(f"fril live: <b>{heat_txt}</b> — {pi.get('phase') or '?'} — {names}")
        if st.get("void_runs"):
            lines.append(f"{st['void_runs']} run(s) kept as void (restarted race) — saved, hidden from the dashboard")
        for p, f in st.get("replay_files", {}).items():
            lines.append(f"{Path(p).name}: {f['status']} ({f['runs']} runs)")
        if st.get("error"):
            lines.append(f"<span style='color:#e55'>{st['error']}</span>")
        self.counts.setText("<br>".join(lines))


# ──────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Race day")
        self.resize(1280, 800)
        self.day = None
        self.dash = None
        self.bridge = Bridge()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.start = StartPage(self)
        self.live_setup = LiveSetupPage(self)
        self.replay_setup = ReplaySetupPage(self)
        self.running = RunningPage(self)
        for p in (self.start, self.live_setup, self.replay_setup, self.running):
            self.stack.addWidget(p)
        self.bridge.preview.connect(self.running.on_preview)
        self.bridge.log.connect(lambda s: (self.running.log.addItem(s), self.running.log.scrollToBottom()))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.running.tick)

    def show_page(self, page):
        if page is self.replay_setup:
            self.replay_setup.refresh()
        self.stack.setCurrentWidget(page)

    def start_day(self, tp: TrackPaths, cfg: DayConfig, live_source=None, replay_files=None):
        try:
            provider = FrilLiveProvider() if cfg.mode == "live" and cfg.fril_live else None
            self.day = RaceDay(tp, cfg, provider=provider,
                               log=lambda s: self.bridge.log.emit(time.strftime("%H:%M:%S  ") + s))
            self.day.on_preview = lambda f: self.bridge.preview.emit(f)
            self.day.start()
        except Exception as e:
            self.day = None
            return QMessageBox.critical(self, "Can't start", str(e))
        self.dash = Dashboard(self.day).start()
        self.running.set_dashboard(self.dash.url)
        self.running.title.setText(f"{'LIVE' if cfg.mode == 'live' else 'REPLAYS'} — {tp.name}")
        self.running.add_more.setVisible(cfg.mode == "replays")
        if live_source is not None:
            self.day.start_live(live_source)
        if replay_files:
            self.day.add_replay_files(replay_files)
        self.timer.start(500)
        self.show_page(self.running)

    def stop_day(self):
        if self.day and QMessageBox.question(self, "Stop", "Stop recording / analysing and close the dashboard?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.timer.stop()
        if self.day:
            self.day.stop()
        if self.dash:
            self.dash.stop()
        self.day = self.dash = None
        self.running.frame = None
        self.running.log.clear()
        self.show_page(self.start)

    def closeEvent(self, e):
        if self.day:
            self.day.stop()
        if self.dash:
            self.dash.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
