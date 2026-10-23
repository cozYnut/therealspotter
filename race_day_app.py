#!/usr/bin/env python3
"""
Race-day desktop app (operator) — see docs/RACE_DAY_PLAN.md.

    python race_day_app.py

Start screen: FPVTrackside event, Live capture or Replays.
  FPVTrackside event: an event folder of FPVTrackside — every finished heat
            is listed; learn the track from chosen heats, check each pilot's
            laps against the video, write corrections back (race_day/event_ui.py).
  Live capture: a track folder (new, existing or one of the last 3),
            single or 2×2 feed and its channels, the capture card → the
            operator records heats (Record / Stop, 3 min at most) into
            <track>/videos/, picks heats to analyse and to learn the track from;
            the app's data is in <track>/lazySpotter/ (race_day/live_ui.py).
  Replays:  a track folder: its videos (<track>/videos/, single pilot or
            2×2) listed as heats, analysed and learned from on demand, the
            results in the browser — the live capture screen without the live
            feed (data in <track>/lazySpotter/).

The browser dashboard (read-only) is served on the local network; its
address is shown in the top bar.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from dataset_paths import data_root
from race_day.capture import DeviceSource, FileSource, HeatRecorder, list_devices, open_device
from race_day.event_ui import EventPage
from race_day.live_ui import LivePage, feed_pixmap
from race_day.server import Dashboard
from race_day.trackside import LIVE_SUBDIR

CHANNELS = [f"{b}{i}" for b in "RFEAL" for i in range(1, 9)]
VIDEO_FILTER = "Videos (*.mp4 *.mov *.mkv *.avi *.MP4 *.MOV)"


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


def _saved_setup(track_dir: Path) -> dict:
    """A track's live-capture setup (<track>/lazySpotter/race_day.json "setup"), or
    the channels and gate count of the old live mode's <track>/race_day.json."""
    for p, key in ((Path(track_dir) / LIVE_SUBDIR / "race_day.json", "setup"),
                   (Path(track_dir) / "race_day.json", "config")):
        try:
            d = json.loads(p.read_text(encoding="utf-8")).get(key) or {}
        except (OSError, ValueError):
            continue
        if d:
            return d if key == "setup" else {"channels": d.get("layout"), "n_gates": d.get("n_gates")}
    return {}


RECENT = 3          # past tracks offered on the setup page


def _recent_tracks() -> List[str]:
    return [p for p in _app_settings().get("recent_tracks", []) if Path(p).is_dir()][:RECENT]


def _remember_track(track_dir: Path):
    rest = [p for p in _app_settings().get("recent_tracks", []) if p != str(track_dir)]
    _save_app_settings(recent_tracks=[str(track_dir)] + rest[:9])


def _app_settings() -> dict:
    p = Path.home() / ".race_day_app.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_app_settings(**kw):
    d = _app_settings()
    d.update(kw)
    try:
        (Path.home() / ".race_day_app.json").write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


class Bridge(QObject):
    """Worker threads → GUI thread."""
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
        event = _btn("⏱  FPVTrackside event\ncheck every heat's laps in the video", True)
        live = _btn("●  Live capture\nrecord from the capture card")
        rep = _btn("▶  Replays\na track folder's videos")
        for b in (event, live, rep):
            b.setMinimumSize(300, 120)
        event.clicked.connect(lambda: win.choose_event("live"))
        live.clicked.connect(lambda: win.show_page(win.live_setup))
        rep.clicked.connect(win.choose_replays)
        row.addStretch()
        row.addWidget(event)
        row.addWidget(live)
        row.addWidget(rep)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()


class LayoutEditor(QGroupBox):
    """The channel of each quadrant of a 2×2 feed (or the single pilot's channel)."""

    def __init__(self):
        super().__init__("Channels (as on the screen)")
        g = QGridLayout(self)
        self.boxes, self.labels = [], []
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems(CHANNELS)
            cb.setMinimumWidth(110)
            lab = QLabel(["top left", "top right", "bottom left", "bottom right"][i])
            g.addWidget(lab, r * 2, c)
            g.addWidget(cb, r * 2 + 1, c)
            self.boxes.append(cb)
            self.labels.append(lab)
        self.set(["R1", "R3", "R6", "R8"])
        self.set_single(False)

    def set_single(self, single: bool):
        """A single feed has one channel: only the first box is used."""
        self.single = single
        self.labels[0].setText("channel" if single else "top left")
        for w in self.boxes[1:] + self.labels[1:]:
            w.setVisible(not single)

    def set(self, layout):
        for cb, ch in zip(self.boxes, layout):
            cb.setCurrentText(ch)

    def get(self):
        chans = [cb.currentText().strip() or f"Q{i + 1}" for i, cb in enumerate(self.boxes)]
        return chans[:1] if self.single else chans


class LiveSetupPage(QWidget):
    """Live capture setup: the track folder (chosen, or one of the last
    three), the feed (single or 2×2) and its channels, the capture device.
    Saved with the track (<track>/lazySpotter/race_day.json)."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.track_dir: Optional[Path] = None
        v = QVBoxLayout(self)
        v.addWidget(_big("Live capture — setup"))
        form = QFormLayout()
        track_row = QHBoxLayout()
        self.track = QLineEdit()
        self.track.setReadOnly(True)
        self.track.setPlaceholderText("choose the track folder (a new, empty folder for a new track)")
        browse = _btn("Choose folder…")
        browse.clicked.connect(self._browse)
        track_row.addWidget(self.track, 1)
        track_row.addWidget(browse)
        self.recent_row = QHBoxLayout()
        self.feed = QComboBox()
        self.feed.addItems(["2×2 — four pilots", "Single pilot"])
        self.feed.currentIndexChanged.connect(lambda i: self.layout_ed.set_single(i == 1))
        self.gates = QSpinBox()
        self.gates.setRange(2, 40)
        self.gates.setValue(10)
        self.gates.setToolTip("Used when learning the track (you can change it at Relearn)")
        self.max_len = QSpinBox()
        self.max_len.setRange(10, 900)
        self.max_len.setValue(180)
        self.max_len.setSuffix(" s")
        self.max_len.setToolTip("A recording stops by itself after this long (or earlier with Stop)")
        self.device = QComboBox()
        self._fill_devices()
        form.addRow("Track folder", track_row)
        form.addRow("Recent", self.recent_row)
        form.addRow("Feed", self.feed)
        form.addRow("Number of gates", self.gates)
        dev_row = QHBoxLayout()
        dev_row.addWidget(self.device, 1)
        refresh = _btn("Refresh")
        refresh.setToolTip("List the devices again (after plugging in a capture card)")
        refresh.clicked.connect(self._fill_devices)
        dev_row.addWidget(refresh)
        form.addRow("Capture device", dev_row)
        form.addRow("Longest recording", self.max_len)
        v.addLayout(form)
        self.layout_ed = LayoutEditor()
        v.addWidget(self.layout_ed)
        self.preview = QLabel("Preview")
        self.preview.setMinimumHeight(220)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#666")
        v.addWidget(self.preview, 1)
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
        self.device.addItem("Test: play a video file as if live…", "file")
        for k in range(self.device.count()):         # keep the chosen device after a refresh
            if self.device.itemText(k).split(": ", 1)[-1] == current:
                self.device.setCurrentIndex(k)

    def showEvent(self, e):
        super().showEvent(e)
        self._fill_recent()

    def _fill_recent(self):
        while self.recent_row.count():
            w = self.recent_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        recent = _recent_tracks()
        for p in recent:
            b = QPushButton(Path(p).name)
            b.setToolTip(p)
            b.clicked.connect(lambda _=False, p=p: self._choose(Path(p)))
            self.recent_row.addWidget(b)
        if not recent:
            self.recent_row.addWidget(QLabel("<span style='color:#888'>none yet</span>"))
        self.recent_row.addStretch()

    def _browse(self):
        start = str(self.track_dir or (Path(_recent_tracks()[0]).parent if _recent_tracks() else data_root()))
        d = QFileDialog.getExistingDirectory(self, "Track folder (its videos go in <track>/videos, "
                                             "the app's data in <track>/lazySpotter)", start)
        if d:
            self._choose(Path(d))

    def _choose(self, track_dir: Path):
        self.track_dir = track_dir
        self.track.setText(str(track_dir))
        self._load_saved(track_dir)

    def _load_saved(self, track_dir: Path):
        s = _saved_setup(track_dir)
        if s:
            chans = s.get("channels") or ["R1", "R3", "R6", "R8"]
            self.feed.setCurrentIndex(1 if len(chans) == 1 else 0)
            self.layout_ed.set(chans)
            self.gates.setValue(int(s.get("n_gates") or self.gates.value()))
            self.max_len.setValue(int(s.get("max_s") or 180))
            dev = s.get("device")
            for k in range(self.device.count()):
                if dev and self.device.itemText(k).split(": ", 1)[-1] == dev:
                    self.device.setCurrentIndex(k)

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
            self.preview.setPixmap(feed_pixmap(frame, self.preview.width(), self.preview.height(),
                                               self.layout_ed.get(), False))
            self.preview_info.setText(_size_note(frame.shape[1], frame.shape[0]))
        else:
            self.preview.setText("No picture from this device")
            self.preview_info.setText("")

    def _start(self):
        if not self.track_dir:
            return QMessageBox.warning(self, "Track", "Choose the track folder (or one of the recent tracks).")
        d = self.device.currentData()
        if d == "file":
            p, _ = QFileDialog.getOpenFileName(self, "Test video (played in real time as if live)", str(data_root()),
                                               VIDEO_FILTER)
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
        setup = {"channels": self.layout_ed.get(), "n_gates": self.gates.value(), "max_s": self.max_len.value(),
                 "device": self.device.currentText().split(": ", 1)[-1] if d != "file" else None}
        self.win.start_capture(self.track_dir, setup, source)


def _size_note(w: int, h: int) -> str:
    if (w, h) == (1920, 1080):
        return f"<span style='color:#4caf50'>{w}×{h} ✓</span>"
    return f"<span style='color:#e0a030'>{w}×{h} — the capture card should give 1920×1080; is this the right device?</span>"


# ──────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Race day")
        self.resize(1280, 800)
        self.dash = None
        self.bridge = Bridge()
        self.bridge.log.connect(print)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.start = StartPage(self)
        self.live_setup = LiveSetupPage(self)
        self.event_page = EventPage(self)
        self.live_page = LivePage(self)
        self.replays_page = LivePage(self, live=False)
        self.event_day = None
        for p in (self.start, self.live_setup, self.event_page, self.live_page, self.replays_page):
            self.stack.addWidget(p)

    def show_page(self, page):
        self.stack.setCurrentWidget(page)

    def _log(self, s: str):
        self.bridge.log.emit(time.strftime("%H:%M:%S  ") + s)

    def _start_dashboard(self, page):
        try:
            self.dash = Dashboard(self.event_day).start()
            page.dash_url = self.dash.url
        except OSError as e:                     # port in use: the app works without it
            self.dash = None
            page.dash_url = f"dashboard off ({e})"

    # ── live capture ──────────────────────────────────────────

    def start_capture(self, track_dir: Path, setup: dict, source):
        from race_day.eventday import EventDay
        from race_day.trackside import TrackVideos
        try:
            self.event_day = EventDay(event=TrackVideos(track_dir), mode="capture", log=self._log)
            with self.event_day.lock:
                self.event_day.state["n_gates"] = setup["n_gates"]
                self.event_day.state["setup"] = setup
            self.event_day._save()
            self.event_day.start()
        except Exception as e:
            self.event_day = None
            source.stop()
            return QMessageBox.critical(self, "Can't start", str(e))
        day = self.event_day
        _remember_track(Path(track_dir))

        def saved(path: Path, info: dict):          # a recording is complete: its length and time
            day.event.set_info(path.name, {**day.event.info().get(path.name, {}), **info})
            self._log(f"{path.name} saved ({info['seconds']:.0f} s)")
        rec = HeatRecorder(source, day.event.dir, day.tp.dir / "recording", max_s=float(setup["max_s"]),
                           on_saved=saved).start()
        self._start_dashboard(self.live_page)
        self.live_page.open_track(day, rec, setup)
        self.show_page(self.live_page)

    # ── Replays: a track folder's videos ──────────────────────

    def choose_replays(self):
        recent = _recent_tracks()
        start = str(Path(recent[0]).parent) if recent else str(data_root())
        folder = QFileDialog.getExistingDirectory(self, "Replays: the track folder (its videos are in <track>/videos)",
                                                  start)
        if folder:
            self.start_replays(Path(folder))

    def start_replays(self, track_dir: Path):
        from race_day.eventday import EventDay
        from race_day.trackside import TrackVideos
        try:
            self.event_day = EventDay(event=TrackVideos(track_dir), mode="replays", log=self._log)
            self.event_day.start()
        except Exception as e:
            self.event_day = None
            return QMessageBox.critical(self, "Can't open the track", str(e))
        _remember_track(track_dir)
        self._start_dashboard(self.replays_page)
        self.replays_page.open_track(self.event_day, setup=_saved_setup(track_dir))
        self.show_page(self.replays_page)

    # ── FPVTrackside event ────────────────────────────────────

    def choose_event(self, mode: str):
        start = str(Path.home())
        last = _app_settings().get("event_dir")
        if last and Path(last).exists():
            start = str(Path(last).parent)
        folder = QFileDialog.getExistingDirectory(
            self, "FPVTrackside event folder (…/FPVTrackside/events/<event>)" if mode == "live"
            else "Replays: an FPVTrackside event folder, or a folder of videos", start)
        if folder:
            self.start_event(folder, mode)

    def start_event(self, folder: str, mode: str):
        from race_day.eventday import EventDay
        try:
            self.event_day = EventDay(folder, mode=mode, log=self._log)
            self.event_day.start()
        except Exception as e:
            self.event_day = None
            return QMessageBox.critical(self, "Can't open the event", str(e))
        _save_app_settings(event_dir=folder)
        self._start_dashboard(self.event_page)
        self.event_page.open(self.event_day)
        self.show_page(self.event_page)

    def stop_event(self):
        """Close the event / live track (EventPage's and LivePage's Close)."""
        page = self.stack.currentWidget()
        recording = page is self.live_page and self.live_page.rec and self.live_page.rec.status()["recording"]
        if self.event_day and QMessageBox.question(
                self, "Close", ("A heat is being recorded — it's saved as it is. " if recording else "")
                + "Close? (Nothing is lost — it continues where it was next time.)") \
                != QMessageBox.StandardButton.Yes:
            return
        self._close()
        self.show_page(self.start)

    def _close(self):
        for page in (self.event_page, self.live_page, self.replays_page):
            page.close_day()
        if self.event_day:
            self.event_day.stop()
        if self.dash:
            self.dash.stop()
        self.event_day = self.dash = None

    def closeEvent(self, e):
        self._close()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
