#!/usr/bin/env python3
"""
Race-day desktop app (operator) — see docs/RACE_DAY_PLAN.md.

    python lazy_spotter_v2.py

Start screen: FPVTrackside event, Live capture or Replays.
Events are chosen from the data folder (race_day/picker.py), never browsed for.

  FPVTrackside event: an event folder of FPVTrackside — every finished heat
            is listed; learn the track from chosen heats, check each pilot's
            laps against the video, write corrections back (race_day/event_ui.py).
  Live capture: a track of the data folder (or a new one, or one of the last 3),
            single or 2×2 feed and its channels, the capture card → the
            operator records heats (Record / Stop, 3 min at most) into
            <track>/videos/, picks heats to analyse and to learn the track from;
            the app's data is in <track>/lazySpotter/ (race_day/live_ui.py).
  Replays:  any track or FPVTrackside event of the data folder: a track's
            videos (<track>/videos/, single pilot or 2×2) listed as heats,
            analysed and learned from on demand, the results in the browser —
            the live capture screen without the live feed (data in
            <track>/lazySpotter/); an FPVTrackside event opens in the event
            screen without Auto.

The browser dashboard (read-only) starts with the app and shows every event
of the data folder (the open one live); its address is on the start screen
and in the top bar.

The data folder (tracks, training data — FPV_DATA_ROOT for the whole app)
is asked for the first time, kept in ~/.lazy_spotter_v2.json and can be changed
on the start screen.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from dataset_paths import data_root
from race_day.capture import DeviceSource, FileSource, HeatRecorder, list_devices, open_device
from race_day.event_ui import EventPage
from race_day.live_ui import LivePage, feed_pixmap
from race_day import picker
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


def _use_data_root(folder: Path):
    """Make `folder` the data folder for the whole app — dataset_paths.data_root()
    reads FPV_DATA_ROOT, and the analysis workers inherit it."""
    os.environ["FPV_DATA_ROOT"] = str(folder)


def _ask_data_root(parent, start: Path) -> Optional[Path]:
    d = QFileDialog.getExistingDirectory(parent, "Data folder (tracks, training data — e.g. FPVdatasets)", str(start))
    return Path(d) if d else None


def setup_data_root() -> bool:
    """At start: the data folder saved last time; the first time, ask for one
    (the dialog starts at the current default). False: none chosen."""
    saved = _app_settings().get("data_root")
    if saved and Path(saved).is_dir():
        _use_data_root(Path(saved))
        return True
    QMessageBox.information(None, "Race day", "Choose the data folder: where the tracks and training data are "
                            "(e.g. FPVdatasets). You can change it later on the start screen.")
    folder = _ask_data_root(None, data_root())
    if not folder:
        return False
    _save_app_settings(data_root=str(folder))
    _use_data_root(folder)
    return True


RECENT = 3          # past tracks offered on the setup page


def _recent_tracks() -> List[str]:
    """The last tracks opened that are in the current data folder."""
    root = data_root().resolve()
    return [p for p in _app_settings().get("recent_tracks", [])
            if Path(p).is_dir() and Path(p).resolve().is_relative_to(root)][:RECENT]


def _remember_track(track_dir: Path):
    rest = [p for p in _app_settings().get("recent_tracks", []) if p != str(track_dir)]
    _save_app_settings(recent_tracks=[str(track_dir)] + rest[:9])


SETTINGS = ".lazy_spotter_v2.json"
OLD_SETTINGS = ".race_day_app.json"       # before the app was renamed: read until the new file exists


def _app_settings() -> dict:
    for name in (SETTINGS, OLD_SETTINGS):
        try:
            return json.loads((Path.home() / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _save_app_settings(**kw):
    d = _app_settings()
    d.update(kw)
    try:
        (Path.home() / SETTINGS).write_text(json.dumps(d), encoding="utf-8")
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
        folder_row = QHBoxLayout()
        self.folder = QLabel(f"Data folder: {data_root()}")
        self.folder.setStyleSheet("color:#888")
        change = QPushButton("Change…")
        change.setToolTip("Choose another data folder (tracks, training data)")
        change.clicked.connect(self._change_folder)
        folder_row.addStretch()
        folder_row.addWidget(self.folder)
        folder_row.addWidget(change)
        folder_row.addStretch()
        v.addLayout(folder_row)
        self.dash = QLabel("")
        self.dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dash.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.dash)
        row = QHBoxLayout()
        event = _btn("⏱  FPVTrackside event\ncheck every heat's laps in the video", True)
        live = _btn("●  Live capture\nrecord from the capture card")
        rep = _btn("▶  Replays\nany track or event of the data folder")
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

    def _change_folder(self):
        folder = _ask_data_root(self, data_root())
        if folder:
            _save_app_settings(data_root=str(folder))
            _use_data_root(folder)
            self.folder.setText(f"Data folder: {folder}")


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
        self.track.setPlaceholderText("choose a track of the data folder, or make a new one")
        browse = _btn("Choose track…")
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
        form.addRow("Track", track_row)
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
        root = data_root()
        it = picker.pick(self, "Live capture — choose the track", picker.track_folders(root), root, allow_new=True,
                         empty_text="No tracks yet — make one with New track…")
        if it:
            self._choose(it["path"])

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
        self._start_dashboard()

    def show_page(self, page):
        self.stack.setCurrentWidget(page)

    def _log(self, s: str):
        self.bridge.log.emit(time.strftime("%H:%M:%S  ") + s)

    def _start_dashboard(self):
        """One dashboard for the whole session: every event of the data folder
        (it follows a change of data folder), the open one read live."""
        try:
            self.dash = Dashboard(data_root, lambda: self.event_day).start()
            url = self.dash.url
        except OSError as e:                     # port in use: the app works without it
            self.dash, url = None, f"dashboard off ({e})"
        for page in (self.event_page, self.live_page, self.replays_page):
            page.dash_url = url
        self.start.dash.setText(f"Browser (every event of the data folder): <b>{url}</b>" if self.dash else url)

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
        self.live_page.open_track(day, rec, setup)
        self.show_page(self.live_page)

    # ── Replays: a track folder's videos ──────────────────────

    def choose_replays(self):
        """Any track or FPVTrackside event of the data folder: a track opens in the
        Replays screen, an FPVTrackside event in the event screen (no Auto)."""
        root = data_root()
        it = picker.pick(self, "Replays — choose a track or an event",
                         picker.track_folders(root) + picker.fpvtrackside_events(root), root)
        if not it:
            return
        if it["kind"] == "track":
            self.start_replays(it["path"])
        else:
            self.start_event(str(it["path"]), "replays")

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
        self.replays_page.open_track(self.event_day, setup=_saved_setup(track_dir))
        self.show_page(self.replays_page)

    # ── FPVTrackside event ────────────────────────────────────

    def choose_event(self, mode: str):
        root = data_root()
        it = picker.pick(self, "FPVTrackside event — choose the event", picker.fpvtrackside_events(root), root,
                         empty_text="No FPVTrackside events in the data folder (folders with an Event.json).")
        if it:
            self.start_event(str(it["path"]), mode)

    def start_event(self, folder: str, mode: str):
        from race_day.eventday import EventDay
        try:
            self.event_day = EventDay(folder, mode=mode, log=self._log)
            self.event_day.start()
        except Exception as e:
            self.event_day = None
            return QMessageBox.critical(self, "Can't open the event", str(e))
        _save_app_settings(event_dir=folder)
        st = _app_settings()
        self.event_day.set_fpvtrackside(st.get("fpvtrackside_url") or "http://localhost:8080",
                                        st.get("fpvtrackside_token") or os.environ.get("FPV_API_TOKEN"))
        self.event_day.set_fril(st.get("fril_live_url"))
        self.event_page.open(self.event_day)
        self.show_page(self.event_page)

    def set_fril(self):
        """fril.co.il/live's ingest address, with its token: the race on now and
        the races to come are sent there. Kept in the app's settings."""
        st = _app_settings()
        url, ok = QInputDialog.getText(
            self, "fril live", "The race on now and the races to come are sent to fril.co.il/live, the same way "
            "FPVTrackside sends its live results.\n\nfril's live ingest address, with its token (empty = don't send):",
            text=st.get("fril_live_url") or "https://fril.co.il/api/live/ingest?token=")
        if not ok:
            return
        url = url.strip()
        if url.endswith("?token="):                   # the template, never filled in
            url = ""
        if url and not url.startswith(("http://", "https://")):
            return QMessageBox.warning(self, "fril live", "The address starts with https:// (or http://).")
        _save_app_settings(fril_live_url=url)
        if self.event_day:
            self.event_day.set_fril(url)

    def set_fpvtrackside(self):
        """Where FPVTrackside's web server is, and its API token (only needed
        from another computer). Kept in the app's settings."""
        st = _app_settings()
        url, ok = QInputDialog.getText(
            self, "FPVTrackside", "FPVTrackside's web server (Menu → Application Settings → Web → Auto start HTTP "
            "Server).\nWith the FPVTrackside alpha (its REST API) and this event open there, lap changes are made in "
            "FPVTrackside directly.\n\nAddress:", text=st.get("fpvtrackside_url") or "http://localhost:8080")
        if not ok:
            return
        token, ok = QInputDialog.getText(
            self, "FPVTrackside", "API token — only when FPVTrackside runs on another computer (a web user's API key, "
            "or its FPV_API_TOKEN). Leave empty on the same computer:", text=st.get("fpvtrackside_token") or "")
        if not ok:
            return
        url = url.strip() or "http://localhost:8080"
        if "://" not in url:
            url = "http://" + url
        _save_app_settings(fpvtrackside_url=url, fpvtrackside_token=token.strip())
        if self.event_day:
            self.event_day.set_fpvtrackside(url, token.strip() or os.environ.get("FPV_API_TOKEN"))

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
        """Close the open event (the dashboard keeps running: it shows every event)."""
        for page in (self.event_page, self.live_page, self.replays_page):
            page.close_day()
        if self.event_day:
            self.event_day.stop()
        self.event_day = None

    def closeEvent(self, e):
        self._close()
        if self.dash:
            self.dash.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    if not setup_data_root():
        sys.exit(0)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
