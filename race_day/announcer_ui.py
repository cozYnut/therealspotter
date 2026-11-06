"""
The announcer's settings window (the FPVTrackside screen's "Announcer…").

    ┌ Announcer ───────────────────────────────────────────────────────┐
    │ ☑ The app speaks the race calls (English)                        │
    │ Club [fril ▾]  Track [wingate ▾]          (clubs/<club>/: its words)
    │ Voice [Samantha ▾]  Speed [normal]  [Test]                       │
    │ FPVTrackside's own voice: on   [Turn it off]                     │
    │ ☑ Mid-lap positions from the video  Card [USB Video ▾]  15 fps   │
    │   live video: 14.8 fps · 0 % dropped · 60 ms behind              │
    │ [▶ Replay the selected heat]                                     │
    │ FPVTrackside → the app: Notifier ExtensionMode, NotificationURL  │
    │   http://localhost:8766/   (events received · passed on to fril) │
    └──────────────────────────────────────────────────────── [Close] ┘
"""

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)

from race_day import announcer as ann
from race_day.clubstyle import clubs

SYSTEM_VOICE = "(the Mac's own voice)"
NO_TRACK = "(no track file)"


class AnnouncerDialog(QDialog):
    def __init__(self, win, announcer, relay, save, load):
        """save(**settings) / load(): the app's settings."""
        super().__init__(win)
        self.setWindowTitle("Announcer")
        self.win, self.a, self.relay, self.save, self.load = win, announcer, relay, save, load
        v = QVBoxLayout(self)

        self.on = QCheckBox("The app speaks the race calls (English)")
        self.on.setChecked(announcer.enabled)
        self.on.setToolTip("Lap calls, places, the holeshot, the last lap and who finishes where, from FPVTrackside's\n"
                           "live events; where each pilot is mid-lap, from the live video; and what FPVTrackside\n"
                           "itself would say (next up, results, hurry up…), from its log. Turn FPVTrackside's own\n"
                           "voice off below, or you hear both.")
        self.on.toggled.connect(self._on)
        v.addWidget(self.on)

        row = QHBoxLayout()
        row.addWidget(QLabel("Club"))
        self.club = QComboBox()
        self.club.addItems(clubs())
        self.club.setToolTip("The club's own words for every call: clubs/<club>/ (lines.yaml, pilots.yaml, tracks/).\n"
                             "clubs/example/ shows how to make one.")
        if announcer.style.club:
            self.club.setCurrentText(announcer.style.club)
        self.club.currentTextChanged.connect(self._club)
        row.addWidget(self.club, 1)
        row.addWidget(QLabel("Track"))
        self.track = QComboBox()
        self.track.setToolTip("The track file: what to call its gates (\"the dive gate\") and where pilots go down")
        self._fill_tracks()
        self.track.currentTextChanged.connect(self._style)
        row.addWidget(self.track, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Voice"))
        self.voice = QComboBox()
        self.voice.addItem(SYSTEM_VOICE)
        self.voice.addItems(ann.english_voices())
        if announcer.speaker.voice:
            self.voice.setCurrentText(announcer.speaker.voice)
        self.voice.currentTextChanged.connect(self._voice)
        row.addWidget(self.voice, 1)
        row.addWidget(QLabel("Speed"))
        self.rate = QSpinBox()
        self.rate.setRange(0, 400)
        self.rate.setSingleStep(10)
        self.rate.setSpecialValueText("normal")
        self.rate.setSuffix(" words/min")
        self.rate.setValue(announcer.speaker.rate or 0)
        self.rate.valueChanged.connect(self._voice)
        row.addWidget(self.rate)
        test = QPushButton("Test")
        test.clicked.connect(announcer.test)
        row.addWidget(test)
        v.addLayout(row)

        row = QHBoxLayout()
        self.ft_label = QLabel("")
        row.addWidget(self.ft_label, 1)
        self.ft_btn = QPushButton("")
        self.ft_btn.clicked.connect(self._ft_speech)
        row.addWidget(self.ft_btn)
        v.addLayout(row)

        st = load()
        row = QHBoxLayout()
        self.video = QCheckBox("Mid-lap positions from the video")
        self.video.setChecked(bool(st.get("live_video_on")))
        self.video.setToolTip("FPVTrackside's capture card is read a second time and each pilot's gates are named as\n"
                              "they're flown, on the event's learned track: who takes the lead and where, the gap at a\n"
                              "gate, who is down. Needs the FPVTrackside event open here, with its track learned.\n"
                              "Its own process: ~1.5 GB of memory, and the Mac's Neural Engine.")
        self.video.toggled.connect(self._video)
        row.addWidget(self.video)
        row.addWidget(QLabel("Card"))
        self.device = QComboBox()
        try:
            from race_day.capture import list_devices
            for i, name in list_devices():
                self.device.addItem(f"{i}: {name}", i)
        except Exception:                        # noqa: BLE001 — no camera list: the index alone
            self.device.addItem("0", 0)
        idx = self.device.findData(int(st.get("live_video_device") or 0))
        self.device.setCurrentIndex(max(0, idx))
        self.device.currentIndexChanged.connect(self._video)
        row.addWidget(self.device, 1)
        self.fps = QSpinBox()
        self.fps.setRange(5, 30)
        self.fps.setSuffix(" fps")
        self.fps.setValue(int(st.get("live_video_fps") or 15))
        self.fps.setToolTip("Frames analysed a second per pilot. Fewer: lighter on the Mac, gates named a little later.")
        self.fps.valueChanged.connect(self._video)
        row.addWidget(self.fps)
        v.addLayout(row)
        self.video_state = QLabel("")
        self.video_state.setStyleSheet("color:#888")
        v.addWidget(self.video_state)

        talk = self.a.talk
        row = QHBoxLayout()
        self.talk_on = QCheckBox("Talk between races")
        self.talk_on.setChecked(bool(talk and talk.enabled))
        self.talk_on.setToolTip("Fill the gaps (clubs/<club>/talk.yaml): each race's preview and wrap, the qualifying\n"
                                "results, the bracket explained, the final walk-out, the champion, and filler in a quiet\n"
                                "gap. Never during a race: it stops at the start call.")
        self.talk_on.toggled.connect(self._talk_on)
        row.addWidget(self.talk_on)
        self.open_btn = QPushButton("▶ Opening (about 10 min)")
        self.open_btn.setToolTip("The welcome, the day's plan, safety, the track gate by gate, the rules of qualifying\n"
                                 "and the brackets, the points, the season so far and who to watch — one after another.\n"
                                 "The closing comes by itself when Places & points shows final results.")
        self.open_btn.clicked.connect(lambda: talk and talk.opening())
        row.addWidget(self.open_btn)
        stop = QPushButton("■ Stop talking")
        stop.clicked.connect(lambda: talk and talk.stop())
        row.addWidget(stop)
        v.addLayout(row)
        self.talk_state = QLabel("")
        self.talk_state.setStyleSheet("color:#888")
        self.talk_state.setWordWrap(True)
        v.addWidget(self.talk_state)

        self.replay_btn = QPushButton("▶ Replay the selected heat")
        self.replay_btn.setToolTip("The heat picked on the FPVTrackside screen, played to the announcer as if it were being\n"
                                   "raced: its video for the capture card, its crossings for FPVTrackside's events.\n"
                                   "Nothing goes to fril. Turn the calls on above to hear it.")
        self.replay_btn.clicked.connect(self._replay)
        v.addWidget(self.replay_btn)

        url = f"http://localhost:{relay.port}/" if relay else None
        how = QLabel(
            "<b>FPVTrackside → the app</b>: in FPVTrackside, Menu → Application Settings → <i>Gate / LED POST "
            "notifications</i>: Notifier enabled / mode = <b>ExtensionMode</b>, NotificationURL = "
            f"<b>{url}</b>, then restart FPVTrackside.<br>The app passes every event on to fril "
            "(its <i>fril live…</i> address), so fril's live board works as before — <b>while the app is open</b>."
            if url else "⚠ The app can't take FPVTrackside's events: its port is in use.")
        how.setWordWrap(True)
        how.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(how)
        self.stats = QLabel("")
        self.stats.setStyleSheet("color:#888")
        v.addWidget(self.stats)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        v.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(1000)
        self._refresh()

    def _on(self, on: bool):
        self.a.set_enabled(on)
        self.save(announcer_on=on)

    def _fill_tracks(self):
        from race_day.clubstyle import ClubStyle
        self.track.blockSignals(True)
        self.track.clear()
        self.track.addItem(NO_TRACK)
        self.track.addItems(ClubStyle(self.club.currentText() or None).tracks)
        if self.a.style.track_name:
            self.track.setCurrentText(self.a.style.track_name)
        self.track.blockSignals(False)

    def _club(self, *_):
        self._fill_tracks()
        self._style()

    def _style(self, *_):
        club = self.club.currentText() or None
        track = None if self.track.currentText() == NO_TRACK else self.track.currentText()
        self.a.set_style(club, track)
        self.save(announcer_club=club, announcer_track=track)

    def _voice(self, *_):
        voice = None if self.voice.currentText() == SYSTEM_VOICE else self.voice.currentText()
        self.a.set_voice(voice, self.rate.value() or None)
        self.save(announcer_voice=voice, announcer_rate=self.rate.value() or None)

    def _video(self, *_):
        self.save(live_video_on=self.video.isChecked(), live_video_device=self.device.currentData(),
                  live_video_fps=self.fps.value())
        why = self.win.start_live_video()
        if why and self.video.isChecked():
            self.video_state.setText(f"live video: {why}")

    def _talk_on(self, on: bool):
        if self.a.talk:
            self.a.talk.enabled = bool(on)
            if not on:
                self.a.talk.stop()
        self.save(talk_on=bool(on))

    def _replay(self):
        why = self.win.replay_heat()
        if why and why != "stopped":
            QMessageBox.information(self, "Replay", why)

    def _ft_speech(self):
        now = ann.ft_speech(self.a.profile_dir)
        QMessageBox.information(self, "FPVTrackside's voice", ann.set_ft_speech(self.a.profile_dir, not now))
        self._refresh()

    def _refresh(self):
        now = ann.ft_speech(self.a.profile_dir)
        self.ft_label.setText("FPVTrackside's own voice: " + {True: "on", False: "off (the app speaks)",
                                                               None: "its settings weren't found"}[now])
        self.ft_btn.setText("Turn it off" if now else "Turn it on")
        self.ft_btn.setEnabled(now is not None)
        live = self.win.live.status
        replaying = bool(self.win.replay and not self.win.replay.done.is_set())
        self.replay_btn.setText("■ Stop the replay" if replaying else "▶ Replay the selected heat")
        if live.get("error"):
            self.video_state.setText(f"live video: ⚠ {live['error']}")
        elif live.get("running"):
            s = live.get("stats") or {}
            what = "replaying a heat" if replaying else "on the capture card"
            self.video_state.setText(f"live video ({what}): " + (
                f"{s.get('fps')} fps a pilot · {s.get('dropped')} % of frames dropped · {s.get('delay_ms')} ms behind"
                if s.get("racing") else "waiting for the race to start" if live.get("ready") else "loading…"))
        elif not self.video.isChecked():
            self.video_state.setText("")
        t = self.a.talk
        if t:
            self.talk_state.setText(f"talking ({t.what_job}): “{t.what}”" if t.busy and t.what else
                                    "talk: off" if not t.enabled else
                                    "talk: no lines (clubs/<club>/talk.yaml)" if not self.a.style.talk else "")
        if self.relay:
            s = self.relay.status()
            bits = [f"{s['received']} events from FPVTrackside" + (f" (last {time.strftime('%H:%M:%S', time.localtime(s['last_at']))})"
                                                               if s["last_at"] else "")]
            bits.append("not passed on to fril (no fril live address)" if not s["forwarding"] else
                        f"⚠ fril: {s['forward_error']}" if s["forward_error"] else f"{s['forwarded']} passed on to fril")
            last = self.a.calls[-1] if self.a.calls else None
            self.stats.setText("   ·   ".join(bits) + (f"\nlast call: “{last}”" if last else ""))
