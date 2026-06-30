#!/usr/bin/env python3
"""
Phase 2 — Interactive Gate Learning UI (multi-video, self-contained).

No command-line arguments needed — open everything from inside the UI.

When you open a video, candidate extraction runs automatically in the
background.  Candidates appear on the timeline when extraction finishes.

Workflow:
  1. python learn_ui.py
  2. Open Video → select video 1
     (extraction runs automatically, candidates appear on timeline)
  3. Mark gate passes (G) and lap ends (L) → DEFINE mode
  4. Open Video → select video 2 → confirm ADD mode
     (extraction runs, more candidates appear)
  5. Mark same gates in same order → embeddings added to existing slots
  6. Repeat for more videos
  7. S → Save gate_memory.json

Keyboard shortcuts:
  Space            Play / Pause
  Up / Down        Seek ±1 frame
  Left / Right     Seek ±1 second
  Shift+Left/Right Seek ±10 seconds
  G                Mark gate pass
  L                Mark lap end  (DEFINE mode only)
  D                Delete last   (DEFINE mode only)
  S                Save gate_memory.json
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
        QSizePolicy, QMessageBox, QToolBar, QStatusBar, QFrame, QSpinBox,
        QProgressBar, QScrollArea,
    )
    from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal, QProcess
    from PyQt6.QtGui import (
        QImage, QPixmap, QPainter, QColor, QPen, QFont, QBrush,
        QPolygon, QAction, QKeySequence, QShortcut, QPalette,
    )
except ImportError:
    sys.exit("PyQt6 is required.  Install with:  pip install PyQt6")


# ──────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────

_GATE_COLORS = {
    "square":   QColor(100, 160, 255),
    "arch":     QColor(0,   220, 220),
    "circle":   QColor(100, 220, 100),
    "flagpole": QColor(220, 100, 220),
    "unknown":  QColor(180, 180, 180),
}
_LAP_COLOR      = QColor(255, 200,   0)
_PLAYHEAD_COLOR = QColor(255,  60,  60)
_BG_COLOR       = QColor( 24,  24,  24)
_TRACK_COLOR    = QColor( 55,  55,  55)
_FILL_COLOR     = QColor( 80,  80,  80)
_POINTER_COLOR  = QColor( 60, 220, 120)


def _gate_color(gate_type: str) -> QColor:
    return _GATE_COLORS.get((gate_type or "").lower(), _GATE_COLORS["unknown"])


def _crop_padded(frame: np.ndarray, bbox: list, pad_frac: float = 0.5):
    x1, y1, x2, y2 = map(int, bbox)
    H, W = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(W, x2 + px), min(H, y2 + py)
    return frame[ny1:ny2, nx1:nx2], [nx1, ny1, nx2, ny2]


def _auto_clip_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    idx: int
    t: float
    gate_type: str
    embedding: list
    crop_path: str
    bbox: list
    reason: str


@dataclass
class GateSlot:
    """Persistent gate — accumulates CLIP embeddings across multiple videos."""
    slot_idx: int
    gate_type: str
    embeddings: List[list] = field(default_factory=list)
    crop_paths: List[str]  = field(default_factory=list)
    lap_after: bool = False

    MAX_EMBEDS = 6

    def add_candidate(self, c: Candidate):
        if c.embedding and len(self.embeddings) < self.MAX_EMBEDS:
            self.embeddings.append(c.embedding)
        if c.crop_path and c.crop_path not in self.crop_paths:
            self.crop_paths.append(c.crop_path)

    @property
    def embed_count(self) -> int:
        return len(self.embeddings)

    @property
    def is_full(self) -> bool:
        return len(self.embeddings) >= self.MAX_EMBEDS


@dataclass
class SessionMarker:
    """Marker placed in the current video (shown on timeline, not persisted)."""
    t: float
    is_lap_end: bool = False
    slot_idx: Optional[int] = None
    is_extra_lap: bool = False  # G marker added during lap 2+ (adds to existing slot)


# ──────────────────────────────────────────────────────────────
# Timeline widget
# ──────────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    seeked = pyqtSignal(float)

    _CAND_H = 12
    _GATE_H = 18
    _LAP_H  = 26
    _PAD    = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._duration: float = 0.0
        self._current_t: float = 0.0
        self._candidates: List[Candidate] = []
        self._session_markers: List[SessionMarker] = []
        self._hover_t: Optional[float] = None

    def set_duration(self, d: float):
        self._duration = float(d)
        self.update()

    def set_current_t(self, t: float):
        self._current_t = float(t)
        self.update()

    def set_candidates(self, cands: List[Candidate]):
        self._candidates = cands
        self.update()

    def set_session_markers(self, markers: List[SessionMarker]):
        self._session_markers = markers
        self.update()

    def _t_to_x(self, t: float) -> int:
        if self._duration <= 0:
            return self._PAD
        return int(self._PAD + (t / self._duration) * (self.width() - 2 * self._PAD))

    def _x_to_t(self, x: int) -> float:
        if self._duration <= 0:
            return 0.0
        w = self.width() - 2 * self._PAD
        return max(0.0, min(self._duration, (x - self._PAD) / max(1, w) * self._duration))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        mid = H // 2

        p.fillRect(0, 0, W, H, _BG_COLOR)

        if self._duration <= 0:
            p.setPen(QColor(100, 100, 100))
            p.drawText(QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter, "No video loaded")
            p.end()
            return

        # Track bar + progress fill
        bar_h = 6
        p.fillRect(self._PAD, mid - bar_h // 2, W - 2 * self._PAD, bar_h, _TRACK_COLOR)
        px = self._t_to_x(self._current_t)
        if px > self._PAD:
            p.fillRect(self._PAD, mid - bar_h // 2, px - self._PAD, bar_h, _FILL_COLOR)

        # Candidate ticks
        for c in self._candidates:
            cx = self._t_to_x(c.t)
            col = QColor(_gate_color(c.gate_type))
            col.setAlpha(150)
            p.setPen(QPen(col, 1))
            p.drawLine(cx, mid - self._CAND_H, cx, mid + self._CAND_H)

        # Session markers (gate diamonds + lap lines)
        for m in self._session_markers:
            mx = self._t_to_x(m.t)
            if m.is_lap_end:
                p.setPen(QPen(_LAP_COLOR, 2))
                p.drawLine(mx, mid - self._LAP_H, mx, mid + self._LAP_H)
                p.setFont(QFont("Arial", 7))
                p.setPen(_LAP_COLOR)
                p.drawText(mx - 10, mid - self._LAP_H - 4, "LAP")
            else:
                col = _gate_color("square")
                p.setPen(QPen(col, 2))
                p.drawLine(mx, mid - self._GATE_H, mx, mid + self._GATE_H)
                tip_y = mid - self._GATE_H
                diamond = QPolygon([
                    QPoint(mx,     tip_y - 7),
                    QPoint(mx + 5, tip_y),
                    QPoint(mx,     tip_y + 7),
                    QPoint(mx - 5, tip_y),
                ])
                p.setBrush(QBrush(col))
                p.drawPolygon(diamond)
                p.setBrush(QBrush())
                if m.slot_idx is not None:
                    p.setFont(QFont("Arial", 7))
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(mx - 8, mid + self._GATE_H + 13, f"G{m.slot_idx + 1}")

        # Playhead
        ph = self._t_to_x(self._current_t)
        p.setPen(QPen(_PLAYHEAD_COLOR, 2))
        p.drawLine(ph, 2, ph, H - 2)

        # Hover time
        if self._hover_t is not None:
            hx = self._t_to_x(self._hover_t)
            p.setFont(QFont("Arial", 8))
            p.setPen(QColor(200, 200, 200))
            p.drawText(max(4, min(W - 44, hx - 14)), H - 5, f"{self._hover_t:.2f}s")

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.seeked.emit(self._x_to_t(event.pos().x()))

    def mouseMoveEvent(self, event):
        self._hover_t = self._x_to_t(event.pos().x())
        self.update()

    def leaveEvent(self, _event):
        self._hover_t = None
        self.update()


# ──────────────────────────────────────────────────────────────
# Video player
# ──────────────────────────────────────────────────────────────

class VideoPlayer(QWidget):
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._duration: float = 0.0
        self._current_t: float = 0.0
        self._playing: bool = False
        self._last_frame: Optional[np.ndarray] = None

        lv = QVBoxLayout(self)
        lv.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: black;")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lv.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def current_t(self) -> float:
        return self._current_t

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_playing(self) -> bool:
        return self._playing

    def load(self, path: str) -> bool:
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = total / max(self._fps, 1.0)
        self._current_t = 0.0
        self._render_next()
        return True

    def play(self):
        if not self._cap:
            return
        self._playing = True
        self._timer.start(max(1, int(1000.0 / self._fps)))

    def pause(self):
        self._playing = False
        self._timer.stop()

    def toggle(self):
        self.pause() if self._playing else self.play()

    def seek(self, t: float):
        if not self._cap:
            return
        t = max(0.0, min(self._duration, t))
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        self._current_t = t
        self._render_next()
        self.position_changed.emit(self._current_t)

    def seek_frames(self, delta: int):
        """Seek by an exact number of frames."""
        if not self._cap:
            return
        frame_sec = 1.0 / max(self._fps, 1.0)
        self.seek(self._current_t + delta * frame_sec)

    def _tick(self):
        if not self._cap:
            return
        self._render_next()
        self._current_t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.position_changed.emit(self._current_t)
        if self._current_t >= self._duration - 0.05:
            self.pause()

    def _render_next(self):
        if not self._cap:
            return
        ok, frame = self._cap.read()
        if not ok:
            return
        self._last_frame = frame
        self._push(frame)

    def _push(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self._label.width() or 640, self._label.height() or 360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._push(self._last_frame)


# ──────────────────────────────────────────────────────────────
# Manage Clips window
# ──────────────────────────────────────────────────────────────

class ManageClipsDialog(QWidget):
    """Standalone window listing every stored clip across all gate slots.
    The user can delete individual clips; changes apply immediately."""

    clips_changed = pyqtSignal()

    def __init__(self, gate_slots, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Manage Clips")
        self.resize(980, 620)
        self.setStyleSheet("background:#1a1a1a;")
        self._gate_slots = gate_slots

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel(
            "Click  × Delete  to remove a clip from a gate slot. "
            "Deletes the image and its embedding. Changes are immediate."
        )
        hint.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;}")
        layout.addWidget(self._scroll)

        self._content = QWidget()
        self._vbox = QVBoxLayout(self._content)
        self._vbox.setContentsMargins(4, 4, 4, 4)
        self._vbox.setSpacing(8)
        self._scroll.setWidget(self._content)

        self._refresh()

    def refresh(self):
        self._refresh()

    def _refresh(self):
        while self._vbox.count():
            item = self._vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for slot in self._gate_slots:
            hdr = QLabel(
                f"G{slot.slot_idx + 1}  —  {slot.gate_type}  "
                f"[{slot.embed_count}/{GateSlot.MAX_EMBEDS} embeddings  |  "
                f"{len(slot.crop_paths)} crops]"
            )
            hdr.setStyleSheet(
                "font-weight:bold; font-size:13px; color:#60dc78; "
                "background:#252525; padding:5px 6px; border-radius:3px;"
            )
            self._vbox.addWidget(hdr)

            if not slot.crop_paths:
                lbl = QLabel("   (no clips)")
                lbl.setStyleSheet("color:#444; font-size:11px;")
                self._vbox.addWidget(lbl)
                continue

            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(8)

            n_embeds = len(slot.embeddings)
            for i, crop_path in enumerate(list(slot.crop_paths)):
                cell = QWidget()
                cell.setFixedWidth(152)
                cell.setStyleSheet(
                    "background:#222; border:1px solid #3a3a3a; border-radius:4px;"
                )
                cv = QVBoxLayout(cell)
                cv.setContentsMargins(4, 4, 4, 4)
                cv.setSpacing(3)

                img_lbl = QLabel()
                img_lbl.setFixedSize(140, 98)
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet("background:#111; border:none;")
                if os.path.exists(crop_path):
                    img_lbl.setPixmap(
                        QPixmap(crop_path).scaled(
                            140, 98,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                else:
                    img_lbl.setText("file missing")
                    img_lbl.setStyleSheet("background:#111; color:#555; border:none;")
                cv.addWidget(img_lbl)

                has_embed = i < n_embeds
                info = QLabel(f"Clip {i + 1}" + ("" if has_embed else "  ⚠ no embed"))
                info.setStyleSheet(
                    f"color:{'#bbb' if has_embed else '#a06020'}; font-size:9px; border:none;"
                )
                info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cv.addWidget(info)

                del_btn = QPushButton("× Delete")
                del_btn.setFixedHeight(22)
                del_btn.setStyleSheet(
                    "background:#5a2020; color:white; font-size:10px; "
                    "border:none; border-radius:2px;"
                )
                del_btn.clicked.connect(lambda _, s=slot, idx=i: self._delete_clip(s, idx))
                cv.addWidget(del_btn)

                row_h.addWidget(cell)

            row_h.addStretch()
            self._vbox.addWidget(row_w)

        self._vbox.addStretch()

    def _delete_clip(self, slot: "GateSlot", idx: int):
        if idx < len(slot.embeddings):
            slot.embeddings.pop(idx)
        if idx < len(slot.crop_paths):
            slot.crop_paths.pop(idx)
        self.clips_changed.emit()
        self._refresh()


# ──────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV Gate Learning UI")
        self.resize(1440, 860)

        # ── Persistent across video loads ─────────────────────
        self._gate_slots: List[GateSlot] = []
        self._match_window: float = 120.0  # frames
        self._det_model_path: Optional[str] = self._find_model_auto()
        self._clip_device: str = _auto_clip_device()

        # ── Per-video session ─────────────────────────────────
        self._candidates: List[Candidate] = []
        self._used_idxs: set = set()
        self._session_markers: List[SessionMarker] = []
        self._slot_pointer: int = 0
        self._define_second_lap: bool = False   # True after first L in define mode
        self._define_lap_num: int = 1           # current lap number in define mode
        self._video_count: int = 0
        self._mode: str = "define"          # "define" | "add"
        self._current_video_path: Optional[str] = None

        # ── Background extraction (QProcess) ─────────────────
        self._proc: Optional[QProcess] = None
        self._proc_json_path: str = ""

        # ── Force Clip CLIP embedder (lazy-loaded on first use) ─
        self._clip_embedder = None
        self._manage_clips_win: Optional[ManageClipsDialog] = None

        self._build_ui()
        self._build_shortcuts()

    # ── Model auto-detection ──────────────────────────────────

    def _find_model_auto(self) -> Optional[str]:
        here = Path(__file__).parent
        default = here / "current_best_non_vocab.pt"
        return str(default) if default.exists() else None

    def _ensure_model(self) -> bool:
        if self._det_model_path and Path(self._det_model_path).exists():
            return True
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO detector model", "", "PyTorch Models (*.pt)"
        )
        if path:
            self._det_model_path = path
            return True
        return False

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction("📂 Open Video", self)
        open_act.triggered.connect(self._on_open_video)
        tb.addAction(open_act)

        model_act = QAction("🔧 Set Model", self)
        model_act.triggered.connect(self._on_set_model)
        tb.addAction(model_act)

        tb.addSeparator()

        save_act = QAction("💾 Save Memory  [S]", self)
        save_act.triggered.connect(self._on_save)
        tb.addAction(save_act)

        tb.addSeparator()
        tb.addWidget(QLabel("  Match ±"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 600)
        self._window_spin.setValue(int(self._match_window))
        self._window_spin.setSuffix("f")
        self._window_spin.setFixedWidth(72)
        self._window_spin.setToolTip("Frame window around a marker to search for candidates")
        self._window_spin.valueChanged.connect(lambda v: setattr(self, "_match_window", float(v)))
        tb.addWidget(self._window_spin)

        tb.addSeparator()
        self._mode_badge = QLabel("  MODE: DEFINE")
        self._mode_badge.setStyleSheet("font-weight:bold; color:#60dc78; font-size:12px;")
        tb.addWidget(self._mode_badge)

        # ── Central layout ────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Left column
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        self._video = VideoPlayer()
        self._video.position_changed.connect(self._on_position)
        lv.addWidget(self._video, stretch=1)

        self._time_label = QLabel("0.00 s  /  0.00 s")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_label.setStyleSheet("color:#999; font-size:11px;")
        lv.addWidget(self._time_label)

        self._timeline = TimelineWidget()
        self._timeline.seeked.connect(self._on_seek)
        lv.addWidget(self._timeline)

        lv.addWidget(self._build_legend())

        # Extraction progress bar (hidden until running)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar{background:#222;border:1px solid #444;border-radius:3px;}"
            "QProgressBar::chunk{background:#2a7a4a;}"
        )
        lv.addWidget(self._progress_bar)

        lv.addLayout(self._build_action_row())
        lv.addLayout(self._build_seek_row())

        root.addWidget(left, stretch=3)

        # Right column
        right = QWidget()
        right.setFixedWidth(295)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(4)

        self._video_status = QLabel("No video loaded")
        self._video_status.setStyleSheet("color:#aaa; font-size:10px;")
        rv.addWidget(self._video_status)

        rv.addWidget(self._lbl("Gate Order", bold=True, size=13))

        self._gate_list = QListWidget()
        self._gate_list.setStyleSheet(
            "QListWidget{background:#222;border:1px solid #444;}"
            "QListWidget::item:selected{background:#2a5a8a;}"
        )
        self._gate_list.currentRowChanged.connect(self._on_slot_selected)
        rv.addWidget(self._gate_list, stretch=1)

        self._stats_label = QLabel("Gates: 0   Videos: 0")
        self._stats_label.setStyleSheet("color:#777; font-size:10px;")
        rv.addWidget(self._stats_label)

        self._manage_clips_btn = QPushButton("🗂  Manage Clips")
        self._manage_clips_btn.setFixedHeight(26)
        self._manage_clips_btn.setStyleSheet(
            "background:#2a2a2a; color:#ccc; font-size:11px; border:1px solid #444;"
        )
        self._manage_clips_btn.clicked.connect(self._on_manage_clips)
        rv.addWidget(self._manage_clips_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        rv.addWidget(sep)

        rv.addWidget(self._lbl("Matched crop:"))
        self._crop_label = QLabel()
        self._crop_label.setFixedSize(270, 185)
        self._crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setStyleSheet("background:#111; border:1px solid #333;")
        rv.addWidget(self._crop_label)

        self._match_info = QLabel("")
        self._match_info.setStyleSheet("color:#777; font-size:9px;")
        self._match_info.setWordWrap(True)
        rv.addWidget(self._match_info)

        root.addWidget(right)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            f"Ready — device: {self._clip_device}  |  "
            f"model: {Path(self._det_model_path).name if self._det_model_path else 'not found'}"
        )

    def _build_legend(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(10)
        h.addWidget(self._lbl("Candidates:"))
        for name, col in _GATE_COLORS.items():
            if name == "unknown":
                continue
            h.addWidget(self._lbl(f"● {name}", color=col))
        h.addWidget(self._lbl("│ lap", color=_LAP_COLOR))
        h.addWidget(self._lbl("◆ gate"))
        h.addStretch()
        return w

    @staticmethod
    def _lbl(text: str, color: Optional[QColor] = None, bold: bool = False, size: int = 10) -> QLabel:
        l = QLabel(text)
        c = f"rgb({color.red()},{color.green()},{color.blue()})" if color else "#999"
        w = "bold" if bold else "normal"
        l.setStyleSheet(f"font-size:{size}px; color:{c}; font-weight:{w};")
        return l

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(34)
        self._play_btn.clicked.connect(self._on_play_pause)
        row.addWidget(self._play_btn)

        _gate_btn_styles = [
            ("Square  [1]",   "square",   "#1a3d6a"),
            ("Arch  [2]",     "arch",     "#005555"),
            ("Circle  [3]",   "circle",   "#1a4a1a"),
            ("Flagpole  [4]", "flagpole", "#4a1a4a"),
        ]
        self._gate_type_btns: dict = {}
        for label, gtype, bg in _gate_btn_styles:
            btn = QPushButton(label)
            btn.setFixedHeight(34)
            btn.setStyleSheet(f"background:{bg}; color:white;")
            btn.clicked.connect(lambda _, t=gtype: self._on_mark_gate(gate_type=t))
            row.addWidget(btn)
            self._gate_type_btns[gtype] = btn

        self._mark_lap_btn = QPushButton("Start/Finish  [L]")
        self._mark_lap_btn.setFixedHeight(34)
        self._mark_lap_btn.setStyleSheet("background:#5a2060; color:white;")
        self._mark_lap_btn.clicked.connect(self._on_mark_start_finish)
        row.addWidget(self._mark_lap_btn)

        self._delete_btn = QPushButton("Delete Last  [D]")
        self._delete_btn.setFixedHeight(34)
        self._delete_btn.setStyleSheet("background:#5a2020; color:white;")
        self._delete_btn.clicked.connect(self._on_delete_last)
        row.addWidget(self._delete_btn)

        self._skip_btn = QPushButton("Skip  [K]")
        self._skip_btn.setFixedHeight(34)
        self._skip_btn.setStyleSheet("background:#1e3f6a; color:white;")
        self._skip_btn.setToolTip(
            "Skip current gate slot without assigning a clip.\n"
            "The next G/Force Clip will go to the following gate."
        )
        self._skip_btn.clicked.connect(self._on_skip)
        row.addWidget(self._skip_btn)

        self._force_btn = QPushButton("Force Clip  [F]")
        self._force_btn.setFixedHeight(34)
        self._force_btn.setStyleSheet("background:#4a3a0a; color:white;")
        self._force_btn.setToolTip(
            "Embed the current frame directly (no candidate search).\n"
            "Uses gate bbox + padding if one is visible, otherwise full frame."
        )
        self._force_btn.clicked.connect(self._on_force_clip)
        row.addWidget(self._force_btn)

        self._cancel_btn = QPushButton("⏹ Cancel Extraction")
        self._cancel_btn.setFixedHeight(34)
        self._cancel_btn.setStyleSheet("background:#5a3000; color:white;")
        self._cancel_btn.clicked.connect(self._cancel_extraction)
        self._cancel_btn.setVisible(False)
        row.addWidget(self._cancel_btn)

        return row

    def _build_seek_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._lbl("Seek:", size=11))
        for label, delta_sec in [("◀◀ -10s", -10), ("◀ -1s", -1), ("+1s ▶", 1), ("+10s ▶▶", 10)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px;")
            btn.clicked.connect(lambda _, d=delta_sec: self._on_seek(self._video.current_t + d))
            row.addWidget(btn)
        for label, delta_frames in [("◀ -1f", -1), ("+1f ▶", 1)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px; background:#2a2a2a;")
            btn.clicked.connect(lambda _, d=delta_frames: self._seek_frame(d))
            row.addWidget(btn)
        return row

    def _build_shortcuts(self):
        pairs = [
            ("Space",        self._on_play_pause),
            ("G",            self._on_mark_gate),           # auto-detect type
            ("1",            lambda: self._on_mark_gate(gate_type="square")),
            ("2",            lambda: self._on_mark_gate(gate_type="arch")),
            ("3",            lambda: self._on_mark_gate(gate_type="circle")),
            ("4",            lambda: self._on_mark_gate(gate_type="flagpole")),
            ("L",            self._on_mark_start_finish),
            ("D",            self._on_delete_last),
            ("K",            self._on_skip),
            ("F",            self._on_force_clip),
            ("S",            self._on_save),
            ("Up",           lambda: self._seek_frame(+1)),
            ("Down",         lambda: self._seek_frame(-1)),
            ("Right",        lambda: self._on_seek(self._video.current_t + 1.0)),
            ("Left",         lambda: self._on_seek(self._video.current_t - 1.0)),
            ("Shift+Right",  lambda: self._on_seek(self._video.current_t + 10.0)),
            ("Shift+Left",   lambda: self._on_seek(self._video.current_t - 10.0)),
        ]
        for key, fn in pairs:
            QShortcut(QKeySequence(key), self).activated.connect(fn)

    # ── Model management ──────────────────────────────────────

    def _on_set_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO detector model", "", "PyTorch Models (*.pt)"
        )
        if path:
            self._det_model_path = path
            self._status.showMessage(f"Model set: {Path(path).name}")

    # ── File loading + auto extraction ────────────────────────

    def _on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.MOV)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        # If slots already defined, confirm ADD mode
        if self._gate_slots:
            ans = QMessageBox.question(
                self, "Add embeddings from new video?",
                f"Track already has {len(self._gate_slots)} gate(s) defined.\n\n"
                "Loading a new video enters ADD mode:\n"
                "  • Press G for each gate in the same order\n"
                "  • Embeddings are added to existing slots (up to 6 per gate)\n"
                "  • Lap End and Delete are disabled in ADD mode\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                return
            self._enter_add_mode()

        if not self._video.load(path):
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            return

        self._current_video_path = path
        self._video_count += 1
        self._candidates = []
        self._used_idxs.clear()
        self._session_markers.clear()
        self._timeline.set_duration(self._video.duration)
        self._timeline.set_candidates([])
        self._timeline.set_session_markers([])
        self.setWindowTitle(f"FPV Gate Learning UI  —  {Path(path).name}")
        self._video_status.setText(f"Video {self._video_count}: {Path(path).name}")
        self._update_mode_ui()

        # Start background extraction
        self._start_extraction(path)

    # ── Background extraction (QProcess) ─────────────────────

    def _start_extraction(self, video_path: str):
        if not self._ensure_model():
            self._status.showMessage("Extraction skipped — no model selected.")
            return

        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()

        stem = Path(video_path).stem
        here = Path(__file__).parent
        out_json  = str(here / f"candidates_{stem}.json")
        crops_dir = str(here / f"candidate_crops_{stem}")
        self._proc_json_path = out_json

        script = str(here / "extract_candidates.py")

        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_extract_output)
        self._proc.finished.connect(self._on_extract_finished)

        args = [
            script,
            "--video",          video_path,
            "--det-model",      self._det_model_path,
            "--output",         out_json,
            "--crops-dir",      crops_dir,
            "--clip-device",    self._clip_device,
        ]
        self._proc.start(sys.executable, args)

        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._status.showMessage("Extracting candidates…  0%")

    def _on_extract_output(self):
        if not self._proc:
            return
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Parse "Progress: 36%  frame=300/826  candidates=6"
            if line.startswith("Progress:"):
                try:
                    pct_str = line.split("%")[0].split()[-1]
                    pct = int(float(pct_str))
                    self._progress_bar.setValue(pct)
                    cand_part = [p for p in line.split() if p.startswith("candidates=")]
                    n = int(cand_part[0].split("=")[1]) if cand_part else 0
                    self._status.showMessage(f"Extracting…  {pct}%   ({n} candidates so far)")
                except Exception:
                    pass
            elif line.startswith("Done."):
                self._status.showMessage(line)

    def _on_extract_finished(self, exit_code: int, _exit_status):
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)

        if exit_code != 0:
            self._status.showMessage("Extraction failed or was cancelled.")
            return

        if not self._proc_json_path or not Path(self._proc_json_path).exists():
            self._status.showMessage("Extraction finished but no output JSON found.")
            return

        self._load_candidates_json(self._proc_json_path)

    def _cancel_extraction(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
        self._progress_bar.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._status.showMessage("Extraction cancelled.")

    def _load_candidates_json(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._status.showMessage(f"Could not load candidates JSON: {e}")
            return

        self._candidates = []
        for c in data.get("candidates", []):
            self._candidates.append(Candidate(
                idx=int(c.get("idx", 0)),
                t=float(c.get("t", 0.0)),
                gate_type=str(c.get("gate_type", "unknown")),
                embedding=c.get("embedding", []),
                crop_path=str(c.get("crop_path", "")),
                bbox=c.get("bbox", []),
                reason=str(c.get("reason", "")),
            ))

        self._timeline.set_candidates(self._candidates)
        self._status.showMessage(
            f"Ready — {len(self._candidates)} candidates found in {Path(path).name}"
        )

    # ── Mode management ───────────────────────────────────────

    def _enter_add_mode(self):
        self._mode = "add"
        self._slot_pointer = 0
        self._define_second_lap = False
        self._mark_lap_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._mark_lap_btn.setStyleSheet("background:#333; color:#555;")
        self._delete_btn.setStyleSheet("background:#333; color:#555;")

    def _update_mode_ui(self):
        if self._mode == "define":
            if not self._define_second_lap:
                self._mode_badge.setText("  MODE: DEFINE")
                self._mode_badge.setStyleSheet("font-weight:bold; color:#60dc78; font-size:12px;")
            else:
                ptr = self._slot_pointer % max(1, len(self._gate_slots))
                remaining = len(self._gate_slots) - ptr
                self._mode_badge.setText(
                    f"  MODE: DEFINE (Lap {self._define_lap_num})  "
                    f"Next: G{ptr + 1}  Remaining: {remaining}"
                )
                self._mode_badge.setStyleSheet("font-weight:bold; color:#a0e870; font-size:12px;")
        else:
            ptr = self._slot_pointer % max(1, len(self._gate_slots))
            remaining = len(self._gate_slots) - ptr
            self._mode_badge.setText(
                f"  MODE: ADD (video {self._video_count})  "
                f"Next: G{ptr + 1}  Remaining: {remaining}"
            )
            self._mode_badge.setStyleSheet("font-weight:bold; color:#ffa028; font-size:12px;")
        self._refresh_gate_list()

    # ── Playback ──────────────────────────────────────────────

    def _on_play_pause(self):
        self._video.toggle()
        self._play_btn.setText("⏸  Pause" if self._video.is_playing else "▶  Play")

    def _on_seek(self, t: float):
        self._video.pause()
        self._play_btn.setText("▶  Play")
        self._video.seek(t)

    def _seek_frame(self, delta: int):
        self._video.pause()
        self._play_btn.setText("▶  Play")
        self._video.seek_frames(delta)

    def _on_position(self, t: float):
        self._timeline.set_current_t(t)
        self._time_label.setText(f"{t:.2f} s  /  {self._video.duration:.2f} s")

    # ── Marking ───────────────────────────────────────────────

    def _on_mark_gate(self, gate_type: Optional[str] = None):
        t = self._video.current_t
        matched = self._find_candidates(t)

        if self._mode == "define" and not self._define_second_lap:
            # First lap: every G creates a new gate slot
            slot_idx = len(self._gate_slots)
            resolved_type = gate_type or (matched[0].gate_type if matched else "unknown")
            slot = GateSlot(slot_idx=slot_idx, gate_type=resolved_type)
            for c in matched:
                slot.add_candidate(c)
                self._used_idxs.add(c.idx)
            self._gate_slots.append(slot)
            self._session_markers.append(SessionMarker(t=t, slot_idx=slot_idx))
            self._status.showMessage(
                f"G{slot_idx + 1} [{resolved_type}] defined at {t:.2f}s  —  {slot.embed_count} embedding(s)"
            )
            if slot.crop_paths:
                self._show_crop(slot)
        else:
            # Lap 2+ (define mode) or ADD mode: add embedding to existing slot
            if not self._gate_slots:
                self._status.showMessage("No gates defined yet.")
                return
            self._slot_pointer = self._slot_pointer % len(self._gate_slots)
            slot = self._gate_slots[self._slot_pointer]
            if slot.is_full:
                self._status.showMessage(
                    f"G{slot.slot_idx + 1} is full (6/6). Skipping."
                )
                self._slot_pointer += 1
                self._update_mode_ui()
                return
            prev = slot.embed_count
            for c in matched:
                slot.add_candidate(c)
                self._used_idxs.add(c.idx)
            added = slot.embed_count - prev
            is_extra = (self._mode == "define")  # tag so D knows not to pop the slot
            self._session_markers.append(SessionMarker(t=t, slot_idx=slot.slot_idx, is_extra_lap=is_extra))
            self._slot_pointer += 1
            self._status.showMessage(
                f"G{slot.slot_idx + 1}: +{added} embedding(s)  "
                f"(total {slot.embed_count}/{GateSlot.MAX_EMBEDS})"
            )
            if slot.crop_paths:
                self._show_crop(slot)

        self._timeline.set_session_markers(self._session_markers)
        self._update_mode_ui()

    def _on_mark_start_finish(self):
        """Mark the Start/Finish gate (always G1 / slot 0) and reset the lap queue to G2."""
        if self._mode not in ("define", "add"):
            return
        t = self._video.current_t
        matched = self._find_candidates(t)

        if self._mode == "add":
            if not self._gate_slots:
                self._status.showMessage("No gates defined yet.")
                return
            slot = self._gate_slots[0]
            if not slot.is_full:
                prev = slot.embed_count
                for c in matched:
                    slot.add_candidate(c)
                    self._used_idxs.add(c.idx)
                added = slot.embed_count - prev
            else:
                added = 0
            self._session_markers.append(SessionMarker(t=t, slot_idx=0))
            self._slot_pointer = 1
            self._status.showMessage(
                f"Start/Finish (G1): +{added}  ({slot.embed_count}/{GateSlot.MAX_EMBEDS})  Next: G2"
            )
            if slot.crop_paths:
                self._show_crop(slot)
            self._timeline.set_session_markers(self._session_markers)
            self._update_mode_ui()
            return

        # ── DEFINE mode ───────────────────────────────────────
        if not self._gate_slots:
            # First ever press: create G1
            resolved_type = matched[0].gate_type if matched else "unknown"
            slot = GateSlot(slot_idx=0, gate_type=resolved_type)
            for c in matched:
                slot.add_candidate(c)
                self._used_idxs.add(c.idx)
            self._gate_slots.append(slot)
            self._session_markers.append(SessionMarker(t=t, slot_idx=0))
            self._status.showMessage(
                f"Start/Finish (G1) [{resolved_type}] defined  —  "
                f"{slot.embed_count} embedding  —  press G for G2, G3, …"
            )
            if slot.crop_paths:
                self._show_crop(slot)

        elif not self._define_second_lap:
            # End of first lap: add to G1, enter second-lap mode
            slot = self._gate_slots[0]
            prev = slot.embed_count
            if not slot.is_full:
                for c in matched:
                    slot.add_candidate(c)
                    self._used_idxs.add(c.idx)
            added = slot.embed_count - prev
            self._define_second_lap = True
            self._slot_pointer = 1   # next G → G2
            self._define_lap_num = 2
            self._session_markers.append(SessionMarker(t=t, slot_idx=0, is_extra_lap=True))
            self._status.showMessage(
                f"Lap 1 end / Lap 2 start  —  G1: +{added}  "
                f"({slot.embed_count}/{GateSlot.MAX_EMBEDS})  Next: G2"
            )
            if slot.crop_paths:
                self._show_crop(slot)

        else:
            # Already in extra-lap mode: add to G1, reset queue to G2
            slot = self._gate_slots[0]
            if slot.is_full:
                self._status.showMessage("G1 (Start/Finish) is full (6/6).")
                self._slot_pointer = 1
                self._update_mode_ui()
                return
            prev = slot.embed_count
            for c in matched:
                slot.add_candidate(c)
                self._used_idxs.add(c.idx)
            added = slot.embed_count - prev
            self._session_markers.append(SessionMarker(t=t, slot_idx=0, is_extra_lap=True))
            self._slot_pointer = 1
            self._define_lap_num += 1
            self._status.showMessage(
                f"Lap {self._define_lap_num - 1} end / Lap {self._define_lap_num} start  —  "
                f"G1: +{added}  ({slot.embed_count}/{GateSlot.MAX_EMBEDS})  Next: G2"
            )
            if slot.crop_paths:
                self._show_crop(slot)

        self._timeline.set_session_markers(self._session_markers)
        self._update_mode_ui()

    def _on_delete_last(self):
        if self._mode != "define" or not self._session_markers:
            return
        m = self._session_markers.pop()
        if not m.is_lap_end:
            if m.is_extra_lap:
                # Lap 2+ marker: remove the embedding we added, don't pop the slot
                if m.slot_idx is not None and 0 <= m.slot_idx < len(self._gate_slots):
                    slot = self._gate_slots[m.slot_idx]
                    if slot.embeddings:
                        slot.embeddings.pop()
                    if slot.crop_paths:
                        removed = slot.crop_paths.pop()
                        for c in self._candidates:
                            if c.crop_path == removed:
                                self._used_idxs.discard(c.idx)
                                break
                self._slot_pointer = max(0, self._slot_pointer - 1)
                # Recompute second-lap state from remaining Start/Finish markers (slot 0 extra-lap)
                sf_markers = [mm for mm in self._session_markers if mm.is_extra_lap and mm.slot_idx == 0]
                self._define_second_lap = len(sf_markers) > 0
                self._define_lap_num = 1 + len(sf_markers)
            elif self._gate_slots:
                # First-lap marker: pop the gate slot entirely
                slot = self._gate_slots.pop()
                for crop in slot.crop_paths:
                    for c in self._candidates:
                        if c.crop_path == crop:
                            self._used_idxs.discard(c.idx)
        elif m.is_lap_end and self._gate_slots:
            # Legacy handler (L-key markers from old sessions)
            self._gate_slots[-1].lap_after = False
            self._define_second_lap = any(s.lap_after for s in self._gate_slots)
            self._define_lap_num = 1 + sum(1 for s in self._gate_slots if s.lap_after)
            if not self._define_second_lap:
                self._slot_pointer = 0
        self._timeline.set_session_markers(self._session_markers)
        self._refresh_gate_list()
        self._status.showMessage("Last marker deleted")
        self._update_mode_ui()

    # ── Skip / Force Clip ─────────────────────────────────────

    def _on_skip(self):
        """Advance the slot pointer by one without assigning any clip.
        In first-lap DEFINE mode, creates an empty placeholder slot instead."""
        t = self._video.current_t

        if self._mode == "define" and not self._define_second_lap:
            # First lap: create an empty slot so the sequence stays intact
            slot_idx = len(self._gate_slots)
            slot = GateSlot(slot_idx=slot_idx, gate_type="unknown")
            self._gate_slots.append(slot)
            self._session_markers.append(SessionMarker(t=t, slot_idx=slot_idx))
            self._timeline.set_session_markers(self._session_markers)
            self._status.showMessage(
                f"Skipped — empty G{slot_idx + 1} placeholder created  "
                f"(assign clips later via ADD mode or Force Clip)"
            )
        else:
            if not self._gate_slots:
                self._status.showMessage("No gates defined yet.")
                return
            ptr = self._slot_pointer % len(self._gate_slots)
            self._slot_pointer += 1
            next_ptr = self._slot_pointer % max(1, len(self._gate_slots))
            self._status.showMessage(
                f"Skipped G{ptr + 1}  →  Next: G{next_ptr + 1}"
            )

        self._update_mode_ui()

    def _on_force_clip(self):
        """Embed the current displayed frame directly and assign it to the next slot.
        Uses gate bbox+padding from the nearest visible candidate; falls back to full frame."""
        frame = self._video._last_frame
        if frame is None:
            self._status.showMessage("No frame loaded.")
            return

        t = self._video.current_t
        fps = max(self._video.fps, 1.0)

        # Find nearest candidate for bbox + gate_type hint (ignores used_idxs and match window)
        nearest = min(self._candidates, key=lambda c: abs(c.t - t), default=None)
        use_bbox = nearest and nearest.bbox and abs(nearest.t - t) * fps <= 60
        gate_type_hint = nearest.gate_type if nearest else "unknown"

        crop, _ = _crop_padded(frame, nearest.bbox) if use_bbox else (frame, [])

        embedder = self._get_clip_embedder()
        if embedder is None:
            return

        self._status.showMessage("Embedding current frame…")
        QApplication.processEvents()
        emb = embedder.embed_bgr(crop)

        # Save crop image alongside other candidates
        stem = Path(self._current_video_path).stem if self._current_video_path else "force"
        crops_dir = Path(__file__).parent / f"candidate_crops_{stem}"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = str(crops_dir / f"force_{int(t * 1000)}.jpg")
        cv2.imwrite(crop_path, crop)

        if self._mode == "define" and not self._define_second_lap:
            # First lap: create a new slot
            slot_idx = len(self._gate_slots)
            slot = GateSlot(slot_idx=slot_idx, gate_type=gate_type_hint)
            slot.embeddings.append(emb.tolist())
            slot.crop_paths.append(crop_path)
            self._gate_slots.append(slot)
            self._session_markers.append(SessionMarker(t=t, slot_idx=slot_idx))
            self._status.showMessage(
                f"Force Clip → G{slot_idx + 1} [{gate_type_hint}]  "
                f"({'bbox+pad' if use_bbox else 'full frame'})"
            )
        else:
            if not self._gate_slots:
                self._status.showMessage("No gates defined yet.")
                return
            ptr = self._slot_pointer % len(self._gate_slots)
            slot = self._gate_slots[ptr]
            if slot.is_full:
                self._status.showMessage(f"G{slot.slot_idx + 1} is full (6/6). Skipping.")
                self._slot_pointer += 1
                self._update_mode_ui()
                return
            if len(slot.embeddings) < GateSlot.MAX_EMBEDS:
                slot.embeddings.append(emb.tolist())
            if crop_path not in slot.crop_paths:
                slot.crop_paths.append(crop_path)
            is_extra = (self._mode == "define")
            self._session_markers.append(
                SessionMarker(t=t, slot_idx=slot.slot_idx, is_extra_lap=is_extra)
            )
            self._slot_pointer += 1
            self._status.showMessage(
                f"Force Clip → G{slot.slot_idx + 1} "
                f"({slot.embed_count}/{GateSlot.MAX_EMBEDS})  "
                f"({'bbox+pad' if use_bbox else 'full frame'})"
            )

        self._show_crop(slot)
        self._timeline.set_session_markers(self._session_markers)
        self._update_mode_ui()

    def _get_clip_embedder(self):
        if self._clip_embedder is None:
            try:
                from lazy_spotter import ClipEmbedder
                self._status.showMessage("Loading CLIP model… (first Force Clip use only)")
                QApplication.processEvents()
                self._clip_embedder = ClipEmbedder(device=self._clip_device)
            except Exception as e:
                self._status.showMessage(f"CLIP load failed: {e}")
                return None
        return self._clip_embedder

    def _on_manage_clips(self):
        if not self._gate_slots:
            self._status.showMessage("No gates defined yet.")
            return
        if self._manage_clips_win is None or not self._manage_clips_win.isVisible():
            self._manage_clips_win = ManageClipsDialog(self._gate_slots)
            self._manage_clips_win.clips_changed.connect(self._on_clips_changed)
            self._manage_clips_win.show()
        else:
            self._manage_clips_win.raise_()
            self._manage_clips_win.activateWindow()

    def _on_clips_changed(self):
        self._refresh_gate_list()

    # ── Candidate matching ────────────────────────────────────

    def _find_candidates(self, t: float) -> List[Candidate]:
        fps = self._video.fps
        nearby = [
            c for c in self._candidates
            if round(abs(c.t - t) * fps) <= self._match_window and c.idx not in self._used_idxs
        ]
        nearby.sort(key=lambda c: round(abs(c.t - t) * fps))
        return nearby[:1]

    # ── Gate list ─────────────────────────────────────────────

    def _refresh_gate_list(self):
        self._gate_list.clear()
        show_ptr = self._mode == "add" or (self._mode == "define" and self._define_second_lap)
        for slot in self._gate_slots:
            is_next = (
                show_ptr
                and slot.slot_idx == self._slot_pointer % max(1, len(self._gate_slots))
            )
            arrow    = "▶ " if is_next else "   "
            icon     = "●" if slot.is_full else ("✓" if slot.embed_count > 0 else "⚠")
            text     = (
                f"{arrow}{icon}  G{slot.slot_idx + 1}  {slot.gate_type}"
                f"  [{slot.embed_count}/{GateSlot.MAX_EMBEDS}]"
            )
            if slot.lap_after:
                text += "  — LAP"
            item = QListWidgetItem(text)
            item.setForeground(_POINTER_COLOR if is_next else _gate_color(slot.gate_type))
            self._gate_list.addItem(item)

        total = sum(s.embed_count for s in self._gate_slots)
        self._stats_label.setText(
            f"Gates: {len(self._gate_slots)}   Embeddings: {total}   Videos: {self._video_count}"
        )

    def _on_slot_selected(self, row: int):
        if row < 0 or row >= len(self._gate_slots):
            return
        slot = self._gate_slots[row]
        self._show_crop(slot)
        self._match_info.setText(
            f"G{slot.slot_idx + 1}  {slot.gate_type}\n"
            f"Embeddings: {slot.embed_count} / {GateSlot.MAX_EMBEDS}\n"
            f"Lap end after: {'yes' if slot.lap_after else 'no'}"
        )

    def _show_crop(self, slot: GateSlot):
        if not slot.crop_paths:
            self._crop_label.setText("No crop yet")
            return
        path = slot.crop_paths[-1]
        if os.path.exists(path):
            self._crop_label.setPixmap(
                QPixmap(path).scaled(
                    270, 185,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._crop_label.setText("Crop not found")

    # ── Save ──────────────────────────────────────────────────

    def _on_save(self):
        if not self._gate_slots:
            QMessageBox.warning(self, "Nothing to save", "Define at least one gate first.")
            return

        empty = [s for s in self._gate_slots if s.embed_count == 0]
        if empty:
            ans = QMessageBox.question(
                self, "Empty gate slots",
                f"{len(empty)} gate(s) have no embeddings and won't match in race mode.\n\nSave anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gate Memory", "gate_memory.json", "JSON Files (*.json)"
        )
        if not path:
            return

        memory = [
            {
                "order_idx":  slot.slot_idx,
                "gate_id":    slot.slot_idx + 1,
                "gate_type":  slot.gate_type,
                "embeds":     slot.embeddings,
                "created_t":  0.0,
                "last_img":   slot.crop_paths[-1] if slot.crop_paths else "",
                "embed_imgs": list(slot.crop_paths),
            }
            for slot in self._gate_slots
        ]

        out = {
            "version":             2,
            "mode":                "learn",
            "race_lookahead":      3,
            "expected_idx":        0,
            "max_embeds_per_gate": GateSlot.MAX_EMBEDS,
            "memory":              memory,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        total = sum(s.embed_count for s in self._gate_slots)
        self._status.showMessage(f"Saved {len(memory)} gates  ({total} embeddings) → {path}")
        QMessageBox.information(
            self, "Saved",
            f"gate_memory.json saved:\n{path}\n\n"
            f"{len(memory)} gates  |  {total} total embeddings  |  {self._video_count} video(s)"
        )


# ──────────────────────────────────────────────────────────────
# Dark palette + entry point
# ──────────────────────────────────────────────────────────────

def _dark_palette() -> QPalette:
    pal = QPalette()
    for role, rgb in [
        (QPalette.ColorRole.Window,          (26,  26,  26)),
        (QPalette.ColorRole.WindowText,      (220, 220, 220)),
        (QPalette.ColorRole.Base,            (34,  34,  34)),
        (QPalette.ColorRole.AlternateBase,   (42,  42,  42)),
        (QPalette.ColorRole.Text,            (220, 220, 220)),
        (QPalette.ColorRole.Button,          (50,  50,  50)),
        (QPalette.ColorRole.ButtonText,      (220, 220, 220)),
        (QPalette.ColorRole.Highlight,       (42,  90, 138)),
        (QPalette.ColorRole.HighlightedText, (255, 255, 255)),
    ]:
        pal.setColor(role, QColor(*rgb))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
