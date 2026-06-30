#!/usr/bin/env python3
"""
Pass Detector Debug / Tuning UI.

Load a video, mark your observed passes manually (ground truth), then run the
pass detector with any parameter combination and compare what it finds against
your marks on the timeline.

All PassDetector parameters are fetched live from the class __init__ signature —
add a new parameter there and it will appear in this UI automatically.

Usage:
    python debug_ui.py

Keyboard shortcuts:
    Space            Play / Pause
    Up / Down        Seek ±1 frame
    Left / Right     Seek ±1 second
    Shift+Left/Right Seek ±10 seconds
    P                Mark a manual pass at current time
    D                Delete last manual mark
    R                Run pass detector
"""

import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
        QSizePolicy, QMessageBox, QToolBar, QStatusBar, QFrame,
        QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFormLayout,
        QSplitter, QProgressBar, QToolButton,
    )
    from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal, QThread
    from PyQt6.QtGui import (
        QImage, QPixmap, QPainter, QColor, QPen, QFont, QBrush,
        QPolygon, QAction, QKeySequence, QShortcut, QPalette,
    )
except ImportError:
    sys.exit("PyQt6 is required.  Install with:  pip install PyQt6")


# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────

_GATE_COLORS = {
    "square":   QColor(100, 160, 255),
    "arch":     QColor(0,   220, 220),
    "circle":   QColor(100, 220, 100),
    "flagpole": QColor(220, 100, 220),
    "unknown":  QColor(180, 180, 180),
}
_USER_MARK_COLOR = QColor( 80, 220, 100)
_ALIGNED_COLOR   = QColor(255, 160,   0)
_PLAYHEAD_COLOR  = QColor(255,  60,  60)
_BG_COLOR        = QColor( 24,  24,  24)
_TRACK_COLOR     = QColor( 55,  55,  55)
_FILL_COLOR      = QColor( 80,  80,  80)


def _gate_color(gate_type: str) -> QColor:
    return _GATE_COLORS.get((gate_type or "").lower(), _GATE_COLORS["unknown"])


# ─────────────────────────────────────────────────────────────────────────────
# Parameter metadata — ranges/steps for known PassDetector params.
# Unknown params (added to PassDetector later) fall through to _auto_meta().
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_META: Dict[str, Dict] = {
    # TimeTracker
    "iou_match_thresh":          {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "ttl_seconds":               {"min": 0.0,  "max": 3.0,  "step": 0.05,  "dec": 2},
    "lock_min_score":            {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "lock_hysteresis":           {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "lock_streak":               {"min": 1,    "max": 20,   "step": 1},
    "ema_alpha":                 {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    # PassDetector
    "min_track_score":           {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "min_area_ratio":            {"min": 0.0,  "max": 1.0,  "step": 0.005, "dec": 3},
    "center_tol":                {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "disappear_timeout":         {"min": 0.0,  "max": 5.0,  "step": 0.05,  "dec": 2},
    "flag_center_tol":           {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "flag_edge_tol":             {"min": 0.0,  "max": 0.5,  "step": 0.01,  "dec": 2},
    "flag_min_edges":            {"min": 1,    "max": 4,    "step": 1},
    "pass_cooldown_sec":         {"min": 0.0,  "max": 10.0, "step": 0.05,  "dec": 2},
    "type_cooldown_sec":         {"min": 0.0,  "max": 10.0, "step": 0.05,  "dec": 2},
    "track_cooldown_sec":        {"min": 0.0,  "max": 30.0, "step": 0.1,   "dec": 1},
    "area_vel_ema_alpha":        {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "min_area_vel_ema":          {"min": 0.0,  "max": 1.0,  "step": 0.005, "dec": 3},
    "aligned_shrink_reset_frac": {"min": 0.0,  "max": 1.0,  "step": 0.01,  "dec": 2},
    "aligned_max_age_sec":       {"min": 0.0,  "max": 60.0, "step": 0.5,   "dec": 1},
    "flag_aligned_shrink_reset_frac": {"min": 0.0, "max": 1.0, "step": 0.01, "dec": 2},
    "gate_pass_area_thresh":          {"min": 0.0, "max": 1.0, "step": 0.005, "dec": 3},
}


_PARAM_DOCS: Dict[str, str] = {
    # ── TimeTracker ──────────────────────────────────────────────────────────
    "iou_match_thresh": (
        "Minimum IOU overlap to match a new YOLO detection to an existing track.\n\n"
        "Lower = more permissive; helps with fast-moving or thin objects like flagpoles.\n\n"
        "Example: 0.15 will match even partly-overlapping boxes. "
        "0.5 requires significant overlap — good for large stable gates, bad for poles."
    ),
    "ttl_seconds": (
        "How long a track stays alive after YOLO stops detecting it (coasting / TTL).\n\n"
        "Lower = tracks die quickly when lost; higher = tracks survive brief occlusions.\n\n"
        "Example: 0.03 ≈ 1 frame at 30 fps (almost no coasting). "
        "0.3 ≈ 9 frames of coasting with a frozen bbox."
    ),
    "lock_min_score": (
        "Minimum YOLO confidence for a track to be locked to a gate type.\n\n"
        "Example: 0.20 means only detections with ≥ 20 % confidence are used "
        "for type assignment."
    ),
    "lock_hysteresis": (
        "Score must drop this far below lock_min_score before the type lock is released. "
        "Prevents flickering between types.\n\n"
        "Example: 0.10 → once locked at 0.20, the type is kept until score drops below 0.10."
    ),
    "lock_streak": (
        "Consecutive frames above lock_min_score required before a type is committed.\n\n"
        "Example: 3 → gate type is only locked after 3 frames in a row above the threshold."
    ),
    "ema_alpha": (
        "Smoothing factor for the score exponential moving average (EMA).\n\n"
        "Higher = reacts faster to new detections. Lower = smoother, slower to update.\n\n"
        "Example: 0.5 weights the current and historical score equally. "
        "0.1 is very smooth; 0.9 is almost raw."
    ),
    # ── PassDetector ─────────────────────────────────────────────────────────
    "min_track_score": (
        "Minimum smoothed YOLO score for the pass detector to consider a track at all.\n\n"
        "Example: 0.22 ignores weak or uncertain detections before they ever reach "
        "the alignment logic."
    ),
    "min_area_ratio": (
        "Minimum fraction of frame area the bbox must cover before a gate is considered "
        "close enough to align.\n\n"
        "Example: 0.03 = gate must cover at least 3 % of the frame. "
        "Increase to ignore distant gates."
    ),
    "center_tol": (
        "Maximum normalised distance from frame centre for a gate to be considered aligned.\n\n"
        "Example: 0.18 → gate centre must be within 18 % of the frame dimension from centre. "
        "Increase for wide approaches."
    ),
    "disappear_timeout": (
        "Seconds after a track disappears during which its disappearance can still trigger a pass.\n\n"
        "Example: 0.25 s → if the gate vanishes within ¼ second of alignment it counts. "
        "Too large = stale passes; too small = missed passes at high speed."
    ),
    "flag_center_tol": (
        "For flagpoles — how close to the horizontal centre (nx = 0.5) the flag must be "
        "to mark it as 'was centred'.\n\n"
        "Example: 0.15 → flag centre must be within 15 % of frame width from centre."
    ),
    "flag_edge_tol": (
        "Fraction of frame dimension used as tolerance when checking if a bbox edge is "
        "near the real camera boundary.\n\n"
        "Example: 0.05 → within 5 % of the frame width/height counts as 'near the edge'."
    ),
    "flag_min_edges": (
        "Minimum number of bbox edges that must be near the real camera edge "
        "for a disappearance to count as a pass (applies to both flags and gates).\n\n"
        "Example: 2 → at least 2 of the 4 bbox sides must be close to the camera boundary. "
        "Set to 1 to be very permissive; 4 requires the bbox to fill the whole frame."
    ),
    "pass_cooldown_sec": (
        "Minimum time between any two passes (global cooldown). "
        "Prevents double-firing on a single pass event.\n\n"
        "Example: 0.2 s → no second pass can fire within 200 ms of the last one."
    ),
    "type_cooldown_sec": (
        "Minimum time between two passes of the same gate type.\n\n"
        "Example: 0.2 s → two 'square' passes cannot fire within 200 ms of each other."
    ),
    "track_cooldown_sec": (
        "Minimum time before the same track ID can fire another pass.\n\n"
        "Example: 1.5 s → a single track won't produce more than one pass per 1.5 seconds."
    ),
    "area_vel_ema_alpha": (
        "Smoothing factor for the area-growth velocity EMA.\n\n"
        "Higher = reacts faster to changes in approach speed.\n\n"
        "Example: 0.35 is a moderate response."
    ),
    "min_area_vel_ema": (
        "Minimum smoothed growth rate (area fraction per second) required to arm alignment. "
        "Filters out static or receding gates.\n\n"
        "Example: 0.015 → gate must be growing by at least 1.5 % of frame area per second."
    ),
    "aligned_shrink_reset_frac": (
        "If the gate shrinks by this fraction relative to its area when alignment fired, "
        "reset to idle. Prevents a gate behind you from firing.\n\n"
        "Example: 0.14 → if area drops 14 % from the moment of alignment, disarm."
    ),
    "aligned_max_age_sec": (
        "If a track stays in aligned stage longer than this without passing, reset to idle. "
        "Safety valve against stuck alignments.\n\n"
        "Example: 10 s → alignment expires after 10 seconds if no pass occurs."
    ),
    "min_aligned_frames": (
        "Gate must remain in aligned stage for at least this many consecutive frames "
        "before a disappearance can count as a pass. "
        "Filters out single-frame glitch alignments.\n\n"
        "Example: 3 → gate must be aligned across 3 frames (~100 ms at 30 fps)."
    ),
    "gate_pass_area_thresh": (
        "Two-step in-frame pass trigger for gates:\n"
        "1. Gate's area_ratio (shown in bbox overlay as 'ar=') rises above this threshold.\n"
        "2. Area then starts shrinking (area velocity EMA < 0) AND edges are near the "
        "camera boundary → pass fires immediately, no disappearance needed.\n\n"
        "0.0 (default) = disabled — only disappear-based passes fire.\n\n"
        "Example: 0.35 → gate must first cover 35 % of the frame, then as soon as it "
        "starts shrinking and is near the edges, pass fires."
    ),
    "flag_aligned_shrink_reset_frac": (
        "Flag-specific shrink reset threshold. If the flag's bbox area drops by this "
        "fraction relative to when it entered aligned, reset to idle.\n\n"
        "0.0 (default) = disabled — flags are not reset by shrinking area, since you fly "
        "past them (not through them) so the bbox may shrink as you pass.\n\n"
        "Example: 0.30 → if the flag bbox shrinks 30 % from its aligned size, disarm."
    ),
    "ignore_flagpoles": (
        "Skip all flagpole detections in this pass detector instance.\n\n"
        "Example: enable this in the GATES detector and disable in the FLAGS detector "
        "so each handles only its own type."
    ),
}


def _auto_meta(default: Any) -> Dict:
    """Fallback range/step for params not in _PARAM_META."""
    if isinstance(default, float):
        if default <= 1.0:
            return {"min": 0.0,   "max": 5.0,   "step": 0.01, "dec": 2}
        if default <= 10.0:
            return {"min": 0.0,   "max": 50.0,  "step": 0.1,  "dec": 2}
        return     {"min": 0.0,   "max": 500.0, "step": 1.0,  "dec": 1}
    if isinstance(default, int):
        return {"min": 0, "max": max(100, default * 10), "step": 1}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-section param filters
# ─────────────────────────────────────────────────────────────────────────────

# Params that only make sense for flags — hide from the GATES section
_GATE_SKIP: set = {
    "flag_center_tol",
    "flag_aligned_shrink_reset_frac",
    "ignore_flagpoles",
    "flag_aligned_shrink_reset_frac",
}

# Params that only make sense for gates — hide from the FLAGS section
_FLAG_SKIP: set = {
    "center_tol",
    "min_area_vel_ema",
    "area_vel_ema_alpha",
    "aligned_shrink_reset_frac",
    "gate_pass_area_thresh",
    "flag_aligned_shrink_reset_frac",
}


# ─────────────────────────────────────────────────────────────────────────────
# Live introspection of PassDetector and TimeTracker parameters
# ─────────────────────────────────────────────────────────────────────────────

def _introspect(cls) -> List[tuple]:
    """Return [(name, default), ...] for all keyword params of cls.__init__."""
    sig = inspect.signature(cls.__init__)
    return [
        (name, param.default)
        for name, param in sig.parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    ]

def _get_passdet_params() -> List[tuple]:
    from pass_detector import PassDetector
    return _introspect(PassDetector)

def _get_tracker_params() -> List[tuple]:
    from lazy_spotter import TimeTracker
    return _introspect(TimeTracker)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserMark:
    t: float


@dataclass
class DetectedEvent:
    t: float
    event_type: str      # "pass" | "aligned"
    gate_type: str
    reason: str = ""
    track_id: int = -1
    bbox: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Timeline widget
# ─────────────────────────────────────────────────────────────────────────────

class TimelineWidget(QWidget):
    seeked = pyqtSignal(float)

    _USER_H  = 22
    _PASS_H  = 16
    _ALIGN_H = 9
    _PAD     = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(105)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._duration: float = 0.0
        self._current_t: float = 0.0
        self._user_marks: List[UserMark] = []
        self._detected_events: List[DetectedEvent] = []
        self._hover_t: Optional[float] = None

    def set_duration(self, d: float):
        self._duration = float(d)
        self.update()

    def set_current_t(self, t: float):
        self._current_t = float(t)
        self.update()

    def set_user_marks(self, marks: List[UserMark]):
        self._user_marks = marks
        self.update()

    def set_detected_events(self, events: List[DetectedEvent]):
        self._detected_events = events
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

        # track bar + progress fill
        bar_h = 6
        p.fillRect(self._PAD, mid - bar_h // 2, W - 2 * self._PAD, bar_h, _TRACK_COLOR)
        px = self._t_to_x(self._current_t)
        if px > self._PAD:
            p.fillRect(self._PAD, mid - bar_h // 2, px - self._PAD, bar_h, _FILL_COLOR)

        # detected events — below the centre line
        for evt in self._detected_events:
            ex = self._t_to_x(evt.t)
            if evt.event_type == "aligned":
                col = QColor(_ALIGNED_COLOR)
                col.setAlpha(170)
                p.setPen(QPen(col, 1))
                p.drawLine(ex, mid + 3, ex, mid + 3 + self._ALIGN_H)
            else:
                col = QColor(_gate_color(evt.gate_type))
                col.setAlpha(210)
                p.setPen(QPen(col, 2))
                p.drawLine(ex, mid - self._PASS_H, ex, mid + self._PASS_H)
                # small diamond below the bar
                dy = mid + self._PASS_H + 2
                diamond = QPolygon([
                    QPoint(ex,     dy),
                    QPoint(ex + 4, dy + 5),
                    QPoint(ex,     dy + 10),
                    QPoint(ex - 4, dy + 5),
                ])
                p.setBrush(QBrush(col))
                p.drawPolygon(diamond)
                p.setBrush(QBrush())

        # user marks — above the centre line
        for m in self._user_marks:
            mx = self._t_to_x(m.t)
            col = _USER_MARK_COLOR
            p.setPen(QPen(col, 2))
            p.drawLine(mx, mid - self._USER_H, mx, mid)
            ty = mid - self._USER_H
            diamond = QPolygon([
                QPoint(mx,     ty - 8),
                QPoint(mx + 5, ty),
                QPoint(mx,     ty + 8),
                QPoint(mx - 5, ty),
            ])
            p.setBrush(QBrush(col))
            p.drawPolygon(diamond)
            p.setBrush(QBrush())
            p.setFont(QFont("Arial", 7))
            p.setPen(QColor(255, 255, 255))
            p.drawText(mx - 3, ty - 10, "P")

        # playhead
        ph = self._t_to_x(self._current_t)
        p.setPen(QPen(_PLAYHEAD_COLOR, 2))
        p.drawLine(ph, 2, ph, H - 2)

        # hover time
        if self._hover_t is not None:
            hx = self._t_to_x(self._hover_t)
            p.setFont(QFont("Arial", 8))
            p.setPen(QColor(200, 200, 200))
            p.drawText(max(4, min(W - 44, hx - 14)), H - 4, f"{self._hover_t:.2f}s")

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


# ─────────────────────────────────────────────────────────────────────────────
# Video player
# ─────────────────────────────────────────────────────────────────────────────

def _qcolor_to_bgr(c: QColor) -> tuple:
    return (c.blue(), c.green(), c.red())


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
        self._overlay_bbox: Optional[list] = None
        self._overlay_label: str = ""
        self._overlay_color: tuple = (80, 220, 100)
        self._annotation_fn = None   # Optional[Callable[[float], List[dict]]]
        self._cam_left_norm: Optional[float] = None
        self._cam_right_norm: Optional[float] = None
        self._show_aligned: bool = True
        self._show_not_aligned: bool = True

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

    def set_overlay(self, bbox=None, label: str = "", color: tuple = (80, 220, 100)):
        self._overlay_bbox  = bbox
        self._overlay_label = label
        self._overlay_color = color
        if self._last_frame is not None:
            self._push(self._last_frame)

    def clear_overlay(self):
        self.set_overlay()

    def set_show_aligned(self, on: bool):
        self._show_aligned = bool(on)
        if self._last_frame is not None:
            self._push(self._last_frame)

    def set_show_not_aligned(self, on: bool):
        self._show_not_aligned = bool(on)
        if self._last_frame is not None:
            self._push(self._last_frame)

    def set_annotation_fn(self, fn):
        """fn(t: float) -> List[dict] — called each frame to get live track boxes."""
        self._annotation_fn = fn

    def clear_annotations(self):
        self._annotation_fn = None

    def set_camera_edges(self, left_norm: float, right_norm: float):
        self._cam_left_norm = left_norm
        self._cam_right_norm = right_norm
        if self._last_frame is not None:
            self._push(self._last_frame)

    def seek_frames(self, delta: int):
        if not self._cap:
            return
        self.seek(self._current_t + delta / max(self._fps, 1.0))

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
        frame = frame.copy()
        # Camera edge lines (blue vertical lines showing real frame boundary)
        if self._cam_left_norm is not None and self._cam_right_norm is not None:
            h, w = frame.shape[:2]
            lx = int(self._cam_left_norm * w)
            rx = int(self._cam_right_norm * w)
            cv2.line(frame, (lx, 0), (lx, h), (255, 100, 0), 1)
            cv2.line(frame, (rx, 0), (rx, h), (255, 100, 0), 1)
        # Live track annotations (from last run)
        if self._annotation_fn is not None:
            for ann in (self._annotation_fn(self._current_t) or []):
                is_aligned = ann.get("stage", "idle") in ("aligned", "passed")
                if is_aligned and not self._show_aligned:
                    continue
                if not is_aligned and not self._show_not_aligned:
                    continue
                _draw_track_box(frame, ann)
        # Single-event overlay (clicked in event list)
        if self._overlay_bbox and len(self._overlay_bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in self._overlay_bbox]
            col = self._overlay_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            if self._overlay_label:
                (tw, th), _ = cv2.getTextSize(
                    self._overlay_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                ly = max(y1 - 6, th + 4)
                cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), col, -1)
                cv2.putText(frame, self._overlay_label, (x1 + 3, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
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


# ─────────────────────────────────────────────────────────────────────────────
# Background runner thread
# ─────────────────────────────────────────────────────────────────────────────

def _draw_track_box(frame: np.ndarray, ann: dict):
    """Draw one tracked bbox on frame with stage-aware colour and label."""
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    stage  = ann.get("stage", "idle")
    gtype  = ann.get("type", "unknown")
    tid    = ann.get("track_id", -1)
    score      = ann.get("score", 0.0)
    area_ratio = ann.get("area_ratio", 0.0)
    centered   = ann.get("flag_centered", False)

    qc = _gate_color(gtype)
    base_col = (qc.blue(), qc.green(), qc.red())

    if stage == "aligned":
        col, thick = (0, 255, 120), 2
    elif stage == "passed":
        col, thick = (0, 220, 255), 3
    else:
        col, thick = base_col, 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

    extra = "  ctr" if centered else ""
    label = f"tid={tid}  {gtype[:4]}  ar={area_ratio:.3f}{extra}  {stage}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ly = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ly - th - 3), (x1 + tw + 6, ly + 2), col, -1)
    cv2.putText(frame, label, (x1 + 3, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ub = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(ua + ub - inter, 1)

class RunnerThread(QThread):
    progress            = pyqtSignal(int, int, int)   # frame, total, n_events
    event_found         = pyqtSignal(dict)            # one event dict as it fires
    frame_data_found    = pyqtSignal(dict)            # {t, tracks} for bbox overlay
    camera_edges_found  = pyqtSignal(float, float)    # left_norm, right_norm
    finished_ok         = pyqtSignal()
    error               = pyqtSignal(str)

    def __init__(self, video_path: str, model_path: str, params: dict, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._model_path = model_path
        self._params     = params
        self._stop_flag  = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._do_run()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _do_run(self):
        from ultralytics import YOLO
        from lazy_spotter import TimeTracker, clamp_bbox, _get_yolo_names, _cls_to_name
        from pass_detector import PassDetector, detect_camera_edges, _is_flag

        det   = YOLO(self._model_path)
        names = _get_yolo_names(det)

        tracker       = TimeTracker(**self._params["tracker"])
        passdet_gates = PassDetector(**self._params["gates"])
        passdet_flags = PassDetector(**self._params["flags"])

        cap   = cv2.VideoCapture(self._video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, first_frame = cap.read()
        if ok:
            left_n, right_n = detect_camera_edges(first_frame)
            passdet_gates.set_camera_edges(left_n, right_n)
            passdet_flags.set_camera_edges(left_n, right_n)
            self.camera_edges_found.emit(left_n, right_n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        emitted_aligned: set = set()       # (tid, aligned_time) already emitted
        # tid → last timestamp when that track IOU-matched a real YOLO detection.
        # Bypasses the tracker TTL so event markers land on the true disappearance frame.
        yolo_seen_times: Dict[int, float] = {}
        n_events = 0
        frame_idx = 0

        while not self._stop_flag:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            H, W = frame.shape[:2]

            res   = det(frame, conf=0.25, verbose=False, max_det=50)[0]
            typed = []
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                bb    = clamp_bbox((x1, y1, x2, y2), W, H)
                conf  = float(b.conf[0])
                cname = _cls_to_name(int(b.cls[0]), names)
                gtype = cname if conf >= 0.20 else "NONE"
                typed.append({"bbox": bb, "det_conf": conf, "type": gtype, "type_score": conf})
            typed = sorted(typed, key=lambda d: d["det_conf"], reverse=True)[:10]

            tracks = tracker.update(typed, t)

            # Record which tids are confirmed by a real YOLO detection this frame.
            det_bboxes = [d["bbox"] for d in typed]
            for tr in tracks:
                if det_bboxes and any(_iou(tr.bbox, db) >= 0.3 for db in det_bboxes):
                    yolo_seen_times[tr.track_id] = t

            # Route tracks to the right passdet by type
            gate_tracks = [tr for tr in tracks if not _is_flag(tr.locked_type or "")]
            flag_tracks = [tr for tr in tracks if     _is_flag(tr.locked_type or "")]
            passdet_gates.update(gate_tracks, t, frame_w=W, frame_h=H)
            passdet_flags.update(flag_tracks, t, frame_w=W, frame_h=H)

            # collect pass events from both passdets
            pass_fired = False
            for pd in (passdet_gates, passdet_flags):
                while True:
                    evt = pd.pop_any_passed()
                    if evt is None:
                        break
                    pass_fired = True
                    evt["event_type"] = "pass"
                    tid = evt["track_id"]
                    st  = pd.states.get(tid)
                    evt["time"] = yolo_seen_times.get(tid, evt.get("time", t))
                    if st:
                        evt["bbox"] = list(st.last_bbox)
                    n_events += 1
                    self.event_found.emit(evt)

            # if any pass fired, clear aligned tracks across both instances
            if pass_fired:
                passdet_gates.clear_all_aligned()
                passdet_flags.clear_all_aligned()

            # alignment events from both passdets
            for pd in (passdet_gates, passdet_flags):
                for tid, st in pd.states.items():
                    if st.stage == "aligned":
                        key = (tid, round(st.aligned_time, 2))
                        if key not in emitted_aligned:
                            emitted_aligned.add(key)
                            n_events += 1
                            self.event_found.emit({
                                "event_type": "aligned",
                                "track_id":   tid,
                                "type":       st.ttype,
                                "time":       yolo_seen_times.get(tid, st.aligned_time),
                                "reason":     "aligned",
                                "bbox":       list(st.last_bbox),
                            })

            # emit per-frame track snapshot for bbox overlay in video player
            # (only tracks actually present this frame — pd.states keeps stale
            # entries alive for up to 2s after a track disappears)
            visible_tids = {int(tr.track_id) for tr in tracks}
            snapshot = []
            for pd in (passdet_gates, passdet_flags):
                for tid, st in pd.states.items():
                    if tid not in visible_tids:
                        continue
                    snapshot.append({
                        "bbox":          list(st.last_bbox),
                        "track_id":      tid,
                        "type":          st.ttype,
                        "stage":         st.stage,
                        "score":         round(st.last_score_ema, 3),
                        "area_ratio":    round(st.last_area_ratio, 3),
                        "flag_centered": st.flag_was_centered,
                        "nx":            round(st.last_nx, 3),
                    })
            self.frame_data_found.emit({"t": t, "tracks": snapshot})

            if frame_idx % 15 == 0:
                self.progress.emit(frame_idx, total, n_events)

        cap.release()
        self.finished_ok.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Defaults persistence
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS_PATH = Path(__file__).parent / "debug_ui_defaults.json"


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pass Detector Debug UI")
        self.resize(1560, 920)

        self._video_path:     Optional[str] = None
        self._det_model_path: Optional[str] = self._find_model_auto()

        self._user_marks:      List[UserMark]      = []
        self._detected_events: List[DetectedEvent] = []
        self._frame_annotations: Dict[str, List[dict]] = {}  # "%.3f" → [ann, ...]

        self._runner: Optional[RunnerThread] = None
        self._tracker_widgets: Dict[str, Any] = {}
        self._gate_widgets:    Dict[str, Any] = {}
        self._flag_widgets:    Dict[str, Any] = {}

        self._build_ui()
        self._build_shortcuts()

    # ── Model ───────────────────────────────────────────────────────────────

    def _find_model_auto(self) -> Optional[str]:
        default = Path(__file__).parent / "current_best_non_vocab.pt"
        return str(default) if default.exists() else None

    def _ensure_model(self) -> bool:
        if self._det_model_path and Path(self._det_model_path).exists():
            return True
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model", "", "PyTorch Models (*.pt)"
        )
        if path:
            self._det_model_path = path
            return True
        return False

    # ── UI construction ─────────────────────────────────────────────────────

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
        self._model_label = QLabel(
            f"  model: {Path(self._det_model_path).name if self._det_model_path else 'not found'}"
        )
        self._model_label.setStyleSheet("color:#aaa; font-size:11px;")
        tb.addWidget(self._model_label)

        # ── Central layout ───────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Left column — video + timeline
        left = QWidget()
        lv   = QVBoxLayout(left)
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

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(14)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar{background:#222;border:1px solid #444;border-radius:3px;}"
            "QProgressBar::chunk{background:#2a4a8a;}"
        )
        lv.addWidget(self._progress_bar)

        lv.addLayout(self._build_action_row())
        lv.addLayout(self._build_seek_row())

        root.addWidget(left, stretch=3)

        # Right column — params (top) + events (bottom) via splitter
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setFixedWidth(380)

        # ── Params panel ────────────────────────────────────────────────────
        param_outer = QWidget()
        pv = QVBoxLayout(param_outer)
        pv.setContentsMargins(4, 4, 4, 4)
        pv.setSpacing(4)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run  [R]")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(
            "background:#1a3d6a; color:white; font-weight:bold; font-size:13px;"
        )
        self._run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("💾 Save Defaults")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setStyleSheet("background:#2a4a2a; color:white;")
        self._stop_btn.clicked.connect(self._on_stop_or_save)
        run_row.addWidget(self._stop_btn)
        pv.addLayout(run_row)

        hint_lbl = QLabel("Parameters auto-loaded from source")
        hint_lbl.setStyleSheet("font-size:10px; color:#666;")
        pv.addWidget(hint_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        param_inner = QWidget()
        pv2 = QVBoxLayout(param_inner)
        pv2.setContentsMargins(2, 2, 2, 4)
        pv2.setSpacing(2)

        for title, params_fn, store, skip in [
            ("TRACKER",       _get_tracker_params,  self._tracker_widgets, set()),
            ("GATES  (square / arch / circle)", _get_passdet_params, self._gate_widgets, _GATE_SKIP),
            ("FLAGS  (flagpole)",               _get_passdet_params, self._flag_widgets, _FLAG_SKIP),
        ]:
            hdr = QLabel(f"  {title}")
            hdr.setStyleSheet(
                "font-size:11px; font-weight:bold; color:#ccc;"
                "background:#303040; padding:3px 0px;"
            )
            pv2.addWidget(hdr)
            form = QFormLayout()
            form.setSpacing(3)
            form.setContentsMargins(4, 2, 4, 6)
            self._build_param_section(form, params_fn(), store, skip)
            pv2.addLayout(form)

        pv2.addStretch()
        scroll.setWidget(param_inner)
        pv.addWidget(scroll, stretch=1)

        right_splitter.addWidget(param_outer)

        # ── Events panel ─────────────────────────────────────────────────────
        evt_outer = QWidget()
        ev = QVBoxLayout(evt_outer)
        ev.setContentsMargins(4, 4, 4, 4)
        ev.setSpacing(2)

        self._events_header = QLabel("Events: —")
        self._events_header.setStyleSheet("font-size:11px; color:#aaa; font-weight:bold;")
        ev.addWidget(self._events_header)

        self._event_list = QListWidget()
        self._event_list.setStyleSheet(
            "QListWidget{background:#1e1e1e; border:1px solid #444; font-size:11px;"
            "  font-family: monospace;}"
            "QListWidget::item:selected{background:#2a4a8a;}"
        )
        self._event_list.currentRowChanged.connect(self._on_event_selected)
        ev.addWidget(self._event_list, stretch=1)

        right_splitter.addWidget(evt_outer)
        right_splitter.setSizes([520, 280])

        root.addWidget(right_splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        model_name = Path(self._det_model_path).name if self._det_model_path else "not found"
        self._status.showMessage(f"Ready  |  model: {model_name}")

        self._load_defaults()

    def _build_param_section(self, form: QFormLayout,
                              params: List[tuple], store: Dict[str, Any],
                              skip: set = None):
        """Add one spinbox/checkbox row per param into form; save widgets into store."""
        skip = skip or set()
        for name, default in params:
            if name in skip:
                continue
            label = name.replace("_", " ")
            if isinstance(default, bool):
                w = QCheckBox()
                w.setChecked(bool(default))
            elif isinstance(default, int):
                meta = _PARAM_META.get(name) or _auto_meta(default)
                w = QSpinBox()
                w.setRange(int(meta.get("min", 0)), int(meta.get("max", 100)))
                w.setSingleStep(int(meta.get("step", 1)))
                w.setValue(int(default))
            elif isinstance(default, float):
                meta = _PARAM_META.get(name) or _auto_meta(default)
                w = QDoubleSpinBox()
                w.setRange(float(meta.get("min", 0.0)), float(meta.get("max", 100.0)))
                w.setSingleStep(float(meta.get("step", 0.01)))
                w.setDecimals(int(meta.get("dec", 3)))
                w.setValue(float(default))
            else:
                continue
            w.setFixedHeight(24)
            store[name] = w

            doc = _PARAM_DOCS.get(name, "")
            if doc:
                info_btn = QToolButton()
                info_btn.setText("ⓘ")
                info_btn.setFixedSize(20, 20)
                info_btn.setStyleSheet(
                    "QToolButton{font-size:12px; color:#5af; border:none;"
                    " background:transparent;}"
                    "QToolButton:hover{color:#9df;}"
                )
                info_btn.clicked.connect(
                    lambda _, n=name, d=doc: QMessageBox.information(self, n, d)
                )
                row_w = QWidget()
                row_h = QHBoxLayout(row_w)
                row_h.setContentsMargins(0, 0, 0, 0)
                row_h.setSpacing(3)
                row_h.addWidget(w)
                row_h.addWidget(info_btn)
                form.addRow(label + ":", row_w)
            else:
                form.addRow(label + ":", w)

    def _build_legend(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(8)
        h.addWidget(self._lbl("Your marks:"))
        h.addWidget(self._lbl("◆ pass", color=_USER_MARK_COLOR))
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#444;")
        h.addWidget(sep)
        h.addWidget(self._lbl("Detector:"))
        for name, col in _GATE_COLORS.items():
            if name == "unknown":
                continue
            h.addWidget(self._lbl(f"◆ {name}", color=col))
        h.addWidget(self._lbl("│ aligned", color=_ALIGNED_COLOR))

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color:#444;")
        h.addWidget(sep2)

        self._show_aligned_cb = QCheckBox("aligned")
        self._show_aligned_cb.setChecked(True)
        self._show_aligned_cb.setStyleSheet("font-size:10px; color:#ccc;")
        self._show_aligned_cb.toggled.connect(self._video.set_show_aligned)
        h.addWidget(self._show_aligned_cb)

        self._show_not_aligned_cb = QCheckBox("not aligned")
        self._show_not_aligned_cb.setChecked(True)
        self._show_not_aligned_cb.setStyleSheet("font-size:10px; color:#ccc;")
        self._show_not_aligned_cb.toggled.connect(self._video.set_show_not_aligned)
        h.addWidget(self._show_not_aligned_cb)

        h.addStretch()
        return w

    @staticmethod
    def _lbl(text: str, color: Optional[QColor] = None, bold: bool = False, size: int = 10) -> QLabel:
        lbl = QLabel(text)
        c   = f"rgb({color.red()},{color.green()},{color.blue()})" if color else "#999"
        fw  = "bold" if bold else "normal"
        lbl.setStyleSheet(f"font-size:{size}px; color:{c}; font-weight:{fw};")
        return lbl

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(34)
        self._play_btn.clicked.connect(self._on_play_pause)
        row.addWidget(self._play_btn)

        mark_btn = QPushButton("◆ Mark Pass  [P]")
        mark_btn.setFixedHeight(34)
        mark_btn.setStyleSheet("background:#1a4a2a; color:white;")
        mark_btn.clicked.connect(self._on_mark_pass)
        row.addWidget(mark_btn)

        del_btn = QPushButton("Delete Last  [D]")
        del_btn.setFixedHeight(34)
        del_btn.setStyleSheet("background:#5a2020; color:white;")
        del_btn.clicked.connect(self._on_delete_last)
        row.addWidget(del_btn)

        clear_btn = QPushButton("Clear Results")
        clear_btn.setFixedHeight(34)
        clear_btn.setStyleSheet("background:#3a2a00; color:white;")
        clear_btn.clicked.connect(self._on_clear_results)
        row.addWidget(clear_btn)

        return row

    def _build_seek_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._lbl("Seek:", size=11))
        for label, ds in [("◀◀ -10s", -10), ("◀ -1s", -1), ("+1s ▶", 1), ("+10s ▶▶", 10)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px;")
            btn.clicked.connect(lambda _, d=ds: self._on_seek(self._video.current_t + d))
            row.addWidget(btn)
        for label, df in [("◀ -1f", -1), ("+1f ▶", 1)]:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setStyleSheet("font-size:11px; background:#2a2a2a;")
            btn.clicked.connect(lambda _, d=df: self._video.seek_frames(d))
            row.addWidget(btn)
        return row

    def _build_shortcuts(self):
        pairs = [
            ("Space",        self._on_play_pause),
            ("P",            self._on_mark_pass),
            ("D",            self._on_delete_last),
            ("R",            self._on_run),
            ("Up",           lambda: self._video.seek_frames(+1)),
            ("Down",         lambda: self._video.seek_frames(-1)),
            ("Right",        lambda: self._on_seek(self._video.current_t + 1.0)),
            ("Left",         lambda: self._on_seek(self._video.current_t - 1.0)),
            ("Shift+Right",  lambda: self._on_seek(self._video.current_t + 10.0)),
            ("Shift+Left",   lambda: self._on_seek(self._video.current_t - 10.0)),
        ]
        for key, fn in pairs:
            QShortcut(QKeySequence(key), self).activated.connect(fn)

    # ── Model management ────────────────────────────────────────────────────

    def _on_set_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model", "", "PyTorch Models (*.pt)"
        )
        if path:
            self._det_model_path = path
            self._model_label.setText(f"  model: {Path(path).name}")
            self._status.showMessage(f"Model set: {Path(path).name}")

    # ── Video loading ───────────────────────────────────────────────────────

    def _on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.MP4 *.MOV)"
        )
        if not path:
            return
        self._stop_runner()
        if not self._video.load(path):
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            return
        self._video_path = path
        self._user_marks.clear()
        self._detected_events.clear()
        self._frame_annotations.clear()
        self._video.clear_annotations()
        self._event_list.clear()
        self._timeline.set_duration(self._video.duration)
        self._timeline.set_user_marks([])
        self._timeline.set_detected_events([])
        self._events_header.setText("Events: —")
        self.setWindowTitle(f"Pass Detector Debug UI  —  {Path(path).name}")
        self._status.showMessage(f"Loaded: {Path(path).name}  ({self._video.duration:.1f}s)")

    # ── Playback ────────────────────────────────────────────────────────────

    def _on_play_pause(self):
        if not self._video.is_playing:
            self._video.clear_overlay()
        self._video.toggle()
        self._play_btn.setText("⏸  Pause" if self._video.is_playing else "▶  Play")

    def _on_seek(self, t: float):
        self._video.pause()
        self._play_btn.setText("▶  Play")
        self._video.seek(t)

    def _on_position(self, t: float):
        self._timeline.set_current_t(t)
        self._time_label.setText(f"{t:.2f} s  /  {self._video.duration:.2f} s")

    # ── User marks ──────────────────────────────────────────────────────────

    def _on_mark_pass(self):
        if not self._video_path:
            return
        t = self._video.current_t
        self._user_marks.append(UserMark(t=t))
        self._timeline.set_user_marks(self._user_marks)
        self._status.showMessage(
            f"Manual mark at {t:.2f}s  ({len(self._user_marks)} total)"
        )

    def _on_delete_last(self):
        if self._user_marks:
            m = self._user_marks.pop()
            self._timeline.set_user_marks(self._user_marks)
            self._status.showMessage(f"Deleted mark at {m.t:.2f}s")

    def _on_clear_results(self):
        self._detected_events.clear()
        self._frame_annotations.clear()
        self._video.clear_annotations()
        self._event_list.clear()
        self._timeline.set_detected_events([])
        self._events_header.setText("Events: —")

    # ── Run ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_widgets(store: Dict[str, Any]) -> dict:
        out = {}
        for name, w in store.items():
            if isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, QDoubleSpinBox):
                out[name] = w.value()
            elif isinstance(w, QSpinBox):
                out[name] = w.value()
        return out

    def _collect_params(self) -> dict:
        return {
            "tracker": self._read_widgets(self._tracker_widgets),
            "gates":   self._read_widgets(self._gate_widgets),
            "flags":   self._read_widgets(self._flag_widgets),
        }

    def _on_run(self):
        if not self._video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if not self._ensure_model():
            return

        self._stop_runner()
        self._detected_events.clear()
        self._event_list.clear()
        self._timeline.set_detected_events([])
        self._events_header.setText("Events: running…")

        self._runner = RunnerThread(
            video_path = self._video_path,
            model_path = self._det_model_path,
            params     = self._collect_params(),
            parent     = self,
        )
        self._frame_annotations.clear()
        self._video.clear_annotations()
        self._runner.progress.connect(self._on_run_progress)
        self._runner.event_found.connect(self._on_event_found)
        self._runner.frame_data_found.connect(self._on_frame_data)
        self._runner.camera_edges_found.connect(self._video.set_camera_edges)
        self._runner.finished_ok.connect(self._on_run_done)
        self._runner.error.connect(self._on_run_error)
        self._runner.start()
        self._video.set_annotation_fn(self._get_annotations_at)

        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._run_btn.setEnabled(False)
        self._stop_btn.setText("⏹ Stop")
        self._stop_btn.setStyleSheet("background:#5a2020; color:white;")
        self._status.showMessage("Running pass detector…")

    def _on_stop_or_save(self):
        if self._runner and self._runner.isRunning():
            self._stop_runner()
            self._status.showMessage("Stopped.")
        else:
            self._on_save_defaults()

    def _stop_runner(self):
        if self._runner and self._runner.isRunning():
            self._runner.request_stop()
            self._runner.wait(4000)
        self._runner = None
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setText("💾 Save Defaults")
        self._stop_btn.setStyleSheet("background:#2a4a2a; color:white;")

    def _on_save_defaults(self):
        data = self._collect_params()
        try:
            _DEFAULTS_PATH.write_text(json.dumps(data, indent=2))
            self._status.showMessage(f"Defaults saved to {_DEFAULTS_PATH.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _load_defaults(self):
        if not _DEFAULTS_PATH.exists():
            return
        try:
            data = json.loads(_DEFAULTS_PATH.read_text())
        except Exception:
            return
        for section, store in [
            ("tracker", self._tracker_widgets),
            ("gates",   self._gate_widgets),
            ("flags",   self._flag_widgets),
        ]:
            for name, value in data.get(section, {}).items():
                w = store.get(name)
                if w is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))

    def _on_run_progress(self, frame: int, total: int, n_events: int):
        pct = int(100 * frame / max(total, 1))
        self._progress_bar.setValue(pct)
        self._status.showMessage(
            f"Running…  {pct}%  ({frame}/{total} frames)  —  {n_events} events"
        )

    def _on_event_found(self, evt: dict):
        t         = float(evt.get("time") or evt.get("t", 0.0))
        etype     = str(evt.get("event_type", "pass"))
        gate_type = str(evt.get("type") or evt.get("gate_type") or "unknown")
        reason    = str(evt.get("reason", ""))
        track_id  = int(evt.get("track_id", -1))
        bbox      = list(evt.get("bbox") or [])

        det = DetectedEvent(
            t=t, event_type=etype, gate_type=gate_type,
            reason=reason, track_id=track_id, bbox=bbox,
        )
        self._detected_events.append(det)
        self._timeline.set_detected_events(self._detected_events)

        if etype == "pass":
            color = _gate_color(gate_type)
            text  = f"◆ {t:7.2f}s  pass     [{gate_type}]  {reason}  tid={track_id}"
        else:
            color = _ALIGNED_COLOR
            text  = f"│ {t:7.2f}s  aligned  [{gate_type}]  {reason}  tid={track_id}"

        item = QListWidgetItem(text)
        item.setForeground(color)
        self._event_list.addItem(item)
        self._event_list.scrollToBottom()

    def _on_run_done(self):
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setText("💾 Save Defaults")
        self._stop_btn.setStyleSheet("background:#2a4a2a; color:white;")
        passes   = sum(1 for e in self._detected_events if e.event_type == "pass")
        aligneds = sum(1 for e in self._detected_events if e.event_type == "aligned")
        self._events_header.setText(f"Events: {passes} passes, {aligneds} alignments")
        self._status.showMessage(
            f"Done — {passes} pass(es)  |  {aligneds} alignment(s)"
        )

    def _on_run_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setText("💾 Save Defaults")
        self._stop_btn.setStyleSheet("background:#2a4a2a; color:white;")
        self._status.showMessage("Error during run")
        QMessageBox.critical(self, "Runner error", msg[:2000])

    def _on_frame_data(self, data: dict):
        key = f"{data['t']:.3f}"
        self._frame_annotations[key] = data.get("tracks", [])

    def _get_annotations_at(self, t: float) -> List[dict]:
        key = f"{t:.3f}"
        if key in self._frame_annotations:
            return self._frame_annotations[key]
        # ±40 ms tolerance search
        t_ms = round(t * 1000)
        for k, v in self._frame_annotations.items():
            try:
                if abs(round(float(k) * 1000) - t_ms) <= 40:
                    return v
            except ValueError:
                pass
        return []

    def _on_event_selected(self, row: int):
        if 0 <= row < len(self._detected_events):
            self._on_seek(self._detected_events[row].t)


# ─────────────────────────────────────────────────────────────────────────────
# Dark palette + entry point
# ─────────────────────────────────────────────────────────────────────────────

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
