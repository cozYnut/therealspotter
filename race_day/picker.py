"""
Choosing an event in lazy_spotter_v2: everything is found in the data folder.

  fpvtrackside_events(root)   FPVTrackside event folders (with an Event.json),
                              anywhere a few levels down (e.g. <root>/events/<id>)
  track_folders(root)         the data folder's own subfolders that are tracks:
                              not FPVTrackside events, not the folder holding
                              them, not training/ or scores/
  EventPicker                 a list to choose from (newest first, filter, and
                              optionally "New track…")
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from dataset_paths import TRAINING_DIR
from race_day.trackside import OLD_SUBDIR, SUBDIR, parse_time, read_json

NOT_TRACKS = {TRAINING_DIR, "scores"}
# never looked into for events: a track's own folders (big, and never events)
SKIP = {TRAINING_DIR, "scores", SUBDIR, OLD_SUBDIR, "videos", "runs", "gt", "test_videos", "memory_videos",
        "learn_videos", "candidates", "2x2 videos", "recording"}
DEPTH = 3


def fpvtrackside_events(root: Path) -> List[dict]:
    out = []
    root = Path(root)
    for d, dirs, files in os.walk(root):
        depth = len(Path(d).relative_to(root).parts)
        if "Event.json" in files and Path(d) != root:
            ev = read_json(Path(d) / "Event.json") or {}
            out.append({"path": Path(d), "name": ev.get("Name") or Path(d).name, "kind": "FPVTrackside event",
                        "time": parse_time(ev.get("Start")) or os.path.getmtime(d)})
            dirs[:] = []                         # an event's heat folders hold no events
            continue
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in SKIP and depth < DEPTH]
    return out


def _holds_events(d: Path) -> bool:
    """A folder of FPVTrackside events (like <root>/events), not a track."""
    try:
        return any((s / "Event.json").exists() for s in d.iterdir() if s.is_dir())
    except OSError:
        return False


def _last_used(d: Path) -> float:
    """When a track was last used: its app state, its newest video, or the folder."""
    times = [os.path.getmtime(d)]
    for p in (d / SUBDIR / "race_day.json", d / "videos"):
        if p.exists():
            times.append(os.path.getmtime(p))
    return max(times)


def track_folders(root: Path) -> List[dict]:
    out = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in NOT_TRACKS:
            continue
        if (d / "Event.json").exists() or _holds_events(d):
            continue
        out.append({"path": d, "name": d.name, "kind": "track", "time": _last_used(d)})
    return out


class EventPicker(QDialog):
    """Choose one of `items` (newest first); with allow_new, "New track…"
    makes a folder in `root`. chosen: the item's dict after exec()."""

    def __init__(self, parent, title: str, items: List[dict], root: Path, allow_new: bool = False,
                 empty_text: str = "Nothing found in the data folder."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 460)
        self.root, self.chosen = Path(root), None
        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"Data folder: {root}"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter…")
        self.filter.textChanged.connect(self._filter)
        v.addWidget(self.filter)
        self.list = QTreeWidget()
        self.list.setHeaderLabels(["Name", "Kind", "Date", "Folder"])
        self.list.setRootIsDecorated(False)
        self.list.itemDoubleClicked.connect(lambda *_: self._open())
        for it in sorted(items, key=lambda x: -x["time"]):
            row = QTreeWidgetItem([it["name"], it["kind"], time.strftime("%Y-%m-%d %H:%M", time.localtime(it["time"])),
                                   str(it["path"].relative_to(self.root)) if it["path"].is_relative_to(self.root)
                                   else str(it["path"])])
            row.setData(0, Qt.ItemDataRole.UserRole, it)
            row.setToolTip(0, str(it["path"]))
            self.list.addTopLevelItem(row)
        for c in range(4):
            self.list.resizeColumnToContents(c)
        if self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(0))
        v.addWidget(self.list, 1)
        if not items:
            v.addWidget(QLabel(f"<span style='color:#888'>{empty_text}</span>"))
        row = QHBoxLayout()
        if allow_new:
            new = QPushButton("New track…")
            new.setToolTip(f"Make a new track folder in {root}")
            new.clicked.connect(self._new)
            row.addWidget(new)
        row.addStretch()
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._open)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        v.addLayout(row)

    def _filter(self, text: str):
        t = text.strip().lower()
        for i in range(self.list.topLevelItemCount()):
            it = self.list.topLevelItem(i)
            it.setHidden(bool(t) and not any(t in it.text(c).lower() for c in range(4)))

    def _open(self):
        it = self.list.currentItem()
        if it is None or it.isHidden():
            return
        self.chosen = it.data(0, Qt.ItemDataRole.UserRole)
        self.accept()

    def _new(self):
        name, ok = QInputDialog.getText(self, "New track", f"Name of the new track (a folder in {self.root}):")
        name = name.strip()
        if not ok or not name:
            return
        if any(c in name for c in "/\\:") or name.startswith("."):
            return QMessageBox.warning(self, "New track", "Use a plain name, without / \\ : or a leading dot.")
        d = self.root / name
        if d.exists():
            return QMessageBox.warning(self, "New track", f"{name} already exists — choose it from the list.")
        d.mkdir(parents=True)
        self.chosen = {"path": d, "name": name, "kind": "track", "time": time.time()}
        self.accept()


def pick(parent, title: str, items: List[dict], root: Path, allow_new: bool = False,
         empty_text: str = "Nothing found in the data folder.") -> Optional[dict]:
    dlg = EventPicker(parent, title, items, root, allow_new, empty_text)
    return dlg.chosen if dlg.exec() == QDialog.DialogCode.Accepted else None
