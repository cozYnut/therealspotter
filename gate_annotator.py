#!/usr/bin/env python3
"""
Gate Annotation Tool — single-file edition.

Usage:
    python gate_annotator.py

Requires: PySide6, opencv-python, ultralytics, numpy, pyyaml
Model:    current_best_non_vocab.pt  (same directory as this file)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
#  Stdlib
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from PySide6.QtCore import Qt, QPointF, QRectF, QRect, QSize, Signal, QObject, QThread
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPixmap, QImage, QPainter, QCursor, QFont,
    QAction, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QStatusBar, QSplitter,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSpinBox,
    QComboBox, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QHBoxLayout, QProgressDialog, QDialog, QDialogButtonBox,
    QCheckBox, QGroupBox, QGraphicsScene, QGraphicsView,
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsPixmapItem,
    QTabWidget, QDoubleSpinBox, QFormLayout, QScrollArea,
    QStyledItemDelegate, QButtonGroup, QStyle,
)

# ═════════════════════════════════════════════════════════════════════════════
#  CLASS / KEYPOINT SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

CLASS_NAMES: dict[int, str] = {0: "square", 1: "arch", 2: "circle", 3: "flagpole"}
CLASS_INDICES: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

KPT_NAMES: dict[str, list[str]] = {
    "square":   ["top_left", "top_right", "bottom_right", "bottom_left"],
    "arch":     ["base_left", "shoulder_left", "apex", "shoulder_right", "base_right"],
    "circle":   ["top", "upper_right", "lower_right", "bottom", "lower_left", "upper_left"],
    "flagpole": ["base", "mid", "tip"],
}

KPT_SKELETON: dict[str, list[tuple[int, int]]] = {
    "square":   [(0, 1), (1, 2), (2, 3), (3, 0)],
    "arch":     [(0, 1), (1, 2), (2, 3), (3, 4)],
    "circle":   [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    "flagpole": [(0, 1), (1, 2)],
}

# ═════════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Keypoint:
    x: float    # pixel coords in the original frame
    y: float
    vis: int    # 0=absent  1=occluded/estimated  2=clearly visible


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    class_idx: int
    class_name: str
    conf: float
    keypoints: list[Keypoint] = field(default_factory=list)

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2.0
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2.0
    @property
    def w(self)  -> float: return self.x2 - self.x1
    @property
    def h(self)  -> float: return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "class_idx": self.class_idx, "class_name": self.class_name, "conf": self.conf,
            "keypoints": [{"x": k.x, "y": k.y, "vis": k.vis} for k in self.keypoints],
        }

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        det = Detection(d["x1"], d["y1"], d["x2"], d["y2"],
                        d["class_idx"], d["class_name"], d["conf"])
        det.keypoints = [Keypoint(k["x"], k["y"], k["vis"]) for k in d.get("keypoints", [])]
        return det


# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO LOADER
# ═════════════════════════════════════════════════════════════════════════════

class VideoLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __len__(self) -> int: return self._total

    @property
    def fps(self) -> float: return self._fps

    @property
    def duration(self) -> float: return self._total / max(self._fps, 1.0)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Frame {idx} out of range [0, {self._total})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx}")
        return frame

    def timestamp_of(self, idx: int) -> float:
        return idx / max(self._fps, 1.0)

    def close(self) -> None:
        self._cap.release()

    def __del__(self) -> None:
        try: self._cap.release()
        except Exception: pass


# ═════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def auto_keypoints(det: Detection, frame_w: int, frame_h: int) -> list[Keypoint]:
    """Heuristic keypoints from the detect bbox — used as the annotation starting guess."""
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    cx, cy, w, h = det.cx, det.cy, det.w, det.h

    def in_frame(px: float, py: float) -> bool:
        return 0 <= px <= frame_w and 0 <= py <= frame_h

    def kp(px: float, py: float, estimated: bool = False) -> Keypoint:
        vis = 1 if estimated else (2 if in_frame(px, py) else 1)
        return Keypoint(px, py, vis)

    name = det.class_name
    if name == "square":
        return [kp(x1, y1), kp(x2, y1), kp(x2, y2), kp(x1, y2)]

    elif name == "circle":
        a, b = w / 2.0, h / 2.0
        s60, c60 = math.sin(math.radians(60)), math.cos(math.radians(60))
        return [
            kp(cx,         cy - b      ),   # top
            kp(cx + a*s60, cy - b*c60  ),   # upper_right
            kp(cx + a*s60, cy + b*c60  ),   # lower_right
            kp(cx,         cy + b      ),   # bottom
            kp(cx - a*s60, cy + b*c60  ),   # lower_left
            kp(cx - a*s60, cy - b*c60  ),   # upper_left
        ]

    elif name == "arch":
        return [
            kp(x1, y2        ),   # base_left
            kp(x1, cy,  True ),   # shoulder_left (estimated)
            kp(cx, y1,  True ),   # apex          (estimated)
            kp(x2, cy,  True ),   # shoulder_right (estimated)
            kp(x2, y2        ),   # base_right
        ]

    elif name == "flagpole":
        if h >= w:
            return [kp(cx, y2), kp(cx, cy, True), kp(cx, y1)]
        else:
            return [kp(x1, cy), kp(cx, cy, True), kp(x2, cy)]

    return []


class InferenceEngine:
    def __init__(
        self,
        det_model_path: str,
        kpt_model_paths: Optional[dict[str, str]] = None,
        device: str = "mps",
        det_conf: float = 0.25,
    ) -> None:
        self.det_conf = det_conf
        self.device   = device
        print(f"[Inference] Loading detect model: {det_model_path}")
        self.det_model = YOLO(det_model_path)
        self.kpt_models: dict[str, YOLO] = {}
        for cls_name, path in (kpt_model_paths or {}).items():
            print(f"[Inference] Loading keypoint model ({cls_name}): {path}")
            self.kpt_models[cls_name] = YOLO(path)

    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        res = self.det_model(frame, conf=self.det_conf, verbose=False, device=self.device)[0]
        detections: list[Detection] = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_idx  = int(box.cls[0])
            cls_name = CLASS_NAMES.get(cls_idx, "unknown")
            conf     = float(box.conf[0])
            det = Detection(x1, y1, x2, y2, cls_idx, cls_name, conf)
            if cls_name in self.kpt_models:
                crop, (ox, oy) = self._crop_padded(frame, det)
                kres = self.kpt_models[cls_name](crop, verbose=False, device=self.device)[0]
                det.keypoints = self._parse_keypoints(kres, ox, oy)
            else:
                det.keypoints = auto_keypoints(det, w, h)
            detections.append(det)
        return detections

    def _crop_padded(self, frame: np.ndarray, det: Detection, pad: float = 0.2):
        fh, fw = frame.shape[:2]
        px, py = det.w * pad, det.h * pad
        x1 = max(0.0, det.x1 - px); y1 = max(0.0, det.y1 - py)
        x2 = min(fw,  det.x2 + px); y2 = min(fh,  det.y2 + py)
        return frame[int(y1):int(y2), int(x1):int(x2)], (x1, y1)

    def _parse_keypoints(self, kres, ox: float, oy: float) -> list[Keypoint]:
        kpts: list[Keypoint] = []
        if kres.keypoints is None:
            return kpts
        for xy, conf in zip(kres.keypoints.xy[0].tolist(), kres.keypoints.conf[0].tolist()):
            kpts.append(Keypoint(float(xy[0]) + ox, float(xy[1]) + oy,
                                 2 if float(conf) >= 0.5 else 1))
        return kpts


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class DatasetManager:
    _IMPORTED = "imported"  # subfolder for images imported from external YOLO datasets

    def __init__(self, root: str = "gate_annotations") -> None:
        self.root = Path(root)
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        (self.root / "labels").mkdir(parents=True, exist_ok=True)

    # ── save / load (video frames) ────────────────────────────────────────────

    def save_frame(self, frame: np.ndarray, video_stem: str,
                   frame_idx: int, detections: list[Detection]) -> None:
        frame_dir = self.root / "frames" / video_stem
        label_dir = self.root / "labels" / video_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        img_path  = frame_dir / f"{frame_idx:06d}.jpg"
        lbl_path  = label_dir / f"{frame_idx:06d}.json"
        cv2.imwrite(str(img_path), frame)

        existing_kpt_approved = False
        if lbl_path.exists():
            try:
                old = json.loads(lbl_path.read_text(encoding="utf-8"))
                existing_kpt_approved = old.get("kpt_approved", False)
            except Exception:
                pass

        payload = {
            "video_stem":    video_stem,
            "frame_idx":     frame_idx,
            "image_path":    str(img_path),
            "bbox_approved": True,
            "kpt_approved":  existing_kpt_approved,
            "gates":         [d.to_dict() for d in detections],
        }
        lbl_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_frame_labels(self, video_stem: str,
                          frame_idx: int) -> Optional[list[Detection]]:
        p = self.root / "labels" / video_stem / f"{frame_idx:06d}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return [Detection.from_dict(d) for d in data["gates"]]

    def annotated_frames(self, video_stem: str) -> list[int]:
        d = self.root / "labels" / video_stem
        return sorted(int(p.stem) for p in d.glob("*.json")) if d.exists() else []

    # ── import from external YOLO detect dataset ──────────────────────────────

    def import_yolo_dataset(
        self,
        src_root: str | Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[int, int, list[str]]:
        """Copy images + labels from a YOLO detect dataset into the master pool.

        Validates class names, deduplicates by MD5 hash, writes provenance.
        Returns (imported, skipped, errors).
        """
        src = Path(src_root)
        errors: list[str] = []

        # Validate class names
        src_classes: list[str] = []
        if (src / "classes.txt").exists():
            src_classes = [l.strip() for l in
                           (src / "classes.txt").read_text().splitlines() if l.strip()]
        elif (src / "data.yaml").exists():
            cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
            names = cfg.get("names", [])
            src_classes = names if isinstance(names, list) else list(names.values())
        else:
            errors.append("No classes.txt or data.yaml in dataset root")
            return 0, 0, errors

        expected = list(CLASS_INDICES.keys())
        if src_classes != expected:
            errors.append(f"Class mismatch — got {src_classes}, expected {expected}")
            return 0, 0, errors

        # Load provenance index
        prov_path = self.root / "provenance.json"
        prov: dict = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        known_hashes = {v.get("file_hash") for v in prov.values() if v.get("file_hash")}

        # Collect all images across splits
        all_imgs: list[tuple[Path, Path, str]] = []
        for split in ("train", "val", "test", ""):
            img_dir = (src / "images" / split) if split else (src / "images")
            lbl_dir = (src / "labels" / split) if split else (src / "labels")
            if not img_dir.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(img_dir.glob(ext)):
                    all_imgs.append((img_path, lbl_dir, split or "root"))

        total = len(all_imgs)
        imported = skipped = 0

        for i, (img_path, lbl_dir, split) in enumerate(all_imgs):
            if progress_cb:
                progress_cb(i, total)

            raw = img_path.read_bytes()
            h = hashlib.md5(raw).hexdigest()[:8]
            if h in known_hashes:
                skipped += 1
                continue

            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"Cannot read: {img_path.name}")
                skipped += 1
                continue
            H, W = img.shape[:2]

            pool_stem = f"{img_path.stem}_{h}"
            out_img = self.root / "frames" / self._IMPORTED / f"{pool_stem}.jpg"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), img)

            txt_path = lbl_dir / f"{img_path.stem}.txt"
            dets = (self._yolo_txt_to_detections(txt_path, W, H)
                    if txt_path.exists() else [])

            out_lbl = self.root / "labels" / self._IMPORTED / f"{pool_stem}.json"
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.write_text(json.dumps({
                "video_stem":    self._IMPORTED,
                "frame_idx":     pool_stem,
                "image_path":    str(out_img),
                "bbox_approved": True,
                "kpt_approved":  False,
                "gates":         [d.to_dict() for d in dets],
            }, indent=2), encoding="utf-8")

            prov[f"{self._IMPORTED}/{pool_stem}"] = {
                "source": "dataset",
                "original_path": str(img_path),
                "dataset_root": str(src),
                "split": split,
                "file_hash": h,
                "date_added": datetime.now().isoformat(),
            }
            known_hashes.add(h)
            imported += 1

        if progress_cb:
            progress_cb(total, total)
        prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        return imported, skipped, errors

    @staticmethod
    def _yolo_txt_to_detections(txt_path: Path, W: int, H: int) -> list[Detection]:
        dets: list[Detection] = []
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            cls_name = CLASS_NAMES.get(cls_idx, "unknown")
            det = Detection(x1, y1, x2, y2, cls_idx, cls_name, 1.0)
            det.keypoints = auto_keypoints(det, W, H)
            dets.append(det)
        return dets

    def list_imported_images(self) -> list[Path]:
        """Return sorted list of label JSON paths for imported images."""
        d = self.root / "labels" / self._IMPORTED
        return sorted(d.glob("*.json")) if d.exists() else []

    def save_imported_label(self, pool_stem: str, dets: list[Detection]) -> None:
        lbl = self.root / "labels" / self._IMPORTED / f"{pool_stem}.json"
        if not lbl.exists():
            return
        payload = json.loads(lbl.read_text(encoding="utf-8"))
        payload["gates"] = [d.to_dict() for d in dets]
        payload["bbox_approved"] = True
        lbl.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def copy_video_frames_to_pool(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[int, int, list[str]]:
        """Copy every saved video frame into the imported pool (originals kept).

        Preserves all annotation data, bbox_approved, and kpt_approved.
        Deduplicates by MD5 hash so running twice is safe.
        """
        errors: list[str] = []

        prov_path = self.root / "provenance.json"
        prov: dict = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        known_hashes = {v.get("file_hash") for v in prov.values() if v.get("file_hash")}

        # Collect all video-frame label JSONs (skip the imported subdir)
        lbl_root = self.root / "labels"
        all_labels: list[tuple[Path, str]] = []
        if lbl_root.exists():
            for vdir in sorted(lbl_root.iterdir()):
                if not vdir.is_dir() or vdir.name == self._IMPORTED:
                    continue
                for lf in sorted(vdir.glob("*.json")):
                    all_labels.append((lf, vdir.name))

        total = len(all_labels)
        copied = skipped = 0

        for i, (lf, video_stem) in enumerate(all_labels):
            if progress_cb:
                progress_cb(i, total)
            try:
                payload  = json.loads(lf.read_text(encoding="utf-8"))
                img_path = Path(payload["image_path"])
                if not img_path.exists():
                    errors.append(f"Image missing: {img_path.name}")
                    skipped += 1
                    continue

                raw = img_path.read_bytes()
                h   = hashlib.md5(raw).hexdigest()[:8]
                if h in known_hashes:
                    skipped += 1
                    continue

                fi = payload["frame_idx"]
                pool_stem = (f"{video_stem}_{fi:06d}_{h}"
                             if isinstance(fi, int) else f"{video_stem}_{fi}_{h}")

                out_img = self.root / "frames" / self._IMPORTED / f"{pool_stem}.jpg"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(img_path), str(out_img))

                out_lbl = self.root / "labels" / self._IMPORTED / f"{pool_stem}.json"
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.write_text(json.dumps({
                    "video_stem":    self._IMPORTED,
                    "frame_idx":     pool_stem,
                    "image_path":    str(out_img),
                    "bbox_approved": payload.get("bbox_approved", True),
                    "kpt_approved":  payload.get("kpt_approved",  False),
                    "gates":         payload.get("gates", []),
                }, indent=2), encoding="utf-8")

                prov[f"{self._IMPORTED}/{pool_stem}"] = {
                    "source":              "video_frame",
                    "original_video":      video_stem,
                    "original_frame_idx":  fi,
                    "original_path":       str(img_path),
                    "file_hash":           h,
                    "date_added":          datetime.now().isoformat(),
                }
                known_hashes.add(h)
                copied += 1
            except Exception as e:
                errors.append(f"{lf.name}: {e}")
                skipped += 1

        if progress_cb:
            progress_cb(total, total)
        prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        return copied, skipped, errors

    def set_image_approval(
        self,
        label_path: "str | Path",
        bbox_approved: Optional[bool] = None,
        kpt_approved: Optional[bool] = None,
    ) -> None:
        p = Path(label_path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if bbox_approved is not None:
            data["bbox_approved"] = bbox_approved
        if kpt_approved is not None:
            data["kpt_approved"] = kpt_approved
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_all_images(self) -> list[dict]:
        """Return all annotated images from every subdir, with approval + source metadata."""
        root = self.root / "labels"
        if not root.exists():
            return []
        results = []
        for vdir in sorted(root.iterdir()):
            if not vdir.is_dir():
                continue
            src = "imported" if vdir.name == self._IMPORTED else "video"
            for lf in sorted(vdir.glob("*.json")):
                try:
                    data = json.loads(lf.read_text(encoding="utf-8"))
                    results.append({
                        "label_path":    str(lf),
                        "image_path":    data.get("image_path", ""),
                        "bbox_approved": data.get("bbox_approved", True),
                        "kpt_approved":  data.get("kpt_approved",  False),
                        "source":        src,
                        "stem":          lf.stem,
                    })
                except Exception:
                    pass
        return results

    # ── frozen validation split ────────────────────────────────────────────────

    @staticmethod
    def _entry_id(entry: dict) -> str:
        vs = entry["video_stem"]
        fi = entry["frame_idx"]
        return f"{vs}/{fi:06d}" if isinstance(fi, int) else f"{vs}/{fi}"

    def get_val_split(self) -> set[str]:
        p = self.root / "val_split.json"
        return set(json.loads(p.read_text())) if p.exists() else set()

    def create_val_split(self, val_frac: float = 0.15) -> set[str]:
        entries = self._collect_all_labels()
        random.shuffle(entries)
        n_val = max(1, int(len(entries) * val_frac))
        val_ids = {self._entry_id(e) for e in entries[:n_val]}
        (self.root / "val_split.json").write_text(
            json.dumps(sorted(val_ids), indent=2), encoding="utf-8"
        )
        print(f"[Dataset] Frozen val split created: {len(val_ids)}/{len(entries)} images")
        return val_ids

    def get_kpt_val_split(self, class_name: str) -> set[str]:
        p = self.root / f"val_split_kpt_{class_name}.json"
        return set(json.loads(p.read_text())) if p.exists() else set()

    def create_kpt_val_split(self, class_name: str, val_frac: float = 0.15) -> set[str]:
        n_kpts = len(KPT_NAMES[class_name])
        entries = [
            e for e in self._collect_all_labels()
            if any(g["class_name"] == class_name
                   and len(g.get("keypoints", [])) == n_kpts
                   for g in e["gates"])
        ]
        random.shuffle(entries)
        n_val = max(1, int(len(entries) * val_frac))
        val_ids = {self._entry_id(e) for e in entries[:n_val]}
        (self.root / f"val_split_kpt_{class_name}.json").write_text(
            json.dumps(sorted(val_ids), indent=2), encoding="utf-8"
        )
        print(f"[Dataset] Frozen kpt val split ({class_name}): {len(val_ids)}/{len(entries)}")
        return val_ids

    # ── Ultralytics dataset export ────────────────────────────────────────────

    def export_detect_dataset(
        self,
        output_dir: Optional[str] = None,
        val_split: float = 0.15,
        frozen_val_ids: Optional[set] = None,
    ) -> Path:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(output_dir) if output_dir else self.root / "datasets" / "detect" / ts
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        entries = [e for e in self._collect_all_labels() if e.get("bbox_approved", True)]

        if frozen_val_ids is not None:
            val_e   = [e for e in entries if self._entry_id(e) in frozen_val_ids]
            train_e = [e for e in entries if self._entry_id(e) not in frozen_val_ids]
        else:
            random.shuffle(entries)
            n_val = max(1, int(len(entries) * val_split))
            val_e, train_e = entries[:n_val], entries[n_val:]

        for split_name, ents in [("train", train_e), ("val", val_e)]:
            for p in ents:
                fi = p["frame_idx"]
                stem = (f"{p['video_stem']}_{fi:06d}"
                        if isinstance(fi, int) else f"{p['video_stem']}_{fi}")
                dst_img = out / "images" / split_name / f"{stem}.jpg"
                if Path(p["image_path"]).exists():
                    shutil.copy(p["image_path"], dst_img)
                txt = self._detect_txt(p)
                if txt:
                    (out / "labels" / split_name / f"{stem}.txt").write_text(
                        txt, encoding="utf-8"
                    )

        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(CLASS_INDICES)}\nnames: {list(CLASS_INDICES.keys())}\n",
            encoding="utf-8",
        )
        print(f"[Dataset] detect → {out}  ({len(train_e)} train / {len(val_e)} val)")
        return out

    def export_keypoint_dataset(
        self,
        class_name: str,
        output_dir: Optional[str] = None,
        val_split: float = 0.15,
        frozen_val_ids: Optional[set] = None,
    ) -> Path:
        if class_name not in KPT_NAMES:
            raise ValueError(f"Unknown class: {class_name}")
        n_kpts = len(KPT_NAMES[class_name])
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(output_dir) if output_dir else (
            self.root / "datasets" / "keypoints" / class_name / ts)
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        all_entries = [
            p for p in self._collect_all_labels()
            if p.get("kpt_approved", False)
            and any(g["class_name"] == class_name
                    and len(g.get("keypoints", [])) == n_kpts
                    for g in p["gates"])
        ]
        if not all_entries:
            print(f"[Dataset] No annotated frames for '{class_name}'")
            return out

        if frozen_val_ids is not None:
            val_e   = [e for e in all_entries if self._entry_id(e) in frozen_val_ids]
            train_e = [e for e in all_entries if self._entry_id(e) not in frozen_val_ids]
        else:
            random.shuffle(all_entries)
            n_val = max(1, int(len(all_entries) * val_split))
            val_e, train_e = all_entries[:n_val], all_entries[n_val:]

        for split_name, ents in [("train", train_e), ("val", val_e)]:
            for p in ents:
                img = cv2.imread(p["image_path"])
                if img is None:
                    continue
                H, W = img.shape[:2]
                fi = p["frame_idx"]
                stem = (f"{p['video_stem']}_{fi:06d}"
                        if isinstance(fi, int) else f"{p['video_stem']}_{fi}")
                shutil.copy(p["image_path"], out / "images" / split_name / f"{stem}.jpg")
                lines = [
                    self._pose_line(g, W, H)
                    for g in p["gates"]
                    if g["class_name"] == class_name and len(g.get("keypoints", [])) == n_kpts
                ]
                if lines:
                    (out / "labels" / split_name / f"{stem}.txt").write_text(
                        "\n".join(lines) + "\n", encoding="utf-8"
                    )

        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"nc: 1\nnames: [{class_name}]\nkpt_shape: [{n_kpts}, 3]\n",
            encoding="utf-8",
        )
        print(f"[Dataset] kpt '{class_name}' → {out}  ({len(train_e)} train / {len(val_e)} val)")
        return out

    # ── helpers ──────────────────────────────────────────────────────────────

    def _collect_all_labels(self) -> list[dict]:
        root = self.root / "labels"
        if not root.exists():
            return []
        results = []
        for vdir in sorted(root.iterdir()):
            if not vdir.is_dir():
                continue
            for lf in sorted(vdir.glob("*.json")):
                try:
                    results.append(json.loads(lf.read_text(encoding="utf-8")))
                except Exception:
                    pass
        return results

    @staticmethod
    def _detect_txt(payload: dict) -> str:
        img = cv2.imread(payload["image_path"])
        if img is None:
            return ""
        H, W = img.shape[:2]
        lines = []
        for g in payload["gates"]:
            cls = CLASS_INDICES.get(g["class_name"])
            if cls is None:
                continue
            cx = ((g["x1"] + g["x2"]) / 2) / W
            cy = ((g["y1"] + g["y2"]) / 2) / H
            bw = (g["x2"] - g["x1"]) / W
            bh = (g["y2"] - g["y1"]) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)

    @staticmethod
    def _pose_line(gate: dict, W: int, H: int) -> str:
        cls  = CLASS_INDICES.get(gate["class_name"], 0)
        kpts = gate.get("keypoints", [])
        vis_xs = [k["x"] for k in kpts if k["vis"] > 0]
        vis_ys = [k["y"] for k in kpts if k["vis"] > 0]
        if vis_xs:
            x1, y1 = min(vis_xs), min(vis_ys)
            x2, y2 = max(vis_xs), max(vis_ys)
        else:
            x1, y1, x2, y2 = gate["x1"], gate["y1"], gate["x2"], gate["y2"]
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        bw = max((x2 - x1) / W, 1e-6)
        bh = max((y2 - y1) / H, 1e-6)
        parts = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
        for k in kpts:
            parts.append(f"{k['x']/W:.6f} {k['y']/H:.6f} {k['vis']}")
        return " ".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ═════════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, det_model_path: str, dataset_manager: DatasetManager,
                 output_root: str = "gate_models", device: str = "mps") -> None:
        self.det_model_path = str(det_model_path)
        self.dm             = dataset_manager
        self.output_root    = Path(output_root)
        self.device         = device
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.output_root / "manifest.json"
        self._active_detect = self.det_model_path
        self._active_kpts: dict[str, Optional[str]] = {k: None for k in KPT_NAMES}
        self._best_det_map: float = 0.0
        self._best_kpt_maps: dict[str, float] = {k: 0.0 for k in KPT_NAMES}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            d = json.loads(self._manifest_path.read_text())
            self._active_detect  = d.get("detect",          self.det_model_path)
            self._active_kpts    = d.get("keypoints",       self._active_kpts)
            self._best_det_map   = d.get("best_detect_map", 0.0)
            self._best_kpt_maps  = d.get("best_kpt_maps",   self._best_kpt_maps)

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps({
            "detect":         self._active_detect,
            "keypoints":      self._active_kpts,
            "best_detect_map": self._best_det_map,
            "best_kpt_maps":  self._best_kpt_maps,
        }, indent=2))

    @property
    def active_detect_weights(self) -> str:
        return self._active_detect

    def active_kpt_weights(self, cls: str) -> Optional[str]:
        return self._active_kpts.get(cls)

    # ── fine-tune (returns results; does NOT auto-promote) ────────────────────

    def finetune_detect(
        self,
        epochs: int = 50,
        batch: int = 16,
        lr0: float = 0.002,
        patience: int = 10,
        warmup_epochs: float = 0.5,
        imgsz: int = 640,
        retrain_from_base: bool = False,
    ) -> dict:
        """Train detect model with frozen val split. Caller decides whether to promote."""
        val_ids = self.dm.get_val_split()
        if not val_ids:
            val_ids = self.dm.create_val_split()

        dataset = self.dm.export_detect_dataset(frozen_val_ids=val_ids)
        if not (dataset / "data.yaml").exists():
            return {"error": "No annotated frames to export"}

        start = self.det_model_path if retrain_from_base else self._active_detect
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        run   = self.output_root / "detect" / ts

        print(f"[Trainer] detect  epochs={epochs}  lr0={lr0}  batch={batch}  start={start}")
        res = YOLO(start).train(
            data=str(dataset / "data.yaml"),
            epochs=epochs, imgsz=imgsz, device=self.device,
            project=str(run), name="train", exist_ok=True,
            lr0=lr0, batch=batch, patience=patience, warmup_epochs=warmup_epochs,
        )
        new_map = self._map(res)
        weights = run / "train" / "weights" / "best.pt"
        data_yaml = str(dataset / "data.yaml")
        old_map = self.eval_on_val(self._active_detect, data_yaml)

        print(f"[Trainer] detect  new={new_map:.4f}  old={old_map:.4f}")
        return {
            "weights":  str(weights) if weights.exists() else None,
            "new_map":  new_map,
            "old_map":  old_map,
        }

    def finetune_keypoints(
        self,
        class_name: str,
        epochs: int = 50,
        batch: int = 16,
        lr0: float = 0.002,
        patience: int = 10,
        warmup_epochs: float = 0.5,
        imgsz: int = 640,
        retrain_from_base: bool = False,
    ) -> dict:
        if class_name not in KPT_NAMES:
            raise ValueError(class_name)

        val_ids = self.dm.get_kpt_val_split(class_name)
        if not val_ids:
            val_ids = self.dm.create_kpt_val_split(class_name)

        dataset = self.dm.export_keypoint_dataset(class_name, frozen_val_ids=val_ids)
        if not (dataset / "data.yaml").exists():
            return {"error": f"No '{class_name}' keypoint annotations to export"}

        current = self._active_kpts.get(class_name)
        start   = "yolo11n-pose.pt" if (retrain_from_base or not current) else current
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run     = self.output_root / "keypoints" / class_name / ts
        n       = len(KPT_NAMES[class_name])

        print(f"[Trainer] {class_name}  kpt_shape=[{n},3]  epochs={epochs}  lr0={lr0}")
        res = YOLO(start).train(
            data=str(dataset / "data.yaml"),
            epochs=epochs, imgsz=imgsz, device=self.device,
            project=str(run), name="train", exist_ok=True,
            lr0=lr0, batch=batch, patience=patience, warmup_epochs=warmup_epochs,
        )
        new_map  = self._map(res)
        weights  = run / "train" / "weights" / "best.pt"
        data_yaml = str(dataset / "data.yaml")
        old_map  = self.eval_on_val(current, data_yaml) if current else 0.0

        print(f"[Trainer] {class_name}  new={new_map:.4f}  old={old_map:.4f}")
        return {
            "weights":  str(weights) if weights.exists() else None,
            "new_map":  new_map,
            "old_map":  old_map,
        }

    # ── promotion (called after user confirms in UI) ──────────────────────────

    def promote_detect(self, weights_path: str, new_map: float) -> None:
        self._active_detect = weights_path
        self._best_det_map  = new_map
        self._save_manifest()

    def promote_keypoints(self, class_name: str, weights_path: str, new_map: float) -> None:
        self._active_kpts[class_name]   = weights_path
        self._best_kpt_maps[class_name] = new_map
        self._save_manifest()

    # ── evaluation helpers ────────────────────────────────────────────────────

    def eval_on_val(self, weights_path: str, data_yaml: str) -> float:
        try:
            res = YOLO(weights_path).val(data=data_yaml, device=self.device, verbose=False)
            return float(res.results_dict.get("metrics/mAP50-95(B)", 0.0))
        except Exception as e:
            print(f"[Trainer] eval failed: {e}")
            return 0.0

    @staticmethod
    def _map(results) -> float:
        try:
            return float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        except Exception:
            return 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  ANNOTATION CANVAS — shared draggable infrastructure
# ═════════════════════════════════════════════════════════════════════════════

HANDLE_RADIUS  = 7.0
SKEL_PEN       = QPen(QColor(255, 255, 255, 180), 1.5)
SKEL_PEN_SEL   = QPen(QColor(255, 220,  50, 220), 1.5)

_VIS_BRUSH = {
    0: QBrush(QColor(130, 130, 130)),
    1: QBrush(QColor(255, 200,   0)),
    2: QBrush(QColor(  0, 210,  80)),
}
_VIS_PEN = {
    0: QPen(QColor( 80,  80,  80), 1.5),
    1: QPen(QColor(180, 140,   0), 1.5),
    2: QPen(QColor(  0, 140,  50), 1.5),
}
CLASS_COLORS: dict[str, QColor] = {
    "square":   QColor( 50, 150, 255),
    "arch":     QColor(255, 100,  50),
    "circle":   QColor(200,  50, 255),
    "flagpole": QColor( 50, 220, 120),
}


class DrawMode(Enum):
    SELECT   = auto()
    ADD_GATE = auto()


# ── draggable handle ──────────────────────────────────────────────────────────

class AnnotationHandle(QGraphicsEllipseItem):
    def __init__(self, idx: int, x: float, y: float,
                 radius: float, owner: "DraggablePointSet") -> None:
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self._idx    = idx
        self._owner  = owner
        self.setPos(x, y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self.setZValue(20)

    def itemChange(self, change, value):
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and self._owner and not self._owner._updating):
            constrained = self._owner.constrain_handle(self._idx, value)
            return constrained if constrained is not None else value
        return super().itemChange(change, value)

    def _notify(self) -> None:
        if self._owner and not self._owner._updating:
            self._owner.on_handle_moved(self._idx, self.pos())

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        self._notify()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._notify()


# ── shared draggable point set ────────────────────────────────────────────────

class DraggablePointSet(QObject):
    """N draggable handles + skeleton lines.  Subclass for bbox and keypoints."""
    changed = Signal()

    def __init__(self, scene: QGraphicsScene,
                 positions: list[tuple[float, float]],
                 skeleton:  list[tuple[int, int]],
                 handle_brush: QBrush, handle_pen: QPen,
                 line_pen: QPen, radius: float = HANDLE_RADIUS) -> None:
        super().__init__()
        self.scene    = scene
        self._skeleton = skeleton
        self._updating = False

        self.handles: list[AnnotationHandle] = []
        for i, (x, y) in enumerate(positions):
            h = self._make_handle(i, x, y, radius)
            h.setBrush(handle_brush)
            h.setPen(handle_pen)
            scene.addItem(h)
            self.handles.append(h)

        self.lines: list[QGraphicsLineItem] = []
        for _ in skeleton:
            li = QGraphicsLineItem()
            li.setPen(line_pen)
            li.setZValue(10)
            scene.addItem(li)
            self.lines.append(li)

        self._update_lines()

    def _make_handle(self, idx: int, x: float, y: float,
                     radius: float) -> AnnotationHandle:
        return AnnotationHandle(idx, x, y, radius, self)

    def constrain_handle(self, idx: int, proposed: QPointF) -> Optional[QPointF]:
        return None

    def on_handle_moved(self, idx: int, new_pos: QPointF) -> None:
        self._update_lines()
        self.changed.emit()

    def _update_lines(self) -> None:
        pts = [h.pos() for h in self.handles]
        for li, (a, b) in zip(self.lines, self._skeleton):
            pa, pb = pts[a], pts[b]
            li.setLine(pa.x(), pa.y(), pb.x(), pb.y())

    @property
    def positions(self) -> list[QPointF]:
        return [h.pos() for h in self.handles]

    def set_positions(self, pts: list[tuple[float, float]]) -> None:
        self._updating = True
        for h, (x, y) in zip(self.handles, pts):
            h.setPos(x, y)
        self._updating = False
        self._update_lines()

    def set_visible(self, visible: bool) -> None:
        for h in self.handles:
            h.setVisible(visible)
        for li in self.lines:
            li.setVisible(visible)

    def set_selected_style(self, selected: bool) -> None:
        pen = SKEL_PEN_SEL if selected else SKEL_PEN
        for li in self.lines:
            li.setPen(pen)

    def remove_from_scene(self) -> None:
        for h in self.handles:
            self.scene.removeItem(h)
        for li in self.lines:
            self.scene.removeItem(li)

    def bounding_rect(self) -> QRectF:
        pts = self.positions
        if not pts:
            return QRectF()
        xs = [p.x() for p in pts]; ys = [p.y() for p in pts]
        return QRectF(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))


# ── axis-aligned bounding box ─────────────────────────────────────────────────

class BboxAnnotation(DraggablePointSet):
    """4-corner axis-aligned bbox.  Corners: TL=0, TR=1, BR=2, BL=3."""

    def __init__(self, scene: QGraphicsScene,
                 x1: float, y1: float, x2: float, y2: float,
                 class_name: str = "square") -> None:
        color = CLASS_COLORS.get(class_name, QColor(50, 150, 255))
        self._class_name = class_name
        super().__init__(
            scene=scene,
            positions=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            skeleton=[(0, 1), (1, 2), (2, 3), (3, 0)],
            handle_brush=QBrush(color),
            handle_pen=QPen(Qt.white, 1.5),
            line_pen=QPen(color, 2.0),
        )

    def on_handle_moved(self, idx: int, new_pos: QPointF) -> None:
        if self._updating:
            return
        self._updating = True
        pts = [h.pos() for h in self.handles]

        if idx == 0:
            new = [new_pos, QPointF(pts[1].x(), new_pos.y()),
                   pts[2],  QPointF(new_pos.x(), pts[3].y())]
        elif idx == 1:
            new = [QPointF(pts[0].x(), new_pos.y()), new_pos,
                   QPointF(new_pos.x(), pts[2].y()), pts[3]]
        elif idx == 2:
            new = [pts[0], QPointF(new_pos.x(), pts[1].y()),
                   new_pos, QPointF(pts[3].x(), new_pos.y())]
        else:
            new = [QPointF(new_pos.x(), pts[0].y()), pts[1],
                   QPointF(pts[2].x(), new_pos.y()), new_pos]

        for h, p in zip(self.handles, new):
            h.setPos(p)
        self._updating = False
        self._update_lines()
        self.changed.emit()

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        pts = self.positions
        tl, br = pts[0], pts[2]
        return (min(tl.x(), br.x()), min(tl.y(), br.y()),
                max(tl.x(), br.x()), max(tl.y(), br.y()))

    def set_selected_style(self, selected: bool) -> None:
        color = CLASS_COLORS.get(self._class_name, QColor(50, 150, 255))
        pen = QPen(QColor(255, 220, 50) if selected else color,
                   2.5 if selected else 2.0)
        for li in self.lines:
            li.setPen(pen)
        h_pen = QPen(QColor(255, 220, 50) if selected else Qt.white, 1.5)
        for h in self.handles:
            h.setPen(h_pen)


# ── keypoint handle (adds right-click visibility cycle) ───────────────────────

class _KptHandle(AnnotationHandle):
    def __init__(self, idx: int, x: float, y: float,
                 radius: float, owner: "KeypointAnnotation") -> None:
        super().__init__(idx, x, y, radius, owner)
        self._kpt_owner = owner
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._kpt_owner._cycle_vis(self._idx)
            event.accept()
        else:
            super().mousePressEvent(event)


# ── keypoint annotation ────────────────────────────────────────────────────────

class KeypointAnnotation(DraggablePointSet):
    """N freely-draggable keypoints.  Right-click cycles vis: 2→1→0→2."""

    def __init__(self, scene: QGraphicsScene, class_name: str,
                 keypoints: list[Keypoint]) -> None:
        self._class_name = class_name
        self._vis: list[int] = [kp.vis for kp in keypoints]
        super().__init__(
            scene=scene,
            positions=[(kp.x, kp.y) for kp in keypoints],
            skeleton=KPT_SKELETON.get(class_name, []),
            handle_brush=_VIS_BRUSH[2],
            handle_pen=_VIS_PEN[2],
            line_pen=SKEL_PEN,
        )
        for i in range(len(keypoints)):
            self._apply_vis(i)

    def _make_handle(self, idx: int, x: float, y: float,
                     radius: float) -> _KptHandle:
        return _KptHandle(idx, x, y, radius, self)

    def _apply_vis(self, idx: int) -> None:
        v = self._vis[idx]
        self.handles[idx].setBrush(_VIS_BRUSH[v])
        self.handles[idx].setPen(_VIS_PEN[v])
        self.handles[idx].setOpacity(0.35 if v == 0 else 1.0)

    def _cycle_vis(self, idx: int) -> None:
        self._vis[idx] = {2: 1, 1: 0, 0: 2}[self._vis[idx]]
        self._apply_vis(idx)
        self.changed.emit()

    @property
    def keypoints(self) -> list[Keypoint]:
        return [Keypoint(h.pos().x(), h.pos().y(), v)
                for h, v in zip(self.handles, self._vis)]


# ── gate annotation (bbox + optional keypoints bundled) ───────────────────────

class GateAnnotation:
    def __init__(self, scene: QGraphicsScene, det: Detection) -> None:
        self.class_idx  = det.class_idx
        self.class_name = det.class_name
        self.conf       = det.conf
        self.bbox = BboxAnnotation(scene, det.x1, det.y1, det.x2, det.y2, det.class_name)

        expected = len(KPT_NAMES.get(det.class_name, []))
        if det.keypoints and len(det.keypoints) == expected:
            self.kpts: Optional[KeypointAnnotation] = KeypointAnnotation(
                scene, det.class_name, det.keypoints)
        else:
            self.kpts = None

    def set_selected(self, selected: bool) -> None:
        self.bbox.set_selected_style(selected)
        if self.kpts:
            self.kpts.set_selected_style(selected)

    def set_visible(self, visible: bool) -> None:
        self.bbox.set_visible(visible)
        if self.kpts:
            self.kpts.set_visible(visible)

    def remove_from_scene(self) -> None:
        self.bbox.remove_from_scene()
        if self.kpts:
            self.kpts.remove_from_scene()

    def to_detection(self) -> Detection:
        x1, y1, x2, y2 = self.bbox.xyxy
        det = Detection(x1, y1, x2, y2, self.class_idx, self.class_name, self.conf)
        if self.kpts:
            det.keypoints = self.kpts.keypoints
        return det


# ── canvas ─────────────────────────────────────────────────────────────────────

def _bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    )


class AnnotationCanvas(QGraphicsScene):
    gate_selected       = Signal(int)
    gate_added          = Signal(object)
    gate_deleted        = Signal(int)
    annotations_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.gates: list[GateAnnotation] = []
        self._selected_idx: int = -1
        self._mode: DrawMode = DrawMode.SELECT
        self._pending_class: str = "square"
        self._drag_start: Optional[QPointF] = None
        self._drag_rect: Optional[QGraphicsRectItem] = None

    def set_frame(self, frame: np.ndarray) -> None:
        px = _bgr_to_qpixmap(frame)
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(px)
            self._pixmap_item.setZValue(-1)
            self.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(px)
        self.setSceneRect(self._pixmap_item.boundingRect())

    def load_detections(self, detections: list[Detection]) -> None:
        self.clear_gates()
        for det in detections:
            gate = GateAnnotation(self, det)
            gate.bbox.changed.connect(self.annotations_changed)
            if gate.kpts:
                gate.kpts.changed.connect(self.annotations_changed)
            self.gates.append(gate)

    def clear_gates(self) -> None:
        for g in self.gates:
            g.remove_from_scene()
        self.gates.clear()
        self._selected_idx = -1

    def select_gate(self, idx: int) -> None:
        if 0 <= self._selected_idx < len(self.gates):
            self.gates[self._selected_idx].set_selected(False)
        self._selected_idx = idx
        if 0 <= idx < len(self.gates):
            self.gates[idx].set_selected(True)
        self.gate_selected.emit(idx)

    def delete_selected(self) -> None:
        idx = self._selected_idx
        if 0 <= idx < len(self.gates):
            self.gates[idx].remove_from_scene()
            self.gates.pop(idx)
            self._selected_idx = -1
            self.gate_deleted.emit(idx)
            self.annotations_changed.emit()

    def current_detections(self) -> list[Detection]:
        return [g.to_detection() for g in self.gates]

    def set_mode(self, mode: DrawMode, pending_class: str = "square") -> None:
        self._mode = mode
        self._pending_class = pending_class

    def _view_transform(self):
        from PySide6.QtGui import QTransform
        return self.views()[0].transform() if self.views() else QTransform()

    def mousePressEvent(self, event) -> None:
        if self._mode == DrawMode.ADD_GATE and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.scenePos()
            pen = QPen(CLASS_COLORS.get(self._pending_class, QColor(255, 255, 0)),
                       2, Qt.PenStyle.DashLine)
            self._drag_rect = QGraphicsRectItem(QRectF(self._drag_start, self._drag_start))
            self._drag_rect.setPen(pen)
            self._drag_rect.setZValue(30)
            self.addItem(self._drag_rect)
            event.accept()
        else:
            item = self.itemAt(event.scenePos(), self._view_transform())
            if item is None or item is self._pixmap_item:
                self.select_gate(-1)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_rect and self._drag_start:
            self._drag_rect.setRect(
                QRectF(self._drag_start, event.scenePos()).normalized()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_rect and self._drag_start and event.button() == Qt.MouseButton.LeftButton:
            rect = QRectF(self._drag_start, event.scenePos()).normalized()
            self.removeItem(self._drag_rect)
            self._drag_rect = None
            self._drag_start = None
            if rect.width() > 5 and rect.height() > 5:
                self._add_gate_from_rect(rect)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _add_gate_from_rect(self, rect: QRectF) -> None:
        cls_name = self._pending_class
        cls_idx  = CLASS_INDICES.get(cls_name, 0)
        pw = int(self._pixmap_item.pixmap().width())  if self._pixmap_item else 1920
        ph = int(self._pixmap_item.pixmap().height()) if self._pixmap_item else 1080
        det = Detection(rect.left(), rect.top(), rect.right(), rect.bottom(),
                        cls_idx, cls_name, 1.0)
        det.keypoints = auto_keypoints(det, pw, ph)
        gate = GateAnnotation(self, det)
        gate.bbox.changed.connect(self.annotations_changed)
        if gate.kpts:
            gate.kpts.changed.connect(self.annotations_changed)
        self.gates.append(gate)
        self.gate_added.emit(gate)
        self.annotations_changed.emit()

    def set_bbox_visible(self, visible: bool) -> None:
        for g in self.gates:
            g.bbox.set_visible(visible)

    def set_kpts_visible(self, visible: bool) -> None:
        for g in self.gates:
            if g.kpts:
                g.kpts.set_visible(visible)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)


# ── zoom-capable view ─────────────────────────────────────────────────────────

class AnnotationView(QGraphicsView):
    def __init__(self, scene: AnnotationCanvas, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._panning   = False
        self._pan_start = None

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning   = True
            self._pan_start = event.position().toPoint()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
#  TIMELINE WIDGET
# ═════════════════════════════════════════════════════════════════════════════

class TimelineWidget(QWidget):
    frame_clicked  = Signal(int)
    PX_PER_FRAME   = 2
    H              = 38

    _COLOR = {
        "green":  QColor( 55, 200,  55),
        "yellow": QColor(220, 170,   0),
        "red":    QColor(210,  50,  50),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._total   = 0
        self._current = 0
        self._counts: dict[int, int] = {}
        self.setFixedHeight(self.H)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def setup(self, total_frames: int) -> None:
        self._total = total_frames
        self._counts.clear()
        self._current = 0
        self.setFixedWidth(max(total_frames * self.PX_PER_FRAME, 400))
        self.update()

    def set_current(self, frame_idx: int) -> None:
        old_x = self._current * self.PX_PER_FRAME
        self._current = frame_idx
        new_x = frame_idx * self.PX_PER_FRAME
        self.update(old_x - 1, 0, self.PX_PER_FRAME + 2, self.H)
        self.update(new_x - 1, 0, self.PX_PER_FRAME + 2, self.H)

    def mark_frame(self, frame_idx: int, count: int) -> None:
        self._counts[frame_idx] = count
        x = frame_idx * self.PX_PER_FRAME
        self.update(x, 0, self.PX_PER_FRAME, self.H)

    def clear_marks(self) -> None:
        self._counts.clear()
        self.update()

    @classmethod
    def _color(cls, count: int) -> QColor:
        if count <= 2:
            return cls._COLOR["green"]
        elif count <= 5:
            return cls._COLOR["yellow"]
        return cls._COLOR["red"]

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(event.rect(), QColor(22, 22, 22))
        bar_h = self.H - 6
        for fidx, cnt in self._counts.items():
            x = fidx * self.PX_PER_FRAME
            p.fillRect(x, 3, self.PX_PER_FRAME, bar_h, self._color(cnt))
        if self._total > 0:
            x = self._current * self.PX_PER_FRAME
            p.setPen(QPen(QColor(255, 255, 255, 230), 1))
            p.drawLine(x, 0, x, self.H)

    def _frame_at(self, x: float) -> int:
        if self._total == 0:
            return 0
        return max(0, min(int(x / self.PX_PER_FRAME), self._total - 1))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.frame_clicked.emit(self._frame_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frame_clicked.emit(self._frame_at(event.position().x()))


# ═════════════════════════════════════════════════════════════════════════════
#  BACKGROUND WORKERS
# ═════════════════════════════════════════════════════════════════════════════

class _AllFramesWorker(QObject):
    frame_done = Signal(int, list)
    progress   = Signal(int, int)
    finished   = Signal()

    def __init__(self, engine: InferenceEngine, video_path: str) -> None:
        super().__init__()
        self._engine     = engine
        self._video_path = video_path
        self._stop_flag  = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        video = VideoLoader(self._video_path)
        total = len(video)
        try:
            for idx in range(total):
                if self._stop_flag:
                    break
                try:
                    frame = video.read_frame(idx)
                    dets  = self._engine.predict_frame(frame)
                    if dets:
                        self.frame_done.emit(idx, dets)
                    self.progress.emit(idx + 1, total)
                except Exception:
                    self.progress.emit(idx + 1, total)
        finally:
            video.close()
            self.finished.emit()


class _ImportWorker(QObject):
    """Imports a YOLO dataset folder into the master pool in a background thread."""
    progress = Signal(int, int)          # done, total
    finished = Signal(int, int, list)    # imported, skipped, errors

    def __init__(self, dm: DatasetManager, src_root: str) -> None:
        super().__init__()
        self._dm       = dm
        self._src_root = src_root

    def run(self) -> None:
        def cb(done: int, total: int) -> None:
            self.progress.emit(done, total)
        try:
            imported, skipped, errors = self._dm.import_yolo_dataset(self._src_root, cb)
        except Exception as e:
            imported, skipped, errors = 0, 0, [str(e)]
        self.finished.emit(imported, skipped, errors)


class _CopyWorker(QObject):
    """Copies saved video frames into the imported pool in a background thread."""
    progress = Signal(int, int)         # done, total
    finished = Signal(int, int, list)   # copied, skipped, errors

    def __init__(self, dm: DatasetManager) -> None:
        super().__init__()
        self._dm = dm

    def run(self) -> None:
        def cb(done: int, total: int) -> None:
            self.progress.emit(done, total)
        try:
            copied, skipped, errors = self._dm.copy_video_frames_to_pool(cb)
        except Exception as e:
            copied, skipped, errors = 0, 0, [str(e)]
        self.finished.emit(copied, skipped, errors)


class _ThumbLoader(QObject):
    """Background worker: generates / reads cached thumbnails and emits them."""
    thumb_ready = Signal(str, object)   # (label_path, QImage) — QImage is thread-safe
    finished    = Signal()
    THUMB_W = 160
    THUMB_H = 120

    def __init__(self, items: list, cache_dir: Path) -> None:
        super().__init__()
        self._items     = items      # list of (label_path, image_path) strings
        self._cache_dir = cache_dir
        self._stop      = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        for label_path, image_path in self._items:
            if self._stop:
                break
            try:
                stem       = Path(label_path).stem
                cache_file = self._cache_dir / f"{stem}.jpg"
                if cache_file.exists():
                    qimg = QImage(str(cache_file))
                    if not qimg.isNull():
                        self.thumb_ready.emit(label_path, qimg)
                        continue
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = min(self.THUMB_W / max(w, 1), self.THUMB_H / max(h, 1))
                nw, nh = int(w * scale), int(h * scale)
                small  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((self.THUMB_H, self.THUMB_W, 3), dtype=np.uint8)
                y0 = (self.THUMB_H - nh) // 2
                x0 = (self.THUMB_W - nw) // 2
                canvas[y0:y0+nh, x0:x0+nw] = small
                cv2.imwrite(str(cache_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
                rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                qimg = QImage(
                    bytes(rgb.data), self.THUMB_W, self.THUMB_H,
                    self.THUMB_W * 3, QImage.Format.Format_RGB888
                ).copy()
                self.thumb_ready.emit(label_path, qimg)
            except Exception:
                pass
        self.finished.emit()


class ThumbnailDelegate(QStyledItemDelegate):
    """Draws thumbnail + source/approval badges + filename in the Full Dataset grid."""
    THUMB_W = 160
    THUMB_H = 120
    PAD     = 4
    BADGE_D = 16
    ITEM_W  = 160 + 4 * 2        # 168
    ITEM_H  = 120 + 4 * 2 + 16  # 148

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.ITEM_W, self.ITEM_H)

    def paint(self, painter, option, index) -> None:
        painter.save()
        rect: QRect = option.rect

        # selection highlight
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, QColor(60, 120, 200, 80))
            painter.setPen(QPen(QColor(60, 120, 200), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

        # thumbnail area
        thumb_rect = QRect(
            rect.left() + self.PAD,
            rect.top()  + self.PAD,
            self.THUMB_W,
            self.THUMB_H,
        )
        px = index.data(Qt.ItemDataRole.UserRole + 4)
        if isinstance(px, QPixmap) and not px.isNull():
            painter.drawPixmap(thumb_rect, px)
        else:
            painter.fillRect(thumb_rect, QColor(50, 50, 50))

        # filename (bottom strip)
        name_rect = QRect(rect.left(), rect.bottom() - 15, rect.width(), 16)
        f = QFont()
        f.setPointSize(8)
        painter.setFont(f)
        painter.setPen(QColor(200, 200, 200))
        stem = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if len(stem) > 22:
            stem = stem[:20] + "…"
        painter.drawText(name_rect,
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                         stem)

        # source badge — top-left of thumbnail
        src = index.data(Qt.ItemDataRole.UserRole + 3) or "video"
        src_color  = QColor(60, 130, 210) if src == "imported" else QColor(70, 170, 70)
        src_label  = "IMP" if src == "imported" else "VID"
        src_rect   = QRect(thumb_rect.left() + 2, thumb_rect.top() + 2, 30, self.BADGE_D)
        painter.fillRect(src_rect, src_color)
        painter.setPen(Qt.GlobalColor.white)
        f.setPointSize(7)
        painter.setFont(f)
        painter.drawText(src_rect, Qt.AlignmentFlag.AlignCenter, src_label)

        # B + K approval badges — bottom-right of thumbnail
        bbox_ok = bool(index.data(Qt.ItemDataRole.UserRole + 1))
        kpt_ok  = bool(index.data(Qt.ItemDataRole.UserRole + 2))

        k_rect = QRect(
            thumb_rect.right() - self.BADGE_D - 1,
            thumb_rect.bottom() - self.BADGE_D - 1,
            self.BADGE_D, self.BADGE_D,
        )
        b_rect = QRect(
            k_rect.left() - self.BADGE_D - 2,
            k_rect.top(),
            self.BADGE_D, self.BADGE_D,
        )
        painter.fillRect(b_rect, QColor(55, 200, 55) if bbox_ok else QColor(200, 55, 55))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, "B")

        painter.fillRect(k_rect, QColor(55, 200, 55) if kpt_ok else QColor(200, 55, 55))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(k_rect, Qt.AlignmentFlag.AlignCenter, "K")

        painter.restore()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════

_HERE = Path(__file__).parent


class MainWindow(QMainWindow):
    DET_MODEL_PATH   = str(_HERE / "current_best_non_vocab.pt")
    ANNOTATIONS_ROOT = str(_HERE / "gate_annotations")
    MODELS_ROOT      = str(_HERE / "gate_models")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gate Annotation Tool")
        self.resize(1400, 960)

        # ── state ────────────────────────────────────────────────────────────
        self._video:     Optional[VideoLoader]      = None
        self._frame_idx: int                        = 0
        self._frame:     Optional[np.ndarray]       = None
        self._unsaved:   bool                       = False
        self._engine:    Optional[InferenceEngine]  = None
        self._worker:    Optional[_AllFramesWorker] = None
        self._bg_thread: Optional[QThread]          = None
        self._dm       = DatasetManager(self.ANNOTATIONS_ROOT)
        self._trainer:   Optional[Trainer]          = None
        self._inference_results: dict[int, list]   = {}

        # dataset-tab state
        self._mode: str                     = "video"   # "video" | "dataset"
        self._ds_current_stem: Optional[str] = None     # pool_stem of shown dataset image
        self._ds_frame: Optional[np.ndarray] = None
        self._import_worker: Optional[_ImportWorker] = None
        self._import_thread: Optional[QThread]       = None
        self._copy_worker:   Optional[_CopyWorker]   = None
        self._copy_thread:   Optional[QThread]       = None
        self._ds_folder: Optional[str]               = None  # last browsed folder path
        self._label_to_row: dict[str, int]           = {}
        self._thumb_worker: Optional[_ThumbLoader]   = None
        self._thumb_thread: Optional[QThread]        = None

        # ── canvas ───────────────────────────────────────────────────────────
        self._canvas = AnnotationCanvas()
        self._canvas.gate_selected.connect(self._on_gate_selected)
        self._canvas.gate_added.connect(self._on_gate_added)
        self._canvas.gate_deleted.connect(self._on_gate_deleted)
        self._canvas.annotations_changed.connect(self._mark_unsaved)
        self._view = AnnotationView(self._canvas)

        self._build_toolbar()
        self._build_central()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open a video to begin, or browse the Full Dataset tab.")

        self.addAction(QAction(self, shortcut=Qt.Key.Key_Left,        triggered=self._prev_frame))
        self.addAction(QAction(self, shortcut=Qt.Key.Key_Right,       triggered=self._next_frame))
        self.addAction(QAction(self, shortcut=QKeySequence("Ctrl+S"), triggered=self._save_frame))
        self.addAction(QAction(self, shortcut=Qt.Key.Key_Delete,      triggered=self._canvas.delete_selected))

        self._update_controls()

    # ── toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("Open Video…", self)
        act_open.triggered.connect(self._open_video)
        tb.addAction(act_open)
        tb.addSeparator()

        self._btn_infer = QPushButton("Run Inference")
        self._btn_infer.setMinimumWidth(120)
        self._btn_infer.clicked.connect(self._toggle_inference)
        tb.addWidget(self._btn_infer)

        self._btn_save = QPushButton("Save  Ctrl+S")
        self._btn_save.clicked.connect(self._save_frame)
        tb.addWidget(self._btn_save)
        tb.addSeparator()

        tb.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        for name in CLASS_INDICES:
            self._class_combo.addItem(name)
        self._class_combo.currentTextChanged.connect(self._on_class_changed)
        tb.addWidget(self._class_combo)

        self._btn_add = QPushButton("✚ Add Gate")
        self._btn_add.setCheckable(True)
        self._btn_add.toggled.connect(self._on_add_toggled)
        tb.addWidget(self._btn_add)
        tb.addSeparator()

        self._btn_tune = QPushButton("Fine-tune…")
        self._btn_tune.clicked.connect(self._show_finetune_dialog)
        tb.addWidget(self._btn_tune)
        tb.addSeparator()

        self._btn_show_bbox = QPushButton("BBox")
        self._btn_show_bbox.setCheckable(True)
        self._btn_show_bbox.setChecked(True)
        self._btn_show_bbox.setToolTip("Toggle bounding box visibility")
        self._btn_show_bbox.toggled.connect(self._toggle_bbox_visibility)
        tb.addWidget(self._btn_show_bbox)

        self._btn_show_kpt = QPushButton("KPT")
        self._btn_show_kpt.setCheckable(True)
        self._btn_show_kpt.setChecked(True)
        self._btn_show_kpt.setToolTip("Toggle keypoint visibility")
        self._btn_show_kpt.toggled.connect(self._toggle_kpt_visibility)
        tb.addWidget(self._btn_show_kpt)

    # ── central layout ────────────────────────────────────────────────────────

    def _build_central(self) -> None:
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ── top: canvas (left) + annotation panel (right) ────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._view)

        right = QWidget()
        vlay  = QVBoxLayout(right)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.addWidget(QLabel("Annotations"))

        self._gate_list = QListWidget()
        self._gate_list.currentRowChanged.connect(self._on_list_row)
        vlay.addWidget(self._gate_list)

        hlay = QHBoxLayout()
        hlay.addWidget(QLabel("Re-class:"))
        self._reclass_combo = QComboBox()
        for name in CLASS_INDICES:
            self._reclass_combo.addItem(name)
        hlay.addWidget(self._reclass_combo)
        btn_rc = QPushButton("Apply")
        btn_rc.clicked.connect(self._reclassify)
        hlay.addWidget(btn_rc)
        vlay.addLayout(hlay)

        btn_del = QPushButton("Delete  (Del)")
        btn_del.clicked.connect(self._canvas.delete_selected)
        vlay.addWidget(btn_del)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        # ── bottom: tabbed panel (Video / Full Dataset) ───────────────────────
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.addTab(self._build_video_tab(),   "Video")
        self._bottom_tabs.addTab(self._build_dataset_tab(), "Full Dataset")
        self._bottom_tabs.currentChanged.connect(self._on_tab_changed)

        # vertical splitter so user can resize canvas vs thumbnail grid
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(splitter)
        v_split.addWidget(self._bottom_tabs)
        v_split.setSizes([680, 250])
        root_lay.addWidget(v_split)

        self.setCentralWidget(root)

    def _build_video_tab(self) -> QWidget:
        tab = QWidget()
        blay = QVBoxLayout(tab)
        blay.setContentsMargins(6, 2, 6, 2)
        blay.setSpacing(2)

        nav = QHBoxLayout()
        nav.setSpacing(4)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.clicked.connect(self._prev_frame)
        nav.addWidget(self._btn_prev)

        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(0)
        self._frame_spin.setMaximum(0)
        self._frame_spin.setFixedWidth(80)
        self._frame_spin.editingFinished.connect(self._on_frame_spin)
        nav.addWidget(self._frame_spin)

        self._frame_label = QLabel("/ 0")
        nav.addWidget(self._frame_label)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(36)
        self._btn_next.clicked.connect(self._next_frame)
        nav.addWidget(self._btn_next)

        nav.addStretch()

        for color, label in [("#37c837", "1–2"), ("#daa800", "3–5"), ("#d23232", "6+")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            nav.addWidget(dot)
            nav.addWidget(QLabel(label))

        blay.addLayout(nav)

        self._timeline = TimelineWidget()
        self._timeline_scroll = QScrollArea()
        self._timeline_scroll.setWidget(self._timeline)
        self._timeline_scroll.setWidgetResizable(False)
        self._timeline_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._timeline_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._timeline_scroll.setFixedHeight(TimelineWidget.H + 18)
        self._timeline.frame_clicked.connect(self._go_to_frame)
        blay.addWidget(self._timeline_scroll)

        return tab

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        vlay = QVBoxLayout(tab)
        vlay.setContentsMargins(6, 4, 6, 4)
        vlay.setSpacing(4)

        # ── Row 1: path + browse + import + refresh + status ─────────────────
        top = QHBoxLayout()

        self._ds_path_label = QLabel("No dataset folder selected")
        self._ds_path_label.setStyleSheet("color: #888; font-size: 11px;")
        top.addWidget(self._ds_path_label, stretch=1)

        btn_browse = QPushButton("Browse Dataset Folder…")
        btn_browse.clicked.connect(self._browse_dataset_folder)
        top.addWidget(btn_browse)

        self._btn_import = QPushButton("Import Images")
        self._btn_import.setEnabled(False)
        self._btn_import.clicked.connect(self._import_dataset)
        top.addWidget(self._btn_import)

        self._btn_copy_video = QPushButton("Copy Video Frames to Pool")
        self._btn_copy_video.setToolTip(
            "Copy every saved video frame into the imported pool.\n"
            "Originals are kept — safe to run multiple times."
        )
        self._btn_copy_video.clicked.connect(self._copy_video_frames)
        top.addWidget(self._btn_copy_video)

        btn_refresh = QPushButton("↻ Refresh")
        btn_refresh.clicked.connect(self._refresh_dataset_list)
        top.addWidget(btn_refresh)

        self._ds_status = QLabel("")
        self._ds_status.setStyleSheet("font-size: 11px;")
        top.addWidget(self._ds_status)
        vlay.addLayout(top)

        # ── Row 2: filter bar ─────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        filter_row.addWidget(QLabel("BBox:"))
        self._bbox_filter_grp = QButtonGroup(tab)
        self._bbox_filter_grp.setExclusive(True)
        for i, txt in enumerate(["Any", "✓", "✗"]):
            btn = QPushButton(txt)
            btn.setCheckable(True)
            btn.setFixedWidth(34)
            self._bbox_filter_grp.addButton(btn, i)
            filter_row.addWidget(btn)
        self._bbox_filter_grp.button(0).setChecked(True)
        self._bbox_filter_grp.idToggled.connect(
            lambda _id, chk: self._apply_filter() if chk else None)

        filter_row.addSpacing(12)
        filter_row.addWidget(QLabel("KPT:"))
        self._kpt_filter_grp = QButtonGroup(tab)
        self._kpt_filter_grp.setExclusive(True)
        for i, txt in enumerate(["Any", "✓", "✗"]):
            btn = QPushButton(txt)
            btn.setCheckable(True)
            btn.setFixedWidth(34)
            self._kpt_filter_grp.addButton(btn, i)
            filter_row.addWidget(btn)
        self._kpt_filter_grp.button(0).setChecked(True)
        self._kpt_filter_grp.idToggled.connect(
            lambda _id, chk: self._apply_filter() if chk else None)

        filter_row.addSpacing(12)
        filter_row.addWidget(QLabel("Source:"))
        self._src_combo = QComboBox()
        self._src_combo.addItems(["All Sources", "Video", "Imported"])
        self._src_combo.currentIndexChanged.connect(lambda _: self._apply_filter())
        filter_row.addWidget(self._src_combo)

        filter_row.addSpacing(12)
        self._showing_label = QLabel("Showing 0/0")
        self._showing_label.setStyleSheet("color: #888; font-size: 11px;")
        filter_row.addWidget(self._showing_label)
        filter_row.addStretch()
        vlay.addLayout(filter_row)

        # ── Row 3: bulk approval actions ──────────────────────────────────────
        bulk_row = QHBoxLayout()
        bulk_row.setSpacing(4)

        self._btn_mark_bbox_ok = QPushButton("Mark BBox ✓")
        self._btn_mark_bbox_ok.clicked.connect(lambda: self._bulk_set_approval(bbox=True))
        self._btn_mark_bbox_ok.setEnabled(False)
        bulk_row.addWidget(self._btn_mark_bbox_ok)

        self._btn_mark_bbox_no = QPushButton("Mark BBox ✗")
        self._btn_mark_bbox_no.clicked.connect(lambda: self._bulk_set_approval(bbox=False))
        self._btn_mark_bbox_no.setEnabled(False)
        bulk_row.addWidget(self._btn_mark_bbox_no)

        bulk_row.addSpacing(12)

        self._btn_mark_kpt_ok = QPushButton("Mark KPT ✓")
        self._btn_mark_kpt_ok.clicked.connect(lambda: self._bulk_set_approval(kpt=True))
        self._btn_mark_kpt_ok.setEnabled(False)
        bulk_row.addWidget(self._btn_mark_kpt_ok)

        self._btn_mark_kpt_no = QPushButton("Mark KPT ✗")
        self._btn_mark_kpt_no.clicked.connect(lambda: self._bulk_set_approval(kpt=False))
        self._btn_mark_kpt_no.setEnabled(False)
        bulk_row.addWidget(self._btn_mark_kpt_no)

        bulk_row.addStretch()
        vlay.addLayout(bulk_row)

        # ── Row 4: thumbnail grid ─────────────────────────────────────────────
        self._ds_image_list = QListWidget()
        self._ds_image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._ds_image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._ds_image_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self._ds_image_list.setMovement(QListWidget.Movement.Static)
        self._ds_image_list.setItemDelegate(
            ThumbnailDelegate(self._ds_image_list))
        self._ds_image_list.setSpacing(4)
        self._ds_image_list.setUniformItemSizes(True)
        self._ds_image_list.itemSelectionChanged.connect(
            self._on_ds_selection_changed)
        vlay.addWidget(self._ds_image_list, stretch=1)

        return tab

    # ── tab switching ─────────────────────────────────────────────────────────

    def _on_tab_changed(self, idx: int) -> None:
        if self._unsaved:
            self._save_frame(silent=True)
        if idx == 0:
            self._mode = "video"
            if self._video and self._frame is not None:
                self._canvas.set_frame(self._frame)
                saved = self._dm.load_frame_labels(
                    Path(self._video.path).stem, self._frame_idx)
                if saved is not None:
                    self._canvas.load_detections(saved)
                    self._apply_visibility_to_canvas()
                elif self._frame_idx in self._inference_results:
                    self._canvas.load_detections(self._inference_results[self._frame_idx])
                    self._apply_visibility_to_canvas()
                else:
                    self._canvas.clear_gates()
            else:
                self._canvas.clear_gates()
        else:
            self._mode = "dataset"
            self._refresh_dataset_list()
        self._update_controls()

    # ── video / frame navigation ──────────────────────────────────────────────

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)"
        )
        if not path:
            return
        if self._video:
            self._video.close()
        self._video = VideoLoader(path)
        n = len(self._video)
        self._frame_spin.setMaximum(n - 1)
        self._frame_label.setText(f"/ {n - 1}")
        self._frame_idx = 0
        self._inference_results.clear()
        self._timeline.setup(n)
        self.statusBar().showMessage(
            f"Opened: {Path(path).name}  ({n} frames  {self._video.fps:.1f} fps)"
        )
        if self._engine is None:
            self._engine  = InferenceEngine(self.DET_MODEL_PATH)
            self._trainer = Trainer(self.DET_MODEL_PATH, self._dm, self.MODELS_ROOT)
        # Switch to video tab
        self._bottom_tabs.setCurrentIndex(0)
        self._mode = "video"
        self._go_to_frame(0)
        self._update_controls()

    def _go_to_frame(self, idx: int) -> None:
        if self._video is None:
            return
        idx = max(0, min(idx, len(self._video) - 1))
        if self._unsaved:
            self._save_frame(silent=True)
        self._frame_idx = idx
        self._frame = self._video.read_frame(idx)
        self._canvas.set_frame(self._frame)

        saved = self._dm.load_frame_labels(Path(self._video.path).stem, idx)
        if saved is not None:
            self._canvas.load_detections(saved)
            self._apply_visibility_to_canvas()
            self.statusBar().showMessage(
                f"Frame {idx} — {len(saved)} saved annotation(s)")
        elif idx in self._inference_results:
            dets = self._inference_results[idx]
            self._canvas.load_detections(dets)
            self._apply_visibility_to_canvas()
            self.statusBar().showMessage(
                f"Frame {idx} — {len(dets)} inference result(s)  [unsaved]")
        else:
            self._canvas.clear_gates()
            self.statusBar().showMessage(f"Frame {idx} — no detections")

        self._frame_spin.setValue(idx)
        self._refresh_list()
        self._unsaved = False
        self._update_controls()

        self._timeline.set_current(idx)
        vp_w = self._timeline_scroll.viewport().width()
        x    = idx * TimelineWidget.PX_PER_FRAME
        self._timeline_scroll.horizontalScrollBar().setValue(
            max(0, x - vp_w // 2))

    def _prev_frame(self) -> None: self._go_to_frame(self._frame_idx - 1)
    def _next_frame(self) -> None: self._go_to_frame(self._frame_idx + 1)
    def _on_frame_spin(self) -> None: self._go_to_frame(self._frame_spin.value())

    # ── inference ─────────────────────────────────────────────────────────────

    def _toggle_inference(self) -> None:
        if self._bg_thread and self._bg_thread.isRunning():
            self._stop_inference()
        else:
            self._start_inference()

    def _start_inference(self) -> None:
        if self._engine is None or self._video is None:
            return
        self._inference_results.clear()
        self._timeline.clear_marks()

        self._btn_infer.setText("■ Stop")
        self._btn_infer.setStyleSheet("color: #d23232; font-weight: bold;")
        self.statusBar().showMessage("Running inference on all frames…")

        self._worker = _AllFramesWorker(self._engine, self._video.path)
        thread = QThread(self)
        self._worker.moveToThread(thread)
        thread.started.connect(self._worker.run)
        self._worker.frame_done.connect(self._on_frame_done)
        self._worker.progress.connect(self._on_infer_progress)
        self._worker.finished.connect(self._on_infer_finished)
        self._worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._bg_thread = thread
        thread.start()

    def _stop_inference(self) -> None:
        if self._worker:
            self._worker.stop()

    def _on_frame_done(self, frame_idx: int, dets: list) -> None:
        self._inference_results[frame_idx] = dets
        self._timeline.mark_frame(frame_idx, len(dets))
        if frame_idx == self._frame_idx and self._mode == "video":
            self._canvas.load_detections(dets)
            self._apply_visibility_to_canvas()
            self._refresh_list()

    def _on_infer_progress(self, done: int, total: int) -> None:
        pct = 100 * done // max(total, 1)
        frames_with = len(self._inference_results)
        self.statusBar().showMessage(
            f"Inference  {done}/{total}  ({pct}%)  —  {frames_with} frame(s) with detections"
        )

    def _on_infer_finished(self) -> None:
        self._worker     = None
        self._bg_thread  = None
        self._btn_infer.setText("Run Inference")
        self._btn_infer.setStyleSheet("")
        n = len(self._inference_results)
        self.statusBar().showMessage(
            f"Inference complete — {n} frame(s) with detections"
        )

    # ── save ──────────────────────────────────────────────────────────────────

    def _save_frame(self, silent: bool = False) -> None:
        dets = self._canvas.current_detections()
        if self._mode == "video":
            if not self._video or self._frame is None:
                return
            self._dm.save_frame(self._frame, Path(self._video.path).stem,
                                self._frame_idx, dets)
            if not silent:
                self.statusBar().showMessage(
                    f"Saved frame {self._frame_idx} — {len(dets)} annotation(s)")
        elif self._mode == "dataset" and self._ds_current_stem:
            self._dm.save_imported_label(self._ds_current_stem, dets)
            # update bbox_approved=True in the list item
            for item in self._ds_image_list.selectedItems():
                item.setData(Qt.ItemDataRole.UserRole + 1, True)
            if not silent:
                self.statusBar().showMessage(
                    f"Saved dataset image — {len(dets)} annotation(s)")
        self._unsaved = False

    def _mark_unsaved(self) -> None:
        self._unsaved = True

    # ── gate list ─────────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self._gate_list.blockSignals(True)
        self._gate_list.clear()
        for i, g in enumerate(self._canvas.gates):
            self._gate_list.addItem(
                QListWidgetItem(f"{i+1}. {g.class_name}  conf={g.conf:.2f}")
            )
        self._gate_list.blockSignals(False)

    def _on_gate_selected(self, idx: int) -> None:
        if 0 <= idx < self._gate_list.count():
            self._gate_list.blockSignals(True)
            self._gate_list.setCurrentRow(idx)
            self._gate_list.blockSignals(False)

    def _on_list_row(self, row: int) -> None:
        self._canvas.select_gate(row)

    def _on_gate_added(self, _gate: GateAnnotation) -> None:
        self._refresh_list()
        self._canvas.select_gate(len(self._canvas.gates) - 1)

    def _on_gate_deleted(self, _idx: int) -> None:
        self._refresh_list()

    # ── add-gate mode ─────────────────────────────────────────────────────────

    def _on_add_toggled(self, checked: bool) -> None:
        if checked:
            self._canvas.set_mode(DrawMode.ADD_GATE, self._class_combo.currentText())
            self._btn_add.setText("✖ Cancel")
        else:
            self._canvas.set_mode(DrawMode.SELECT)
            self._btn_add.setText("✚ Add Gate")

    def _on_class_changed(self, cls: str) -> None:
        if self._btn_add.isChecked():
            self._canvas.set_mode(DrawMode.ADD_GATE, cls)

    # ── reclassify ────────────────────────────────────────────────────────────

    def _reclassify(self) -> None:
        idx = self._canvas._selected_idx
        if not (0 <= idx < len(self._canvas.gates)):
            return
        new_cls = self._reclass_combo.currentText()
        gate = self._canvas.gates[idx]
        gate.class_name = new_cls
        gate.class_idx  = CLASS_INDICES[new_cls]
        pw = int(self._canvas._pixmap_item.pixmap().width())  if self._canvas._pixmap_item else 1920
        ph = int(self._canvas._pixmap_item.pixmap().height()) if self._canvas._pixmap_item else 1080
        x1, y1, x2, y2 = gate.bbox.xyxy
        det = Detection(x1, y1, x2, y2, gate.class_idx, new_cls, gate.conf)
        det.keypoints = auto_keypoints(det, pw, ph)
        if gate.kpts:
            gate.kpts.remove_from_scene()
        expected = len(KPT_NAMES.get(new_cls, []))
        if det.keypoints and len(det.keypoints) == expected:
            gate.kpts = KeypointAnnotation(self._canvas, new_cls, det.keypoints)
            gate.kpts.changed.connect(self._mark_unsaved)
        else:
            gate.kpts = None
        self._refresh_list()
        self._mark_unsaved()

    # ── canvas visibility toggles ─────────────────────────────────────────────

    def _toggle_bbox_visibility(self, checked: bool) -> None:
        self._canvas.set_bbox_visible(checked)

    def _toggle_kpt_visibility(self, checked: bool) -> None:
        self._canvas.set_kpts_visible(checked)

    def _apply_visibility_to_canvas(self) -> None:
        self._canvas.set_bbox_visible(self._btn_show_bbox.isChecked())
        self._canvas.set_kpts_visible(self._btn_show_kpt.isChecked())

    # ── dataset tab methods ───────────────────────────────────────────────────

    def _browse_dataset_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select YOLO Dataset Folder", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        self._ds_folder = folder
        self._ds_path_label.setText(folder)
        self._ds_path_label.setStyleSheet("color: #ccc; font-size: 11px;")

        # Validate classes
        src = Path(folder)
        src_classes: list[str] = []
        if (src / "classes.txt").exists():
            src_classes = [l.strip() for l in
                           (src / "classes.txt").read_text().splitlines() if l.strip()]
        elif (src / "data.yaml").exists():
            try:
                cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
                names = cfg.get("names", [])
                src_classes = names if isinstance(names, list) else list(names.values())
            except Exception:
                pass

        expected = list(CLASS_INDICES.keys())
        if src_classes == expected:
            self._btn_import.setEnabled(True)
            self._ds_status.setText("✓ Classes match")
            self._ds_status.setStyleSheet("color: #37c837; font-size: 11px;")
        elif src_classes:
            self._btn_import.setEnabled(False)
            self._ds_status.setText(f"✗ Class mismatch: {src_classes}")
            self._ds_status.setStyleSheet("color: #d23232; font-size: 11px;")
        else:
            self._btn_import.setEnabled(False)
            self._ds_status.setText("✗ No classes.txt or data.yaml found")
            self._ds_status.setStyleSheet("color: #d23232; font-size: 11px;")

    def _import_dataset(self) -> None:
        if not self._ds_folder or self._import_thread:
            return
        self._btn_import.setEnabled(False)
        self._ds_status.setText("Importing…")
        self._ds_status.setStyleSheet("color: #daa800; font-size: 11px;")

        self._import_worker = _ImportWorker(self._dm, self._ds_folder)
        thread = QThread(self)
        self._import_worker.moveToThread(thread)
        thread.started.connect(self._import_worker.run)
        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._import_thread = thread
        thread.start()

    def _on_import_progress(self, done: int, total: int) -> None:
        self.statusBar().showMessage(f"Importing dataset images… {done}/{total}")

    def _copy_video_frames(self) -> None:
        if self._copy_thread and self._copy_thread.isRunning():
            return
        self._btn_copy_video.setEnabled(False)
        self._ds_status.setText("Copying video frames…")
        self._ds_status.setStyleSheet("color: #daa800; font-size: 11px;")

        self._copy_worker = _CopyWorker(self._dm)
        thread = QThread(self)
        self._copy_worker.moveToThread(thread)
        thread.started.connect(self._copy_worker.run)
        self._copy_worker.progress.connect(self._on_copy_progress)
        self._copy_worker.finished.connect(self._on_copy_finished)
        self._copy_worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._copy_thread = thread
        thread.start()

    def _on_copy_progress(self, done: int, total: int) -> None:
        self.statusBar().showMessage(f"Copying video frames to pool… {done}/{total}")

    def _on_copy_finished(self, copied: int, skipped: int, errors: list) -> None:
        self._copy_worker = None
        self._copy_thread = None
        self._btn_copy_video.setEnabled(True)

        msg = f"✓ {copied} copied, {skipped} skipped"
        if errors:
            msg += f", {len(errors)} errors"
        self._ds_status.setText(msg)
        self._ds_status.setStyleSheet("color: #37c837; font-size: 11px;")
        self.statusBar().showMessage(f"Copy done — {msg}")

        if errors:
            QMessageBox.warning(self, "Copy Warnings",
                                "\n".join(errors[:20]) +
                                (f"\n…and {len(errors)-20} more" if len(errors) > 20 else ""))
        self._refresh_dataset_list()

    def _on_import_finished(self, imported: int, skipped: int, errors: list) -> None:
        self._import_worker = None
        self._import_thread = None
        self._btn_import.setEnabled(bool(self._ds_folder))

        msg = f"✓ {imported} imported, {skipped} skipped"
        if errors:
            msg += f", {len(errors)} errors"
        self._ds_status.setText(msg)
        self._ds_status.setStyleSheet("color: #37c837; font-size: 11px;")
        self.statusBar().showMessage(f"Import done — {msg}")

        if errors:
            QMessageBox.warning(self, "Import Warnings",
                                "\n".join(errors[:20]) +
                                (f"\n…and {len(errors)-20} more" if len(errors) > 20 else ""))
        self._refresh_dataset_list()

    def _refresh_dataset_list(self) -> None:
        # stop any running thumb loader
        if self._thumb_worker:
            self._thumb_worker.stop()
        if self._thumb_thread:
            self._thumb_thread.quit()
            self._thumb_thread.wait(500)
        self._thumb_worker = None
        self._thumb_thread = None

        self._ds_image_list.blockSignals(True)
        self._ds_image_list.clear()
        self._label_to_row.clear()

        all_imgs = self._dm.list_all_images()
        placeholder = self._make_placeholder()
        thumb_items: list = []

        for row, entry in enumerate(all_imgs):
            lp      = entry["label_path"]
            ip      = entry["image_path"]
            item    = QListWidgetItem(entry["stem"])
            item.setData(Qt.ItemDataRole.UserRole,     lp)
            item.setData(Qt.ItemDataRole.UserRole + 1, entry["bbox_approved"])
            item.setData(Qt.ItemDataRole.UserRole + 2, entry["kpt_approved"])
            item.setData(Qt.ItemDataRole.UserRole + 3, entry["source"])
            item.setData(Qt.ItemDataRole.UserRole + 4, placeholder)
            self._ds_image_list.addItem(item)
            self._label_to_row[lp] = row
            thumb_items.append((lp, ip))

        self._ds_image_list.blockSignals(False)
        n = len(all_imgs)
        self._ds_status.setText(f"{n} image(s) in pool")
        self._apply_filter()

        if thumb_items:
            cache_dir = Path(self.ANNOTATIONS_ROOT) / "thumbs"
            self._thumb_worker = _ThumbLoader(thumb_items, cache_dir)
            thread = QThread(self)
            self._thumb_worker.moveToThread(thread)
            thread.started.connect(self._thumb_worker.run)
            self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
            self._thumb_worker.finished.connect(self._on_thumb_finished)
            self._thumb_worker.finished.connect(thread.quit)
            thread.finished.connect(thread.deleteLater)
            self._thumb_thread = thread
            thread.start()

    def _on_ds_selection_changed(self) -> None:
        selected = self._ds_image_list.selectedItems()
        has_sel  = len(selected) > 0
        self._btn_mark_bbox_ok.setEnabled(has_sel)
        self._btn_mark_bbox_no.setEnabled(has_sel)
        self._btn_mark_kpt_ok.setEnabled(has_sel)
        self._btn_mark_kpt_no.setEnabled(has_sel)
        if len(selected) == 1:
            row = self._ds_image_list.row(selected[0])
            self._load_dataset_row(row)

    def _load_dataset_row(self, row: int) -> None:
        if row < 0:
            return
        if self._unsaved:
            self._save_frame(silent=True)
        item = self._ds_image_list.item(row)
        if item is None:
            return
        lf_path = Path(item.data(Qt.ItemDataRole.UserRole))
        try:
            payload = json.loads(lf_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.statusBar().showMessage(f"Error reading label: {e}")
            return
        img_path = Path(payload["image_path"])
        if not img_path.exists():
            self.statusBar().showMessage(f"Image missing: {img_path}")
            return
        frame = cv2.imread(str(img_path))
        if frame is None:
            self.statusBar().showMessage(f"Cannot read: {img_path}")
            return
        self._ds_current_stem = str(payload["frame_idx"])
        self._ds_frame = frame
        self._canvas.set_frame(frame)
        dets = [Detection.from_dict(d) for d in payload.get("gates", [])]
        self._canvas.load_detections(dets)
        self._apply_visibility_to_canvas()
        self._refresh_list()
        self._unsaved = False
        bbox_ok = payload.get("bbox_approved", True)
        kpt_ok  = payload.get("kpt_approved",  False)
        self.statusBar().showMessage(
            f"Dataset: {lf_path.stem}  ({len(dets)} annotation(s))"
            f"  BBox={'✓' if bbox_ok else '✗'}  KPT={'✓' if kpt_ok else '✗'}"
        )

    def _apply_filter(self) -> None:
        bbox_id = self._bbox_filter_grp.checkedId()   # 0=Any 1=✓ 2=✗
        kpt_id  = self._kpt_filter_grp.checkedId()    # 0=Any 1=✓ 2=✗
        src_idx = self._src_combo.currentIndex()       # 0=All 1=Video 2=Imported
        shown = total = 0
        for row in range(self._ds_image_list.count()):
            item = self._ds_image_list.item(row)
            if item is None:
                continue
            total += 1
            bbox_ok = bool(item.data(Qt.ItemDataRole.UserRole + 1))
            kpt_ok  = bool(item.data(Qt.ItemDataRole.UserRole + 2))
            src     = item.data(Qt.ItemDataRole.UserRole + 3) or "video"
            hide = (
                (bbox_id == 1 and not bbox_ok) or
                (bbox_id == 2 and     bbox_ok) or
                (kpt_id  == 1 and not kpt_ok)  or
                (kpt_id  == 2 and     kpt_ok)  or
                (src_idx == 1 and src != "video")    or
                (src_idx == 2 and src != "imported")
            )
            item.setHidden(hide)
            if not hide:
                shown += 1
        self._showing_label.setText(f"Showing {shown}/{total}")

    def _bulk_set_approval(
        self, bbox: Optional[bool] = None, kpt: Optional[bool] = None
    ) -> None:
        for item in self._ds_image_list.selectedItems():
            lp = item.data(Qt.ItemDataRole.UserRole)
            if not lp:
                continue
            try:
                self._dm.set_image_approval(lp, bbox_approved=bbox, kpt_approved=kpt)
                if bbox is not None:
                    item.setData(Qt.ItemDataRole.UserRole + 1, bbox)
                if kpt is not None:
                    item.setData(Qt.ItemDataRole.UserRole + 2, kpt)
            except Exception:
                pass
        self._ds_image_list.viewport().update()
        self._apply_filter()

    def _on_thumb_ready(self, label_path: str, qimg) -> None:
        row = self._label_to_row.get(label_path)
        if row is None:
            return
        item = self._ds_image_list.item(row)
        if item is None:
            return
        item.setData(Qt.ItemDataRole.UserRole + 4, QPixmap.fromImage(qimg))

    def _on_thumb_finished(self) -> None:
        self._thumb_worker = None
        self._thumb_thread = None

    def _make_placeholder(self) -> QPixmap:
        px = QPixmap(ThumbnailDelegate.THUMB_W, ThumbnailDelegate.THUMB_H)
        px.fill(QColor(45, 45, 45))
        return px

    # ── fine-tune ─────────────────────────────────────────────────────────────

    def _show_finetune_dialog(self) -> None:
        if not self._trainer:
            QMessageBox.information(self, "Fine-tune",
                                    "Open a video first to initialise the trainer.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Fine-tune")
        dlg.setMinimumWidth(420)
        vlay = QVBoxLayout(dlg)

        # ── pool stats ────────────────────────────────────────────────────────
        all_imgs = self._dm.list_all_images()
        n_bbox   = sum(1 for e in all_imgs if e["bbox_approved"])
        n_kpt    = sum(1 for e in all_imgs if e["kpt_approved"])
        pool_lbl = QLabel(
            f"Pool: {n_bbox} bbox-approved | {n_kpt} kpt-approved "
            f"(of {len(all_imgs)} total)"
        )
        pool_lbl.setStyleSheet("color: #9cf; font-size: 11px;")
        vlay.addWidget(pool_lbl)

        # ── hyperparameters ───────────────────────────────────────────────────
        grp_hp = QGroupBox("Hyperparameters (shared)")
        form = QFormLayout(grp_hp)

        epochs_spin = QSpinBox()
        epochs_spin.setRange(1, 500); epochs_spin.setValue(50)
        form.addRow("Epochs:", epochs_spin)

        lr0_spin = QDoubleSpinBox()
        lr0_spin.setRange(1e-5, 0.1); lr0_spin.setSingleStep(0.001)
        lr0_spin.setDecimals(4); lr0_spin.setValue(0.002)
        form.addRow("lr0:", lr0_spin)

        batch_spin = QSpinBox()
        batch_spin.setRange(1, 256); batch_spin.setValue(16)
        form.addRow("Batch:", batch_spin)

        patience_spin = QSpinBox()
        patience_spin.setRange(0, 200); patience_spin.setValue(10)
        form.addRow("Patience:", patience_spin)

        warmup_spin = QDoubleSpinBox()
        warmup_spin.setRange(0.0, 5.0); warmup_spin.setSingleStep(0.1)
        warmup_spin.setDecimals(1); warmup_spin.setValue(0.5)
        form.addRow("Warmup epochs:", warmup_spin)

        vlay.addWidget(grp_hp)

        # ── model selection ───────────────────────────────────────────────────
        grp_det = QGroupBox("Detect model")
        chk_det = QCheckBox("Fine-tune detect model")
        chk_det.setChecked(True)
        QVBoxLayout(grp_det).addWidget(chk_det)
        vlay.addWidget(grp_det)

        grp_kpt = QGroupBox("Keypoint models")
        kpt_checks: dict[str, QCheckBox] = {}
        klay = QVBoxLayout(grp_kpt)
        for cls in KPT_NAMES:
            chk = QCheckBox(f"Fine-tune {cls} keypoints")
            kpt_checks[cls] = chk
            klay.addWidget(chk)
        vlay.addWidget(grp_kpt)

        # ── val split note ────────────────────────────────────────────────────
        has_split = bool(self._dm.get_val_split())
        note = QLabel(
            "✓ Frozen val split exists — will reuse it." if has_split
            else "⚠ No frozen val split yet — will be created from current pool."
        )
        note.setStyleSheet("color: #888; font-size: 11px;")
        vlay.addWidget(note)

        bbox_btn = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox_btn.accepted.connect(dlg.accept)
        bbox_btn.rejected.connect(dlg.reject)
        vlay.addWidget(bbox_btn)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        run_det = chk_det.isChecked()
        run_kpt = [cls for cls, chk in kpt_checks.items() if chk.isChecked()]
        if not run_det and not run_kpt:
            return

        epochs   = epochs_spin.value()
        lr0      = lr0_spin.value()
        batch    = batch_spin.value()
        patience = patience_spin.value()
        warmup   = warmup_spin.value()

        prog = QProgressDialog("Fine-tuning…", None, 0, 0, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0)
        prog.show()
        QApplication.processEvents()

        finetune_results: dict[str, dict] = {}

        if run_det:
            try:
                finetune_results["detect"] = self._trainer.finetune_detect(
                    epochs=epochs, batch=batch, lr0=lr0,
                    patience=patience, warmup_epochs=warmup,
                )
            except Exception as exc:
                finetune_results["detect"] = {"error": str(exc)}

        for cls in run_kpt:
            try:
                finetune_results[cls] = self._trainer.finetune_keypoints(
                    cls, epochs=epochs, batch=batch, lr0=lr0,
                    patience=patience, warmup_epochs=warmup,
                )
            except Exception as exc:
                finetune_results[cls] = {"error": str(exc)}

        prog.close()

        # ── promotion dialog ──────────────────────────────────────────────────
        self._show_promotion_dialog(finetune_results)

        # Reload inference engine with whatever is now active
        kpt_paths = {cls: self._trainer.active_kpt_weights(cls)
                     for cls in KPT_NAMES if self._trainer.active_kpt_weights(cls)}
        self._engine = InferenceEngine(self._trainer.active_detect_weights, kpt_paths)
        self.statusBar().showMessage("Fine-tune complete — engine updated.")

    def _show_promotion_dialog(self, results: dict) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Fine-tune Results — Promote?")
        dlg.setMinimumWidth(480)
        vlay = QVBoxLayout(dlg)

        vlay.addWidget(QLabel(
            "Review results. Check models to promote to active.\n"
            "Unchecked models stay on disk but are not used."
        ))

        promote_checks: dict[str, tuple[QCheckBox, dict]] = {}

        for key, res in results.items():
            row = QHBoxLayout()
            if "error" in res:
                row.addWidget(QLabel(f"✗  {key}:  ERROR — {res['error']}"))
            else:
                new_map = res.get("new_map", 0.0)
                old_map = res.get("old_map", 0.0)
                weights = res.get("weights")
                if not weights:
                    row.addWidget(QLabel(f"✗  {key}:  no weights produced"))
                else:
                    arrow = "▲" if new_map > old_map else ("▼" if new_map < old_map else "=")
                    lbl = (f"{key}   new mAP={new_map:.4f}   "
                           f"old mAP={old_map:.4f}   {arrow}")
                    chk = QCheckBox(lbl)
                    chk.setChecked(new_map >= old_map)
                    promote_checks[key] = (chk, res)
                    row.addWidget(chk)
            vlay.addLayout(row)

        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        bbox.button(QDialogButtonBox.StandardButton.Ok).setText("Promote selected")
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        vlay.addWidget(bbox)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        for key, (chk, res) in promote_checks.items():
            if not chk.isChecked():
                continue
            weights = res["weights"]
            new_map = res["new_map"]
            if key == "detect":
                self._trainer.promote_detect(weights, new_map)
            else:
                self._trainer.promote_keypoints(key, weights, new_map)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_controls(self) -> None:
        has_video = self._video is not None
        running   = bool(self._bg_thread and self._bg_thread.isRunning())
        in_video  = self._mode == "video"

        self._btn_prev.setEnabled(in_video and has_video and not running and self._frame_idx > 0)
        self._btn_next.setEnabled(
            in_video and has_video and not running and
            self._frame_idx < len(self._video) - 1
        )
        self._btn_infer.setEnabled(in_video and has_video)
        self._btn_save.setEnabled(has_video or self._ds_current_stem is not None)
        self._btn_add.setEnabled(not running)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
        if self._thumb_worker:
            self._thumb_worker.stop()
        if self._copy_thread and self._copy_thread.isRunning():
            self._copy_thread.quit()
            self._copy_thread.wait(1000)
        if self._unsaved:
            self._save_frame(silent=True)
        if self._video:
            self._video.close()
        super().closeEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Gate Annotation Tool")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
