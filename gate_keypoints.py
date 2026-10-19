#!/usr/bin/env python3
"""
Gate keypoints for the Full Dataset of gate_annotator.py.

One small pose model per class, run on a padded crop around each gate's box
(the same crop gate_annotator's InferenceEngine uses):

  square    top_left, top_right, bottom_right, bottom_left   (outer corners; a
            U-shaped gate's feet are bottom_right / bottom_left)
  arch      base_left, shoulder_left, apex, shoulder_right, base_right
  flagpole  base, mid, tip

Ground truth is every frame with kpt_approved (OK, green in the annotator).
A fixed part of them is held out for testing (kpt_test_split.json) and never
trained on, so every score is honest.

    python gate_keypoints.py drafts            # draft points from the segmentation model's masks
    python gate_keypoints.py evaluate          # score drafts / placeholders / models on the held-out frames
    python gate_keypoints.py train             # train the per-class models (variants), keep the best
    python gate_keypoints.py apply             # write the best models' points into every not-OK frame
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataset_paths import training_dir

KPT_NAMES: Dict[str, List[str]] = {
    "square":   ["top_left", "top_right", "bottom_right", "bottom_left"],
    "arch":     ["base_left", "shoulder_left", "apex", "shoulder_right", "base_right"],
    "circle":   ["top", "upper_right", "lower_right", "bottom", "lower_left", "upper_left"],
    "flagpole": ["base", "mid", "tip"],
}
CLASSES = ["square", "arch", "flagpole"]          # the classes this dataset has
CROP_PAD = 0.2                                    # same as gate_annotator.InferenceEngine._crop_padded
IMGSZ = 320                                       # crops are small: 320 is plenty and fast
TEST_FRAC = 0.4

ANN = training_dir() / "gate_annotations"
LABELS = ANN / "labels"
WORK = training_dir() / "keypoint_work"
MODELS = training_dir() / "gate_models" / "keypoints"
SEG_MODEL = training_dir() / "gate-segmentation-yolo11n-step1200-fp32.onnx"


# ──────────────────────────────────────────────────────────────
# Dataset access
# ──────────────────────────────────────────────────────────────

def all_labels() -> List[Tuple[Path, dict]]:
    out = []
    for f in sorted(LABELS.glob("*/*.json")):
        try:
            out.append((f, json.loads(f.read_text(encoding="utf-8"))))
        except ValueError:
            pass
    return out


def entry_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def _val_id(d: dict) -> str:
    """gate_annotator's id of a frame in its Val split (val_split.json)."""
    fi = d.get("frame_idx")
    return f"{d.get('video_stem')}/{fi:06d}" if isinstance(fi, int) else f"{d.get('video_stem')}/{fi}"


def test_split(labels) -> set:
    """The test frames: OK keypoint frames that are in the annotator's Val
    split ("→ Val" in the Full Dataset tab). Never trained on, so every score
    is honest. (Before any frame was in Val: a fixed 40% of the OK frames.)"""
    vp = ANN / "val_split.json"
    val = set(json.loads(vp.read_text())) if vp.exists() else set()
    test = {entry_id(f) for f, d in labels if d.get("kpt_approved") and _val_id(d) in val}
    if test:
        return test
    p = ANN / "kpt_test_split.json"
    if p.exists():
        return set(json.loads(p.read_text()))
    ids = sorted(entry_id(f) for f, d in labels if d.get("kpt_approved"))
    test = {i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % 100 < TEST_FRAC * 100}
    p.write_text(json.dumps(sorted(test), indent=1))
    return test


def tune_split(labels, test: set, frac: float = 0.1) -> set:
    """Training-time validation (picking the best round): a fixed 10% of the
    OK frames that aren't test frames. Not trained on, never the test set."""
    return {entry_id(f) for f, d in labels if d.get("kpt_approved") and entry_id(f) not in test
            and int(hashlib.md5(("tune|" + entry_id(f)).encode()).hexdigest(), 16) % 100 < frac * 100}


def crop_box(frame_wh, box, pad: float = CROP_PAD):
    """The padded crop around a gate box: (x1, y1, x2, y2) in frame pixels."""
    W, H = frame_wh
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0.0, x1 - px), max(0.0, y1 - py), min(float(W), x2 + px), min(float(H), y2 + py))


def is_placeholder(g: dict) -> bool:
    """Keypoints still at the box-derived starting guess (auto_keypoints)."""
    k = g.get("keypoints") or []
    if g["class_name"] == "square" and len(k) == 4:
        c = {(round(g["x1"]), round(g["y1"])), (round(g["x2"]), round(g["y1"])),
             (round(g["x2"]), round(g["y2"])), (round(g["x1"]), round(g["y2"]))}
        return {(round(p["x"]), round(p["y"])) for p in k} == c
    if g["class_name"] == "flagpole" and len(k) == 3:
        cx = (g["x1"] + g["x2"]) / 2
        cy = (g["y1"] + g["y2"]) / 2
        return all(abs(p["x"] - cx) < 0.5 for p in k) or all(abs(p["y"] - cy) < 0.5 for p in k)
    if g["class_name"] == "arch" and len(k) == 5:
        return abs(k[0]["x"] - g["x1"]) < 0.5 and abs(k[4]["x"] - g["x2"]) < 0.5 and abs(k[2]["y"] - g["y1"]) < 0.5
    return not k


# ──────────────────────────────────────────────────────────────
# Draft points from segmentation masks
# ──────────────────────────────────────────────────────────────

def _order_quad(pts):
    pts = np.array(pts, dtype=float)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]]


def _quad(contour):
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 23):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            q = _order_quad(ap[:, 0, :])
            a_q = cv2.contourArea(np.array(q, dtype=np.float32))
            return q, (cv2.contourArea(hull) / a_q if a_q > 0 else 0.0)
    return None, 0.0


def _vis(p, W, H, m=3):
    """2 = a real point; 0 = on the frame edge (the gate is cut off there)."""
    return 0 if (p[0] <= m or p[1] <= m or p[0] >= W - 1 - m or p[1] >= H - 1 - m) else 2


def draft_from_mask(cls: str, mask: np.ndarray, box, W: int, H: int) -> Optional[List[dict]]:
    """Keypoints for one gate from its mask (frame-size uint8), or None if the
    mask doesn't give a clean shape."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 60:
        return None
    pts = c[:, 0, :].astype(float)
    if cls == "square":
        q, fit = _quad(c)
        if q is None or fit < 0.85:
            return None
        bw, bh = box[2] - box[0], box[3] - box[1]
        qa = cv2.contourArea(np.array(q, dtype=np.float32))
        if not (0.35 * bw * bh <= qa <= 1.3 * bw * bh):
            return None
        return [{"x": float(p[0]), "y": float(p[1]), "vis": _vis(p, W, H)} for p in q]
    if cls == "flagpole":
        ys = pts[:, 1]
        if ys.max() - ys.min() < 20:
            return None
        base = pts[ys.argmax()]
        tip = pts[ys.argmin()]
        midp = (base + tip) / 2
        ys_, xs_ = np.nonzero(mask)
        mpts = np.stack([xs_, ys_], 1).astype(float)
        mid = mpts[np.argmin(((mpts - midp) ** 2).sum(1))]
        return [{"x": float(p[0]), "y": float(p[1]), "vis": _vis(p, W, H)} for p in (base, mid, tip)]
    if cls == "arch":
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        w = x1 - x0
        if w < 20:
            return None
        left = pts[pts[:, 0] <= x0 + 0.35 * w]
        right = pts[pts[:, 0] >= x1 - 0.35 * w]
        if not len(left) or not len(right):
            return None
        bl, br = left[left[:, 1].argmax()], right[right[:, 1].argmax()]
        apex = pts[pts[:, 1].argmin()]

        def shoulder(a, b, side):
            sel = pts[(pts[:, 0] < apex[0]) if side < 0 else (pts[:, 0] > apex[0])]
            sel = sel[sel[:, 1] < max(a[1], b[1])]
            if not len(sel):
                return (a + b) / 2
            d = np.cross(b - a, sel - a) / (np.linalg.norm(b - a) + 1e-6)
            return sel[np.argmax(np.abs(d))]
        sl, sr = shoulder(bl, apex, -1), shoulder(br, apex, 1)
        return [{"x": float(p[0]), "y": float(p[1]), "vis": _vis(p, W, H)} for p in (bl, sl, apex, sr, br)]
    return None


def fix_square_feet(g: dict, kps: List[dict], frame_h: int) -> List[dict]:
    """A square gate's box always reaches its lowest foot: move the lower foot
    down (or up) to the box bottom and the other foot by the same amount.
    Masks and models both tend to put the feet too high. Not done when the
    box touches the frame bottom (the feet are out of view there)."""
    if len(kps) != 4 or g["y2"] >= frame_h - 3:
        return kps
    kps = [dict(k) for k in kps]
    lo = max((2, 3), key=lambda k: kps[k]["y"])
    dy = g["y2"] - kps[lo]["y"]
    for k in (2, 3):
        kps[k]["y"] += dy
    return kps


SEG_CLASSES = {"square": {"square_gate", "arch_gate"}, "arch": {"square_gate", "arch_gate"}, "flagpole": {"flag"}}


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def make_drafts(only_ids: Optional[set] = None, out: Path = WORK / "drafts.json"):
    """Seg-model drafts for every gate of every frame (matched to the frame's
    approved boxes, so the class is the dataset's own)."""
    from ultralytics import YOLO
    seg = YOLO(str(SEG_MODEL), task="segment")
    WORK.mkdir(parents=True, exist_ok=True)
    drafts = json.loads(out.read_text()) if out.exists() else {}
    labels = all_labels()
    t0 = time.time()
    for n, (f, d) in enumerate(labels):
        eid = entry_id(f)
        if eid in drafts or (only_ids is not None and eid not in only_ids):
            continue
        img = cv2.imread(d["image_path"])
        if img is None:
            drafts[eid] = []
            continue
        H, W = img.shape[:2]
        r = seg(cv2.resize(img, (640, 640)), verbose=False, conf=0.25)[0]       # squashed: see the model notes
        inst = []
        if r.masks is not None:
            for b, m in zip(r.boxes, r.masks.data.cpu().numpy()):
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                inst.append((seg.names[int(b.cls)], [x1 * W / 640, y1 * H / 640, x2 * W / 640, y2 * H / 640],
                             (cv2.resize(m, (W, H)) > 0.5).astype(np.uint8)))
        out_g = []
        for gi, g in enumerate(d["gates"]):
            box = [g["x1"], g["y1"], g["x2"], g["y2"]]
            best = max(((_iou(box, ib), im) for nm, ib, im in inst if nm in SEG_CLASSES.get(g["class_name"], ())),
                       default=(0.0, None), key=lambda x: x[0])
            kp = draft_from_mask(g["class_name"], best[1], box, W, H) if best[0] >= 0.5 else None
            out_g.append({"gate": gi, "cls": g["class_name"], "iou": round(best[0], 3), "keypoints": kp})
        drafts[eid] = out_g
        if n % 250 == 0:
            out.write_text(json.dumps(drafts))
            print(f"  drafts {n}/{len(labels)}  {time.time() - t0:.0f}s", flush=True)
    out.write_text(json.dumps(drafts))
    return drafts


# ──────────────────────────────────────────────────────────────
# Scoring against the approved (held-out) frames
# ──────────────────────────────────────────────────────────────

def score_points(pred: List[dict], gold: List[dict], box) -> Optional[List[float]]:
    """Per visible gold point: distance / gate size (box diagonal)."""
    diag = math.hypot(box[2] - box[0], box[3] - box[1]) or 1.0
    if not pred or len(pred) != len(gold):
        return None
    return [math.hypot(p["x"] - g["x"], p["y"] - g["y"]) / diag for p, g in zip(pred, gold) if g["vis"] > 0]


def summarize(errs: List[float], n_gates: int, n_missing: int) -> dict:
    e = np.array(errs) if errs else np.array([1.0])
    return {"gates": n_gates, "no_prediction": n_missing, "points": len(errs),
            "pck05": float((e <= 0.05).mean()), "pck10": float((e <= 0.10).mean()), "median": float(np.median(e))}


def evaluate(predict, cls: str, test_ids: set, labels) -> dict:
    """predict(entry_path, label, gate_index, gate) → keypoints or None."""
    errs, n, miss = [], 0, 0
    for f, d in labels:
        if entry_id(f) not in test_ids:
            continue
        for gi, g in enumerate(d["gates"]):
            if g["class_name"] != cls or len(g.get("keypoints") or []) != len(KPT_NAMES[cls]):
                continue
            n += 1
            s = score_points(predict(f, d, gi, g), g["keypoints"], (g["x1"], g["y1"], g["x2"], g["y2"]))
            if s is None:
                miss += 1
            else:
                errs += s
    return summarize(errs, n, miss)


# ──────────────────────────────────────────────────────────────
# Training sets (crops) and models
# ──────────────────────────────────────────────────────────────

def build_set(cls: str, items: List[Tuple[str, dict, List[dict], int]], out: Path, val_items=None):
    """items: (image path, gate dict, keypoints, repeat). Writes a YOLO pose
    dataset of crops (train, and val if given)."""
    if out.exists():
        shutil.rmtree(out)
    n_k = len(KPT_NAMES[cls])
    for split, its in (("train", items), ("val", val_items or items[: max(1, len(items) // 20)])):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        cache = {}
        for i, (img_path, g, kps, rep) in enumerate(its):
            img = cache.get(img_path)
            if img is None:
                img = cv2.imread(img_path)
                cache = {img_path: img}
            if img is None:
                continue
            H, W = img.shape[:2]
            cr = crop_box((W, H), (g["x1"], g["y1"], g["x2"], g["y2"]))
            x1, y1, x2, y2 = map(int, cr)
            c = img[y1:y2, x1:x2]
            if c.size == 0 or c.shape[0] < 8 or c.shape[1] < 8:
                continue
            ch, cw = c.shape[:2]
            # the gate's box inside the crop
            bx1, by1, bx2, by2 = g["x1"] - x1, g["y1"] - y1, g["x2"] - x1, g["y2"] - y1
            parts = [f"0 {((bx1 + bx2) / 2) / cw:.6f} {((by1 + by2) / 2) / ch:.6f} {(bx2 - bx1) / cw:.6f} {(by2 - by1) / ch:.6f}"]
            for k in kps:
                kx, ky, kv = k["x"] - x1, k["y"] - y1, k["vis"]
                if not (0 <= kx < cw and 0 <= ky < ch):
                    kx, ky, kv = 0.0, 0.0, 0
                parts.append(f"{min(max(kx / cw, 0), 1):.6f} {min(max(ky / ch, 0), 1):.6f} {kv}")
            for r in range(rep):
                name = f"{i:06d}_{r}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{name}.txt").write_text(" ".join(parts) + "\n")
    flip = {"square": [1, 0, 3, 2], "arch": [4, 3, 2, 1, 0], "flagpole": [0, 1, 2], "circle": [0, 5, 4, 3, 2, 1]}[cls]
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\n"
                                   f"names: [{cls}]\nkpt_shape: [{n_k}, 3]\nflip_idx: {flip}\n")
    return out


def train(data: Path, base: str, run: Path, epochs: int, lr0: float = 0.002, freeze: Optional[int] = None) -> Path:
    from ultralytics import YOLO
    import torch
    m = YOLO(base)
    m.add_callback("on_train_epoch_end", lambda _: torch.mps.empty_cache())
    kw = dict(data=str(data), epochs=epochs, imgsz=IMGSZ, device="mps", project=str(run.parent), name=run.name,
              exist_ok=True, lr0=lr0, batch=32, patience=max(8, epochs // 4), plots=False, verbose=False,
              mosaic=0.0, degrees=8.0, translate=0.08, scale=0.25, fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
              workers=4)
    if freeze:
        kw["freeze"] = freeze
    m.train(**kw)
    return run / "weights" / "best.pt"


class KptPredictor:
    """Runs a class's pose model(s) on the padded crop of a known gate box.
    weights: one model, or a list whose points are averaged (models that find
    nothing are left out). Square gates get their feet moved to the box
    bottom (fix_square_feet)."""

    def __init__(self, weights, fix_feet: bool = True):
        self.members = [_OneModel(w) for w in ([weights] if isinstance(weights, (str, Path)) else weights)]
        self.fix_feet = fix_feet

    def __call__(self, img: np.ndarray, g: dict) -> Optional[List[dict]]:
        outs = [o for o in (m(img, g) for m in self.members) if o]
        if not outs:
            return None
        kps = [{"x": float(np.mean([o[i]["x"] for o in outs])), "y": float(np.mean([o[i]["y"] for o in outs])),
                "vis": max(o[i]["vis"] for o in outs)} for i in range(len(outs[0]))]
        if self.fix_feet and g["class_name"] == "square":
            kps = fix_square_feet(g, kps, img.shape[0])
        return kps


class _OneModel:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.m = YOLO(str(weights))
        args = getattr(self.m, "ckpt", {}) or {}
        self.imgsz = int((args.get("train_args") or {}).get("imgsz") or IMGSZ)

    def __call__(self, img: np.ndarray, g: dict) -> Optional[List[dict]]:
        H, W = img.shape[:2]
        x1, y1, x2, y2 = map(int, crop_box((W, H), (g["x1"], g["y1"], g["x2"], g["y2"])))
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            return None
        r = self.m(c, imgsz=self.imgsz, verbose=False, device="mps", conf=0.01)[0]
        if r.keypoints is None or not len(r.boxes):
            return None
        # the detection that best matches the gate box
        gb = [g["x1"] - x1, g["y1"] - y1, g["x2"] - x1, g["y2"] - y1]
        j = max(range(len(r.boxes)), key=lambda i: _iou(r.boxes.xyxy[i].tolist(), gb) + 0.1 * float(r.boxes.conf[i]))
        xy, cf = r.keypoints.xy[j].tolist(), r.keypoints.conf[j].tolist()
        return [{"x": float(p[0]) + x1, "y": float(p[1]) + y1, "vis": 2 if c_ >= 0.5 else 1} for p, c_ in zip(xy, cf)]


# ──────────────────────────────────────────────────────────────
# Models in use, applying them, retraining
# ──────────────────────────────────────────────────────────────

MANIFEST = training_dir() / "gate_models" / "manifest.json"


def weights_exist(w) -> bool:
    """One model path, or a list of them (averaged)."""
    ws = [w] if isinstance(w, (str, Path)) else (w or [])
    return bool(ws) and all(Path(x).exists() for x in ws)


def weights_name(w) -> str:
    ws = [w] if isinstance(w, (str, Path)) else w
    return " + ".join(f"{Path(x).parts[-4]}/{Path(x).parts[-3]}" for x in ws)


def active_models() -> Dict[str, object]:
    """The keypoint model(s) in use per class (gate_annotator's manifest)."""
    try:
        d = json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}
    return {c: p for c, p in (d.get("keypoints") or {}).items() if p and weights_exist(p)}


def set_active(cls: str, weights, score: float):
    d = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    d.setdefault("keypoints", {})[cls] = str(weights) if isinstance(weights, (str, Path)) else [str(w) for w in weights]
    d.setdefault("best_kpt_maps", {})[cls] = float(score)       # here: share of points within 5% of the gate size
    MANIFEST.write_text(json.dumps(d, indent=2))


def backup_labels() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ANN / "label_backups" / f"labels_{ts}"
    dst.parent.mkdir(exist_ok=True)
    return Path(shutil.make_archive(str(dst), "zip", root_dir=str(LABELS)))


def apply(models: Optional[Dict[str, str]] = None, progress=None, skip_recent_s: float = 0.0) -> dict:
    """Write the models' keypoints into every frame whose keypoints aren't OK
    (kpt_approved false). OK frames and boxes are never touched; all labels
    are backed up first. Each gate records which model made its points."""
    models = models or active_models()
    preds = {c: KptPredictor(w) for c, w in models.items()}
    if not preds:
        return {"error": "no keypoint model in use"}
    backup = backup_labels()
    now = time.time()
    todo, recent = [], []
    for f, d in all_labels():
        if d.get("kpt_approved"):
            continue
        if skip_recent_s and now - f.stat().st_mtime < skip_recent_s:
            recent.append(entry_id(f))               # being edited right now: leave it
            continue
        todo.append((f, d))
    counts = defaultdict(int)
    for i, (f, d) in enumerate(todo):
        img = cv2.imread(d.get("image_path", ""))
        if img is None:
            counts["missing image"] += 1
            continue
        changed = False
        for g in d["gates"]:
            p = preds.get(g["class_name"])
            if p is None:
                continue
            kp = p(img, g)
            if kp and len(kp) == len(KPT_NAMES[g["class_name"]]):
                g["keypoints"] = kp
                g["kpt_source"] = weights_name(models[g["class_name"]])
                counts[g["class_name"]] += 1
                changed = True
        if changed:
            f.write_text(json.dumps(d, indent=2), encoding="utf-8")
        if progress and i % 50 == 0:
            progress(i, len(todo))
    return {"frames": len(todo), "gates": dict(counts), "backup": str(backup), "skipped_recent": recent}


# what the Oct 2026 tuning on ~1000 OK frames showed: each OK gate once per round
# (10× repeats only cost time), validate on held-out OK frames, the small model
# at 320 px; several models averaged beat any single one
RECIPE = {"gold_repeat": 1, "base": "yolo11s-pose.pt", "epochs": 60}


def _gold_items(cls, labels, ids, rep):
    return [(d["image_path"], g, g["keypoints"], rep) for f, d in labels
            if d.get("kpt_approved") and entry_id(f) in ids
            for g in d["gates"] if g["class_name"] == cls and len(g.get("keypoints") or []) == len(KPT_NAMES[cls])]


def _draft_items(cls, labels):
    p = WORK / "drafts.json"
    drafts = json.loads(p.read_text()) if p.exists() else {}
    sizes = {}

    def fixed(d, g, kps):
        if cls != "square":
            return kps
        if d["image_path"] not in sizes:
            im = cv2.imread(d["image_path"])
            sizes[d["image_path"]] = im.shape[0] if im is not None else 10 ** 6
        return fix_square_feet(g, kps, sizes[d["image_path"]])
    return [(d["image_path"], d["gates"][dr["gate"]], fixed(d, d["gates"][dr["gate"]], dr["keypoints"]), 1)
            for f, d in labels if not d.get("kpt_approved")
            for dr in drafts.get(entry_id(f), []) if dr["cls"] == cls and dr["keypoints"] and dr["gate"] < len(d["gates"])]


def _predict_fn(weights):
    p = KptPredictor(weights)
    cache = {}

    def pred(f, d, gi, g):
        if d["image_path"] not in cache:
            cache.clear()
            cache[d["image_path"]] = cv2.imread(d["image_path"])
        img = cache[d["image_path"]]
        return p(img, g) if img is not None else None
    return pred


def retrain(cls: str, start=None, recipe: Optional[dict] = None) -> dict:
    """Fine-tune the class's model(s) in use on every OK frame (except the
    held-out ones), then score the new and the current one(s) on the
    held-out OK frames. With no model in use yet: train one from the
    segmentation drafts first. Returns {"weights", "new_map", "old_map", ...};
    scores are the share of points within 5% of the gate's size."""
    rc = dict(RECIPE, **(recipe or {}))
    labels = all_labels()
    test = test_split(labels)
    tune = tune_split(labels, test)
    approved = {entry_id(f) for f, d in labels if d.get("kpt_approved")}
    gold = _gold_items(cls, labels, approved - test - tune, rc["gold_repeat"])
    val = _gold_items(cls, labels, tune, 1) or None        # picks the best round; never trained on
    current = start if start is not None else active_models().get(cls)
    members = ([current] if isinstance(current, (str, Path)) else list(current)) if current else []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not members:                                    # first model: from the drafts
        drafts = _draft_items(cls, labels)
        if not drafts:
            return {"error": f"no '{cls}' keypoints to train on"}
        d0 = build_set(cls, drafts, WORK / "sets" / f"{cls}_{ts}_drafts", val)
        members = [str(train(d0, str(training_dir() / rc["base"]), MODELS / cls / f"{ts}_drafts", epochs=rc["epochs"]))]
        shutil.rmtree(d0, ignore_errors=True)
    if not gold:
        new_w = members
    else:
        data = build_set(cls, gold, WORK / "sets" / f"{cls}_{ts}", val)
        new_w = [str(train(data, str(m), MODELS / cls / f"{ts}_{k}", epochs=rc["epochs"], lr0=0.001))
                 for k, m in enumerate(members)]
        shutil.rmtree(data, ignore_errors=True)
    new_w = new_w[0] if len(new_w) == 1 else new_w
    new = evaluate(_predict_fn(new_w), cls, test, labels)
    old = evaluate(_predict_fn(current), cls, test, labels) if current else None
    return {"weights": new_w, "new_map": new["pck05"], "old_map": old["pck05"] if old else 0.0,
            "new": new, "old": old, "trained_on_ok_gates": len(gold)}
