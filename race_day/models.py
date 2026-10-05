"""
Detector models for race day.

Analysis workers run the .pt detector on the GPU (mps): results are identical
to the CPU (checked on reviewed videos with score.py) and 3–4× faster.

A Core ML copy (Neural Engine) is even faster, but on the reviewed videos it
made 10–24 errors where the .pt made 7 (tried fp16/fp32, with and without
built-in NMS, square and 384×640 input), so it's off by default
(DayConfig.use_coreml). When on, the copy is made once from the .pt and kept
in <data_root>/training/<model>.mlpackage.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from dataset_paths import training_dir

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DET_MODEL = REPO / "cyn_current-20260715_best.pt"


def coreml_path(pt_path: Path) -> Path:
    return training_dir() / f"{Path(pt_path).stem}.mlpackage"


def ensure_coreml(pt_path: Path) -> Optional[Path]:
    """The Core ML copy of a .pt detector, exported on first use (≈10 s).
    None if Core ML isn't available on this machine."""
    out = coreml_path(pt_path)
    if out.exists():
        return out
    try:
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory() as tmp:       # export writes next to the .pt
            src = Path(tmp) / Path(pt_path).name
            shutil.copy2(pt_path, src)
            exported = Path(YOLO(str(src)).export(format="coreml", imgsz=640, half=True, nms=True))
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(out))
        return out
    except Exception as e:                               # no coremltools, not macOS, …
        print(f"[models] Core ML export failed ({e}) — using the GPU only")
        return None


def worker_devices(pt_path: Path, n_workers: int = 4, use_coreml: bool = False) -> List[Tuple[str, Optional[str]]]:
    """(model path, device) per analysis worker: all on the GPU, or with
    use_coreml half on the Neural Engine (faster, but less accurate so far)."""
    import torch
    gpu = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cml = ensure_coreml(pt_path) if use_coreml else None
    out = []
    for i in range(n_workers):
        if cml is not None and i % 2 == 0:
            out.append((str(cml), None))
        else:
            out.append((str(pt_path), gpu))
    return out
