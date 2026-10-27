"""
Detector models for race day.

By default the detector runs as a Core ML copy of the .pt on the Neural Engine
(not the GPU): exported at exactly the input the analysis gives it (a 960×540
picture → 384×640 after YOLO's letterbox), fp16 (what the Neural Engine
computes in), and without built-in NMS, so YOLO's own filtering runs on its
output as it does for the .pt. On 2027Race1 (4 heats, 16 pilots) its laps and
lap check against LapRF were identical to the .pt on the GPU, with ~2% fewer
mid-lap gate passes, about twice as fast and without PyTorch's GPU memory
(~1.3 GB for a heat instead of ~3.2 GB). The copy is made once from the .pt
(≈30 s) and kept in <data_root>/training/<model>.ane384x640.mlpackage.

(An earlier copy — square 640 input with NMS built in — matched far worse; it
is not used.) Without Core ML (not macOS, no coremltools), the .pt runs on the
GPU (mps), as before.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from dataset_paths import training_dir

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DET_MODEL = REPO / "cyn_current-20261021_best.pt"     # the pass scorer is trained on its boxes


COREML_IMGSZ = [384, 640]      # a 960×540 picture after YOLO's letterbox (640 wide, padded to a multiple of 32)


def coreml_path(pt_path: Path) -> Path:
    return training_dir() / f"{Path(pt_path).stem}.ane{COREML_IMGSZ[0]}x{COREML_IMGSZ[1]}.mlpackage"


def ensure_coreml(pt_path: Path) -> Optional[Path]:
    """The Core ML copy of a .pt detector, exported on first use (≈30 s).
    None if Core ML isn't available on this machine."""
    out = coreml_path(pt_path)
    if out.exists():
        return out
    try:
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory() as tmp:       # export writes next to the .pt
            src = Path(tmp) / Path(pt_path).name
            shutil.copy2(pt_path, src)
            exported = Path(YOLO(str(src)).export(format="coreml", imgsz=COREML_IMGSZ, half=True, nms=False))
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(out))
        return out
    except Exception as e:                               # no coremltools, not macOS, …
        print(f"[models] Core ML export failed ({e}) — using the GPU only")
        return None


def worker_devices(pt_path: Path, n_workers: int = 1, use_coreml: bool = True) -> List[Tuple[str, Optional[str]]]:
    """(model path, device) per analysis worker: the Core ML copy on the Neural
    Engine (device None), or — without Core ML, or use_coreml off — the .pt on
    the GPU."""
    cml = ensure_coreml(pt_path) if use_coreml else None
    if cml is not None:
        return [(str(cml), None)] * n_workers
    import torch
    gpu = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return [(str(pt_path), gpu)] * n_workers
