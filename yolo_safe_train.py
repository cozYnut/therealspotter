#!/usr/bin/env python3
"""
Train a YOLO model without letting memory run away.

Long Ultralytics runs on Apple Silicon kept growing (to ~80 GB with swap):
the MPS cache was never limited, and 8 data-loader processes each held their
own image buffers. Here training runs in its own process with:

  - a hard cap on GPU (MPS) memory: PYTORCH_MPS_HIGH_WATERMARK_RATIO — the
    run fails cleanly instead of swapping
  - 2 data-loader workers (not 8), no image cache in RAM
  - gc + MPS cache emptied after every round and every validation
  - a watchdog in this (parent) process: the training process and its
    workers are measured every few seconds (RAM + GPU memory); over the
    limit (default 14 GB) the training is stopped
  - memory.csv in the run folder: memory after every round, to see any growth

    from yolo_safe_train import safe_train
    res = safe_train(dict(model="current_best_non_vocab.pt", data=".../data.yaml", epochs=50,
                          imgsz=640, batch=8, project=..., name=...), mem_limit_gb=14)

    python yolo_safe_train.py --child args.json     (internal: the training process)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

MPS_WATERMARK = "0.45"      # share of the GPU's recommended working set (~16 GB on a 24 GB Mac) → ~7 GB
MPS_LOW_WATERMARK = "0.35"  # PyTorch requires low ≤ high (its default low, 1.4, is above our cap)
DEFAULTS = dict(workers=2, cache=False, device="mps", plots=False, verbose=True)


def _tree_rss(pid: int) -> int:
    import psutil
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total


def safe_train(args: dict, mem_limit_gb: float = 14.0, on_line: Optional[Callable[[str], None]] = None,
               poll_s: float = 3.0) -> dict:
    """Run model.train(**args) in a child process under the memory limit.
    Returns {"ok", "save_dir", "best", "peak_gb", "stopped"}; on_line gets
    the child's output lines (progress)."""
    args = dict(DEFAULTS, **args)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(args, f)
        argfile = f.name
    env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO=MPS_WATERMARK,
               PYTORCH_MPS_LOW_WATERMARK_RATIO=MPS_LOW_WATERMARK, PYTORCH_ENABLE_MPS_FALLBACK="1")
    p = subprocess.Popen([sys.executable, __file__, "--child", argfile], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
    state = {"peak": 0, "stopped": None, "mps": 0}
    mps_file = Path(argfile + ".mps")

    def watch():
        while p.poll() is None:
            rss = _tree_rss(p.pid)
            try:
                state["mps"] = int(mps_file.read_text() or 0)
            except (OSError, ValueError):
                pass
            used = rss + state["mps"]
            state["peak"] = max(state["peak"], used)
            if used > mem_limit_gb * 1024 ** 3 and not state["stopped"]:
                state["stopped"] = f"memory {used / 1024 ** 3:.1f} GB > limit {mem_limit_gb} GB"
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except OSError:
                    pass
            time.sleep(poll_s)
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    save_dir = None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("SAFE_TRAIN_SAVE_DIR "):
            save_dir = line.split(" ", 1)[1]
        if on_line:
            on_line(line)
    p.wait()
    t.join(timeout=poll_s * 2)
    for f_ in (argfile, str(mps_file)):
        try:
            os.unlink(f_)
        except OSError:
            pass
    best = Path(save_dir) / "weights" / "best.pt" if save_dir else None
    return {"ok": p.returncode == 0 and not state["stopped"], "returncode": p.returncode, "save_dir": save_dir,
            "best": str(best) if best and best.exists() else None, "peak_gb": round(state["peak"] / 1024 ** 3, 2),
            "stopped": state["stopped"]}


def _child(argfile: str):
    import gc
    import torch
    from ultralytics import YOLO
    import psutil
    args = json.loads(Path(argfile).read_text())
    mps_file = Path(argfile + ".mps")
    model = YOLO(args.pop("model"))
    proc = psutil.Process()
    log = {"path": None}

    def free(_=None):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def note(trainer, where):
        free()
        mps = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
        mps_file.write_text(str(mps))
        if log["path"] is None:
            log["path"] = Path(trainer.save_dir) / "memory.csv"
            log["path"].write_text("time,epoch,where,ram_gb,gpu_gb\n")
            print("SAFE_TRAIN_SAVE_DIR", trainer.save_dir, flush=True)
        rss = sum(q.memory_info().rss for q in [proc] + proc.children(recursive=True))
        with open(log["path"], "a") as f:
            f.write(f"{time.time():.0f},{getattr(trainer, 'epoch', -1) + 1},{where},{rss / 1024 ** 3:.2f},{mps / 1024 ** 3:.2f}\n")

    model.add_callback("on_train_start", lambda tr: note(tr, "start"))
    model.add_callback("on_train_epoch_end", lambda tr: note(tr, "train"))
    model.add_callback("on_fit_epoch_end", lambda tr: note(tr, "val"))

    def batch(tr):
        if getattr(tr, "_sb", 0) % 50 == 0 and torch.backends.mps.is_available():
            mps_file.write_text(str(torch.mps.driver_allocated_memory()))
        tr._sb = getattr(tr, "_sb", 0) + 1
    model.add_callback("on_train_batch_end", batch)
    model.train(**args)
    free()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child(sys.argv[2])
    else:
        print(__doc__)
