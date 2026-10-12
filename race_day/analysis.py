"""
Analysis workers: one process per channel, so a heat's 4 runs are analysed
at the same time. Half the workers run the detector as Core ML (Neural
Engine), half as .pt on the GPU — measured fastest on an M5 Pro.

Each run goes through extract_race.run_race_extraction with a blank gate
memory: this finds the passes (pass scorer) and their gate images; gate IDs
and laps are added afterwards by labeling.py, once the track is learned.
"""

import contextlib
import json
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

_W = {}          # per worker process: model path, device, clip device


def _init(slots, clip_device: str):
    path, device = slots.get()
    _W.update(model=path, device=device, clip=clip_device)


def _blank_memory(n_gates: int) -> str:
    mem = {"version": 2, "mode": "blank", "race_lookahead": 3, "expected_idx": 0, "max_embeds_per_gate": 6,
           "memory": [{"order_idx": i, "gate_id": i + 1, "gate_type": "unknown", "embeds": [],
                       "created_t": 0.0, "last_img": "", "embed_imgs": []} for i in range(n_gates)]}
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(mem, f)
    return p


def _analyse(video: str, out_json: str, n_gates: int, log_path: str) -> dict:
    from extract_race import run_race_extraction
    t0 = time.time()
    mem = _blank_memory(n_gates)
    try:
        with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_race_extraction(video, _W["model"], mem, out_json, clip_device=_W["clip"],
                                gate_id_logic="greedy", det_device=_W["device"])
    finally:
        os.unlink(mem)
    return {"seconds": round(time.time() - t0, 1),
            "device": "coreml" if _W["model"].endswith(".mlpackage") else _W["device"]}


class AnalysisPool:
    def __init__(self, devices: List[Tuple[str, Optional[str]]], clip_device: str = "mps"):
        ctx = mp.get_context("spawn")
        self._mgr = ctx.Manager()
        slots = self._mgr.Queue()
        for d in devices:
            slots.put(d)
        self.pool = ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx,
                                        initializer=_init, initargs=(slots, clip_device))
        self.n = len(devices)

    def submit(self, video: Path, out_json: Path, n_gates: int) -> Future:
        log = Path(out_json).with_name(Path(out_json).name.replace(".race_data.json", ".analysis.log"))
        return self.pool.submit(_analyse, str(video), str(out_json), n_gates, str(log))

    def shutdown(self, wait: bool = True):
        self.pool.shutdown(wait=wait, cancel_futures=not wait)
        self._mgr.shutdown()
