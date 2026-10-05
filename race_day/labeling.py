"""
Gate IDs, laps and learning for race-day runs (fast: no video is read).

  label_run(tp, stem)        gate IDs + laps of one analysed run, with the
                             track's learned gates and timing
  learn(tp, stems, n_gates)  learn the track from analysed runs
                             (learn_track.py) and make it the track's memory
"""

import bisect
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import gate_decoder as gd
import learn_track as lt
from dataset_paths import TrackPaths


def _write_json(path: Path, data: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def is_learned(tp: TrackPaths) -> bool:
    return tp.gate_memory.exists() and (tp.learned_timing.exists() or any(tp.gt_dir.glob("*.gt.json"))
                                        if tp.gt_dir.exists() else tp.learned_timing.exists())


def label_run(tp: TrackPaths, stem: str) -> dict:
    """Write gate IDs and laps into runs/<stem>.race_data.json (the per-frame
    overlays too, for race_ui) and return a short summary."""
    path = tp.race_data(stem)
    race = json.loads(path.read_text(encoding="utf-8"))
    source = race.get("candidate_passes") or race.get("passes", [])
    race["candidate_passes"] = source                    # the analysis output, kept for relabelling
    passes, laps, info = gd.label_race(source, tp.gate_memory, tp)
    frames = race.get("frames", [])
    ts = [e["t"] for e in frames]
    for e in frames:
        e.pop("passes", None)
        e.pop("laps", None)

    def at(t):
        i = min(max(bisect.bisect_left(ts, t), 0), len(ts) - 1)
        if i > 0 and abs(ts[i - 1] - t) < abs(ts[i] - t):
            i -= 1
        return frames[i]
    if frames:
        for p in passes:
            at(p["t"]).setdefault("passes", []).append(p)
        for lp in laps:
            at(lp["t1"]).setdefault("laps", []).append({"lap": lp["lap"], "t": lp["t1"]})
    race.update(passes=passes, laps=laps, gate_memory=str(tp.gate_memory), **info)
    _write_json(path, race)
    return {"passes": sum(p["gate_id"] >= 1 for p in passes), "laps": len(laps)}


def learn(tp: TrackPaths, stems: List[str], n_gates: int, verbose: bool = False) -> dict:
    """Learn the track's gates and timing from analysed runs and install it
    as the track's gate_memory.json."""
    videos, raw = [], {}
    for s in stems:
        race = json.loads(tp.race_data(s).read_text(encoding="utf-8"))
        passes = sorted(race.get("candidate_passes") or race.get("passes", []), key=lambda p: p["t"])
        raw[s] = passes
        videos.append(gd.VideoSeq(s, [gd.Obs(p["t"], gd._norm(p.get("query_embedding")),
                                              str(p.get("gate_type", "")), p.get("p_pass")) for p in passes]))
    m, labels = lt.learn(videos, n_gates, gd.Cfg(start_g1=10.0), verbose=verbose)
    lt.save(tp, videos, raw, labels, m, n_gates)
    shutil.copy2(tp.learned_memory, tp.gate_memory)
    laps = [len(gd.build_laps([{"gate_id": g, "t": o.t} for o, g in zip(v.obs, labels[v.stem]) if g > 0], n_gates))
            for v in videos]
    return {"runs": len(stems), "laps": sum(laps),
            "legs_s": [round(float(gd.np.exp(m.log_mu[g])), 3) for g in range(1, n_gates + 1)]}
