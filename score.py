#!/usr/bin/env python3
"""
Score the current system against reviewed ground truth (Milestone 1).

For every track in the dataset (see dataset_paths.py) this compares
runs/<video>.race_data.json with gt/<video>.gt.json and prints:

  Pass detection   a detector pass within ±tol s (0.15) of a real pass is
                   on time; within ±off-tol (0.5) it is off-time (fired for
                   the gate, wrong moment) — not a miss plus a false pass;
                   recall, precision and timing error
  Gate ID          on every real pass the detector fired for: share given
                   a gate, and share correct
                   (videos in memory_videos/ are left out — the memory
                   was learned from them)
  Laps             completed laps found vs marked
  Missed gates     gates the pilot skipped (the system can't flag them yet)
  Per tag          pass recall and gate-ID accuracy for crash/clipped and
                   bad-video marks; "unsure" marks are left out of all scores

Real tracks and their velocidrone/ versions are totalled separately.
Only gt files marked "complete" are scored (--include-incomplete for all).

Pass detection doesn't depend on the gate memory, but gate ID does.  The
saved runs may be older than the track's gate_memory.json, so --rematch
replays the gate matching (GateDB, same settings as extract_race.py) on the
saved CLIP embeddings with the current memory.  No YOLO run is needed.

Usage:
    python score.py                       # all tracks, saved runs
    python score.py track1 --rematch      # one track, current memory
    python score.py --list                # also list every miss / false pass / wrong gate
    python score.py --save baseline       # also write <data_root>/scores/<date>_baseline.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from dataset_paths import SIM_DIR, TrackPaths, data_root, list_tracks

TAGS = ("crash_clipped", "bad_video")


# ──────────────────────────────────────────────────────────────
# Gate matching replay
# ──────────────────────────────────────────────────────────────

def rematch(passes: List[dict], memory_path: Path, args):
    """Re-run GateDB race matching on saved pass embeddings, in time order.
    Returns (copies of the passes with gate_id / source / sim replaced, laps)."""
    from gate_db import replay_race
    return replay_race(passes, str(memory_path), sim_thresh=args.sim_thresh, min_match_margin=args.min_margin,
                       g1_sim_thresh=args.g1_sim_thresh, g1_margin=args.g1_margin,
                       require_same_type=args.require_same_type)


def clip_top1(passes: List[dict], memory_path: Path) -> dict:
    """Index → most similar gate over the whole memory (no order, no threshold)."""
    mem = json.loads(memory_path.read_text(encoding="utf-8")).get("memory", [])
    mem = sorted(mem, key=lambda g: int(g.get("order_idx", 0)))
    banks = []
    for g in mem:
        e = np.asarray(g.get("embeds") or [], dtype=np.float32)
        banks.append(e / np.linalg.norm(e, axis=1, keepdims=True) if len(e) else None)
    best = {}
    for i, p in enumerate(passes):
        emb = p.get("query_embedding")
        if not emb:
            continue
        q = np.asarray(emb, dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = [float((b @ q).max()) if b is not None else -1.0 for b in banks]
        best[i] = int(np.argmax(sims)) + 1
    return best


# ──────────────────────────────────────────────────────────────
# One video
# ──────────────────────────────────────────────────────────────

def match_by_time(gt_t: List[float], pr_t: List[float], tol: float):
    """One-to-one pairs (gt_idx, pred_idx), closest first, within ±tol."""
    pairs = sorted(
        (abs(g - p), gi, pi)
        for gi, g in enumerate(gt_t) for pi, p in enumerate(pr_t) if abs(g - p) <= tol
    )
    used_g, used_p, out = set(), set(), []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi))
    return out


def score_video(tp: TrackPaths, gt_path: Path, args, passes: Optional[List[dict]] = None) -> Optional[dict]:
    """Score one video. `passes` replaces the saved race_data passes (e.g. pass_scorer output)."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    stem = gt_path.name[: -len(".gt.json")]
    if gt.get("status") != "complete" and not args.include_incomplete:
        return {"video": stem, "skipped": f"status {gt.get('status')}"}
    race_path = tp.race_data(stem)
    if not race_path.exists():
        return {"video": stem, "skipped": "no race_data"}
    race = json.loads(race_path.read_text(encoding="utf-8"))
    sys_laps = race.get("laps", []) if passes is None else []
    if passes is None:
        passes = race.get("passes", [])
    if args.rematch:
        passes, sys_laps = rematch(passes, tp.gate_memory, args)
    top1 = clip_top1(passes, tp.gate_memory) if args.rematch else {}

    marks = gt.get("marks", [])
    # "unsure" marks and marks still pending review are left out of every score
    unsure = [m for m in marks if m.get("tag") == "unsure" or m.get("verdict") == "pending"]
    real = [m for m in marks if m["kind"] == "pass" and m not in unsure]
    skipped = [m for m in marks if m["kind"] == "skipped"]
    unsure_t = [m["t"] for m in unsure]
    # detector passes on an "unsure" mark or a tagged-unsure false pass count for nothing
    ignore_t = unsure_t + [fp["t"] for fp in gt.get("false_passes", []) if fp.get("tag") == "unsure"]
    preds = [(i, p) for i, p in enumerate(passes)
             if not any(abs(p["t"] - t) <= args.tol for t in ignore_t)]

    pairs = match_by_time([m["t"] for m in real], [p["t"] for _, p in preds], args.tol)
    # second round: the detector fired for this gate, but outside ±tol
    left_g = [gi for gi in range(len(real)) if gi not in {g for g, _ in pairs}]
    left_p = [pi for pi in range(len(preds)) if pi not in {p for _, p in pairs}]
    late = [(left_g[a], left_p[b]) for a, b in match_by_time(
        [real[gi]["t"] for gi in left_g], [preds[pi][1]["t"] for pi in left_p], args.off_tol)]
    hit_g = {gi: preds[pi] for gi, pi in pairs + late}
    hit_p = {pi for _, pi in pairs + late}

    # where the video is now (it may have been moved after review)
    memory_video = any(f.stem == stem for f in tp.memory_videos.glob("*")) if tp.memory_videos.exists() else False
    v = {
        "video": stem, "memory_video": memory_video,
        "real": len(real), "found": len(pairs), "off_time": [
            (real[gi]["t"], preds[pi][1]["t"] - real[gi]["t"]) for gi, pi in late],
        "missed": [m["t"] for gi, m in enumerate(real) if gi not in hit_g],
        "false": [p["t"] for pi, (_, p) in enumerate(preds) if pi not in hit_p],
        "false_reasons": [p.get("reason", "") for pi, (_, p) in enumerate(preds) if pi not in hit_p],
        "time_err": [abs(real[gi]["t"] - p["t"]) for gi, (_, p) in hit_g.items()],
        "assigned": 0, "correct": 0, "clip_top1": 0, "wrong": [],
        "laps_marked": max(0, sum(1 for m in marks if m["kind"] == "pass" and m["gate_id"] == 1) - 1),
        "laps_found": len(sys_laps),
        "skipped_marks": len(skipped),
        "unsure": len(unsure),
        "tags": {t: {"real": 0, "found": 0, "correct": 0} for t in TAGS},
    }
    # gate ID is judged on every real pass the detector fired for, on time or not
    v["fired"] = len(hit_g)
    for gi, (pidx, p) in hit_g.items():
        m = real[gi]
        if p.get("gate_id", -1) >= 1:
            v["assigned"] += 1
        if p.get("gate_id") == m["gate_id"]:
            v["correct"] += 1
        else:
            v["wrong"].append((m["t"], m["gate_id"], p.get("gate_id", -1), p.get("source", "")))
        if top1.get(pidx) == m["gate_id"]:
            v["clip_top1"] += 1
    for gi, m in enumerate(real):
        if m.get("tag") in TAGS:
            tg = v["tags"][m["tag"]]
            tg["real"] += 1
            if gi in hit_g:
                tg["found"] += 1
                if hit_g[gi][1].get("gate_id") == m["gate_id"]:
                    tg["correct"] += 1
    return v


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100.0 * a / b:.0f}%" if b else "–"


def ms(xs: List[float], q: float) -> str:
    if not xs:
        return "–"
    xs = sorted(xs)
    return f"{1000 * xs[min(len(xs) - 1, int(q * len(xs)))]:.0f} ms"


def total(vs: List[dict]) -> dict:
    t = {"videos": len(vs), "real": 0, "found": 0, "off_time": 0, "missed": 0, "false": 0, "time_err": [],
         "id_found": 0, "assigned": 0, "correct": 0, "clip_top1": 0,
         "laps_marked": 0, "laps_found": 0, "skipped_marks": 0, "unsure": 0,
         "tags": {k: {"real": 0, "found": 0, "correct": 0} for k in TAGS}}
    for v in vs:
        t["real"] += v["real"]
        t["found"] += v["found"]
        t["off_time"] += len(v["off_time"])
        t["missed"] += len(v["missed"])
        t["false"] += len(v["false"])
        t["time_err"] += v["time_err"]
        t["laps_marked"] += v["laps_marked"]
        t["laps_found"] += v["laps_found"]
        t["skipped_marks"] += v["skipped_marks"]
        t["unsure"] += v["unsure"]
        for k in TAGS:
            for f in ("real", "found", "correct"):
                t["tags"][k][f] += v["tags"][k][f]
        if not v["memory_video"]:
            t["id_found"] += v["fired"]
            t["assigned"] += v["assigned"]
            t["correct"] += v["correct"]
            t["clip_top1"] += v["clip_top1"]
    return t


def print_video(v: dict, args):
    if "skipped" in v:
        print(f"  {v['video']:<22} skipped ({v['skipped']})")
        return
    mem = " (memory)" if v["memory_video"] else ""
    print(f"  {v['video'] + mem:<22} passes {v['found']:3}/{v['real']:<3} off-time {len(v['off_time']):2}  "
          f"missed {len(v['missed']):2}  false {len(v['false']):2}  |  gate ID {v['correct']:3}/{v['fired']:<3} "
          f"(given {v['assigned']:3})  |  laps {v['laps_found']}/{v['laps_marked']}  "
          f"|  time p90 {ms(v['time_err'], 0.9)}")
    if args.list:
        for t in v["missed"]:
            print(f"      missed pass    {t:8.3f}s")
        for t, dt in v["off_time"]:
            print(f"      off-time pass  {t:8.3f}s  detector {'late' if dt > 0 else 'early'} by {abs(dt):.3f}s")
        for t, r in zip(v["false"], v["false_reasons"]):
            print(f"      false pass     {t:8.3f}s  ({r})")
        for t, g, p, src in v["wrong"]:
            print(f"      gate ID        {t:8.3f}s  real G{g}  →  {'G' + str(p) if p >= 1 else src}")


def print_total(name: str, t: dict, args):
    print(f"\n  {name} — {t['videos']} video(s)")
    fired = t["found"] + t["off_time"]
    print(f"    Pass recall      {pct(t['found'], t['real'])} on time (±{args.tol:.2f}s)   "
          f"{pct(fired, t['real'])} incl. {t['off_time']} off-time (±{args.off_tol:.2f}s)   missed {t['missed']}")
    print(f"    Pass precision   {pct(fired, fired + t['false'])}      (false {t['false']})")
    print(f"    Pass timing      median {ms(t['time_err'], 0.5)}, p90 {ms(t['time_err'], 0.9)}  (all fired passes)")
    print(f"    Gate ID          correct {pct(t['correct'], t['id_found'])}   given a gate {pct(t['assigned'], t['id_found'])}"
          f"   correct when given {pct(t['correct'], t['assigned'])}   (memory videos left out)")
    if args.rematch:
        print(f"    CLIP top-1       {pct(t['clip_top1'], t['id_found'])}   (most similar gate, no order, no threshold)")
    if getattr(args, "passes", "runs") == "scorer" and not args.rematch:
        print(f"    Laps             marked {t['laps_marked']} (add --rematch to rebuild laps from the new passes)")
    else:
        print(f"    Laps             found {t['laps_found']} / marked {t['laps_marked']}")
    print(f"    Missed gates     {t['skipped_marks']} marked — the current system can't flag them yet")
    for k in TAGS:
        g = t["tags"][k]
        if g["real"]:
            print(f"    Tag {k:<13} recall {pct(g['found'], g['real'])}   gate ID {pct(g['correct'], g['found'])}")
    if t["unsure"]:
        print(f"    Unsure           {t['unsure']} mark(s) left out")


def main():
    ap = argparse.ArgumentParser(description="Score race_data against reviewed gt.json for every track.")
    ap.add_argument("tracks", nargs="*", help="Track folder names (default: all)")
    ap.add_argument("--tol", type=float, default=0.15, help="Max time gap for an on-time detector pass (s)")
    ap.add_argument("--off-tol", type=float, default=0.5,
                    help="Wider gap: a detector pass this close is 'off-time', not a miss + false pass (s)")
    ap.add_argument("--rematch", action="store_true",
                    help="Redo gate matching on saved embeddings with the current gate_memory.json")
    ap.add_argument("--sim-thresh", type=float, default=0.88)
    ap.add_argument("--min-margin", type=float, default=0.03)
    ap.add_argument("--g1-sim-thresh", type=float, default=None)
    ap.add_argument("--g1-margin", type=float, default=None)
    ap.add_argument("--require-same-type", action="store_true")
    ap.add_argument("--include-incomplete", action="store_true", help="Also score gt files still in progress")
    ap.add_argument("--passes", choices=["runs", "live", "scorer"], default="runs",
                    help="runs: the passes saved in race_data (what race_ui / Review show); "
                         "live: the live PassDetector's passes kept in race_data; "
                         "scorer: pass_scorer.py decisions, each video by a model trained on the other videos")
    ap.add_argument("--scorer-logic", default=None, help="pass_scorer logic (default: its BEST_LOGIC)")
    ap.add_argument("--list", action="store_true", help="List every missed pass, false pass and wrong gate")
    ap.add_argument("--save", metavar="LABEL", help="Write results to <data_root>/scores/<date>_<LABEL>.json")
    args = ap.parse_args()

    tracks = [tp for tp in list_tracks() if not args.tracks or tp.dir.name in args.tracks]
    if args.tracks:
        missing = set(args.tracks) - {tp.dir.name for tp in tracks}
        if missing:
            ap.error(f"no such track(s) in {data_root()}: {', '.join(sorted(missing))}")
    units = []
    for tp in tracks:
        units.append(tp)
        if (tp.dir / SIM_DIR).is_dir():
            units.append(TrackPaths(tp.dir / SIM_DIR))

    print(f"Data root: {data_root()}")
    print(f"Gate matching: {'replayed with current gate memory' if args.rematch else 'as saved in runs/'}")
    print("Passes: " + {"runs": "as saved in runs/ (pass_logic of each run)",
                        "live": "live PassDetector (live_passes in runs/)",
                        "scorer": "pass_scorer, leave-one-video-out"}[args.passes])
    results = {"real": [], "sim": []}
    per_track = {}
    for tp in units:
        gts = sorted(tp.gt_dir.glob("*.gt.json")) if tp.gt_dir.exists() else []
        if not gts:
            continue
        if args.rematch and not tp.gate_memory.exists():
            print(f"\n{tp.name}: no gate_memory.json — skipped")
            continue
        print(f"\n{tp.name}")
        override = {}
        if args.passes == "live":
            for g in gts:
                stem = g.name[: -len(".gt.json")]
                if tp.race_data(stem).exists():
                    race = json.loads(tp.race_data(stem).read_text(encoding="utf-8"))
                    override[stem] = race.get("live_passes", race.get("passes", []))
        if args.passes == "scorer":
            import pass_scorer
            videos, cands = pass_scorer.prepare(tp, include_incomplete=args.include_incomplete)
            override = pass_scorer.lovo_passes(videos, cands, args.scorer_logic or pass_scorer.BEST_LOGIC)
        vs = [score_video(tp, g, args, passes=override.get(g.name[: -len(".gt.json")])) for g in gts]
        for v in vs:
            print_video(v, args)
        scored = [v for v in vs if "skipped" not in v]
        if scored:
            print_total(f"{tp.name} total", total(scored), args)
        per_track[tp.name] = vs
        results["sim" if tp.dir.name == SIM_DIR else "real"] += scored

    print("\n" + "═" * 70)
    for kind, label in (("real", "ALL REAL TRACKS"), ("sim", "ALL VELOCIDRONE TRACKS")):
        if results[kind]:
            print_total(label, total(results[kind]), args)
    if not results["real"] and not results["sim"]:
        print("\nNothing scored — review some test videos first (learn_ui → 🔍 Review Video).")

    if args.save:
        out_dir = data_root() / "scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{datetime.now():%Y%m%d_%H%M}_{args.save}.json"
        payload = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "totals": {k: {kk: vv for kk, vv in total(v).items() if kk != "time_err"}
                       for k, v in results.items() if v},
            "tracks": per_track,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
