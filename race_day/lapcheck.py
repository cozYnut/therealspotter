"""
Marshal assist: a pilot's laps from the timing system (LapRF) against the
laps seen in the video.

A lap boundary in the video is where the pilot crosses the start/finish
gate. It's found from the gate sequence, not only from G1 itself (the
detector misses some passes): a G1 pass gives the exact moment; G13 → G2
with G1 not seen means the line was crossed somewhere between those two
passes. Each boundary is a time window [a, b].

Only disagreements are reported, each with a confidence:

  missed    the video shows the pilot cross the start/finish where the
            timing system has no detection: G1 seen, a whole lap (most gates)
            in a believable time, and the pilot flies on afterwards → a lap
            to add
  false     a timing detection with an impossible lap time and no
            start/finish crossing near it in the video → a lap to remove
  unsure    anything else that doesn't match: the video can't tell for
            sure (a missed or mislabelled gate, a crash at the line, a bad
            picture, a DJI feed…) — for a person to look at

Only "missed" and "false" are high confidence, and those are the only ones
Auto mode writes. On 2027Race1 every looser rule (e.g. "the video shows
the pilot mid-lap") produced wrong corrections, so the rules stay strict.

Times: video passes are converted to the clock with the heat's frame times,
plus a fixed offset (the timing antenna isn't exactly at the visual gate),
measured as the median difference over matched laps.
"""

from statistics import median
from typing import List, Optional

TOL_S = 1.0           # slack around a boundary window when matching a detection
PAIR_S = 2.5          # wider window used only to measure the offset
MAX_GAP_S = 6.0       # an inferred boundary (G1 not seen) wider than this isn't trusted
MIN_GATES = 0.6       # a lap in the video needs this share of the gates
SHORT = 0.5           # a lap shorter than this × the usual lap is impossible
SAME_G1_S = 2.0       # two G1 passes this close are the same crossing


def g1_passes(race: dict) -> List[dict]:
    return sorted((p for p in race.get("passes", []) if p.get("gate_id") == 1), key=lambda p: p["t"])


def pairs_offset(det_times: List[float], video_clock: List[float]) -> List[float]:
    """Detection − video differences of the closest pairs (for the offset)."""
    out = []
    for d in det_times:
        if video_clock:
            v = min(video_clock, key=lambda x: abs(x - d))
            if abs(d - v) <= PAIR_S:
                out.append(d - v)
    return out


def boundaries(passes: List[dict], n_gates: int) -> List[dict]:
    """Start/finish crossings in the video: {"a", "b" (video s), "exact",
    "gates": distinct gates seen since the previous crossing}."""
    ps = sorted((p for p in passes if p.get("gate_id", -1) >= 1), key=lambda p: p["t"])
    out, seen = [], set()
    prev = None
    for p in ps:
        g = p["gate_id"]
        if g == 1:
            if out and out[-1]["exact"] and p["t"] - out[-1]["b"] < SAME_G1_S:
                prev = p
                continue                                  # the same crossing seen twice
            out.append({"a": p["t"], "b": p["t"], "exact": True, "gates": len(seen)})
            seen = set()
        elif prev is not None and prev["gate_id"] != 1 and g < prev["gate_id"]:
            # the sequence wrapped (… G13, G14 → G2 …): the line was crossed in between
            out.append({"a": prev["t"], "b": p["t"], "exact": False, "gates": len(seen)})
            seen = set()
        seen.add(g)
        prev = p
    return out


def _mid_lap(passes: List[dict], t: float, n_gates: int, win: float = 2.0) -> bool:
    """The pilot is clearly in the middle of the lap at video time t: passes
    of middle gates (not near the start/finish) just before and just after."""
    lo, hi = 3, n_gates - 1
    before = [p for p in passes if t - win <= p["t"] < t and lo <= p.get("gate_id", -1) <= hi]
    after = [p for p in passes if t < p["t"] <= t + win and lo <= p.get("gate_id", -1) <= hi]
    return bool(before and after)


NO_VIDEO_SHARE = 0.8   # no picture for this share of the race → the run can't be checked ("no video");
                       # a run with less is checked where it has a picture


def _share(intervals, a: float, b: float) -> float:
    if b <= a:
        return 0.0
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in intervals) / (b - a)


def check_pilot(race: dict, dets: List[dict], to_clock, n_gates: int, offset: float,
                start: float, end: float, lap_ref: Optional[float], low_quality: bool = False,
                gray: Optional[List[List[float]]] = None, race_video: Optional[tuple] = None) -> dict:
    """race: the pilot's labelled race_data; dets: the timing system's valid
    detections for this channel ({"t", "holeshot", "id"}); to_clock: video
    seconds → clock seconds. gray: video intervals with no picture (a flat
    gray feed); race_video: the race's start and end in video seconds.
    Returns laps from both sides, the findings, and `confirmed`: the timing
    crossings the video showed (a crossing neither confirmed nor in a finding
    wasn't checked — e.g. while there was no picture). With no picture for most
    of the race the run is "no video" and nothing is reported; otherwise
    nothing is reported while the picture is missing."""
    gray = gray or []
    if race_video and _share(gray, *race_video) >= NO_VIDEO_SHARE:
        return {"timing_laps": max(0, len(dets) - 1), "video_laps": 0, "matched": 0, "findings": [],
                "no_video": True, "confirmed": []}
    passes = [p for p in race.get("passes", []) if p.get("gate_id", -1) >= 1]
    lo, hi = start - 2.0, end + 30.0
    bnd = []
    for b in boundaries(passes, n_gates):
        ca, cb = float(to_clock(b["a"])) + offset, float(to_clock(b["b"])) + offset
        if lo <= cb and ca <= hi and (b["exact"] or cb - ca <= MAX_GAP_S):
            bnd.append(dict(b, ca=ca, cb=cb, video_t=b["b"] if b["exact"] else (b["a"] + b["b"]) / 2))
    det = [d for d in dets if lo <= d["t"] <= hi]

    # match detections to boundary windows (closest first, one to one)
    def dist(d, b):
        return 0.0 if b["ca"] <= d["t"] <= b["cb"] else min(abs(d["t"] - b["ca"]), abs(d["t"] - b["cb"]))
    cand = sorted((dist(d, b), i, j) for i, d in enumerate(det) for j, b in enumerate(bnd) if dist(d, b) <= TOL_S)
    md, mb = {}, {}
    for _, i, j in cand:
        if i not in md and j not in mb:
            md[i], mb[j] = j, i

    det_times = [d["t"] for d in det]
    laps_det = [b - a for a, b in zip(det_times, det_times[1:])]
    ref = lap_ref or (median(laps_det) if laps_det else None)
    findings = []
    clock0 = float(to_clock(0.0))

    def video_time(clock_t):                     # video time of a clock time (for jumping there)
        return clock_t - offset - clock0

    # timing detections the video doesn't confirm
    for i, d in enumerate(det):
        if i in md:
            continue
        prev = det[i - 1]["t"] if i > 0 else None
        lap = d["t"] - prev if prev is not None else None
        vt = video_time(d["t"])
        f = {"kind": "false", "t": d["t"], "detection_id": d["id"], "lap_time": lap,
             "holeshot": d["holeshot"], "video_t": vt}
        near_crossing = any(b["ca"] - 3.0 <= d["t"] <= b["cb"] + 3.0 for b in bnd)
        if lap is not None and ref and lap < SHORT * ref and not near_crossing:
            f.update(confidence="high", reason=f"lap of {lap:.1f}s is impossible (usual {ref:.1f}s) "
                                                "and the video shows no start/finish crossing")
        elif _mid_lap(passes, vt, n_gates):
            f.update(kind="unsure", confidence="low",
                     reason="the video seems to show the pilot mid-lap here — check it")
        else:
            f.update(kind="unsure", confidence="low",
                     reason="timing lap not confirmed by the video — check it")
        findings.append(f)

    # video lap boundaries the timing system doesn't have
    for j, b in enumerate(bnd):
        if j in mb:
            continue
        prev_t = max([x["t"] for x in det if x["t"] < b["ca"]] +
                     [x["cb"] for x in bnd[:j]], default=None)
        lap = b["cb"] - prev_t if prev_t is not None else None
        holeshot = not det or b["cb"] < det[0]["t"] - TOL_S
        f = {"kind": "missed", "t": b["cb"] if b["exact"] else (b["ca"] + b["cb"]) / 2, "video_t": b["video_t"],
             "lap_time": lap, "holeshot": holeshot, "gates_seen": b["gates"], "exact": b["exact"]}
        good = (b["exact"] and j > 0 and b["gates"] >= MIN_GATES * n_gates and ref and lap is not None
                and 0.75 * ref <= lap <= 1.6 * ref)
        flies_on = len({p["gate_id"] for p in passes
                        if b["b"] < p["t"] <= b["b"] + 10.0 and p["gate_id"] != 1}) >= 2
        if good and flies_on and not low_quality:
            f.update(confidence="high", reason=f"lap completed in the video ({b['gates']}/{n_gates} gates, "
                                                f"{lap:.1f}s) and the pilot flies on, but not in the timing")
        elif good:
            f.update(kind="unsure", confidence="low", reason="a lap completed at the end of the flight that the "
                     "timing doesn't have (a crash at the line?) — check it")
        else:
            f.update(kind="unsure", confidence="low", reason="a start/finish crossing in the video that the "
                     "timing doesn't have — check it")
        findings.append(f)

    blind = [f for f in findings if f.get("video_t") is not None
             and any(a - 1.0 <= f["video_t"] <= b + 1.0 for a, b in gray)]
    findings = [f for f in findings if f not in blind]
    findings.sort(key=lambda f: f["t"])
    return {"timing_laps": max(0, len(det) - 1), "video_laps": max(0, len(bnd) - 1),
            "matched": len(md), "findings": findings, "no_video_findings": len(blind),
            "confirmed": [det[i]["id"] for i in md]}        # the timing crossings the video showed


def heat_offset(per_pilot_diffs: List[float]) -> Optional[float]:
    return round(median(per_pilot_diffs), 3) if len(per_pilot_diffs) >= 3 else None
