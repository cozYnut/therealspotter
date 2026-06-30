# pass_detector.py
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np


def _center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _norm_type(s: str) -> str:
    return (s or "").strip().lower()


def _is_flag(t: str) -> bool:
    t = _norm_type(t)
    return ("flag" in t) or ("pole" in t)


def detect_camera_edges(frame: np.ndarray, black_thresh: int = 15) -> Tuple[float, float]:
    """
    Scan the first video frame column-by-column to find where real camera
    content starts and ends (handles 4:3-in-16:9 black bar letterboxing).
    Returns (left_norm, right_norm) in [0.0, 1.0].
    Falls back to (0.0, 1.0) if no black bars are detected.
    """
    gray = frame if frame.ndim == 2 else frame.mean(axis=2)
    W = gray.shape[1]
    col_means = gray.mean(axis=0)

    left = 0
    for i in range(W):
        if col_means[i] > black_thresh:
            left = i
            break

    right = W - 1
    for i in range(W - 1, -1, -1):
        if col_means[i] > black_thresh:
            right = i
            break

    return (float(left) / W, float(right) / W)


def _count_edges_near_real_frame(
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
    cam_left_norm: float, cam_right_norm: float,
    tol: float,
) -> int:
    """
    Count how many of the 4 bbox edges are within tol (as a fraction of frame
    dimensions) of the real camera boundary. Left/right edges account for black
    bars; top/bottom use the full frame height.
    """
    cam_left_px  = cam_left_norm  * frame_w
    cam_right_px = cam_right_norm * frame_w
    tol_px_x = tol * frame_w
    tol_px_y = tol * frame_h

    count = 0
    if x1 <= cam_left_px  + tol_px_x:  count += 1
    if x2 >= cam_right_px - tol_px_x:  count += 1
    if y1 <= tol_px_y:                  count += 1
    if y2 >= frame_h - tol_px_y:       count += 1
    return count


@dataclass
class TrackPassState:
    track_id: int
    ttype: str
    stage: str = "idle"  # idle -> aligned -> passed
    last_seen_time: float = 0.0

    # approach / alignment (pixel-space)
    peak_area: float = 0.0
    last_area: float = 0.0
    last_cx: float = 0.0
    last_cy: float = 0.0

    # flag heuristic: was the flag ever near frame center while close enough?
    flag_was_centered: bool = False

    # cooldown / pass bookkeeping
    passed_time: float = 0.0

    # ----------------------------
    # Debug values (per update)
    # ----------------------------
    last_nx: float = 0.0
    last_ny: float = 0.0
    last_center_dist: float = 0.0
    last_area_ratio: float = 0.0
    last_flag_span: float = 0.0

    last_score_ema: float = 0.0
    last_reason_attempted: str = ""
    last_pass_blocked_by: str = ""  # "", "global", "type", "track", "already_passed"
    last_updated_time: float = 0.0

    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # ----------------------------
    # NEW: growth-based stats
    # ----------------------------
    prev_area_ratio: float = 0.0
    area_vel: float = 0.0          # (area_ratio delta) / sec
    area_vel_ema: float = 0.0      # smoothed velocity
    aligned_area_ratio: float = 0.0
    aligned_time: float = 0.0
    aligned_frames: int = 0
    gate_area_peaked: bool = False
    gate_peaked_area_ratio: float = 0.0


class PassDetector:
    """
    Vision-only gate pass detector, based on tracked boxes.
    Works with your Track objects (needs bbox + locked_type + score_ema).

    Call:
      pd.update(tracks, now, frame_w, frame_h)

    Then:
      pd.pop_any_passed() -> event dict (track_id, type, time, reason)
      pd.get_debug(now) -> dict[track_id] of debug stats for visualization
    """

    def __init__(
        self,
        min_track_score: float = 0.2,
        min_area_ratio: float = 0.03,
        center_tol: float = 0.51,
        disappear_timeout: float = 0.09,

        # flagpole pass knobs
        flag_center_tol: float = 0.15,
        flag_edge_tol: float = 0.11,
        flag_min_edges: int = 2,

        # cooldown knobs
        pass_cooldown_sec: float = 0.09,
        type_cooldown_sec: float = 0.09,
        track_cooldown_sec: float = 0.2,

        # behavior switches
        ignore_flagpoles: bool = False,

        # growth-based alignment knobs
        area_vel_ema_alpha: float = 0.36,
        min_area_vel_ema: float = 0.015,
        aligned_shrink_reset_frac: float = 0.14,
        aligned_max_age_sec: float = 2,
        min_aligned_frames: int = 2,
        flag_aligned_shrink_reset_frac: float = 0.0,
        gate_pass_area_thresh: float = 0.3,
        peak_area_max_jump: float = 6.0,
    ):
        self.min_track_score = float(min_track_score)
        self.min_area_ratio = float(min_area_ratio)
        self.center_tol = float(center_tol)
        self.disappear_timeout = float(disappear_timeout)

        self.flag_center_tol = float(flag_center_tol)
        self.flag_edge_tol   = float(flag_edge_tol)
        self.flag_min_edges  = int(flag_min_edges)

        # camera edges — default full frame; call set_camera_edges() after auto-detection
        self.cam_left_norm:  float = 0.0
        self.cam_right_norm: float = 1.0
        self._frame_w: int = 1
        self._frame_h: int = 1

        self.pass_cooldown_sec = float(pass_cooldown_sec)
        self.type_cooldown_sec = float(type_cooldown_sec)
        self.track_cooldown_sec = float(track_cooldown_sec)

        self.ignore_flagpoles = bool(ignore_flagpoles)

        self.area_vel_ema_alpha = float(area_vel_ema_alpha)
        self.min_area_vel_ema = float(min_area_vel_ema)
        self.aligned_shrink_reset_frac = float(aligned_shrink_reset_frac)
        self.aligned_max_age_sec = float(aligned_max_age_sec)
        self.min_aligned_frames = int(min_aligned_frames)
        self.flag_aligned_shrink_reset_frac = float(flag_aligned_shrink_reset_frac)
        self.gate_pass_area_thresh = float(gate_pass_area_thresh)
        self.peak_area_max_jump = float(peak_area_max_jump)

        self.states: Dict[int, TrackPassState] = {}
        self._just_passed: Dict[int, dict] = {}

        # cooldown history
        self._last_pass_time_global: float = -1e9
        self._last_pass_time_by_type: Dict[str, float] = {}
        self._last_pass_time_by_track: Dict[int, float] = {}

    # ----------------------------
    # Public debug API
    # ----------------------------
    def get_debug(self, now: float) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for tid, st in self.states.items():
            tkey = _norm_type(st.ttype)

            rem_global = max(0.0, self.pass_cooldown_sec - (now - self._last_pass_time_global))
            rem_type = max(0.0, self.type_cooldown_sec - (now - self._last_pass_time_by_type.get(tkey, -1e9)))
            rem_track = max(0.0, self.track_cooldown_sec - (now - self._last_pass_time_by_track.get(tid, -1e9)))

            out[tid] = {
                "track_id": st.track_id,
                "type": st.ttype,
                "stage": st.stage,
                "last_seen_age": float(now - st.last_seen_time),

                "score_ema": float(st.last_score_ema),

                "area_ratio": float(st.last_area_ratio),
                "center_dist": float(st.last_center_dist),
                "nx": float(st.last_nx),
                "ny": float(st.last_ny),

                "flag_was_centered": bool(st.flag_was_centered),

                # growth-based debug
                "area_vel": float(st.area_vel),
                "area_vel_ema": float(st.area_vel_ema),
                "min_area_vel_ema": float(self.min_area_vel_ema),
                "aligned_area_ratio": float(st.aligned_area_ratio),
                "aligned_age": float(now - st.aligned_time) if st.aligned_time > 0 else 0.0,

                # thresholds
                "min_area_ratio": float(self.min_area_ratio),
                "center_tol": float(self.center_tol),
                "disappear_timeout": float(self.disappear_timeout),
                "flag_center_tol": float(self.flag_center_tol),
                "flag_edge_tol": float(self.flag_edge_tol),
                "flag_min_edges": int(self.flag_min_edges),
                "cam_left_norm": float(self.cam_left_norm),
                "cam_right_norm": float(self.cam_right_norm),
                "aligned_shrink_reset_frac": float(self.aligned_shrink_reset_frac),
                "aligned_max_age_sec": float(self.aligned_max_age_sec),

                "cooldown_remaining_global": float(rem_global),
                "cooldown_remaining_type": float(rem_type),
                "cooldown_remaining_track": float(rem_track),

                "last_reason_attempted": st.last_reason_attempted,
                "last_pass_blocked_by": st.last_pass_blocked_by,

                "bbox": st.last_bbox,
            }
        return out

    def set_camera_edges(self, left_norm: float, right_norm: float):
        """Set the real camera left/right edges (normalized 0..1), used for flagpole edge detection."""
        self.cam_left_norm  = float(left_norm)
        self.cam_right_norm = float(right_norm)
        print(f"[PassDetector] camera edges: left={left_norm:.3f}  right={right_norm:.3f}"
              f"  (real width = {(right_norm - left_norm) * 100:.1f}% of video width)")

    # ----------------------------
    # Main logic
    # ----------------------------
    def update(self, tracks, now: float, frame_w: int, frame_h: int):
        frame_area = float(frame_w * frame_h)
        self._frame_w = frame_w
        self._frame_h = frame_h
        seen_ids = set()

        # update seen tracks
        for tr in tracks:
            tid = int(tr.track_id)
            ttype = tr.locked_type or "NONE"
            score_ema = float(getattr(tr, "score_ema", 0.0))

            # ignore junk
            if ttype == "NONE":
                continue
            if score_ema < self.min_track_score:
                continue
            if self.ignore_flagpoles and _is_flag(ttype):
                continue

            seen_ids.add(tid)

            st = self.states.get(tid)
            if st is None:
                st = TrackPassState(track_id=tid, ttype=ttype)
                st.last_updated_time = now
                self.states[tid] = st

            # if type changed (rare), reset state machine
            if _norm_type(st.ttype) != _norm_type(ttype):
                st = TrackPassState(track_id=tid, ttype=ttype)
                st.last_updated_time = now
                self.states[tid] = st

            # time delta for velocity
            dt = max(1e-6, float(now - st.last_updated_time))
            st.last_updated_time = now

            st.last_seen_time = now
            st.last_score_ema = score_ema

            bbox = tr.bbox
            st.last_bbox = bbox

            cx, cy = _center(bbox)
            a = _area(bbox)
            st.last_area = a
            st.peak_area = max(st.peak_area, a)
            st.last_cx = cx
            st.last_cy = cy

            # normalize to [0..1]
            nx = (cx / max(frame_w, 1))
            ny = (cy / max(frame_h, 1))
            dx = abs(nx - 0.5)
            dy = abs(ny - 0.5)
            center_dist = (dx * dx + dy * dy) ** 0.5
            area_ratio = a / max(frame_area, 1.0)

            st.last_nx = nx
            st.last_ny = ny
            st.last_center_dist = center_dist
            st.last_area_ratio = area_ratio

            st.last_reason_attempted = ""
            st.last_pass_blocked_by = ""

            # --------------------------------
            # Growth stats update
            # --------------------------------
            prev = float(st.prev_area_ratio)
            vel = (area_ratio - prev) / dt
            st.area_vel = vel
            st.area_vel_ema = (1.0 - self.area_vel_ema_alpha) * st.area_vel_ema + self.area_vel_ema_alpha * vel
            st.prev_area_ratio = area_ratio

            # --------------------------------
            # flag logic — enters aligned when centered (no growth requirement)
            # --------------------------------
            if _is_flag(ttype):
                if st.stage == "idle":
                    if area_ratio >= (self.min_area_ratio * 0.60) and abs(nx - 0.5) <= self.flag_center_tol:
                        st.stage = "aligned"
                        st.aligned_area_ratio = area_ratio
                        st.aligned_time = now
                        st.aligned_frames = 1
                        st.flag_was_centered = True

                elif st.stage == "aligned":
                    st.aligned_frames += 1

                    # max age safety
                    if self.aligned_max_age_sec > 0:
                        if st.aligned_time > 0 and (now - st.aligned_time) > self.aligned_max_age_sec:
                            st.stage = "idle"
                            st.aligned_area_ratio = 0.0
                            st.aligned_time = 0.0
                            st.aligned_frames = 0
                            st.flag_was_centered = False
                continue

            st.last_flag_span = 0.0

            # --------------------------------
            # Growth-based alignment logic (gates only)
            # --------------------------------
            growth_ok = (st.area_vel_ema >= self.min_area_vel_ema)

            if st.stage == "idle":
                if area_ratio >= self.min_area_ratio and center_dist <= self.center_tol and growth_ok:
                    st.stage = "aligned"
                    st.aligned_area_ratio = area_ratio
                    st.aligned_time = now
                    st.aligned_frames = 1

            elif st.stage == "aligned":
                st.aligned_frames += 1

                if st.aligned_area_ratio > 0 and not st.gate_area_peaked:
                    drop_frac = (st.aligned_area_ratio - area_ratio) / max(st.aligned_area_ratio, 1e-9)
                    if drop_frac >= self.aligned_shrink_reset_frac:
                        st.stage = "idle"
                        st.aligned_area_ratio = 0.0
                        st.aligned_time = 0.0
                        st.aligned_frames = 0
                        st.gate_area_peaked = False
                        st.gate_peaked_area_ratio = 0.0

                # in-frame pass: gate peaked above threshold (with good edges), now shrinking
                if st.stage == "aligned" and self.gate_pass_area_thresh > 0:
                    jump_ratio = area_ratio / prev if prev > 0 else 1.0
                    if (area_ratio >= self.gate_pass_area_thresh
                            and not st.gate_area_peaked
                            and jump_ratio <= self.peak_area_max_jump):
                        # latch only if edges are good RIGHT NOW at the peak
                        x1, y1, x2, y2 = bbox
                        edges_near = _count_edges_near_real_frame(
                            x1, y1, x2, y2,
                            frame_w, frame_h,
                            self.cam_left_norm, self.cam_right_norm,
                            self.flag_edge_tol,
                        )
                        if edges_near >= self.flag_min_edges:
                            st.gate_area_peaked = True
                            st.gate_peaked_area_ratio = area_ratio
                    if st.gate_area_peaked:
                        # track the true running maximum so fire triggers on real shrink
                        st.gate_peaked_area_ratio = max(st.gate_peaked_area_ratio, area_ratio)
                    if (st.gate_area_peaked
                            and area_ratio < st.gate_peaked_area_ratio
                            and st.aligned_frames >= self.min_aligned_frames):
                        st.last_reason_attempted = "area_peak_shrink"
                        self._mark_passed(st, now, reason="area_peak_shrink")

                if st.stage == "aligned" and self.aligned_max_age_sec > 0:
                    if st.aligned_time > 0 and (now - st.aligned_time) > self.aligned_max_age_sec:
                        print(" aligned too long reset  id ", tid)
                        st.stage = "idle"
                        st.aligned_area_ratio = 0.0
                        st.aligned_time = 0.0
                        st.aligned_frames = 0
                        st.gate_area_peaked = False
                        st.gate_peaked_area_ratio = 0.0

        # handle disappeared tracks: if aligned recently => PASS
        for tid, st in list(self.states.items()):
            if tid in seen_ids:
                continue

            if st.stage == "aligned":
                elapsed = now - st.last_seen_time
                if elapsed <= self.disappear_timeout:
                    if st.aligned_frames >= self.min_aligned_frames:
                        x1, y1, x2, y2 = st.last_bbox
                        edges_near = _count_edges_near_real_frame(
                            x1, y1, x2, y2,
                            self._frame_w, self._frame_h,
                            self.cam_left_norm, self.cam_right_norm,
                            self.flag_edge_tol,
                        )
                        if edges_near >= self.flag_min_edges:
                            reason = "flag_disappear_edge" if _is_flag(st.ttype) else "disappear_after_align"
                            st.last_reason_attempted = reason
                            self._mark_passed(st, now, reason=reason)
                else:
                    # disappear window expired without firing — reset to idle so ghost bbox disappears
                    st.stage = "idle"
                    st.aligned_frames = 0
                    st.aligned_area_ratio = 0.0
                    st.aligned_time = 0.0
                    st.flag_was_centered = False
                    st.gate_area_peaked = False
                    st.gate_peaked_area_ratio = 0.0

            # cleanup old states
            if (now - st.last_seen_time) > 2.0:
                self.states.pop(tid, None)

        self._gc_cooldowns(now)

    # ----------------------------
    # Cooldown helpers
    # ----------------------------
    def _cooldown_ok(self, st: TrackPassState, now: float) -> Tuple[bool, str]:
        if st.stage == "passed":
            return (False, "already_passed")

        if (now - self._last_pass_time_global) < self.pass_cooldown_sec:
            return (False, "global")

        tkey = _norm_type(st.ttype)
        last_t = self._last_pass_time_by_type.get(tkey, -1e9)
        if (now - last_t) < self.type_cooldown_sec:
            return (False, "type")

        last_tr = self._last_pass_time_by_track.get(st.track_id, -1e9)
        if (now - last_tr) < self.track_cooldown_sec:
            return (False, "track")

        return (True, "")

    def _mark_passed(self, st: TrackPassState, now: float, reason: str):
        ok, blocked_by = self._cooldown_ok(st, now)
        if not ok:
            st.last_pass_blocked_by = blocked_by
            print("mark passed blocked by cooldown! ", st.track_id)
            return

        st.stage = "passed"
        st.passed_time = now

        self._last_pass_time_global = now
        self._last_pass_time_by_type[_norm_type(st.ttype)] = now
        self._last_pass_time_by_track[st.track_id] = now

        self._just_passed[st.track_id] = {
            "track_id": st.track_id,
            "type": st.ttype,
            "time": now,
            "reason": reason,
        }

        # Reset every other aligned track to idle so they must re-establish
        # alignment from scratch after a pass fires.
        for other_tid, other_st in self.states.items():
            if other_tid != st.track_id and other_st.stage == "aligned":
                other_st.stage = "idle"
                other_st.aligned_area_ratio = 0.0
                other_st.aligned_time = 0.0
                other_st.aligned_frames = 0
                other_st.flag_was_centered = False
                other_st.gate_area_peaked = False
                other_st.gate_peaked_area_ratio = 0.0

    def clear_all_aligned(self):
        """Reset every aligned track to idle. Call on sibling instances after a pass fires."""
        for st in self.states.values():
            if st.stage == "aligned":
                st.stage = "idle"
                st.aligned_area_ratio = 0.0
                st.aligned_time = 0.0
                st.aligned_frames = 0
                st.flag_was_centered = False
                st.gate_area_peaked = False
                st.gate_peaked_area_ratio = 0.0

    def _gc_cooldowns(self, now: float):
        horizon = max(5.0, self.track_cooldown_sec * 4.0)
        for tid in list(self._last_pass_time_by_track.keys()):
            if (now - self._last_pass_time_by_track[tid]) > horizon:
                self._last_pass_time_by_track.pop(tid, None)

        for tkey in list(self._last_pass_time_by_type.keys()):
            if (now - self._last_pass_time_by_type[tkey]) > horizon:
                self._last_pass_time_by_type.pop(tkey, None)

    # ----------------------------
    # Event consumption
    # ----------------------------
    def consume_passed(self, track_id: int) -> Optional[dict]:
        return self._just_passed.pop(int(track_id), None)

    def pop_any_passed(self) -> Optional[dict]:
        for k in list(self._just_passed.keys()):
            return self._just_passed.pop(k)
        return None
