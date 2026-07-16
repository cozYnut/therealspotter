"""
Shared pipeline configuration.

Reads debug_ui_defaults.json so that extract_candidates.py, extract_race.py,
and lazy_spotter.py all run with identical PassDetector / TimeTracker parameters.

Press P in the debug UI (lazy_spotter.py) to save the current values.
"""

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).parent / "debug_ui_defaults.json"

DEFAULTS: Dict[str, Any] = {
    "tracker": {
        "iou_match_thresh": 0.1,
        "ttl_seconds":      0.03,
        "lock_min_score":   0.2,
        "lock_hysteresis":  0.1,
        "lock_streak":      3,
        "ema_alpha":        0.4,
        "min_det_conf":     0.0,
    },
    "gates": {
        "min_track_score":               0.2,
        "min_area_ratio":                0.03,
        "center_tol":                    0.3,
        "disappear_timeout":             0.09,
        "flag_edge_tol":                 0.08,
        "flag_min_edges":                2,
        "pass_cooldown_sec":             0.09,
        "type_cooldown_sec":             0.09,
        "track_cooldown_sec":            0.2,
        "area_vel_ema_alpha":            0.34,
        "min_area_vel_ema":              0.015,
        "aligned_shrink_reset_frac":     0.35,
        "aligned_max_age_sec":           2.0,
        "min_aligned_frames":            4,
        "aligned_shrink_disappear_frac": 0.35,
        "pass_area_ratio":               0.34,
        "aligned_area_jump_frac":        5.0,
        "high_conf_score":               0.0,
        "high_conf_top_tol":             0.35,
    },
    "flags": {
        "min_track_score":             0.14,
        "min_area_ratio":              0.01,
        "disappear_timeout":           0.09,
        "flag_center_tol":             0.2,
        "flag_edge_tol":               0.07,
        "flag_min_edges":              2,
        "pass_cooldown_sec":           0.09,
        "type_cooldown_sec":           0.09,
        "track_cooldown_sec":          0.2,
        "aligned_max_age_sec":         2.0,
        "min_aligned_frames":          3,
        "pass_area_ratio":             0.3,
        "flag_aligned_area_jump_frac": 5.0,
    },
}


def load() -> Dict[str, Any]:
    """Return config dict merged over DEFAULTS. Missing keys fall back."""
    cfg = {
        "tracker": dict(DEFAULTS["tracker"]),
        "gates":   dict(DEFAULTS["gates"]),
        "flags":   dict(DEFAULTS["flags"]),
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for section in ("tracker", "gates", "flags"):
                if section in saved:
                    cfg[section].update(saved[section])
        except Exception as e:
            print(f"[pipeline_cfg] Could not read {CONFIG_PATH}: {e} — using defaults")
    return cfg


def save(cfg: Dict[str, Any]) -> None:
    """Write config to debug_ui_defaults.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[pipeline_cfg] Saved to {CONFIG_PATH}")


def tracker_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg["tracker"]
    return {
        "iou_match_thresh": float(t.get("iou_match_thresh", DEFAULTS["tracker"]["iou_match_thresh"])),
        "ttl_seconds":      float(t.get("ttl_seconds",      DEFAULTS["tracker"]["ttl_seconds"])),
        "lock_min_score":   float(t.get("lock_min_score",   DEFAULTS["tracker"]["lock_min_score"])),
        "lock_hysteresis":  float(t.get("lock_hysteresis",  DEFAULTS["tracker"]["lock_hysteresis"])),
        "lock_streak":      int(t.get("lock_streak",        DEFAULTS["tracker"]["lock_streak"])),
        "ema_alpha":        float(t.get("ema_alpha",         DEFAULTS["tracker"]["ema_alpha"])),
        "min_det_conf":     float(t.get("min_det_conf",     DEFAULTS["tracker"]["min_det_conf"])),
    }


def gates_passdet_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    g = cfg["gates"]
    return {
        "min_track_score":               float(g.get("min_track_score",               DEFAULTS["gates"]["min_track_score"])),
        "min_area_ratio":                float(g.get("min_area_ratio",                DEFAULTS["gates"]["min_area_ratio"])),
        "center_tol":                    float(g.get("center_tol",                    DEFAULTS["gates"]["center_tol"])),
        "disappear_timeout":             float(g.get("disappear_timeout",             DEFAULTS["gates"]["disappear_timeout"])),
        "flag_edge_tol":                 float(g.get("flag_edge_tol",                 DEFAULTS["gates"]["flag_edge_tol"])),
        "flag_min_edges":                int(g.get("flag_min_edges",                  DEFAULTS["gates"]["flag_min_edges"])),
        "pass_cooldown_sec":             float(g.get("pass_cooldown_sec",             DEFAULTS["gates"]["pass_cooldown_sec"])),
        "type_cooldown_sec":             float(g.get("type_cooldown_sec",             DEFAULTS["gates"]["type_cooldown_sec"])),
        "track_cooldown_sec":            float(g.get("track_cooldown_sec",            DEFAULTS["gates"]["track_cooldown_sec"])),
        "area_vel_ema_alpha":            float(g.get("area_vel_ema_alpha",            DEFAULTS["gates"]["area_vel_ema_alpha"])),
        "min_area_vel_ema":              float(g.get("min_area_vel_ema",              DEFAULTS["gates"]["min_area_vel_ema"])),
        "aligned_shrink_reset_frac":     float(g.get("aligned_shrink_reset_frac",     DEFAULTS["gates"]["aligned_shrink_reset_frac"])),
        "aligned_max_age_sec":           float(g.get("aligned_max_age_sec",           DEFAULTS["gates"]["aligned_max_age_sec"])),
        "min_aligned_frames":            int(g.get("min_aligned_frames",              DEFAULTS["gates"]["min_aligned_frames"])),
        "aligned_shrink_disappear_frac": float(g.get("aligned_shrink_disappear_frac", DEFAULTS["gates"]["aligned_shrink_disappear_frac"])),
        "pass_area_ratio":               float(g.get("pass_area_ratio",               DEFAULTS["gates"]["pass_area_ratio"])),
        "aligned_area_jump_frac":        float(g.get("aligned_area_jump_frac",        DEFAULTS["gates"]["aligned_area_jump_frac"])),
        "high_conf_score":               float(g.get("high_conf_score",               DEFAULTS["gates"]["high_conf_score"])),
        "high_conf_top_tol":             float(g.get("high_conf_top_tol",             DEFAULTS["gates"]["high_conf_top_tol"])),
        "ignore_flagpoles":              True,
    }


def flags_passdet_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    f = cfg["flags"]
    return {
        "min_track_score":             float(f.get("min_track_score",             DEFAULTS["flags"]["min_track_score"])),
        "min_area_ratio":              float(f.get("min_area_ratio",              DEFAULTS["flags"]["min_area_ratio"])),
        "disappear_timeout":           float(f.get("disappear_timeout",           DEFAULTS["flags"]["disappear_timeout"])),
        "flag_center_tol":             float(f.get("flag_center_tol",             DEFAULTS["flags"]["flag_center_tol"])),
        "flag_edge_tol":               float(f.get("flag_edge_tol",               DEFAULTS["flags"]["flag_edge_tol"])),
        "flag_min_edges":              int(f.get("flag_min_edges",                DEFAULTS["flags"]["flag_min_edges"])),
        "pass_cooldown_sec":           float(f.get("pass_cooldown_sec",           DEFAULTS["flags"]["pass_cooldown_sec"])),
        "type_cooldown_sec":           float(f.get("type_cooldown_sec",           DEFAULTS["flags"]["type_cooldown_sec"])),
        "track_cooldown_sec":          float(f.get("track_cooldown_sec",          DEFAULTS["flags"]["track_cooldown_sec"])),
        "ignore_flagpoles":            False,
        "aligned_max_age_sec":         float(f.get("aligned_max_age_sec",         DEFAULTS["flags"]["aligned_max_age_sec"])),
        "min_aligned_frames":          int(f.get("min_aligned_frames",            DEFAULTS["flags"]["min_aligned_frames"])),
        "pass_area_ratio":             float(f.get("pass_area_ratio",             DEFAULTS["flags"]["pass_area_ratio"])),
        "flag_aligned_area_jump_frac": float(f.get("flag_aligned_area_jump_frac", DEFAULTS["flags"]["flag_aligned_area_jump_frac"])),
    }
