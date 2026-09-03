"""Fast headless sim — step-based motion, fewer physics settle steps."""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("ZIMA_SIM_FAST", "").lower() in ("1", "true", "yes")


def apply_config(cfg: dict) -> dict:
    if not enabled():
        return cfg
    sim = cfg.setdefault("sim", {})
    sim.setdefault("grasp", {}).update(
        {
            "descend_steps": 6,
            "travel_steps": 4,
            "close_steps": 40,
            "settle_steps": 12,
        }
    )
    return cfg


def piece_settle_steps() -> int:
    return 8 if enabled() else 40


# Video clock: larger MuJoCo dt, same move duration in sim-seconds, so
# joint velocity is unchanged. Shortening duration (old motion_scale) yanks
# the piece off the jaws. Default 10x when FAST=1.
_MIN_CLOCK = 1.0
_MAX_CLOCK = 20.0
_DEFAULT_CLOCK = 2.0


def clock_scale() -> float:
    """How many times faster the integrator runs (video speedup)."""
    if not enabled():
        return 1.0
    raw = os.environ.get("ZIMA_SIM_CLOCK")
    if raw:
        try:
            return max(_MIN_CLOCK, min(_MAX_CLOCK, float(raw)))
        except ValueError:
            pass
    return _DEFAULT_CLOCK


# Kept so callers that still read a duration scale get 1.0 (clock handles speed).
_MIN_SCALE = 1.0


def motion_scale() -> float:
    """Move duration scale. Always 1 in clock mode so paths stay physically the same."""
    return 1.0
