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


def motion_scale() -> float:
    """Scale down move durations in fast mode."""
    return 0.35 if enabled() else 1.0
