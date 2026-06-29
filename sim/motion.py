"""Trajectory helpers — interpolate joints, run pose sequences."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arm import SimArm


def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def interpolate(start: dict[str, float], end: dict[str, float], u: float) -> dict[str, float]:
    u = smoothstep(u)
    keys = set(start) | set(end)
    return {k: start.get(k, 0.0) + (end.get(k, start.get(k, 0.0)) - start.get(k, 0.0)) * u for k in keys}


def run_sequence(arm: SimArm, pose_names: list[str], duration: float = 2.0) -> bool:
    from poses import load_all

    poses = load_all()
    for name in pose_names:
        if name not in poses:
            raise KeyError(f"Unknown pose '{name}'. Available: {list(poses)}")
        ok = arm.go_pose(name, duration=duration)
        if not ok:
            return False
    return True


def hold_for(arm: SimArm, positions: dict[str, float], seconds: float):
    """Hold position with physics stepping (anti-drift)."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        arm.hold(positions)
        arm.step()
