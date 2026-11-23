"""Opt-in realism profile: rougher physics and a noisy depth sensor. Off by default.

  ZIMA_REALISM=1 python sim/run.py --suite core --walls real    # or run.py --realism

The default sim is a best case: grippy felt (friction 1.2 on pieces and board),
a gripper servo that squeezes with the full STS3215 torque, and exact depth.
The profile moves each towards the real arm:
  piece and board sliding friction   0.4   (plastic on a wooden board)
  jaw pad friction                   0.6   (printed PLA/TPU pads)
  gripper torque                     50 %  (lerobot torque-limits the gripper)
  depth noise                        sigma 1 mm at 20 cm, growing with z^2 (stereo), plus 1 % dropouts

Rows from a run with the profile carry variant "realism", so `compare.py`
never lines them up against the default ("base") baselines.
"""

from __future__ import annotations

import os

import numpy as np

PIECE_FRICTION = 0.4
BOARD_FRICTION = 0.4
PAD_FRICTION = 0.6
GRIPPER_TORQUE_FRAC = 0.5
DEPTH_SIGMA_M = 0.001
DEPTH_SIGMA_AT_M = 0.20
DEPTH_DROPOUT = 0.01


def active() -> bool:
    return os.environ.get("ZIMA_REALISM", "").strip().lower() not in ("", "0", "off", "false", "no")


def variant() -> str:
    return "realism" if active() else "base"


def apply_model(model) -> None:
    """Rough contact and a torque-limited gripper on a freshly loaded model."""
    if not active():
        return
    import mujoco

    free_bodies = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)}
    board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board")
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in free_bodies:
            model.geom_friction[g, 0] = min(float(model.geom_friction[g, 0]), PIECE_FRICTION)
        elif g == board:
            model.geom_friction[g, 0] = min(float(model.geom_friction[g, 0]), BOARD_FRICTION)
        elif int(model.geom_priority[g]) > 0:
            model.geom_friction[g, 0] = min(float(model.geom_friction[g, 0]), PAD_FRICTION)
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if a >= 0:
        model.actuator_forcelimited[a] = 1
        model.actuator_forcerange[a] = np.asarray(model.actuator_forcerange[a]) * GRIPPER_TORQUE_FRAC


_rng = None


def noisy_depth(depth: np.ndarray) -> np.ndarray:
    """Stereo-like depth noise plus invalid (zero) pixels; repeatable per process."""
    global _rng
    if not active():
        return depth
    if _rng is None:
        _rng = np.random.default_rng(int(os.environ.get("ZIMA_REALISM_SEED", "0")))
    z = np.asarray(depth, dtype=np.float64)
    sigma = DEPTH_SIGMA_M * (z / DEPTH_SIGMA_AT_M) ** 2
    out = z + _rng.standard_normal(z.shape) * sigma
    out[_rng.random(z.shape) < DEPTH_DROPOUT] = 0.0
    return out.astype(depth.dtype, copy=False)
