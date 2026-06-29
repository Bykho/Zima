"""Keyboard helpers for sim/interact.py."""

from __future__ import annotations

from arm import JOINT_NAMES

KEY_IS_DOWN = 1

SELECT_KEYS = {ord(str(i)): i - 1 for i in range(1, 7)}

# Hold keys → joint. Signs tuned from above_board over the chess board.
BOARD_HOLD = (
    (("a",), "shoulder_pan", -1),
    (("d",), "shoulder_pan", 1),
    (("i",), "shoulder_lift", -1),  # negative lift raises EE over board
    (("k",), "shoulder_lift", 1),
    (("w",), "elbow_flex", 1),
    (("s",), "elbow_flex", -1),
    (("q",), "gripper", 1),
    (("e",), "gripper", -1),
)

ARROW_TO_JOINT = {
    "up": ("wrist_flex", 1),
    "down": ("wrist_flex", -1),
    "left": ("wrist_roll", -1),
    "right": ("wrist_roll", 1),
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def nudge(
    targets: dict[str, float],
    limits: dict[str, tuple[float, float]],
    joint: str,
    delta: float,
) -> dict[str, float]:
    lo, hi = limits[joint]
    out = dict(targets)
    out[joint] = clamp(targets[joint] + delta, lo, hi)
    return out


def _held(keys: dict, code: int) -> bool:
    return bool(keys.get(code, 0) & KEY_IS_DOWN)


def apply_hold_controls(
    keys: dict,
    targets: dict[str, float],
    limits: dict[str, tuple[float, float]],
    step: float,
    arrow_codes: dict[str, int] | None = None,
) -> tuple[dict[str, float], bool]:
    """A/D pan, I/K shoulder lift, W/S elbow, Q/E gripper, arrows wrist — all while held."""
    arrow_codes = arrow_codes or {}
    changed = False
    out = dict(targets)

    for chars, joint, sign in BOARD_HOLD:
        codes = [ord(c) for c in chars] + [ord(c.upper()) for c in chars]
        if any(_held(keys, c) for c in codes):
            out = nudge(out, limits, joint, sign * step)
            changed = True

    for name, (joint, sign) in ARROW_TO_JOINT.items():
        code = arrow_codes.get(name)
        if code is not None and _held(keys, code):
            out = nudge(out, limits, joint, sign * step)
            changed = True

    return out, changed


def gripper_is_open(targets: dict[str, float]) -> bool:
    return targets.get("gripper", 0.0) > 0.4


def apply_fine_controls(
    keys: dict,
    targets: dict[str, float],
    limits: dict[str, tuple[float, float]],
    selected: int,
    step: float,
) -> tuple[dict[str, float], bool]:
    """U/J nudge the selected joint (after picking with 1–6)."""
    joint = JOINT_NAMES[selected]
    if _held(keys, ord("u")) or _held(keys, ord("U")):
        return nudge(targets, limits, joint, step), True
    if _held(keys, ord("j")) or _held(keys, ord("J")):
        return nudge(targets, limits, joint, -step), True
    return targets, False


def apply_key_event(
    targets: dict[str, float],
    limits: dict[str, tuple[float, float]],
    home: dict[str, float],
    *,
    selected: int,
    step: float,
    key: int,
    gripper_open: bool,
    above_board: dict[str, float] | None = None,
) -> tuple[dict[str, float], int, float, bool, bool]:
    """Returns (targets, selected, step, gripper_open, changed)."""
    changed = False
    if key in SELECT_KEYS:
        return targets, SELECT_KEYS[key], step, gripper_open, True
    if key in (ord("+"), ord("=")):
        return targets, selected, min(0.25, step * 1.5), gripper_open, True
    if key == ord("-"):
        return targets, selected, max(0.005, step / 1.5), gripper_open, True
    if key == ord("r"):
        return {n: home.get(n, 0.0) for n in JOINT_NAMES}, selected, step, home.get("gripper", 0.8) > 0.4, True
    if key == ord("b") and above_board:
        t = {n: above_board.get(n, home.get(n, 0.0)) for n in JOINT_NAMES}
        return t, selected, step, t.get("gripper", 0.8) > 0.4, True
    if key == ord("g"):
        gripper_open = not gripper_open
        t = dict(targets)
        t["gripper"] = 0.8 if gripper_open else 0.0
        return t, selected, step, gripper_open, True
    if key == ord("p"):
        from poses import save_pose

        name = f"saved_{int(targets['shoulder_pan']*100)}"
        save_pose(name, targets)
        return targets, selected, step, gripper_open, True
    return targets, selected, step, gripper_open, changed
