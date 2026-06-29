"""
Live interactive SO-101 sim.  Run: ./sim/run_gui.sh

Movement (hold):
  A/D          shoulder pan
  I/K          shoulder lift (raise / lower arm)
  W/S          elbow
  ↑/↓          wrist up / down
  ←/→          wrist roll
  Q/E          gripper open / close

Fine tune:
  1-6          select joint
  U/J          nudge selected joint
  +/-          step size

Other:
  B            above_board pose
  R            home
  G            gripper toggle
  P            save pose
  [ / ]        prev / next named pose
  V            wrist camera (Q in that window to close)
  Esc          quit
"""

from __future__ import annotations

import time

import pybullet as p

from arm import JOINT_NAMES, SimArm, load_config
from board import add_board_visuals
from camera import WristCamera
from controls import apply_fine_controls, apply_hold_controls, apply_key_event, gripper_is_open
from pieces import START_FEN, BoardPieces
from poses import list_poses

STEP = 0.08
ESCAPE = getattr(p, "B3G_ESCAPE", 65307)
ARROWS = {
    "up": getattr(p, "B3G_UPARROW", 65297),
    "down": getattr(p, "B3G_DOWNARROW", 65298),
    "left": getattr(p, "B3G_LEFTARROW", 65299),
    "right": getattr(p, "B3G_RIGHTARROW", 65300),
}


def print_status(selected: int, step: float, targets: dict[str, float], actual: dict[str, float], pose_i: int, pose_names: list[str]):
    pose_label = pose_names[pose_i] if pose_names else "-"
    print(f"\n[{JOINT_NAMES[selected]}] step={step:.3f}  pose={pose_label}")
    print("  A/D pan  I/K lift  W/S elbow  ↑↓ wrist  ←→ roll  Q/E grip  U/J fine  B board  Esc quit")
    for name in JOINT_NAMES:
        mark = ">" if name == JOINT_NAMES[selected] else " "
        print(f" {mark} {name:14s} target {targets[name]:+.3f}   actual {actual[name]:+.3f}")


def main():
    cfg = load_config()
    home = cfg.get("poses", {}).get("home", {n: 0.0 for n in JOINT_NAMES})
    above = cfg.get("poses", {}).get("above_board", home)
    pose_names = list_poses()
    pose_i = 0

    arm = SimArm(gui=True)
    cam = WristCamera(arm, cfg)
    limits = arm.joint_limits()
    targets = {n: above.get(n, home.get(n, 0.0)) for n in JOINT_NAMES}
    arm.reset_pose(targets)
    arm.hold(targets)

    if cfg.get("sim", {}).get("board", {}).get("show_grid", True) or cfg.get("sim", {}).get("board", {}).get("show_checker", True):
        add_board_visuals(cfg)

    board_cfg = cfg.get("sim", {}).get("board", {})
    if board_cfg.get("show_pieces", True):
        fen = board_cfg.get("start_fen", START_FEN)
        pieces = BoardPieces.from_fen(cfg, fen)
        print(f"spawned {len(pieces.by_square)} chess pieces")

    selected = 0
    step = STEP
    gripper_open = targets["gripper"] > 0.4
    show_wrist = False
    last_print = 0.0
    exit_reason = ""

    print("Click the PyBullet window. A/D pan, I/K lift, W/S elbow, arrows wrist, Q/E gripper.")
    print_status(selected, step, targets, arm.joint_states(), pose_i, pose_names)

    try:
        while True:
            if not p.isConnected():
                exit_reason = "3D window closed"
                break

            keys = p.getKeyboardEvents()
            moved = False

            targets, ch = apply_hold_controls(keys, targets, limits, step, ARROWS)
            moved |= ch
            if ch:
                gripper_open = gripper_is_open(targets)
            targets, ch = apply_fine_controls(keys, targets, limits, selected, step)
            moved |= ch

            quit_now = False
            for key, state in keys.items():
                if not (state & p.KEY_WAS_TRIGGERED):
                    continue
                if key == ESCAPE:
                    exit_reason = "Esc pressed"
                    quit_now = True
                    break
                if key == ord("[") and pose_names:
                    pose_i = (pose_i - 1) % len(pose_names)
                    targets = arm.resolve_pose(pose_names[pose_i])
                    gripper_open = targets["gripper"] > 0.4
                    moved = True
                if key == ord("]") and pose_names:
                    pose_i = (pose_i + 1) % len(pose_names)
                    targets = arm.resolve_pose(pose_names[pose_i])
                    gripper_open = targets["gripper"] > 0.4
                    moved = True
                if key == ord("v"):
                    show_wrist = not show_wrist
                    if not show_wrist:
                        WristCamera.close_window()
                    else:
                        print("  wrist camera on (Q in preview window to close)")
                    moved = True
                    continue
                if key == ord("p"):
                    targets, selected, step, gripper_open, ch = apply_key_event(
                        targets,
                        limits,
                        home,
                        selected=selected,
                        step=step,
                        key=key,
                        gripper_open=gripper_open,
                        above_board=above,
                    )
                    if ch:
                        pose_names = list_poses()
                        print("  saved pose -> data/sim_poses.json")
                    moved = ch
                    continue
                targets, selected, step, gripper_open, changed = apply_key_event(
                    targets,
                    limits,
                    home,
                    selected=selected,
                    step=step,
                    key=key,
                    gripper_open=gripper_open,
                    above_board=above,
                )
                if changed:
                    moved = True

            if quit_now:
                break

            if not p.isConnected():
                exit_reason = "3D window closed"
                break

            arm.hold(targets)
            arm.step(1 / 240)
            if show_wrist:
                if not cam.show():
                    show_wrist = False
                    print("  wrist camera closed")
            now = time.time()
            if moved or now - last_print > 0.4:
                print_status(selected, step, targets, arm.joint_states(), pose_i, pose_names)
                last_print = now
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
    finally:
        WristCamera.close_window()
        if p.isConnected():
            arm.disconnect()
        if exit_reason:
            print(f"\nSim ended ({exit_reason}). Thread messages above are normal PyBullet shutdown.")
        else:
            print("\nSim ended.")


if __name__ == "__main__":
    main()
