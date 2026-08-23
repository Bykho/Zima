"""Automated sim tests — run: python sim/test_sim.py [--fast]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

FAST = "--fast" in sys.argv
if FAST:
    os.environ["ZIMA_SIM_FAST"] = "1"

from arm import JOINT_NAMES, SimArm, load_config  # noqa: E402
from board import parse_square, square_to_world  # noqa: E402
from controls import apply_hold_controls, nudge, apply_key_event  # noqa: E402
from fast import motion_scale  # noqa: E402
from motion import run_sequence, interpolate  # noqa: E402
from poses import load_all, save_pose  # noqa: E402


def arm() -> SimArm:
    return SimArm(gui=False, fast=FAST)


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def test_controls_pure():
    limits = {"shoulder_pan": (-1.0, 1.0), "gripper": (0.0, 1.0)}
    t = {n: 0.0 for n in JOINT_NAMES}
    t["gripper"] = 0.8
    t2 = nudge(t, limits, "shoulder_pan", 0.5)
    ok = abs(t2["shoulder_pan"] - 0.5) < 1e-6
    check("nudge joint", ok)
    home = {n: 0.0 for n in JOINT_NAMES}
    home["gripper"] = 0.8
    _, sel, _, _, ch = apply_key_event(t, limits, home, selected=0, step=0.05, key=ord("2"), gripper_open=True)
    check("select joint 2", sel == 1 and ch)
    keys = {ord("d"): 1, ord("w"): 1}
    limits = {
        "shoulder_pan": (-1.0, 1.0),
        "shoulder_lift": (-1.0, 1.0),
        "elbow_flex": (-1.0, 1.0),
        "gripper": (0.0, 1.0),
    }
    t3, moved = apply_hold_controls(keys, t, limits, 0.1)
    check("hold A/D pan", moved and t3["shoulder_pan"] > 0)
    keys = {ord("i"): 1}
    t4, moved = apply_hold_controls(keys, t3, limits, 0.1)
    check("hold I/K lift", moved and t4["shoulder_lift"] < t3["shoulder_lift"])
    keys = {ord("w"): 1}
    t5, moved = apply_hold_controls(keys, t4, limits, 0.1)
    check("hold W/S elbow", moved and t5["elbow_flex"] > t4["elbow_flex"])
    return ok


def test_interact_source_clean():
    src = (ROOT / "sim" / "interact.py").read_text()
    ok = "setUserDebugParameter" not in src
    return check("no setUserDebugParameter", ok)


def test_urdf_loads():
    a = arm()
    ok = len(a.joint_indices) == len(JOINT_NAMES) and len(a.joint_limits()) == len(JOINT_NAMES)
    a.disconnect()
    return check("URDF loads 6 joints", ok)


def test_hold_home():
    home = load_config()["poses"]["home"]
    a = arm()
    a.reset_pose(home)
    for _ in range(60 if FAST else 120):
        a.hold(home)
        a.step(1 / 240)
    err = max(abs(a.joint_states()[n] - home[n]) for n in JOINT_NAMES)
    a.disconnect()
    return check("hold home no drift", err < 0.02, f"err={err:.4f}")


def test_each_joint_moves():
    a = arm()
    all_ok = True
    dur = 0.4 * motion_scale() if FAST else 1.5
    targets = {"shoulder_pan": 0.5, "shoulder_lift": -0.6, "elbow_flex": 0.8, "wrist_flex": -0.4, "wrist_roll": 0.5, "gripper": 0.3}
    for name in JOINT_NAMES:
        a.reset_pose({n: 0.0 for n in JOINT_NAMES})
        target = targets[name]
        ok = a.move_to({name: target}, duration=dur, tolerance=0.05)
        actual = a.joint_states()[name]
        joint_ok = ok and abs(actual - target) < 0.08
        if not check(f"move {name}", joint_ok, f"target={target:.2f} actual={actual:.2f}"):
            all_ok = False
    a.disconnect()
    return all_ok


def test_named_poses_and_sequence():
    a = arm()
    a.reset_pose()
    ok = run_sequence(a, ["reach", "home"], duration=0.8 if FAST else 1.0)
    a.disconnect()
    return check("pose sequence", ok)


def test_pose_save_load():
    from poses import POSES_FILE, delete_pose, load_saved

    had = "test_pose" in load_saved()
    save_pose("test_pose", {n: 0.1 * i for i, n in enumerate(JOINT_NAMES)})
    ok = "test_pose" in load_all()
    if not had:
        delete_pose("test_pose")
    return check("save/load pose", ok)


def test_board_math():
    cfg = load_config()
    f, r = parse_square("e4")
    ok = f == 4 and r == 4
    check("parse e4", ok)
    w = square_to_world("a1", cfg)
    ok2 = w[2] > 0 and w[0] > 0
    return check("square_to_world", ok2, str(w))


def test_board_reach():
    cfg = load_config()
    a = arm()
    a.reset_pose(cfg["poses"].get("above_board", cfg["poses"]["home"]))
    from board import reach_square

    ok = reach_square(a, "e4", cfg, duration=0.8 if FAST else 2.5)
    a.disconnect()
    return check("IK reach e4", ok)


def test_fen_spawn():
    from pieces import START_FEN, BoardPieces, parse_fen

    cfg = load_config()
    board = parse_fen(START_FEN)
    ok = len(board) == 32 and board["e2"] == ("white", "pawn")
    check("parse start FEN", ok, f"{len(board)} pieces")

    a = arm()
    pieces = BoardPieces.from_fen(cfg, START_FEN)
    a.disconnect()
    return check("spawn 32 pieces", len(pieces.by_square) == 32)


def test_e2e4_move():
    from board import add_board_visuals
    from pieces import BoardPieces
    from pick import execute_move, piece_near_square

    cfg = load_config()
    a = arm()
    add_board_visuals(cfg)
    a.reset_pose(cfg["poses"]["above_board"])
    pieces = BoardPieces(cfg=cfg)
    pieces.add("e2", "white", "pawn")
    dur = 1.0 if FAST else 2.0
    ok = execute_move(a, pieces, "e2e4", cfg, duration=dur)
    body = pieces.at("e4")
    placed = body is not None and piece_near_square(pieces, body, "e4", cfg)
    a.disconnect()
    return check("e2e4 pick/place", ok and placed)


def test_wrist_camera_aim():
    from board import add_board_visuals
    from camera import WristCamera

    cfg = load_config()
    a = arm()
    add_board_visuals(cfg)
    a.reset_pose(cfg["poses"]["above_board"])
    cam = WristCamera(a, cfg)
    rgb = cam.capture()
    a.disconnect()
    if rgb is None:
        return check("wrist camera aim", False, "capture failed")
    yellow = ((rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 150) & (rgb[:, :, 2] < 120)).mean()
    ok = yellow < 0.05 and float(rgb.std()) > 40.0
    return check("wrist camera aim", ok, f"yellow={yellow:.1%} std={rgb.std():.1f}")


def test_board_pose_fit():
    import numpy as np
    from board_pose import expected_marker_xyz, fit_pose

    cfg = load_config()
    expected = expected_marker_xyz(cfg)
    pts = np.array([expected[n] for n in ("a1", "h1", "a8", "h8")])
    pose = fit_pose(pts, cfg)
    origin = cfg["sim"]["board"]["origin"]
    dxy = ((pose.origin[0] - origin[0]) ** 2 + (pose.origin[1] - origin[1]) ** 2) ** 0.5
    dyaw = abs(pose.yaw - float(cfg["sim"]["board"].get("yaw", 0.0)))
    dsize = abs(pose.square_size - float(cfg["sim"]["board"]["square_size"]))
    ok = dxy < 0.003 and dyaw < 0.02 and dsize < 0.002
    return check("board pose fit", ok, f"dxy={dxy:.4f} dyaw={dyaw:.4f} dsize={dsize:.4f}")


def test_locate_board():
    from board import add_board_visuals
    from board_pose import locate
    from camera import WristCamera
    from pieces import START_FEN, BoardPieces

    cfg = load_config()
    prior = list(cfg["sim"]["board"]["origin"])
    prior_yaw = float(cfg["sim"]["board"].get("yaw", 0.0))
    prior_size = float(cfg["sim"]["board"]["square_size"])
    a = arm()
    add_board_visuals(cfg)
    BoardPieces.from_fen(cfg, START_FEN)
    a.reset_pose(cfg["poses"]["above_board"])
    cam = WristCamera(a, cfg)
    try:
        pose = locate(a, cam, cfg)
    except ValueError as e:
        a.disconnect()
        return check("locate board from red dots", False, str(e))
    a.disconnect()
    dxy = ((pose.origin[0] - prior[0]) ** 2 + (pose.origin[1] - prior[1]) ** 2) ** 0.5
    dyaw = abs(pose.yaw - prior_yaw)
    dsize = abs(pose.square_size - prior_size)
    ok = pose.n_views >= 1 and dxy < 0.02 and dyaw < 0.12 and dsize < 0.004
    return check(
        "locate board from red dots",
        ok,
        f"views={pose.n_views} dxy={dxy:.3f} dyaw={dyaw:.3f} dsize={dsize:.4f}",
    )


def test_board_voxels():
    import numpy as np
    from board import add_board_visuals
    from board_pose import locate
    from camera import WristCamera
    from pieces import START_FEN, BoardPieces
    from voxels import scan, table_z_board, voxelize

    # synthetic: a block on e2 in board frame
    cfg = load_config()
    size = float(cfg["sim"]["board"]["square_size"])
    rng = np.random.default_rng(0)
    xs = 4.5 * size + rng.uniform(-0.008, 0.008, 80)
    ys = 1.5 * size + rng.uniform(-0.008, 0.008, 80)
    zs = 0.03 + rng.uniform(0.0, 0.02, 80)
    local = np.stack([xs, ys, zs], axis=1)
    cols = np.full((80, 3), 40, dtype=np.uint8)
    occ, _ = voxelize(local, cols, 0.004, 8 * size, 0.02, 0.07)
    ok_synth = int(occ.sum()) > 5
    check("voxelize synthetic e2", ok_synth, f"occ={int(occ.sum())}")

    a = arm()
    add_board_visuals(cfg)
    BoardPieces.from_fen(cfg, START_FEN)
    a.reset_pose(cfg["poses"]["above_board"])
    cam = WristCamera(a, cfg)
    try:
        pose = locate(a, cam, cfg)
        grid = scan(a, cam, cfg, pose)
    except ValueError as e:
        a.disconnect()
        return check("board voxel scan", False, str(e))
    a.disconnect()
    z_clear = table_z_board(pose, cfg) + float(cfg.get("sim", {}).get("voxels", {}).get("z_clear", 0.008))
    occ8 = grid.occupancy_8x8(z_clear)
    back = int(occ8[:, 0:2].sum() + occ8[:, 6:8].sum())
    mid = int(occ8[:, 2:6].sum())
    ok = grid.n_views >= 3 and len(grid) > 80 and back >= 16 and mid <= 8
    return check(
        "board voxel scan",
        ok,
        f"views={grid.n_views} occ={len(grid)} shape={grid.shape} back={back} mid={mid}",
    )


def test_interpolate():
    a = {n: 0.0 for n in JOINT_NAMES}
    b = {n: 1.0 for n in JOINT_NAMES}
    mid = interpolate(a, b, 0.5)
    return check("interpolate mid", abs(mid["shoulder_pan"] - 0.5) < 0.01)


def main():
    mode = " (fast)" if FAST else ""
    print(f"Zima sim tests{mode}\n")
    results = [
        test_controls_pure(),
        test_interact_source_clean(),
        test_urdf_loads(),
        test_hold_home(),
    ]
    if not FAST:
        results.append(test_each_joint_moves())
    results += [
        test_named_poses_and_sequence(),
        test_pose_save_load(),
        test_board_math(),
        test_interpolate(),
        test_board_reach(),
        test_wrist_camera_aim(),
        test_board_pose_fit(),
        test_locate_board(),
        test_board_voxels(),
    ]
    if not FAST:
        results.append(test_fen_spawn())
    results.append(test_e2e4_move())
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} test groups passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
