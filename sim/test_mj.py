"""MuJoCo sim tests — python sim/test_mj.py [--quick]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from ik import solve_ik
from mj import DEBUG_DIR, JOINT_NAMES, Sim, load_config
from squares import square_to_world


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def test_load_and_render():
    sim = Sim()
    q, _ = sim.queen_pose()
    frames = []
    for cam in ("debug_overview", "debug_side", "debug_top", "wrist_cam"):
        path = sim.save_frame(cam, DEBUG_DIR / "smoke" / f"{cam}.png", width=480, height=360)
        frames.append(path)
    rgb = sim.render("debug_overview", 64, 48)
    sim.close()
    ok = q[2] > 0.015 and rgb.std() > 10
    return check("load + render", ok, f"queen_z={q[2]:.3f} frames={len(frames)}")


def test_joints_move():
    sim = Sim()
    sim.set_qpos({n: 0.0 for n in JOINT_NAMES})
    all_ok = True
    targets = {
        "shoulder_pan": 0.4,
        "shoulder_lift": -0.3,
        "elbow_flex": 0.5,
        "wrist_flex": 0.4,
        "wrist_roll": 0.5,
        "gripper": 0.6,
    }
    for name, target in targets.items():
        start = {n: 0.0 for n in JOINT_NAMES}
        sim.set_qpos(start)
        sim.hold(start)
        cmd = {**start, name: target}
        sim.set_ctrl(cmd)
        for _ in range(80):
            sim.step()
        actual = sim.qpos()[name]
        joint_ok = abs(actual - target) < 0.12
        if not check(f"move {name}", joint_ok, f"target={target:.2f} actual={actual:.2f}"):
            all_ok = False
    sim.close()
    return all_ok


def test_gripper_gap():
    sim = Sim()
    sim.set_qpos({**{n: 0.0 for n in JOINT_NAMES}, "gripper": -0.1})
    for _ in range(60):
        sim.step()
    closed = sim.jaw_gap()
    sim.set_ctrl({**sim.qpos(), "gripper": 0.8})
    for _ in range(100):
        sim.step()
    opened = sim.jaw_gap()
    sim.close()
    ok = opened > closed + 0.03
    return check("gripper opens", ok, f"closed={closed*1000:.1f}mm open={opened*1000:.1f}mm")


def test_arm_base_applied():
    import mujoco

    sim = Sim()
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    origin = [float(x) for x in sim.model.body_pos[bid]]
    sim.apply_arm_base([origin[0] + 0.08, origin[1] + 0.03, origin[2]])
    moved = [float(x) for x in sim.model.body_pos[bid]]
    sim.apply_arm_base(origin)
    restored = [float(x) for x in sim.model.body_pos[bid]]
    sim.close()
    ok = abs(moved[0] - origin[0] - 0.08) < 1e-9 and abs(restored[0] - origin[0]) < 1e-9
    return check("arm base position", ok, f"origin={origin[:2]} moved={moved[:2]}")


def test_ik_above_d1():
    sim = Sim()
    cfg = load_config()
    seed = cfg["poses"]["vertical_seed"]
    sim.set_qpos(seed)
    xy = square_to_world("d1", cfg)
    target = [xy[0], xy[1], 0.12]
    ok_solve = solve_ik(sim, target)
    pos, rot = sim.site_pose("gripperframe")
    err = float(((pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2 + (pos[2] - target[2]) ** 2) ** 0.5)
    down = float(rot[2, 0])
    sim.save_frame("debug_overview", DEBUG_DIR / "smoke" / "ik_d1.png")
    sim.save_frame("debug_side", DEBUG_DIR / "smoke" / "ik_d1_side.png")
    sim.close()
    ok = ok_solve and err < 0.008
    return check("IK above d1", ok, f"err={err*1000:.1f}mm  x_z={down:.2f}")


def test_wrist_localize():
    import numpy as np
    from ik import move_ik
    from perception import estimate_target_pose
    from pickup import open_gripper, seed_vertical, settle_queen
    from sensor import capture as capture_wrist

    cfg = load_config()
    sim = Sim(cfg=cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])
    xy = square_to_world("d1", cfg)
    sim.place_queen(square_to_world("d1", cfg, z=surface))
    seed_vertical(sim, cfg)
    open_gripper(sim, 0.45, steps=30)
    settle_queen(sim, 40)
    move_ik(sim, [xy[0], xy[1], 0.08], duration=0.6)
    obs = capture_wrist(sim)
    sim.save_frame("wrist_cam", DEBUG_DIR / "perc" / "localize_wrist.png", 320, 240)
    est = estimate_target_pose(obs, cfg, "d1")
    gt, _ = sim.queen_pose()
    sim.close()
    if est is None:
        return check("wrist localize queen", False, "no estimate")
    err = float(np.linalg.norm(est[:2] - gt[:2]))
    return check(
        "wrist localize queen",
        err < 0.005,
        f"xy_err={err*1000:.1f}mm est={np.round(est[:2], 4).tolist()} gt={np.round(gt[:2], 4).tolist()}",
    )


def test_unstack_policy():
    from sequence import (
        Move,
        UnstackFrame,
        choose_blocker,
        choose_park,
        forward_blockers,
        plan_unstack_move,
        restore_order,
    )

    occ = {"g1": "proxy_g1", "g2": "proxy_g2", "h2": "proxy_h2"}
    home = choose_blocker("h1", occ, {"h1", "d4"}, {"rook"})
    ok_pawn = check("unstack prefers pawn", home == "g2", f"home={home}")
    occupied = {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
        "d4",
    }
    park = choose_park({}, occupied)
    ok_park = check("unstack parks a quiet empty square", park == "f4", f"park={park}")
    home2 = choose_blocker("a2", {"a1": "proxy_a1", "b2": "proxy_b2"}, {"a2"}, {"pawn"})
    ok_skip = check("unstack skips same type", home2 == "a1", f"home={home2}")
    moves = [
        Move("pawn", "g2", "d3"),
        Move("knight", "g1", "e3"),
    ]
    rest = restore_order(moves)
    ok_lifo = check(
        "unstack LIFO restore list",
        rest[0].src == "e3" and rest[0].dst == "g1" and rest[1].src == "d3" and rest[1].dst == "g2",
        f"rest={[ (m.kind, m.src, m.dst) for m in rest ]}",
    )
    planned = plan_unstack_move("h1", "rook", occ, {"h1", "d4"}, {"rook"}, occupied)
    ok_plan = check(
        "plan_unstack_move parks pawn",
        planned is not None and planned.kind == "pawn" and planned.src == "g2" and planned.dst == "f4",
        f"move={planned}",
    )
    stack = [
        UnstackFrame("pawn", "g2", "d3", [0.0, 0.0, 0.0], True),
        UnstackFrame("knight", "g1", "e3", [0.0, 0.0, 0.0], True),
    ]
    last = stack.pop()
    ok_frame = check("unstack LIFO restore", last.kind == "knight" and last.home == "g1", last.home)
    front = forward_blockers("g1", {"g2": "proxy_g2", "f1": "proxy_f1", "h2": "proxy_h2"}, {"knight"})
    ok_front = check("forward blockers are center-rank pawns", set(front) == {"g2", "h2"}, f"front={front}")
    none = forward_blockers("a2", {"a1": "proxy_a1", "b2": "proxy_b2"}, {"pawn"})
    ok_none = check("forward blockers skip same type", none == [], f"front={none}")
    from pickup import bowl_contains

    ok_bowl = check(
        "bowl drop is not a chess square",
        bowl_contains([0.35, -0.26, 0.024], [0.35, -0.26, 0.024])
        and not bowl_contains([0.351, 0.006, 0.02], [0.35, -0.26, 0.024]),
    )
    return ok_pawn and ok_park and ok_skip and ok_lifo and ok_plan and ok_frame and ok_front and ok_none and ok_bowl


def test_home_clears_pieces():
    import numpy as np
    from ik import EE_SITE, solve_ik
    from pickup import FULL_BOARD_XML, queen_upright, seed_vertical
    from squares import bowl_xyz

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    seed_vertical(sim, cfg, snap=True)
    q0, _ = sim.object_pose("queen_d1")
    b = bowl_xyz(cfg)
    ok_ik = solve_ik(sim, [b[0], b[1], 0.04])
    sim.hold(sim.qpos())
    for _ in range(16):
        sim.step()
    seed_vertical(sim, cfg, snap=False)
    q1, quat1 = sim.object_pose("queen_d1")
    gpos, _ = sim.site_pose(EE_SITE)
    drift = float(np.linalg.norm(q1[:2] - q0[:2]))
    sim.close()
    return check(
        "home path clears the queen",
        bool(ok_ik) and drift < 0.008 and queen_upright(quat1),
        f"drift={drift * 1000:.1f}mm ee_z={gpos[2]:.3f}",
    )


def test_cart_speed_near_far():
    import numpy as np
    from ik import FAR_VEL, NEAR_M, NEAR_VEL, _path_time

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.20, 0.0, 0.0])
    t_cruise = _path_time(a, b, np.array([1.0, 0.0, 0.0]), False)
    t_near = _path_time(a, np.array([0.03, 0.0, 0.0]), np.array([0.03, 0.0, 0.0]), False)
    expect_cruise = 0.20 / FAR_VEL
    ok = (
        abs(t_cruise - expect_cruise) < 0.04
        and t_near > 0.03 / NEAR_VEL * 0.7
        and t_cruise < t_near * (0.20 / 0.03)
        and NEAR_M == 0.0508
    )
    return check(
        "cruise faster than last 2 inches",
        ok,
        f"t_20cm_far={t_cruise:.2f}s t_3cm_near={t_near:.2f}s",
    )


def test_column_descend_tracks_xy():
    import numpy as np
    from ik import EE_SITE, servo_lin, solve_ik
    from pickup import seed_vertical

    cfg = load_config()
    sim = Sim(cfg=cfg)
    seed_vertical(sim, cfg, snap=True)
    xy = square_to_world("d4", cfg)
    high = [xy[0], xy[1], 0.12]
    low = [xy[0], xy[1], 0.05]
    ok_ik = solve_ik(sim, high, lock_joints=("wrist_flex",))
    drift = [0.0]
    z0 = [None]
    prev = sim.on_step

    def _hook():
        if prev:
            prev()
        ee, _ = sim.site_pose(EE_SITE)
        if z0[0] is None:
            z0[0] = float(ee[2])
        if abs(float(ee[2]) - z0[0]) >= 0.003:
            drift[0] = max(drift[0], float(np.hypot(ee[0] - xy[0], ee[1] - xy[1])))

    sim.on_step = _hook
    ok_move = servo_lin(
        sim, low, physics=True, lock_joints=("wrist_flex",), column=True, ref=low
    )
    ee, _ = sim.site_pose(EE_SITE)
    z_err = abs(float(ee[2]) - 0.05)
    sim.close()
    ok = bool(ok_ik) and bool(ok_move) and drift[0] < 0.006 and z_err < 0.015
    return check(
        "column descend holds XY",
        ok,
        f"xy_drift={drift[0]*1000:.1f}mm z_err={z_err*1000:.1f}mm",
    )


def main():
    quick = "--quick" in sys.argv
    print("Zima MuJoCo tests" + (" (quick)" if quick else "") + "\n")
    results = [
        test_load_and_render(),
        test_joints_move(),
        test_gripper_gap(),
        test_arm_base_applied(),
        test_ik_above_d1(),
        test_unstack_policy(),
        test_home_clears_pieces(),
        test_cart_speed_near_far(),
        test_column_descend_tracks_xy(),
    ]
    if not quick:
        results.append(test_wrist_localize())
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
