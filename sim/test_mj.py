"""MuJoCo sim tests — python sim/test_mj.py [--quick]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from ik import pose_reaches, reachable_approaches, solve_ik
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
        sim.pose_arm_for_setup(start)
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


def _pose_by_ik(sim, target, **kw) -> bool:
    """Test setup: start the arm at an IK pose (placed before the episode, not moved there)."""
    with sim.probe():
        ok = solve_ik(sim, target, **kw)
        q = sim.qpos()
    sim.pose_arm_for_setup(q)
    return ok


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


def test_look_fits_piece_shape():
    """Post-release look on the real wall: an off-centre knight is located at its base, not its centroid."""
    import numpy as np
    import pickup
    from robot import open_robot
    from scenario import HIDE_XYZ, PIECE_BODIES

    cfg = load_config()
    surface = float(cfg["sim"]["board"]["surface_z"])
    world = Sim(xml_path=pickup.PIECES_XML, cfg=cfg)
    for n in PIECE_BODIES:
        if n != "knight":
            world.place_object(n, HIDE_XYZ)
    c = square_to_world("f3", cfg, z=surface)
    world.place_object("knight", [c[0] - 0.008, c[1] + 0.006, c[2]])
    world.step(60)
    sim = open_robot(world, "real")
    sim.quiet = True
    sim.target_id = "knight"
    pickup.seed_vertical(sim, cfg)
    look = pickup.verify_piece(sim, cfg, "knight", "f3")
    truth = world.object_pose("knight")[0]
    sim.close()
    world.close()
    if not look.get("seen"):
        return check("look fits piece shape", False, "not seen")
    err = float(np.hypot(look["xy"][0] - truth[0], look["xy"][1] - truth[1]))
    return check("look fits piece shape", bool(look.get("fitted")) and err < 0.0005, f"xy_err={err*1000:.2f}mm fitted={look.get('fitted')}")


def test_unstack_policy():
    from sequence import UnstackFrame
    from stages.unstack import choose_blocker, choose_park, plan_unstack_move

    occ = {"g1": "knight_g1", "g2": "pawn_g2", "h2": "pawn_h2"}
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
    home2 = choose_blocker("a2", {"a1": "rook_a1", "b2": "pawn_b2"}, {"a2"}, {"pawn"})
    ok_skip = check("unstack skips same type", home2 == "a1", f"home={home2}")
    planned = plan_unstack_move("h1", "rook", occ, {"h1", "d4"}, {"rook"}, occupied)
    ok_plan = check(
        "plan_unstack_move parks pawn",
        planned is not None and planned.kind == "pawn" and planned.src == "g2" and planned.dst == "f4",
        f"move={planned}",
    )
    stack = [
        UnstackFrame("pawn", "g2", "d3"),
        UnstackFrame("knight", "g1", "e3"),
    ]
    last = stack.pop()
    ok_frame = check("unstack LIFO restore", last.kind == "knight" and last.home == "g1", last.home)
    from pickup import bowl_contains, in_trash, in_well

    ok_bowl = check(
        "bowl drop is not a chess square",
        bowl_contains([0.35, -0.26, 0.024], [0.35, -0.26, 0.024])
        and not bowl_contains([0.351, 0.006, 0.02], [0.35, -0.26, 0.024]),
    )
    cfg = load_config()
    ok_trash = check(
        "bowl occupants are trash, not board collateral",
        in_trash([0.35, -0.26, 0.024], cfg) and not in_trash([0.351, 0.006, 0.02], cfg),
    )
    ok_well = check(
        "well occupants are off-board, not board collateral",
        in_well([0.22, -0.26, 0.024], cfg) and not in_well([0.351, 0.006, 0.02], cfg),
    )
    return ok_pawn and ok_park and ok_skip and ok_plan and ok_frame and ok_bowl and ok_trash and ok_well


def test_home_clears_pieces():
    import numpy as np
    from ik import EE_SITE
    from pickup import FULL_BOARD_XML, queen_upright, seed_vertical
    from squares import bowl_xyz

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    seed_vertical(sim, cfg)
    q0, _ = sim.object_pose("queen_d1")
    b = bowl_xyz(cfg)
    ok_ik = _pose_by_ik(sim, [b[0], b[1], 0.04])
    sim.hold(sim.qpos())
    for _ in range(16):
        sim.step()
    seed_vertical(sim, cfg)
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


def test_reachable_approach_near_base():
    """Near-base hover is dead; pinch and a lower/outward approach are not."""
    import numpy as np
    from pickup import (
        find_pregrasp_pose,
        ray_approach_candidates,
        seed_vertical,
    )
    from stages.recovery import _scoop_axis

    cfg = load_config()
    sim = Sim(cfg=cfg)
    seed_vertical(sim, cfg)
    a5 = square_to_world("a5", cfg)
    d4 = square_to_world("d4", cfg)
    pre_a5 = find_pregrasp_pose(sim, cfg, a5)
    pre_d4 = find_pregrasp_pose(sim, cfg, d4)
    b5 = square_to_world("b5", cfg)
    pinch = [float(b5[0]), float(b5[1]), 0.030]
    hits = []
    pinch_ok = False
    for sign in (1.0, -1.0):
        axis = _scoop_axis(np.array([1.0, 0.0]), sign)
        ikw = dict(lock_joints=("wrist_roll",), x_axis=axis, ori_weight=0.45, ori_steps=8)
        pinch_ok = pose_reaches(sim, pinch, **ikw)
        if not pinch_ok:
            continue
        hits = reachable_approaches(sim, ray_approach_candidates(pinch, axis[:2]), pinch, **ikw)
        if hits:
            break
    far_hits = reachable_approaches(sim, [[0.90, 0.90, 0.11]], [0.90, 0.90, 0.03])
    sim.close()
    mid_high = pre_d4 is not None and float(pre_d4[2]) >= 0.10
    ok = pre_a5 is not None and mid_high and pinch_ok and len(hits) > 0 and far_hits == []
    return check(
        "approach search near base",
        ok,
        f"a5={None if pre_a5 is None else [round(v, 3) for v in pre_a5]} "
        f"d4_z={None if pre_d4 is None else round(pre_d4[2], 3)} "
        f"b5_pinch={pinch_ok} b5_hits={len(hits)} far={len(far_hits)}",
    )


def test_column_descend_tracks_xy():
    """The live descend: `_descend_column` plans a plumb chain straight down; following it holds XY."""
    import numpy as np
    from ik import EE_SITE, move_ik
    from pickup import seed_vertical
    from stages.approach import _descend_column, _follow_q_path

    cfg = load_config()
    sim = Sim(cfg=cfg)
    seed_vertical(sim, cfg)
    xy = square_to_world("d4", cfg)
    high = [xy[0], xy[1], 0.08]
    low = [xy[0], xy[1], 0.03]
    ok_ik = move_ik(sim, high, duration=1.0, physics=True, down=True)
    down = _descend_column(sim, low, ())
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
    ok_move = down is not None and _follow_q_path(sim, down, 1.2)
    ee, _ = sim.site_pose(EE_SITE)
    z_err = abs(float(ee[2]) - low[2])
    sim.close()
    ok = bool(ok_ik) and bool(ok_move) and drift[0] < 0.006 and z_err < 0.015
    return check(
        "column descend holds XY",
        ok,
        f"planned={down is not None} xy_drift={drift[0]*1000:.1f}mm z_err={z_err*1000:.1f}mm",
    )


def test_nest_stands_fallen():
    """Scoop a fallen man into the cup; same path for every piece."""
    import numpy as np
    from pickup import (
        FULL_BOARD_XML,
        in_well,
        queen_tilt_deg,
        queen_upright,
        seed_vertical,
    )
    from stages.recovery import recover_to_nest
    from squares import nest_xyz

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    seed_vertical(sim, cfg)
    xy = square_to_world("d4", cfg, z=0.028)
    half = 0.5 * np.pi
    quat = (float(np.cos(half / 2.0)), 0.0, float(np.sin(half / 2.0)), 0.0)
    sim.place_object("knight_g8", [xy[0], xy[1], 0.028], quat)
    sim.target_id = "knight_g8"
    for _ in range(80):
        sim.step()
    rec = recover_to_nest(sim, cfg, "knight_g8")
    qpos, q = sim.object_pose("knight_g8")
    nest = nest_xyz(cfg)
    dist = float(np.hypot(qpos[0] - nest[0], qpos[1] - nest[1]))
    sim.close()
    ok = bool(rec.get("ok")) and queen_upright(q, 0.55) and in_well(qpos, cfg) and dist < 0.02
    return check(
        "nest cup stands a fallen knight",
        ok,
        f"ok={rec.get('ok')} reason={rec.get('reason')} tilt={rec.get('tilt_deg')} "
        f"xy_mm={dist * 1000:.1f}",
    )


def test_place_b5_no_collateral():
    """One Opera-like place: b7→b5 with a pawn on c6. No full game.

    The old 190 mm carry floor parked the piece on the ring (~c5) and
    scraped in at table height, knocking c6.
    """
    import numpy as np
    from pickup import FULL_BOARD_XML, execute_move, queen_upright, seed_vertical

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    sim.quiet = True
    seed_vertical(sim, cfg)
    sim.place_object("pawn_c7", square_to_world("c6", cfg, z=0.02))
    for _ in range(40):
        sim.step()
    c6_0, _ = sim.object_pose("pawn_c7")
    out = execute_move(
        sim,
        cfg,
        use_vision=False,
        scenario={
            "target_id": "pawn_b7",
            "source": "b7",
            "dest": "b5",
            "perception": "gt",
            "quiet": True,
        },
    )
    c6_1, _ = sim.object_pose("pawn_c7")
    q, quat = sim.object_pose("pawn_b7")
    dest = square_to_world("b5", cfg)
    err = float(np.hypot(q[0] - dest[0], q[1] - dest[1]) * 1000)
    c6_mm = float(np.hypot(c6_1[0] - c6_0[0], c6_1[1] - c6_0[1]) * 1000)
    sim.close()
    coll = out.get("collateral") or []
    ok = (
        bool(out.get("success"))
        and not coll
        and queen_upright(quat, 0.5)
        and err < 18.0
        and c6_mm < 8.0
    )
    return check(
        "b7-b5 place does not knock c6",
        ok,
        f"ok={out.get('success')} reason={out.get('failure_reason')} "
        f"err={err:.1f}mm c6={c6_mm:.1f}mm coll={coll} "
        f"standoff={out.get('transport_standoff')}",
    )


def test_nxb5_capture():
    """Opera Nxb5 as one capture: dump b5, land the knight, leave c6/c4/b3."""
    import numpy as np
    from episode import play_move
    from pickup import FULL_BOARD_XML, queen_upright, seed_vertical

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    sim.quiet = True
    seed_vertical(sim, cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])

    def put(body, sq):
        sim.place_object(body, square_to_world(sq, cfg, z=surface))

    put("pawn_b7", "b5")
    put("knight_b1", "c3")
    put("pawn_c7", "c6")
    put("bishop_c1", "c4")
    put("queen_d1", "b3")
    for _ in range(40):
        sim.step()
    nb0 = {n: sim.object_pose(n)[0][:2].copy() for n in ("pawn_c7", "bishop_c1", "queen_d1")}
    out = play_move(sim, cfg, "c3 b5", use_vision=False)
    kn, kq = sim.object_pose("knight_b1")
    dest = square_to_world("b5", cfg)
    err = float(np.hypot(kn[0] - dest[0], kn[1] - dest[1]) * 1000)

    def disp(name):
        p, _ = sim.object_pose(name)
        return float(np.hypot(p[0] - nb0[name][0], p[1] - nb0[name][1]) * 1000)

    c6, c4, b3 = disp("pawn_c7"), disp("bishop_c1"), disp("queen_d1")
    upright = queen_upright(kq, 0.5)
    sim.close()
    ok = (
        bool(out.get("success"))
        and not (out.get("collateral") or [])
        and upright
        and err < 18.0
        and c6 < 8.0
        and c4 < 8.0
        and b3 < 8.0
    )
    return check(
        "Nxb5 capture does not knock queenside",
        ok,
        f"ok={out.get('success')} reason={out.get('failure_reason')} "
        f"stage={out.get('failure_stage')} err={err:.1f}mm "
        f"c6={c6:.1f} c4={c4:.1f} b3={b3:.1f}",
    )


def _queenside_capture(move: str, attacker: str, src: str, label: str):
    """One Opera capture on b5 with c4/b3 still occupied. Not a full game."""
    import numpy as np
    from episode import play_move
    from pickup import FULL_BOARD_XML, queen_upright, seed_vertical

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    sim.quiet = True
    seed_vertical(sim, cfg)
    surface = float(cfg["sim"]["board"]["surface_z"])

    def put(body, sq):
        sim.place_object(body, square_to_world(sq, cfg, z=surface))

    put("pawn_b7", "b5")
    put(attacker, src)
    put("pawn_c7", "c6")
    put("bishop_c1", "c4")
    put("queen_d1", "b3")
    for _ in range(40):
        sim.step()
    neighbors = ("pawn_c7", "bishop_c1", "queen_d1")
    # The attacker itself is not a neighbor once it leaves src.
    neighbors = tuple(n for n in neighbors if n != attacker)
    nb0 = {n: sim.object_pose(n)[0][:2].copy() for n in neighbors}
    out = play_move(sim, cfg, move, use_vision=False)
    kn, kq = sim.object_pose(attacker)
    dest = square_to_world("b5", cfg)
    err = float(np.hypot(kn[0] - dest[0], kn[1] - dest[1]) * 1000)

    def disp(name):
        p, _ = sim.object_pose(name)
        return float(np.hypot(p[0] - nb0[name][0], p[1] - nb0[name][1]) * 1000)

    dmm = {n: disp(n) for n in neighbors}
    upright = queen_upright(kq, 0.5)
    sim.close()
    ok = (
        bool(out.get("success"))
        and not (out.get("collateral") or [])
        and upright
        and err < 18.0
        and all(v < 8.0 for v in dmm.values())
    )
    extra = " ".join(f"{n.split('_')[0]}={v:.1f}" for n, v in dmm.items())
    return check(
        label,
        ok,
        f"ok={out.get('success')} reason={out.get('failure_reason')} "
        f"stage={out.get('failure_stage')} err={err:.1f}mm {extra}",
    )


def test_bxb5_capture():
    """Opera Bxb5+: bishop from c4, dump b5, leave c6 and Qb3."""
    return _queenside_capture("c4 b5", "bishop_c1", "c4", "Bxb5 capture does not knock queenside")


def test_cxb5_capture():
    """Opera cxb5: pawn from c6, dump b5, leave bishop c4 and Qb3."""
    return _queenside_capture("c6 b5", "pawn_c7", "c6", "cxb5 capture does not knock queenside")


def test_queen_near_base_place():
    """b3 pinch is reachable; place on the square, not the 190 mm ring."""
    from pickup import FULL_BOARD_XML, execute_move, queen_upright, seed_vertical

    cfg = load_config()
    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    sim.quiet = True
    seed_vertical(sim, cfg)
    sim.place_object("queen_d1", square_to_world("f3", cfg, z=0.02))
    for _ in range(40):
        sim.step()
    out = execute_move(
        sim,
        cfg,
        use_vision=False,
        scenario={
            "target_id": "queen_d1",
            "source": "f3",
            "dest": "b3",
            "perception": "gt",
            "quiet": True,
        },
    )
    _, quat = sim.object_pose("queen_d1")
    err = float(out.get("final_position_error_mm") or 99.0)
    sim.close()
    ok = bool(out.get("success")) and queen_upright(quat, 0.5) and err < 18.0
    return check(
        "queen f3-b3 near-base place",
        ok,
        f"ok={out.get('success')} reason={out.get('failure_reason')} err={err:.1f}mm",
    )


def test_world_is_physics_only():
    """Once a robot is open, only physics moves the world: the controller cannot edit it,
    and a piece or arm written outside physics fails the next step."""
    import numpy as np
    from pickup import FULL_BOARD_XML
    from robot import WorldWrite, open_robot

    cfg = load_config()
    world = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    world.place_object("pawn_e2", square_to_world("e3", cfg, z=0.02))
    world.step(10)
    sim = open_robot(world, "real")
    refused = []
    for call in (lambda: sim.place_object("pawn_e3", [0.3, 0.0, 0.02]), lambda: sim.apply_arm_base([0, 0, 0]), sim.reset):
        try:
            call()
        except WorldWrite:
            refused.append(True)
    try:
        world.place_object("pawn_d2", square_to_world("d3", cfg, z=0.02))
        locked = False
    except RuntimeError:
        locked = True
    sim.step(5)
    adr = int(world.model.jnt_qposadr[world.model.joint("pawn_d2_free").id])
    world.data.qpos[adr] += 0.01
    try:
        sim.step()
        caught = False
    except RuntimeError as e:
        caught = "pawn_d2" in str(e)
    world.data.qpos[adr] -= 0.01
    with world.scene_setup():
        world.place_object("pawn_d2", square_to_world("d4", cfg, z=0.02))
    sim.step()
    moved = float(np.hypot(*(world.body_pose("pawn_d2")[0][:2] - np.asarray(square_to_world("d4", cfg)[:2]))))
    sim.close()
    ok = len(refused) == 3 and locked and caught and moved < 0.002
    return check("world is physics-only after open_robot", ok, f"refused={len(refused)}/3 locked={locked} teleport_caught={caught}")


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
        test_reachable_approach_near_base(),
        test_column_descend_tracks_xy(),
        test_world_is_physics_only(),
    ]
    if not quick:
        results.append(test_wrist_localize())
        results.append(test_look_fits_piece_shape())
        results.append(test_nest_stands_fallen())
        results.append(test_place_b5_no_collateral())
        results.append(test_nxb5_capture())
        results.append(test_bxb5_capture())
        results.append(test_cxb5_capture())
        results.append(test_queen_near_base_place())
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
