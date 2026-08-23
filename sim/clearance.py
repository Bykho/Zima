"""Pickup approach clearance from occupancy + gripper envelope.

No square-name special cases, no motion planner. Camera/perception unused.
When occupied neighbors sit along the open-jaw axis, the approach aperture
is the smallest change that keeps those fingers out of neighbor columns.
Grasp EE XY, close target, and empty-board opening stay frozen.
"""

from __future__ import annotations

import numpy as np

from ik import solve_ik
from squares import shift_square, square_to_world

JAW_BODIES = ("gripper", "moving_jaw_so101_v1")
CAMERA_BODIES = ("camera_mount",)
_MARGIN = 0.002
_GRIPPER_SAMPLES = 12


def neighbor_squares(source: str) -> list[str]:
    out = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            sq = shift_square(source, df, dr)
            if sq:
                out.append(sq)
    return out


def occupied_neighbor_bodies(sim, cfg: dict, source: str, scenario: dict | None = None) -> dict[str, str]:
    """Map occupied neighbor squares of `source` to a collision body name."""
    import mujoco

    want = set(neighbor_squares(source))
    found: dict[str, str] = {}
    for nb in (scenario or {}).get("neighbors") or []:
        sq = nb.get("square")
        if sq in want:
            found[sq] = nb.get("id", "pawn")
    for i in range(sim.model.nbody):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or not name.startswith("proxy_"):
            continue
        sq = name[6:]
        if sq in want and float(sim.data.xpos[i][2]) > 0.0:
            found[sq] = name
    return found


def _gaps_to(sim, bodies: list[str]) -> list[float]:
    import mujoco

    def _body_geoms(name: str) -> list[int]:
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return []
        return [i for i in range(sim.model.ngeom) if int(sim.model.geom_bodyid[i]) == bid]

    jaw_geoms: list[int] = []
    for name in JAW_BODIES + CAMERA_BODIES:
        jaw_geoms.extend(_body_geoms(name))
    fromto = np.zeros(6)
    gaps = []
    for name in bodies:
        other = _body_geoms(name)
        best = 1.0
        for i in jaw_geoms:
            for j in other:
                d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.12, fromto)
                if d < best:
                    best = d
        gaps.append(float(best))
    return gaps


def finger_sweep_bodies(sim, cfg: dict, source: str, occ: dict[str, str]) -> list[str]:
    """Occupied neighbors that lie along the open-jaw axis (not the palm)."""
    if not occ:
        return []
    size = float(cfg.get("sim", {}).get("board", {}).get("square_size", 0.0375))
    src = np.asarray(square_to_world(source, cfg)[:2], dtype=float)
    fix = sim.geom_xpos("fixed_jaw_sph_tip1")[:2]
    mov = sim.geom_xpos("moving_jaw_sph_tip1")[:2]
    od = mov - fix
    n = float(np.linalg.norm(od)) + 1e-12
    od = od / n
    out = []
    for sq, body in occ.items():
        v = np.asarray(square_to_world(sq, cfg)[:2], dtype=float) - src
        if float(np.dot(v, od)) > 0.35 * size:
            out.append(body)
    return out


def choose_approach_gripper(
    sim, grasp_xyz, sweep_bodies: list[str], open_g: float, min_g: float = 0.0
) -> float | None:
    """Widest approach opening whose fingers clear sweep_bodies at the grasp pose."""
    if not sweep_bodies:
        return None
    start = sim.qpos()
    lo = float(min_g)
    hi = float(open_g)
    samples = np.unique(np.concatenate(([hi], np.linspace(lo, hi, _GRIPPER_SAMPLES))))
    samples.sort()
    best = None
    best_gap = -1e9
    widest_clear = None
    for g in samples:
        sim.set_qpos(start, hold=True)
        q = sim.qpos()
        q["gripper"] = float(g)
        sim.set_qpos(q, hold=True)
        if not solve_ik(sim, grasp_xyz):
            continue
        gap = min(_gaps_to(sim, sweep_bodies)) if sweep_bodies else 1.0
        if gap > best_gap:
            best_gap = gap
            best = float(g)
        if gap >= _MARGIN:
            widest_clear = float(g)
    sim.set_qpos(start, hold=True)
    if widest_clear is not None:
        if abs(widest_clear - hi) < 0.02:
            return None
        return widest_clear
    if best is None or abs(best - hi) < 0.02:
        return None
    return best


def apply_approach_gripper(sim, hover_xyz, gripper: float) -> bool:
    """Set approach aperture at hover, keeping EE on the hover point."""
    q = sim.qpos()
    q["gripper"] = float(gripper)
    sim.set_qpos(q, hold=True)
    sim.set_ctrl({**sim.qpos(), "gripper": float(gripper)})
    return bool(solve_ik(sim, hover_xyz))


# Clutter-aware approach: small candidate set around the existing grasp.
# Grasp XY is frozen. No camera, no RRT. Wrist-roll + aperture only.
APPROACH_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount", "wrist")
_ROLL_DELTAS = (0.35, -0.35, 0.70, -0.70, 1.05, -1.05, 1.57, -1.57, 2.10, -2.10)
_DESCEND_U = (0.30, 0.55, 0.80)
_APPROACH_MARGIN = 0.0005
_MIN_APPROACH_G = 0.12


# --- transport: lift above the board envelope, then translate in XY ---

TRANSIT_MARGIN = 0.010
CARRY_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount")
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")


def _collision_geoms(sim, body: str) -> list[int]:
    import mujoco

    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return []
    out = []
    for i in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[i]) != bid:
            continue
        if int(sim.model.geom_contype[i]) == 0 and int(sim.model.geom_conaffinity[i]) == 0:
            continue
        if int(sim.model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        out.append(i)
    return out


def _geom_z_range(sim, gid: int) -> tuple[float, float]:
    import mujoco

    pos = np.asarray(sim.data.geom_xpos[gid], dtype=float)
    mat = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
    sz = sim.model.geom_size[gid]
    t = int(sim.model.geom_type[gid])
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        r = float(sz[0])
        return float(pos[2] - r), float(pos[2] + r)
    if t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        axis = mat[:, 2]
        r, h = float(sz[0]), float(sz[1])
        az = float(axis[2])
        radial = float(np.sqrt(max(0.0, 1.0 - az * az))) * r
        zc = float(pos[2])
        return zc - abs(az) * h - radial, zc + abs(az) * h + radial
    if t == G.mjGEOM_BOX:
        hx, hy, hz = (float(sz[0]), float(sz[1]), float(sz[2]))
        zmin, zmax = 1e9, -1e9
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for szz in (-hz, hz):
                    z = float((pos + mat @ np.array([sx, sy, szz]))[2])
                    zmin = min(zmin, z)
                    zmax = max(zmax, z)
        return zmin, zmax
    if t == G.mjGEOM_MESH:
        mid = int(sim.model.geom_dataid[gid])
        adr = int(sim.model.mesh_vertadr[mid]) if mid >= 0 else -1
        n = int(sim.model.mesh_vertnum[mid]) if mid >= 0 else 0
        if n > 0 and adr >= 0:
            verts = np.asarray(sim.model.mesh_vert[adr : adr + n], dtype=float).reshape(-1, 3)
            zs = verts @ mat.T[:, 2] + pos[2]
            return float(zs.min()), float(zs.max())
        r = float(sim.model.geom_rbound[gid])
        return float(pos[2] - r), float(pos[2] + r)
    r = float(sim.model.geom_rbound[gid])
    return float(pos[2] - r), float(pos[2] + r)


def _robot_body_ids(sim) -> set[int]:
    import mujoco

    seed = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    ids: set[int] = set()
    p = int(seed)
    while p >= 0:
        ids.add(p)
        parent = int(sim.model.body_parentid[p])
        if parent == p:
            break
        p = parent
    for i in range(sim.model.nbody):
        p = i
        seen = set()
        while p > 0 and p not in seen:
            if p in ids:
                ids.add(i)
                break
            seen.add(p)
            p = int(sim.model.body_parentid[p])
    return ids


def obstacle_bodies(sim, skip: set[str] | None = None) -> list[str]:
    """Collision bodies on the board that are not the robot or the carried target."""
    import mujoco

    skip = set(skip or ())
    robot = _robot_body_ids(sim)
    out = []
    for i in range(sim.model.nbody):
        if i in robot:
            continue
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or name in skip or name in ("world", "floor"):
            continue
        if float(sim.data.xpos[i][2]) < 0.015:
            continue
        xy = sim.data.xpos[i][:2]
        board_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "board")
        if board_id >= 0:
            c = sim.data.xpos[board_id][:2]
            if abs(float(xy[0] - c[0])) > 0.16 or abs(float(xy[1] - c[1])) > 0.16:
                continue
        geoms = _collision_geoms(sim, name)
        if not geoms:
            continue
        zmax = max(_geom_z_range(sim, g)[1] for g in geoms)
        if zmax <= 0.0:
            continue
        out.append(name)
    return out


def board_top_z(sim, obstacles: list[str]) -> float:
    """Highest world-Z of obstacle collision geometry (piece/proxy/board envelope)."""
    top = 0.0
    for name in obstacles:
        for gid in _collision_geoms(sim, name):
            top = max(top, _geom_z_range(sim, gid)[1])
    return float(top)


def carry_zmin(sim, target_id: str) -> float:
    """Lowest world-Z of carried object + gripper/palm/camera collision envelope."""
    zmin = 1e9
    for name in CARRY_BODIES + (target_id,):
        for gid in _collision_geoms(sim, name):
            zmin = min(zmin, _geom_z_range(sim, gid)[0])
    return float(zmin)


def safe_transit_ee_z(sim, ee_z: float, target_id: str, obstacles: list[str], margin: float = TRANSIT_MARGIN):
    """EE height that puts the carry envelope above the board envelope + margin."""
    top = board_top_z(sim, obstacles)
    bottom = carry_zmin(sim, target_id)
    hang = float(ee_z) - bottom
    needed = top + hang + float(margin)
    z = max(float(ee_z), needed)
    return z, {
        "board_top_z": top,
        "carry_zmin": bottom,
        "hang_below_ee": hang,
        "margin": float(margin),
        "ee_z": z,
    }


def _min_gap(sim, a_bodies: tuple[str, ...], b_bodies: list[str]) -> float:
    import mujoco

    fromto = np.zeros(6)
    best = 1.0
    a_geoms: list[int] = []
    for name in a_bodies:
        a_geoms.extend(_collision_geoms(sim, name))
    for other in b_bodies:
        b_geoms = _collision_geoms(sim, other)
        for i in a_geoms:
            for j in b_geoms:
                d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.20, fromto)
                if d < best:
                    best = d
    return float(best)


def transit_clearances(sim, target_id: str, obstacles: list[str]) -> dict[str, float]:
    pieces = [n for n in obstacles if n != "board"]
    if not pieces:
        return {"queen_mm": 1e6, "gripper_mm": 1e6, "camera_mm": 1e6}
    return {
        "queen_mm": _min_gap(sim, (target_id,), pieces) * 1000.0,
        "gripper_mm": _min_gap(sim, GRIPPER_BODIES, pieces) * 1000.0,
        "camera_mm": _min_gap(sim, CAMERA_BODIES, pieces) * 1000.0,
    }


_CARRY_CONTACT = set(CARRY_BODIES + GRIPPER_BODIES + CAMERA_BODIES)


def transit_nontarget_hit(sim, target_id: str, obstacles: list[str]) -> dict | None:
    """Robot/carried-object contact with a board piece — ignore world/board seating."""
    skip = {target_id, "board", "floor", "world", None}
    robot = _robot_body_ids(sim)
    import mujoco

    def _is_robot(name: str | None) -> bool:
        if not name:
            return False
        if name in _CARRY_CONTACT or name == target_id:
            return True
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return bid >= 0 and int(bid) in robot

    for name in obstacles:
        if name in skip:
            continue
        for h in sim.contacts_with(name):
            other = h.get("other")
            if other in skip or not _is_robot(other):
                continue
            return {
                "body": name,
                "other": other,
                "dist_mm": float(h["dist"] * 1000),
                "camera_involved": other in CAMERA_BODIES
                or "camera" in str(other)
                or str(other).startswith("d405"),
            }
    return None


def neighbor_approach_gap(sim, neighbors: list[str]) -> float:
    """Min gap from gripper/palm/wrist/camera to neighboring pieces (not the target)."""
    if not neighbors:
        return 1.0
    return _min_gap(sim, APPROACH_BODIES, neighbors)


def _clip_roll(sim, value: float) -> float:
    lo, hi = sim.model.jnt_range[sim._joint["wrist_roll"]]
    return float(np.clip(value, lo, hi))


def _pose_for_candidate(sim, xyz, gripper: float, roll: float | None) -> tuple[bool, tuple[str, ...]]:
    q = sim.qpos()
    q["gripper"] = float(gripper)
    lock: tuple[str, ...] = ()
    if roll is not None:
        q["wrist_roll"] = _clip_roll(sim, roll)
        lock = ("wrist_roll",)
    sim.set_qpos(q, hold=True)
    return bool(solve_ik(sim, xyz, lock_joints=lock)), lock


def _candidate_clear(sim, pre, grasp, neighbors: list[str], gripper: float, roll: float | None) -> str:
    """'ok', 'ik', or 'collision'. Grasp XY itself is not collision-tested (jaws must enter the square)."""
    here, _ = sim.site_pose("gripperframe")
    ok, lock = _pose_for_candidate(sim, pre, gripper, roll)
    if not ok:
        return "ik"
    if neighbor_approach_gap(sim, neighbors) < _APPROACH_MARGIN:
        return "collision"
    pre = np.asarray(pre, dtype=float)
    grasp = np.asarray(grasp, dtype=float)
    path = [here + (pre - here) * u for u in (0.4, 0.8)]
    path.extend(pre + (grasp - pre) * float(u) for u in _DESCEND_U)
    for xyz in path:
        if not solve_ik(sim, xyz, lock_joints=lock):
            return "ik"
        if neighbor_approach_gap(sim, neighbors) < _APPROACH_MARGIN:
            return "collision"
    return "ok"


def plan_approach(
    sim,
    pre,
    grasp,
    open_g: float,
    neighbors: list[str],
    source: str | None = None,
    cfg: dict | None = None,
    scenario: dict | None = None,
) -> dict:
    """First collision-free reachable approach around the frozen grasp XY.

    Candidates: unconstrained then a few wrist-roll deltas × open/narrow aperture.
    Empty occupancy returns the current default (unlocked roll, open gripper).
    """
    start = sim.qpos()

    def restore():
        sim.set_qpos(start, hold=True)

    narrow = None
    if neighbors and source and cfg is not None:
        occ = occupied_neighbor_bodies(sim, cfg, source, scenario)
        sweep = finger_sweep_bodies(sim, cfg, source, occ) if occ else []
        if sweep:
            restore()
            narrow = choose_approach_gripper(sim, grasp, sweep, open_g, min_g=_MIN_APPROACH_G)
            restore()
    grippers = [float(open_g)]
    if narrow is not None and abs(float(narrow) - float(open_g)) > 0.02:
        grippers.append(float(narrow))

    restore()
    q = sim.qpos()
    q["gripper"] = float(open_g)
    sim.set_qpos(q, hold=True)
    ik0 = bool(solve_ik(sim, pre))
    base_roll = float(sim.qpos()["wrist_roll"]) if ik0 else float(start["wrist_roll"])
    restore()

    candidates: list[tuple[int, float | None, float]] = [(0, None, grippers[0])]
    rank = 1
    for g in grippers[1:]:
        candidates.append((rank, None, g))
        rank += 1
    for droll in _ROLL_DELTAS:
        roll = _clip_roll(sim, base_roll + droll)
        for g in grippers:
            candidates.append((rank, roll, g))
            rank += 1

    reachable = 0
    tried = 0
    if not neighbors:
        restore()
        return {
            "ok": True,
            "roll": None,
            "gripper": float(open_g),
            "lock_joints": (),
            "n_wp": 1,
            "tried": 1,
            "reachable": 1,
            "rank": 0,
        }

    for rank, roll, gripper in candidates:
        tried += 1
        restore()
        status = _candidate_clear(sim, pre, grasp, neighbors, gripper, roll)
        if status != "ik":
            reachable += 1
        if status == "ok":
            restore()
            lock = ("wrist_roll",) if roll is not None else ()
            return {
                "ok": True,
                "roll": None if roll is None else float(roll),
                "gripper": float(gripper),
                "lock_joints": lock,
                "n_wp": 8,
                "tried": tried,
                "reachable": reachable,
                "rank": rank,
            }

    restore()
    return {
        "ok": False,
        "roll": None,
        "gripper": float(open_g),
        "lock_joints": (),
        "n_wp": 1,
        "tried": tried,
        "reachable": reachable,
        "rank": None,
        "reason": "collision" if reachable else "ik",
    }
