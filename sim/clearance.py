"""Pickup approach clearance from occupancy + gripper envelope.

No square-name special cases, no motion planner. Camera/perception unused.
When occupied neighbors sit along the open-jaw axis, the approach aperture
is the smallest change that keeps those fingers out of neighbor columns.
Grasp EE XY, close target, and empty-board opening stay frozen.
"""

from __future__ import annotations

import numpy as np

from ik import solve_ik, solve_ik_down, solve_ik_upright
from scenario import body_kind
from squares import parse_square, shift_square, square_to_world

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
    size = float(cfg.get("sim", {}).get("board", {}).get("square_size", 0.0375))
    for i in range(sim.model.nbody):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or float(sim.data.xpos[i][2]) <= 0.0:
            continue
        if name.startswith("proxy_"):
            sq = name[6:]
        elif body_kind(name):
            sq = None
            best = 0.55 * size
            for nsq in want:
                p = square_to_world(nsq, cfg)
                d = float(np.hypot(float(sim.data.xpos[i][0]) - p[0], float(sim.data.xpos[i][1]) - p[1]))
                if d < best:
                    best = d
                    sq = nsq
        else:
            continue
        if sq in want:
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
PALM_BODIES = ("gripper", "camera_mount", "wrist")
_ROLL_GRID_RAD = 0.35
_DESCEND_U = (0.30, 0.55, 0.80)
_APPROACH_MARGIN = 0.0005
# Kinematic 0.5 mm still physics-clips on locked-wrist +Z (palm vs neighbor).
_EXTRACT_MARGIN = 0.005
_MIN_APPROACH_G = 0.12
_PRE_OFFSET_SCALE = 0.55


# --- transport: lift above the board envelope, then translate in XY ---

TRANSIT_MARGIN = 0.010
CARRY_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount")
GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")


def _collision_geoms(sim, body: str) -> list[int]:
    """Collision-enabled, non-plane geoms of a body. Model-static, so cached per Sim."""
    import mujoco

    cache = sim.__dict__.setdefault("_collision_geom_cache", {})
    hit = cache.get(body)
    if hit is not None:
        return hit
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body)
    out: list[int] = []
    if bid >= 0:
        m = sim.model
        mask = (
            (m.geom_bodyid == bid)
            & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
            & (m.geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE))
        )
        out = [int(i) for i in np.flatnonzero(mask)]
    cache[body] = out
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
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "board")
            half = 0.20
            if gid >= 0:
                half = float(max(sim.model.geom_size[gid][0], sim.model.geom_size[gid][1])) + 0.02
            if abs(float(xy[0] - c[0])) > half or abs(float(xy[1] - c[1])) > half:
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


def closest_pair(sim, a_bodies, b_bodies) -> dict | None:
    """Nearest (a_body, b_body) pair by geom distance. Diagnostics only."""
    import mujoco

    fromto = np.zeros(6)
    best = 1.0
    pair = None
    for a in a_bodies:
        for i in _collision_geoms(sim, a):
            for b in b_bodies:
                for j in _collision_geoms(sim, b):
                    d = mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.20, fromto)
                    if d < best:
                        best = d
                        pair = (a, b, float(d))
    if pair is None:
        return None
    return {
        "robot": pair[0],
        "body": pair[1],
        "clearance_mm": round(pair[2] * 1000.0, 2),
    }


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


def _pose_for_candidate(
    sim, xyz, gripper: float, roll: float | None, lock_flex: bool
) -> tuple[bool, tuple[str, ...]]:
    q = sim.qpos()
    q["gripper"] = float(gripper)
    lock: tuple[str, ...] = ()
    if roll is not None:
        q["wrist_roll"] = _clip_roll(sim, roll)
        lock = ("wrist_roll",)
    sim.set_qpos(q, hold=True)
    if lock_flex:
        # "Vertical" family: gripper +X world-down at every checked point,
        # flex free for the solver. Mirrors what pickup executes (down=True).
        if solve_ik_down(sim, xyz, lock_joints=lock):
            return True, lock
        return False, lock
    if solve_ik_upright(sim, xyz, lock_joints=lock):
        return True, lock
    return False, lock


_LEVEL_PADS_TILT_DEG = 8.0


def _level_pads_roll(sim, grasp, base_roll: float) -> float:
    """Roll that puts the pinch axis horizontal when a vertical pinch is out of reach.

    On the g/h files the wrist cannot point straight down (18-30 deg tilt in
    the arm's radial plane). If the pinch axis lies in that plane one pad
    lands lower than the other and closing tips the piece 20-30 deg. Rolling
    so gripper local Z is horizontal puts both pads at the same height; the
    remaining tilt is along the pad width, which a cylinder does not mind.
    Below the tilt threshold the natural roll is kept.
    """
    q = sim.qpos()
    q["wrist_roll"] = float(base_roll)
    sim.set_qpos(q, hold=True)
    if not solve_ik_down(sim, grasp, lock_joints=("wrist_roll",)):
        return base_roll
    _, R = sim.site_pose("gripperframe")
    tilt = float(np.degrees(np.arccos(np.clip(-R[2, 0], -1.0, 1.0))))
    if tilt < _LEVEL_PADS_TILT_DEG:
        return base_roll
    y_z, z_z = float(R[2, 1]), float(R[2, 2])
    if abs(y_z) < 1e-6 and abs(z_z) < 1e-6:
        return base_roll
    # Rolling by phi about local X: z(phi) = cos(phi) z0 + sin(phi) y0 (or the
    # mirror, depending on joint sign). Test both and keep the level one.
    best_roll, best_err = base_roll, abs(z_z)
    for phi in (np.arctan2(-z_z, y_z), np.arctan2(z_z, -y_z), np.arctan2(-z_z, y_z) + np.pi, np.arctan2(z_z, -y_z) + np.pi):
        cand = _clip_roll(sim, base_roll + float(phi))
        q["wrist_roll"] = cand
        sim.set_qpos(q, hold=True)
        if not solve_ik_down(sim, grasp, lock_joints=("wrist_roll",)):
            continue
        _, Rc = sim.site_pose("gripperframe")
        err = abs(float(Rc[2, 2]))
        if err < best_err - 1e-3:
            best_roll, best_err = cand, err
    return float(best_roll)


def _solve_path_point(sim, xyz, lock, vertical: bool) -> bool:
    if vertical:
        return solve_ik_down(sim, xyz, lock_joints=lock)
    return solve_ik_upright(sim, xyz, lock_joints=lock)


def approach_xy_offsets(source: str | None, occ: dict[str, str], cfg: dict | None) -> list[tuple[float, float]]:
    """Pregrasp XY shifts away from occupied neighbors. Grasp XY stays frozen.

    Starting-rank clutter is usually denser toward the board interior, so the
    first nonzero offset is typically off the back rank — still generic: it
    follows occupancy, not square names.
    """
    out: list[tuple[float, float]] = [(0.0, 0.0)]
    if not source or not occ:
        return out
    size = float((cfg or {}).get("sim", {}).get("board", {}).get("square_size", 0.0375))
    # Gripper envelope does not shrink with the board. Floor the step so
    # tighter squares still lean the pregrasp off the neighboring pawn.
    step = max(0.032, _PRE_OFFSET_SCALE * size)
    file_i, rank = parse_square(source)
    acc = np.zeros(2)
    for sq in occ:
        fi, r = parse_square(sq)
        acc += np.array([float(fi - file_i), float(r - rank)])
    if float(np.linalg.norm(acc)) < 0.25:
        acc = np.array([file_i - 3.5, rank - 4.5])
    else:
        acc = -acc
    n = float(np.linalg.norm(acc))
    if n < 1e-9:
        return out
    u = acc / n
    raw = [
        (float(u[0] * step), float(u[1] * step)),
        (float(u[0] * step * 1.8), float(u[1] * step * 1.8)),
        (float(u[0] * step), 0.0),
        (0.0, float(u[1] * step)),
    ]
    seen = {(0.0, 0.0)}
    for dx, dy in raw:
        key = (round(dx, 4), round(dy, 4))
        if key in seen or (abs(dx) < 1e-4 and abs(dy) < 1e-4):
            continue
        seen.add(key)
        out.append((dx, dy))
    return out


_WORKSPACE_R_M = 0.19


def _arm_base_xy(cfg: dict | None) -> np.ndarray:
    raw = (cfg or {}).get("sim", {}).get("arm_base_position") or [0.08, 0.045, 0.0]
    return np.asarray(raw[:2], dtype=float)


def _extract_zs(cfg: dict | None) -> tuple[float, ...]:
    lift = 0.08
    if cfg:
        lift = float(cfg.get("sim", {}).get("pickup", {}).get("lift_z", lift))
    # Palm clips neighbors in the first centimetres of +Z, not at full lift height.
    # z=0 is the closed-finger grasp pose the lift actually starts from.
    zs = (0.0, 0.01, 0.02, 0.04, float(lift))
    return tuple(sorted({round(float(z), 4) for z in zs if z >= -1e-9}))


def _extract_xyzs(grasp, cfg: dict | None) -> list[np.ndarray]:
    """Locked-wrist lift samples, including the workspace-floor XY the live lift uses."""
    grasp = np.asarray(grasp, dtype=float)
    zs = _extract_zs(cfg)
    lift = float(zs[-1]) if zs else 0.08
    base = _arm_base_xy(cfg)
    p = grasp[:2]
    v = p - base
    n = float(np.linalg.norm(v))
    extra = max(0.0, _WORKSPACE_R_M - n)
    if extra > 0.001:
        if n < 1e-9:
            v, n = np.array([1.0, 0.0], dtype=float), 1.0
        lift_xy = p + v * (extra / n)
    else:
        lift_xy = p
    out = []
    for z in zs:
        u = 0.0 if lift < 1e-9 else float(z) / lift
        xy = p + u * (lift_xy - p)
        out.append(np.array([float(xy[0]), float(xy[1]), float(grasp[2] + z)], dtype=float))
    return out


def _extract_clear(
    sim, grasp, neighbors: list[str], extract_lock, cfg: dict | None, vertical: bool = False
) -> str | None:
    """None if extract is clear, else 'ik' / 'extract'.

    Jaws are allowed to occupy the source column at grasp. Extract flags
    palm/wrist/camera vs a neighbor — the locked-wrist lift clip.
    """
    for xyz in _extract_xyzs(grasp, cfg):
        ok = (
            solve_ik_down(sim, xyz, lock_joints=extract_lock)
            if vertical
            else solve_ik(sim, xyz, lock_joints=extract_lock)
        )
        if not ok:
            return "ik"
        if _min_gap(sim, PALM_BODIES, neighbors) < _EXTRACT_MARGIN:
            return "extract"
    return None


def _candidate_clear(
    sim,
    pre,
    grasp,
    neighbors: list[str],
    gripper: float,
    roll: float | None,
    cfg: dict | None = None,
    require_extract: bool = True,
) -> str:
    """'ok', 'ik', 'collision', or 'extract'. Grasp XY is not tested (jaws must enter).

    Descend uses the approach roll lock. Extract uses that same candidate's
    lift lock (flex+roll) and a wider margin — palm vs a neighbor column is
    a lift hit. A different roll is allowed when it extracts under its own
    lock; the live lift holds that roll after close.
    """
    here, _ = sim.site_pose("gripperframe")
    start = sim.qpos()
    pre = np.asarray(pre, dtype=float)
    grasp = np.asarray(grasp, dtype=float)
    path = [here + (pre - here) * u for u in (0.4, 0.8)]
    path.extend(pre + (grasp - pre) * float(u) for u in _DESCEND_U)
    last = "ik"
    used_lock: tuple[str, ...] = ()
    for lock_flex in (True, False):
        sim.set_qpos(start, hold=True)
        ok, lock = _pose_for_candidate(sim, pre, gripper, roll, lock_flex)
        if not ok:
            continue
        used_lock = lock
        descend_hit = neighbor_approach_gap(sim, neighbors) < _APPROACH_MARGIN
        path_ok = True
        for xyz in path:
            if not _solve_path_point(sim, xyz, lock, lock_flex):
                path_ok = False
                break
            if neighbor_approach_gap(sim, neighbors) < _APPROACH_MARGIN:
                descend_hit = True
        if not path_ok:
            last = "ik"
            continue
        if not _solve_path_point(sim, grasp, lock, lock_flex):
            last = "ik"
            continue
        if require_extract:
            # Live lift is vertical (down=True, roll lock only) with a
            # locked-wrist fallback; accept if either extract is clear.
            hold_lock = tuple(dict.fromkeys((*lock, "wrist_roll")))
            hit = _extract_clear(sim, grasp, neighbors, hold_lock, cfg, vertical=True)
            if hit:
                extract_lock = tuple(dict.fromkeys((*hold_lock, "wrist_flex")))
                hit = _extract_clear(sim, grasp, neighbors, extract_lock, cfg)
            if hit:
                last = hit
                continue
        if descend_hit:
            last = "collision"
            continue
        return "ok", used_lock
    return last, used_lock


def plan_approach(
    sim,
    pre,
    grasp,
    open_g: float,
    neighbors: list[str],
    source: str | None = None,
    cfg: dict | None = None,
    scenario: dict | None = None,
    require_extract: bool = True,
    force_roll: float | None = None,
    lock_roll: bool = False,
    grasp_for_roll=None,
) -> dict:
    """First collision-free reachable approach around the frozen grasp XY.

    Candidates: unconstrained then wrist-roll × aperture, then the same at a
    few pregrasp XY offsets away from occupied neighbors. Grasp XY is frozen.
    A candidate must also clear locked-wrist +Z extract. Empty occupancy
    returns the current default (unlocked roll, open gripper).
    """
    start = sim.qpos()
    pre0 = np.asarray(pre, dtype=float)
    grasp = np.asarray(grasp, dtype=float)

    def restore():
        sim.set_qpos(start, hold=True)

    def pre_at(dx: float, dy: float) -> list[float]:
        return [float(pre0[0] + dx), float(pre0[1] + dy), float(pre0[2])]

    occ: dict[str, str] = {}
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
    xy_list = approach_xy_offsets(source, occ, cfg) if neighbors else [(0.0, 0.0)]

    restore()
    q = sim.qpos()
    q["gripper"] = float(open_g)
    sim.set_qpos(q, hold=True)
    base_roll = float(start["wrist_roll"])
    for dx, dy in xy_list:
        if solve_ik(sim, pre_at(dx, dy)):
            base_roll = float(sim.qpos()["wrist_roll"])
            break
    restore()
    level_roll = _level_pads_roll(sim, grasp, base_roll)
    restore()

    # Roll is redundant for a vertical pinch, so a "free" roll drifts by up to
    # a radian between planning and the executed pre/descend, and the pinch
    # axis (hence the fixed-pad grasp offset) drifts with it. Always plan and
    # execute an explicit roll; the natural one is what position IK lands on.
    # Where the pinch cannot be vertical, the pad-leveling roll goes first.
    # The natural roll is path dependent (it is whatever the observe pose left),
    # so deltas around it made success depend on history. Scan an absolute
    # grid over the joint range instead, ordered by distance from the natural
    # roll; the pad-leveling roll goes first when the pinch cannot be vertical.
    rolls: list[float] = []
    if abs(level_roll - base_roll) > 0.1:
        rolls.append(level_roll)
    rolls.append(base_roll)
    lo, hi = sim.joint_limits()["wrist_roll"]
    grid = np.arange(lo + 0.05, hi - 0.05 + 1e-9, _ROLL_GRID_RAD)
    for r in sorted(grid, key=lambda v: abs(float(v) - base_roll)):
        r = float(r)
        if all(abs(r - x) > 0.15 for x in rolls):
            rolls.append(r)
    roll_cands: list[tuple[int, float | None, float]] = []
    rank = 0
    for roll in rolls:
        for g in grippers:
            roll_cands.append((rank, roll, g))
            rank += 1
    # Far pregrasp XY: every other grid roll.
    offset_cands: list[tuple[int, float | None, float]] = []
    for roll in rolls[::2]:
        for g in grippers:
            offset_cands.append((rank, roll, g))
            rank += 1

    reachable = 0
    tried = 0
    if not neighbors and not lock_roll:
        restore()
        grasp0 = grasp if grasp_for_roll is None else np.asarray(grasp_for_roll(base_roll), dtype=float)
        restore()
        return {
            "ok": True,
            "roll": float(base_roll),
            "gripper": float(open_g),
            "lock_joints": ("wrist_roll",),
            "n_wp": 1,
            "tried": 1,
            "reachable": 1,
            "rank": 0,
            "pre": pre_at(0.0, 0.0),
            "grasp": [float(v) for v in grasp0],
            "pre_offset_xy": [0.0, 0.0],
        }

    ordered: list[tuple[float, float, int, float | None, float]] = []
    for crank, roll, gripper in roll_cands:
        ordered.append((0.0, 0.0, crank, roll, gripper))
    for dx, dy in xy_list:
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        for crank, roll, gripper in offset_cands:
            ordered.append((dx, dy, crank, roll, gripper))
    if lock_roll:
        ordered = [(0.0, 0.0, 0, force_roll, grippers[0])]

    first_block = None
    extract_block = None
    grasp_cache: dict[float | None, np.ndarray] = {}

    def grasp_at(roll: float | None) -> np.ndarray:
        if grasp_for_roll is None:
            return grasp
        key = None if roll is None else round(float(roll), 4)
        if key not in grasp_cache:
            restore()
            grasp_cache[key] = np.asarray(grasp_for_roll(roll), dtype=float)
        return grasp_cache[key]

    for dx, dy, crank, roll, gripper in ordered:
        tried += 1
        restore()
        pre_i = pre_at(dx, dy)
        grasp_i = grasp_at(roll)
        status, cand_lock = _candidate_clear(
            sim, pre_i, grasp_i, neighbors, gripper, roll, cfg, require_extract=require_extract
        )
        if status != "ik":
            reachable += 1
        if status == "extract" and extract_block is None:
            extract_block = closest_pair(sim, PALM_BODIES, neighbors) or closest_pair(
                sim, APPROACH_BODIES, neighbors
            )
        if status == "collision" and first_block is None:
            first_block = closest_pair(sim, APPROACH_BODIES, neighbors)
        if status == "ok":
            restore()
            # Keep the lock the candidate was verified with. When flex was
            # locked (the vertical wrist set before planning) the descend
            # inherits it, so the pinch closes at the planned tilt.
            lock = tuple(cand_lock) if cand_lock else (("wrist_roll",) if roll is not None else ())
            return {
                "ok": True,
                "roll": None if roll is None else float(roll),
                "gripper": float(gripper),
                "lock_joints": lock,
                "n_wp": 8,
                "tried": tried,
                "reachable": reachable,
                "rank": crank,
                "pre": pre_i,
                "grasp": [float(v) for v in grasp_i],
                "pre_offset_xy": [float(dx), float(dy)],
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
        "pre": pre_at(0.0, 0.0),
        "pre_offset_xy": [0.0, 0.0],
        "block": extract_block or first_block,
    }
