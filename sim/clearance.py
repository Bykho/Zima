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


# Clutter-aware approach: small candidate set around the existing grasp.
# Grasp XY is frozen. No camera, no RRT. Wrist-roll + aperture only.
APPROACH_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount", "wrist", "lower_arm")
PALM_BODIES = ("gripper", "camera_mount", "wrist")
JAW_ONLY_BODIES = ("gripper", "moving_jaw_so101_v1")
_ROLL_GRID_RAD = 0.35
# Whole descend, grasp point included. The old (0.30, 0.55, 0.80) stopped
# short of the grasp, so a moving jaw that ended 3 mm inside the neighbor
# passed as "clear" and knocked it over on every 7th-rank pawn.
_DESCEND_U = tuple(round(0.1 * i, 2) for i in range(1, 11))
# Kinematic clearance the plan must keep to every bystander along the
# approach. Covers ctrl tracking error (~1-2 mm) plus close-range perception
# error (~1-3 mm). Overridable via sim.pickup.approach_margin_mm.
_APPROACH_MARGIN = 0.003
# The jaws straddle the target on the way down. A kinematic brush of up to
# 2 mm against the target's head is harmless (the pinch closes on it right
# after); a jaw landing on top of the piece shows up as several mm of
# interference and is rejected.
_TARGET_MARGIN = -0.002
# Below this the approach is a physical hit, not a close pass: never chosen.
HARD_MARGIN = 0.0005
_HARD_MARGIN = HARD_MARGIN
# Floor for the extract (lift) margin; the approach margin applies if larger.
_EXTRACT_MARGIN = 0.002
_MIN_APPROACH_G = 0.12
# Aperture ladder tried per roll, widest first. Open jaws span ~63 mm at 0.6
# while squares are 37.5 mm, so the moving jaw reaches into the neighbor
# along the pinch axis unless the roll points it at an empty square or the
# aperture comes down to ~40 mm.
_APERTURE_LADDER = (0.6, 0.42, 0.30, 0.20)
_PRE_OFFSET_SCALE = 0.55
# Bystanders farther than this from the grasp cannot be reached by the
# gripper/wrist envelope during approach; skipping them keeps the dense
# check cheap.
_NEAR_R_M = 0.13


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


def _local_aabb(sim, gid: int) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half-extents) of a geom's AABB in its own frame. Cached."""
    import mujoco

    cache = sim.__dict__.setdefault("_local_aabb_cache", {})
    hit = cache.get(gid)
    if hit is not None:
        return hit
    m = sim.model
    t = int(m.geom_type[gid])
    sz = np.asarray(m.geom_size[gid], dtype=float)
    G = mujoco.mjtGeom
    if t == G.mjGEOM_MESH:
        mid = int(m.geom_dataid[gid])
        adr, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[adr : adr + n], dtype=float).reshape(-1, 3)
        lo, hi = v.min(0), v.max(0)
        out = (0.5 * (lo + hi), 0.5 * (hi - lo))
    elif t == G.mjGEOM_BOX:
        out = (np.zeros(3), sz.copy())
    elif t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        r, h = float(sz[0]), float(sz[1])
        out = (np.zeros(3), np.array([r, r, h + (r if t == G.mjGEOM_CAPSULE else 0.0)]))
    else:
        r = float(m.geom_rbound[gid])
        out = (np.zeros(3), np.array([r, r, r]))
    cache[gid] = out
    return out


def _local_aabb_arrays(sim) -> tuple[np.ndarray, np.ndarray]:
    """Local AABB (centre, half) for every geom, as (ngeom,3) arrays. Cached."""
    hit = sim.__dict__.get("_local_aabb_arrays")
    if hit is not None:
        return hit
    n = int(sim.model.ngeom)
    c = np.zeros((n, 3))
    h = np.zeros((n, 3))
    for g in range(n):
        c[g], h[g] = _local_aabb(sim, g)
    sim._local_aabb_arrays = (c, h)
    return c, h


def _world_aabbs(sim, gids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World AABB (centre, half-extents) of the given geoms, vectorized."""
    c, h = _local_aabb_arrays(sim)
    R = np.asarray(sim.data.geom_xmat[gids], dtype=float).reshape(-1, 3, 3)
    p = np.asarray(sim.data.geom_xpos[gids], dtype=float) + np.einsum("nij,nj->ni", R, c[gids])
    e = np.einsum("nij,nj->ni", np.abs(R), h[gids])
    return p, e


def _aabb_gap_matrix(sim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) matrix of world-AABB gaps: lower bounds on geom distance."""
    pa, ea = _world_aabbs(sim, a)
    pb, eb = _world_aabbs(sim, b)
    sep = np.abs(pa[:, None, :] - pb[None, :, :]) - ea[:, None, :] - eb[None, :, :]
    np.maximum(sep, 0.0, out=sep)
    return np.linalg.norm(sep, axis=2)


def _geom_hull_points(sim, gid: int) -> np.ndarray | None:
    """World-frame vertex cloud whose convex hull is the geom (box, mesh). Cached locally."""
    import mujoco

    m = sim.model
    t = int(m.geom_type[gid])
    cache = sim.__dict__.setdefault("_hull_local_cache", {})
    loc = cache.get(gid)
    if loc is None:
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            s = np.asarray(m.geom_size[gid], dtype=float)
            loc = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
        elif t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[gid])
            adr, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            loc = np.asarray(m.mesh_vert[adr : adr + n], dtype=float).reshape(-1, 3)
        else:
            loc = False
        cache[gid] = loc
    if loc is False:
        return None
    R = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
    return loc @ R.T + np.asarray(sim.data.geom_xpos[gid], dtype=float)


def _simplex_closest(W: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Closest point to the origin on the simplex conv(W) (|W| <= 4) and its supporting face."""
    best = None
    n = len(W)
    for mask in range(1, 1 << n):
        idx = [k for k in range(n) if mask >> k & 1]
        P = np.asarray([W[k] for k in idx])
        if len(idx) == 1:
            lam = np.array([1.0])
        else:
            # min ||P^T lam|| s.t. sum lam = 1  ->  KKT system
            G = P @ P.T
            A = np.zeros((len(idx) + 1, len(idx) + 1))
            A[:-1, :-1] = 2.0 * G
            A[:-1, -1] = 1.0
            A[-1, :-1] = 1.0
            rhs = np.zeros(len(idx) + 1)
            rhs[-1] = 1.0
            try:
                lam = np.linalg.solve(A, rhs)[:-1]
            except np.linalg.LinAlgError:
                continue
            if np.any(lam < -1e-9):
                continue
        v = lam @ P
        d = float(v @ v)
        if best is None or d < best[0] - 1e-15:
            best = (d, v, [W[k] for k in idx])
    return best[1], best[2]


def gjk_distance(A: np.ndarray, B: np.ndarray, tol: float = 1e-6, max_iter: int = 64) -> float:
    """Distance between conv(A) and conv(B) (0 when they intersect). Plain GJK."""
    def support(d):
        return A[int(np.argmax(A @ d))] - B[int(np.argmax(B @ (-d)))]

    v = support(np.array([1.0, 0.0, 0.0]))
    W = [v]
    dist = float(np.linalg.norm(v))
    for _ in range(max_iter):
        if dist < tol:
            return 0.0
        w = support(-v)
        if dist * dist - float(v @ w) <= tol * dist:
            return dist
        W.append(w)
        v, W = _simplex_closest(W)
        dist = float(np.linalg.norm(v))
        if len(W) == 4:
            return 0.0
    return dist


def _exact_or_raw(sim, i: int, j: int, d: float) -> float:
    """mj_geomDistance returns exactly 0 for some separated box/mesh pairs.

    Only the box-box special collider and the convex (mesh) path glitch this
    way; primitives pairs are fine. Recompute those with GJK on the hulls.
    """
    if d != 0.0:
        return d
    A = _geom_hull_points(sim, i)
    B = _geom_hull_points(sim, j)
    if A is None or B is None:
        return d
    return float(gjk_distance(A, B))


def _geoms_of(sim, bodies) -> tuple[np.ndarray, list[str]]:
    gids: list[int] = []
    owner: list[str] = []
    for name in bodies:
        for g in _collision_geoms(sim, name):
            gids.append(g)
            owner.append(name)
    return np.asarray(gids, dtype=int), owner


def _nearest(sim, a_bodies, b_bodies, cap: float = 1.0) -> tuple[float, int, int]:
    """(min signed distance, a_geom, b_geom) with an AABB broadphase.

    AABB gap is a lower bound on the exact distance, so pairs whose gap is
    already above the best exact distance found so far cannot improve it and
    are skipped. Pairs are visited in ascending AABB-gap order, so on a
    crowded board only a handful of exact queries run per pose.
    """
    import mujoco

    a, _ = _geoms_of(sim, a_bodies)
    b, _ = _geoms_of(sim, b_bodies)
    if a.size == 0 or b.size == 0:
        return cap, -1, -1
    gaps = _aabb_gap_matrix(sim, a, b)
    order = np.argsort(gaps, axis=None)
    fromto = np.zeros(6)
    best = float(cap)
    bi = bj = -1
    flat = gaps.ravel()
    nb = b.size
    for k in order:
        lb = float(flat[k])
        # Once penetrating, still visit every zero-gap pair for the deepest hit.
        if lb > max(best, 0.0):
            break
        i, j = int(a[k // nb]), int(b[k % nb])
        d = float(mujoco.mj_geomDistance(sim.model, sim.data, i, j, 0.20, fromto))
        d = _exact_or_raw(sim, i, j, d)
        if lb > 1e-4 and d < lb - 1e-4:
            d = lb  # still bogus: the AABB bound is a valid distance floor
        if d < best:
            best, bi, bj = d, i, j
    return best, bi, bj


def _min_gap(sim, a_bodies: tuple[str, ...], b_bodies: list[str]) -> float:
    return float(_nearest(sim, a_bodies, b_bodies)[0])


def closest_pair(sim, a_bodies, b_bodies) -> dict | None:
    """Nearest (a_body, b_body) pair by geom distance. Diagnostics only."""
    d, i, j = _nearest(sim, a_bodies, b_bodies)
    if i < 0:
        return None
    return {
        "robot": sim.body_name(int(sim.model.geom_bodyid[i])),
        "body": sim.body_name(int(sim.model.geom_bodyid[j])),
        "clearance_mm": round(d * 1000.0, 2),
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


_LEVEL_PADS_TILT_DEG = 3.0


def _grasp_tilt_deg(sim, grasp, roll: float) -> float:
    """Gripper +X tilt from world-down at the grasp with `roll` locked (0 if unreachable)."""
    q = sim.qpos()
    q["wrist_roll"] = float(roll)
    sim.set_qpos(q, hold=True)
    if not solve_ik_down(sim, grasp, lock_joints=("wrist_roll",)):
        return 0.0
    _, R = sim.site_pose("gripperframe")
    return float(np.degrees(np.arccos(np.clip(-R[2, 0], -1.0, 1.0))))


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
    # Same margin as the approach: the z=0 sample *is* the grasp pose, so a
    # stricter extract margin would veto approaches the descend check passed.
    margin = max(approach_margin_m(cfg), _EXTRACT_MARGIN)
    for xyz in _extract_xyzs(grasp, cfg):
        ok = (
            solve_ik_down(sim, xyz, lock_joints=extract_lock)
            if vertical
            else solve_ik(sim, xyz, lock_joints=extract_lock)
        )
        if not ok:
            return "ik"
        if _min_gap(sim, PALM_BODIES, neighbors) < margin:
            return "extract"
    return None


def approach_margin_m(cfg: dict | None) -> float:
    mm = (cfg or {}).get("sim", {}).get("pickup", {}).get("approach_margin_mm")
    return _APPROACH_MARGIN if mm is None else float(mm) / 1000.0


def near_bodies(sim, bodies: list[str], xy, radius: float = _NEAR_R_M) -> list[str]:
    """Subset of `bodies` whose origin lies within `radius` (XY) of `xy`."""
    import mujoco

    xy = np.asarray(xy, dtype=float)[:2]
    out = []
    for name in bodies:
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        if float(np.linalg.norm(np.asarray(sim.data.xpos[bid][:2]) - xy)) <= radius:
            out.append(name)
    return out


_CLOSE_SWEEP_STEP = 0.06
# Sweep past the estimated stem by this much: the real piece may sit that far
# from the perception estimate, so the jaw may travel that much further.
_CLOSE_SWEEP_OVERSHOOT_M = 0.006


def _close_sweep_gap(sim, open_g: float, neighbors: list[str], tgt: list[str]) -> float:
    """Min bystander gap while the moving jaw closes from open_g onto the target.

    Arm pose is left as it is (the grasp pose); only the gripper joint moves.
    Restores the aperture afterwards.
    """
    if not neighbors:
        return 1.0
    q = sim.qpos()
    g0 = float(q["gripper"])
    lo = float(sim.joint_limits()["gripper"][0])
    worst = 1.0
    g = float(open_g)
    try:
        while g > lo:
            g = max(lo, g - _CLOSE_SWEEP_STEP)
            q["gripper"] = g
            sim.set_qpos(q, hold=True)
            worst = min(worst, _min_gap(sim, ("moving_jaw_so101_v1",), neighbors))
            if worst < _HARD_MARGIN:
                break
            if tgt and _min_gap(sim, ("moving_jaw_so101_v1",), tgt) < -_CLOSE_SWEEP_OVERSHOOT_M:
                break
    finally:
        q["gripper"] = g0
        sim.set_qpos(q, hold=True)
    return float(worst)


def _candidate_clear(
    sim,
    pre,
    grasp,
    neighbors: list[str],
    gripper: float,
    roll: float | None,
    cfg: dict | None = None,
    require_extract: bool = True,
    target: str | None = None,
    margin: float | None = None,
) -> tuple[str, tuple[str, ...]]:
    """'ok', 'ik', 'collision', 'target' or 'extract' for one (pre, roll, aperture).

    Every point of the approach — transit to pre, the whole descend and the
    grasp pose itself — must keep `margin` from every bystander with the
    gripper, moving jaw, camera, wrist and forearm. The jaws must also not
    land on the target on the way down (they close on it afterwards).

    Extract uses the candidate's lift lock (flex+roll) and a wider margin —
    palm vs a neighbor column is a lift hit.

    Sets sim._cand_min_gap to the smallest bystander gap seen along the
    path of the returned family (for the near-miss fallback).
    """
    margin = approach_margin_m(cfg) if margin is None else float(margin)
    here, _ = sim.site_pose("gripperframe")
    start = sim.qpos()
    pre = np.asarray(pre, dtype=float)
    grasp = np.asarray(grasp, dtype=float)
    transit = [here + (pre - here) * u for u in (0.4, 0.8)]
    descend = [pre + (grasp - pre) * float(u) for u in _DESCEND_U]
    tgt = [target] if target else []
    last = "ik"
    used_lock: tuple[str, ...] = ()
    sim._cand_min_gap = None
    sim._cand_path = None
    best_gap = None
    best_lock: tuple[str, ...] = ()
    best_path: list[dict] | None = None
    for lock_flex in (True, False):
        sim.set_qpos(start, hold=True)
        ok, lock = _pose_for_candidate(sim, pre, gripper, roll, lock_flex)
        if not ok:
            continue
        used_lock = lock
        status = "ok"
        worst = neighbor_approach_gap(sim, neighbors)
        for xyz in transit:
            if not _solve_path_point(sim, xyz, lock, lock_flex):
                status = "ik"
                break
            worst = min(worst, neighbor_approach_gap(sim, neighbors))
        if status == "ik":
            last = "ik"
            continue
        # The verified joint path, pre first. Execution replays exactly these
        # configurations: re-solving IK online from a different seed lands on
        # a different elbow/tilt compromise where the wrist cannot be vertical
        # (g/h files), and that put the palm over a neighbor the plan never saw.
        sim.set_qpos(start, hold=True)
        _pose_for_candidate(sim, pre, gripper, roll, lock_flex)
        path_q: list[dict] = [sim.qpos()]
        for xyz in descend:
            if not _solve_path_point(sim, xyz, lock, lock_flex):
                status = "ik"
                break
            path_q.append(sim.qpos())
            worst = min(worst, neighbor_approach_gap(sim, neighbors))
            if worst < _HARD_MARGIN:
                break  # physical hit: cannot be chosen even as a fallback
            if tgt and _min_gap(sim, JAW_ONLY_BODIES, tgt) < _TARGET_MARGIN:
                status = "target"
                break
        if status == "ik":
            last = "ik"
            continue
        if status != "target" and worst >= _HARD_MARGIN:
            # Close sweep: at the grasp pose the moving jaw swings through
            # everything between the open aperture and the piece. Walk the
            # aperture down until the pad is well into the (estimated) target
            # and keep the bystander gap along the way.
            worst = min(worst, _close_sweep_gap(sim, gripper, neighbors, tgt))
        if status == "ok" and worst < margin:
            status = "collision"
        if status == "target":
            last = status
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
        if status == "ok":
            sim._cand_min_gap = float(worst)
            sim._cand_path = path_q
            return "ok", used_lock
        # collision: remember the widest near-miss, keep looking in the other
        # lock family for a clean one
        last = status
        if best_gap is None or worst > best_gap:
            best_gap, best_lock, best_path = float(worst), lock, path_q
    if last == "collision":
        sim._cand_min_gap = best_gap
        sim._cand_path = best_path
        return last, best_lock
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
    target: str | None = None,
    margin: float | None = None,
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
    # Only bystanders the gripper envelope can actually reach during the
    # approach; the far side of the board is irrelevant and costs time.
    all_neighbors = list(neighbors)
    neighbors = near_bodies(sim, all_neighbors, grasp[:2]) if neighbors else []

    def restore():
        sim.set_qpos(start, hold=True)

    def pre_at(dx: float, dy: float) -> list[float]:
        return [float(pre0[0] + dx), float(pre0[1] + dy), float(pre0[2])]

    occ: dict[str, str] = {}
    if neighbors and source and cfg is not None:
        occ = occupied_neighbor_bodies(sim, cfg, source, scenario)
    # Aperture is a search dimension, widest first: the wide jaw is the most
    # forgiving grasp, but on a crowded board it reaches into the next square
    # along the pinch axis, so each roll is retried narrower before moving on.
    grippers = [float(open_g)]
    if neighbors:
        for g in _APERTURE_LADDER:
            g = float(min(g, open_g))
            if g >= _MIN_APPROACH_G and all(abs(g - x) > 0.05 for x in grippers):
                grippers.append(g)
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
    lo, hi = sim.joint_limits()["wrist_roll"]
    tilt_deg = _grasp_tilt_deg(sim, grasp, base_roll)
    restore()
    # Pinch tilt for a roll r is asin(sin(tilt) * sin(r - level)). When the
    # wrist cannot point down (g/h files, rank 7: 6-30 deg) a roll picked for
    # clearance alone tips the piece on close and it hangs crooked through
    # the carry, so order rolls by that cost first: the leveling roll and its
    # 180 deg flip (moving jaw on the other side) before anything else.
    if tilt_deg >= 3.0:
        for r in (level_roll, level_roll + np.pi, level_roll - np.pi):
            if lo + 0.02 <= r <= hi - 0.02 and all(abs(r - x) > 0.15 for x in rolls):
                rolls.append(float(r))

        def _key(v: float):
            pinch = tilt_deg * abs(np.sin(float(v) - level_roll))
            return (round(pinch / 2.0), abs(float(v) - base_roll))
    else:

        def _key(v: float):
            return (0, abs(float(v) - base_roll))

    if all(abs(base_roll - x) > 0.15 for x in rolls):
        rolls.append(base_roll)
    grid = np.arange(lo + 0.05, hi - 0.05 + 1e-9, _ROLL_GRID_RAD)
    for r in sorted(grid, key=_key):
        r = float(r)
        if all(abs(r - x) > 0.15 for x in rolls):
            rolls.append(r)
    if tilt_deg >= 3.0:
        rolls = sorted(rolls, key=_key)
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
    near_miss: dict | None = None
    grasp_cache: dict[float | None, np.ndarray] = {}
    status_count: dict[str, int] = {}

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
            sim,
            pre_i,
            grasp_i,
            neighbors,
            gripper,
            roll,
            cfg,
            require_extract=require_extract,
            target=target,
            margin=margin,
        )
        status_count[status] = status_count.get(status, 0) + 1
        if status != "ik":
            reachable += 1
        if status == "extract" and extract_block is None:
            extract_block = closest_pair(sim, PALM_BODIES, neighbors) or closest_pair(
                sim, APPROACH_BODIES, neighbors
            )
        if status == "collision" and first_block is None:
            first_block = closest_pair(sim, APPROACH_BODIES, neighbors)
        gap = getattr(sim, "_cand_min_gap", None)
        if status == "collision" and gap is not None and gap >= _HARD_MARGIN:
            if near_miss is None or gap > near_miss["min_gap"]:
                near_miss = {
                    "roll": roll,
                    "gripper": gripper,
                    "lock": cand_lock,
                    "rank": crank,
                    "pre": pre_i,
                    "grasp": grasp_i,
                    "dxdy": (dx, dy),
                    "min_gap": gap,
                    "path_q": getattr(sim, "_cand_path", None),
                }
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
                "statuses": status_count,
                "near": neighbors,
                "min_gap_mm": None if gap is None else round(gap * 1000.0, 2),
                "path_q": getattr(sim, "_cand_path", None),
            }

    restore()
    if near_miss is not None and not lock_roll:
        # Nothing keeps the preferred margin; take the widest near-miss that
        # is still physically clear. The descend guard covers the residual.
        nm = near_miss
        lock = tuple(nm["lock"]) if nm["lock"] else (("wrist_roll",) if nm["roll"] is not None else ())
        return {
            "ok": True,
            "degraded": True,
            "roll": None if nm["roll"] is None else float(nm["roll"]),
            "gripper": float(nm["gripper"]),
            "lock_joints": lock,
            "n_wp": 8,
            "tried": tried,
            "reachable": reachable,
            "rank": nm["rank"],
            "pre": nm["pre"],
            "grasp": [float(v) for v in nm["grasp"]],
            "pre_offset_xy": [float(nm["dxdy"][0]), float(nm["dxdy"][1])],
            "statuses": status_count,
            "near": neighbors,
            "min_gap_mm": round(nm["min_gap"] * 1000.0, 2),
            "path_q": nm["path_q"],
        }
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
        "statuses": status_count,
        "near": neighbors,
    }


def verify_approach(
    sim,
    pre,
    grasp,
    neighbors: list[str],
    gripper: float,
    roll: float | None,
    lock: tuple[str, ...],
    cfg: dict | None = None,
    target: str | None = None,
    margin: float | None = None,
) -> str:
    """Re-check one already-chosen approach against a refined grasp point.

    Same test the planner ran, restricted to the lock family it chose, so a
    close-range perception update that moved the grasp a few mm gets caught
    before the jaws go down. Returns the candidate status ('ok', ...).
    """
    start = sim.qpos()
    near = near_bodies(sim, list(neighbors), np.asarray(grasp)[:2]) if neighbors else []
    margin = approach_margin_m(cfg) if margin is None else float(margin)
    sim._cand_path = None
    try:
        lock_flex = "wrist_flex" in tuple(lock)
        ok, used = _pose_for_candidate(sim, pre, gripper, roll, lock_flex)
        if not ok:
            return "ik"
        pre_a = np.asarray(pre, dtype=float)
        grasp_a = np.asarray(grasp, dtype=float)
        path_q = [sim.qpos()]
        for u in _DESCEND_U:
            xyz = pre_a + (grasp_a - pre_a) * float(u)
            if not _solve_path_point(sim, xyz, used, lock_flex):
                return "ik"
            path_q.append(sim.qpos())
            if neighbor_approach_gap(sim, near) < margin:
                return "collision"
            if target and _min_gap(sim, JAW_ONLY_BODIES, [target]) < _TARGET_MARGIN:
                return "target"
        if _close_sweep_gap(sim, gripper, near, [target] if target else []) < margin:
            return "collision"
        sim._cand_path = path_q
        return "ok"
    finally:
        sim.set_qpos(start, hold=True)
