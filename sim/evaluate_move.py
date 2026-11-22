"""World-side move scoring, shared by every runner, CLI and test.

Reads the physics world only (`world_of(sim)`), never feeds control. A move
counts as done when the mover stands upright on its destination, a captured
piece is gone from the board, and no other piece was disturbed. Knocking over
or shoving a bystander fails the move even if the mover landed.
"""

from __future__ import annotations

import numpy as np

DISPLACED_MM = 8.0
TILTED_DEG = 30.0


def tilt_deg(q) -> float:
    _, x, y, _ = q
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def snapshot(world, bodies) -> dict:
    return {b: (world.body_pose(b)[0].copy(), world.body_pose(b)[1].copy()) for b in bodies}


def disturbance(p0, q0, p, q) -> tuple[float, float, bool]:
    """(xy shove mm, tilt deg, disturbed). A piece already lying down counts only if shoved."""
    mm = float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(p0[:2]))) * 1000
    tilt = tilt_deg(q)
    tipped = tilt > TILTED_DEG and tilt_deg(q0) <= TILTED_DEG
    return mm, tilt, bool(mm > DISPLACED_MM or tipped)


def disturbed(world, before: dict, skip=()) -> list[dict]:
    """Bystanders shoved or knocked over since `before` (a `snapshot`)."""
    out = []
    for b, (p0, q0) in before.items():
        if b in skip:
            continue
        p, q = world.body_pose(b)
        mm, tilt, hit = disturbance(p0, q0, p, q)
        if hit:
            out.append({"body": b, "mm": round(mm, 1), "tilt": round(tilt, 1)})
    return out


def score_board_move(world, cfg: dict, before: dict, mover: str | None, victim: str | None, dst: str) -> dict:
    """Verdict for one chess move. `before` snapshots every piece on the board before the move."""
    from sequence import occupied_squares
    from squares import square_to_world

    occ_after = occupied_squares(world, cfg)
    mover_ok = mover is not None and occ_after.get(dst) == mover and tilt_deg(world.body_pose(mover)[1]) < TILTED_DEG
    victim_gone = victim is None or victim not in occ_after.values()
    hits = disturbed(world, before, skip=(mover, victim))
    out = {"ok": bool(mover_ok and victim_gone and not hits), "mover_ok": bool(mover_ok), "victim_gone": bool(victim_gone), "disturbed": hits}
    if mover is not None:
        p, q = world.body_pose(mover)
        dxy = square_to_world(dst, cfg, z=0.0)
        out["mover_final"] = {
            "xyz_mm": [round(float(v) * 1000, 1) for v in p],
            "off_dest_mm": round(float(np.hypot(*(p[:2] - np.asarray(dxy[:2])))) * 1000, 1),
            "tilt": round(tilt_deg(q), 1),
        }
    return out


ROBOT_JAWS = ("gripper", "moving_jaw_so101_v1")


class NeighborTracker:
    """Peak shove, tilt and contact of named bystanders over an episode, from the world.

    Install `step` as the controller's `on_step`; call `finish()` after the episode
    for the per-neighbour peaks plus the final jaw clearance.
    """

    def __init__(self, world, ids, every: int = 4):
        self.world, self.ids, self.every, self.n = world, list(ids), int(every), 0
        self.start = {b: np.asarray(world.body_pose(b)[0][:2], dtype=float).copy() for b in self.ids}
        self.peak = {b: {"contact": False, "robot_contact": False, "xy_disp_mm": 0.0, "tilt_deg": 0.0} for b in self.ids}

    def step(self) -> None:
        self.n += 1
        if self.n % self.every == 0:
            self.sample()

    def sample(self) -> None:
        for b in self.ids:
            p, q = self.world.body_pose(b)
            hits = [h for h in self.world.contacts_with(b) if h["other"] not in ("board", None)]
            pk = self.peak[b]
            pk["contact"] = pk["contact"] or bool(hits)
            pk["robot_contact"] = pk["robot_contact"] or any(h["other"] in ROBOT_JAWS or (h["other"] or "").startswith("link") for h in hits)
            pk["xy_disp_mm"] = max(pk["xy_disp_mm"], float(np.linalg.norm(np.asarray(p[:2]) - self.start[b])) * 1000)
            pk["tilt_deg"] = max(pk["tilt_deg"], tilt_deg(q))

    def finish(self) -> list[dict]:
        from pickup import min_body_gap

        self.sample()
        out = []
        for b in self.ids:
            gap = min(min_body_gap(self.world, b, j) for j in ROBOT_JAWS)
            pk = self.peak[b]
            out.append({"id": b, "contact": pk["contact"], "robot_contact": pk["robot_contact"],
                        "xy_disp_mm": round(pk["xy_disp_mm"], 1), "tilt_deg": round(pk["tilt_deg"], 1),
                        "clearance_mm": round(gap * 1000, 1)})
        return out


LIFTED_MM = 10.0


def score_lift(world, cfg: dict, before: dict, mover: str, dst: str | None = None) -> dict:
    """Verdict for a pick-and-lift: the mover is off the board surface, or (when the
    episode ran on to a place) upright on `dst`, and no bystander was disturbed."""
    from sequence import occupied_squares

    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    p, q = world.body_pose(mover)
    lifted = (float(p[2]) - surface) * 1000 > LIFTED_MM
    placed = dst is not None and occupied_squares(world, cfg).get(dst) == mover and tilt_deg(q) < TILTED_DEG
    hits = disturbed(world, before, skip=(mover,))
    return {"ok": bool((lifted or placed) and not hits), "lifted": bool(lifted), "placed": bool(placed), "disturbed": hits,
            "mover_final": {"xyz_mm": [round(float(v) * 1000, 1) for v in p], "tilt": round(tilt_deg(q), 1)}}


def board_snapshot(world, cfg: dict) -> tuple[dict, dict]:
    """(square -> body, snapshot of those bodies) for the pieces now on the board."""
    from sequence import occupied_squares

    occ = occupied_squares(world, cfg)
    return occ, snapshot(world, list(occ.values()))
