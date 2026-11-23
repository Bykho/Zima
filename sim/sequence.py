"""LIFO unstack: park blocking neighbors, then restore them in reverse.

On a full board each square is its own free body, so a pawn can park a
neighboring pawn. Same-type skip is only when the caller passes skip_kinds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import stages
from clearance import obstacle_bodies
from ik import EE_SITE, grasp_xy_delta, solve_ik
from mj import probing
from scenario import PIECES, STARTING_AT, body_kind


@dataclass(frozen=True)
class Move:
    """One square-to-square transfer. Sequencer output; execute_move consumes it."""

    kind: str
    src: str
    dst: str


@dataclass
class UnstackFrame:
    kind: str
    home: str
    park: str
    body: str = ""


def piece_kind(square: str, body: str) -> str:
    return body_kind(body) or STARTING_AT.get(square, "pawn")


@probing
def probe_approach(
    sim, cfg: dict, source: str, tid: str, scenario: dict, at=None,
    require_extract: bool = True, force_roll: float | None = None, lock_roll: bool = False,
) -> dict:
    """The selected approach stage's plan for the live occupancy. Restores arm qpos.

    `at` overrides the piece pose so an empty destination square can be
    checked with the same extract envelope as a pickup.
    """
    p = cfg.get("sim", {}).get("pickup", {})
    geom = dict(scenario.get("geometry") or PIECES.get(body_kind(tid) or tid) or {})
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    open_g = float(p.get("gripper_open", 0.45))
    pre_z = float(p.get("pregrasp_z", 0.10))
    grasp_z = float(geom.get("preferred_grasp_height", p.get("grasp_z", 0.016)))
    off = scenario.get(
        "grasp_xy_offset",
        geom.get("grasp_xy_offset", p.get("grasp_xy_offset", [0.0, 0.0])),
    )
    if at is not None:
        xyz = np.asarray(at, dtype=float)
    else:
        xyz, _ = sim.object_pose(tid)
    q0 = sim.qpos()
    if solve_ik(sim, [float(xyz[0]), float(xyz[1]), surface + grasp_z]):
        _, grot = sim.site_pose(EE_SITE)
        d = grasp_xy_delta(grot, off)
        dx, dy = float(d[0]), float(d[1])
    else:
        dx, dy = float(off[0]), float(off[1])
    sim.set_qpos(q0, hold=True)
    pre = [float(xyz[0] + dx), float(xyz[1] + dy), surface + pre_z]
    grasp = [float(xyz[0] + dx), float(xyz[1] + dy), surface + grasp_z]
    clutter = [n for n in obstacle_bodies(sim, skip={tid}) if n != "board"]
    plan = stages.resolve(cfg)["approach"].plan(
        sim, pre, grasp, open_g, clutter, source=source, cfg=cfg, scenario=scenario,
        require_extract=require_extract, force_roll=force_roll, lock_roll=lock_roll,
        target=tid,
    )
    sim.set_qpos(q0, hold=True)
    return plan
