"""Controller-side board perception: wrist scan → classified pieces → belief.

Data association keeps belief body names stable: a detection takes the body
of its kind and color whose belief pose is nearest, then a forgotten body near
where it was last seen, else the one whose home square it sits on, else any
spare. Templates are upright, so a lying detection's kind is not trusted and
only its color and position count. Bodies nobody claims leave the board.
"""

from __future__ import annotations

import math

import numpy as np

import diag
import scan
from scenario import PIECES, body_kind
from sensor import capture

LOOK_AGAIN_SEEN = 0.5  # classify patch less covered than this → close look
LOOK_AGAIN_MARGIN_M = 0.001
UPRIGHT_FRAC = 0.7
LOST_COST_M = 0.1  # a forgotten body's last-seen spot ranks behind any current belief


def _white_body(name: str) -> bool:
    return name.split("_")[1][1] in "12"


def _piece_bodies(sim) -> list[str]:
    return sorted(getattr(sim, "_piece_names", None) or [])


def look_again(sim, cfg: dict, board: scan.Scan, pieces: list[dict]) -> int:
    """Close wrist look at each low-confidence piece, fused into the same map."""
    from pickup import _reach_observe

    geoms = scan.arm_geoms(sim)
    surface = float(cfg["sim"]["board"]["surface_z"])
    n = 0
    for p in pieces:
        if _reach_observe(sim, cfg, [p["xy"][0], p["xy"][1], surface], "") is None:
            continue
        pts, lum = scan._observe(sim, capture(sim), geoms)
        scan.scene.integrate_points(board.hm, pts)
        n += 1
    return n


def classify_board(sim, cfg: dict, board: scan.Scan) -> list[dict]:
    templates = scan.piece_templates(cfg)
    out = []
    for p in board.pieces:
        m = scan.match(p["patch"], templates)
        height = float(PIECES[m["kind"]]["height"])
        out.append({**p, **{k: m[k] for k in ("kind", "score", "margin", "yaw", "seen")},
                    "upright": p["top_h"] > UPRIGHT_FRAC * height})
    return out


def perceive_board(sim, cfg: dict, refine: bool = True) -> dict:
    """Scan, classify, and write the result into the belief. Arm ends at the last scan pose."""
    diag.emit("scan_start", sim)
    out = _perceive_board(sim, cfg, refine)
    diag.emit("scan_end", sim)
    return out


def _perceive_board(sim, cfg: dict, refine: bool) -> dict:
    from pickup import _js_raise_ee

    _js_raise_ee(sim, z_min=0.14)
    board = scan.scan_board(sim, cfg)
    pieces = classify_board(sim, cfg, board)
    unsure = [p for p in pieces if p["seen"] < LOOK_AGAIN_SEEN or p["margin"] < LOOK_AGAIN_MARGIN_M]
    looked = 0
    if refine and unsure:
        looked = look_again(sim, cfg, board, unsure)
        board.pieces = scan.detect_pieces(board, cfg)
        pieces = classify_board(sim, cfg, board)
    assigned = associate(sim, cfg, pieces)
    return {"pieces": pieces, "assigned": assigned, "views": board.views, "looked": looked, "scan": board}


def associate(sim, cfg: dict, pieces: list[dict]) -> dict[str, str]:
    """Detections → belief bodies. Returns square -> body; unclaimed bodies are forgotten."""
    surface = float(cfg["sim"]["board"]["surface_z"])
    bodies = _piece_bodies(sim)
    known = set(getattr(sim, "known", set()))
    prior = {b: sim.body_pose(b)[0][:2] for b in bodies if b in known}
    lost = {b: xy for b, xy in getattr(sim, "last_seen", {}).items() if b not in known}
    free = set(bodies)
    claims: list[tuple[float, int, str]] = []
    for i, p in enumerate(pieces):
        xy = np.asarray(p["xy"])
        for b in bodies:
            if p["upright"] and body_kind(b) != p["kind"]:
                continue
            if p["white"] is not None and _white_body(b) != p["white"]:
                continue
            if b in prior:
                cost = float(np.linalg.norm(prior[b] - xy))
            elif b in lost:
                cost = LOST_COST_M + float(np.linalg.norm(lost[b] - xy))
            else:
                cost = 1.0 + (0.0 if b.split("_")[1] == p["square"] else 0.5)
            claims.append((cost, i, b))
    claims.sort()
    body_of: dict[int, str] = {}
    for cost, i, b in claims:
        if i in body_of or b not in free:
            continue
        body_of[i] = b
        free.discard(b)
    for i, p in enumerate(pieces):
        if i in body_of:
            continue
        spare = sorted(b for b in free if p["white"] is None or _white_body(b) == p["white"]) or sorted(free)
        if spare:
            body_of[i] = spare[0]
            free.discard(spare[0])
    out = {}
    for i, b in body_of.items():
        p = pieces[i]
        yaw = float(p["yaw"])
        if p["upright"]:
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        else:
            quat = (0.7071068, 0.7071068, 0.0, 0.0)
        sim.believe(b, [p["xy"][0], p["xy"][1], surface], quat)
        if p["square"]:
            out[p["square"]] = b
    for b in free:
        if b in known and b != getattr(sim, "held", None):
            sim.forget(b)
    return out
