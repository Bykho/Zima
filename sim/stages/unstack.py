"""Unstack policy: which neighbour to park, and where, so the arm can reach a square.

Move-level stage (`episode.run_episode`), occupancy only; no arm motion.
"""

from __future__ import annotations

import numpy as np

from occupancy import neighbor_squares, occupied_squares
from sequence import Move, piece_kind
from squares import parse_square
from stages import register

MAX_UNSTACK = 4
_PARK_ORDER = tuple(f"{f}{r}" for r in (3, 4, 5, 6) for f in "decfbga")


def plan_unstack_move(
    square: str,
    kind: str,
    occ: dict[str, str],
    skip_squares: set[str],
    skip_kinds: set[str],
    occupied: set[str] | None = None,
    prefer: str | None = None,
) -> Move | None:
    """Next park move to free `square`. Occupancy only; no arm motion."""
    _ = kind
    neigh = {sq: body for sq, body in occ.items() if sq in set(neighbor_squares(square))}
    if prefer:
        if (
            prefer not in neigh
            or prefer in skip_squares
            or piece_kind(prefer, neigh[prefer]) in skip_kinds
        ):
            return None
        home = prefer
    else:
        home = choose_blocker(square, neigh, skip_squares, skip_kinds)
    if not home:
        return None
    blocked = set(occupied if occupied is not None else occ) | skip_squares | {home}
    park = choose_park({}, blocked, occupied if occupied is not None else occ)
    if not park:
        return None
    return Move(kind=piece_kind(home, neigh[home]), src=home, dst=park)


def plan_next_unstack(
    sim,
    cfg: dict,
    square: str,
    kind: str,
    skip_squares: set[str],
    skip_kinds: set[str],
    prefer: str | None = None,
) -> Move | None:
    occ = occupied_squares(sim, cfg)
    return plan_unstack_move(square, kind, occ, skip_squares, skip_kinds, set(occ), prefer=prefer)


def choose_park(cfg: dict, blocked: set[str], occ: dict | set | None = None) -> str | None:
    """Empty square with the fewest occupied neighbors. Rank-3 was too tight
    once every man is a real mesh, not a proxy cylinder."""
    _ = cfg
    sitting = set(occ if occ is not None else blocked) | set(blocked)
    if occ is not None and not isinstance(occ, set):
        sitting = set(occ) | set(blocked)
    candidates = [sq for sq in _PARK_ORDER if sq not in blocked]
    if not candidates:
        return None

    def clutter(sq: str) -> tuple[int, int]:
        n = sum(1 for nb in neighbor_squares(sq) if nb in sitting)
        return (n, _PARK_ORDER.index(sq))

    candidates.sort(key=clutter)
    return candidates[0]


def choose_blocker(
    source: str,
    occ: dict[str, str],
    skip_squares: set[str],
    skip_kinds: set[str],
) -> str | None:
    """Neighbor most aligned with the clutter cluster. Pawns first. No same-type."""
    candidates = []
    for sq, body in occ.items():
        if sq in skip_squares:
            continue
        if piece_kind(sq, body) in skip_kinds:
            continue
        candidates.append(sq)
    if not candidates:
        return None
    fi, rank = parse_square(source)
    acc = np.zeros(2)
    for sq in candidates:
        f2, r2 = parse_square(sq)
        acc += np.array([float(f2 - fi), float(r2 - rank)])
    if float(np.linalg.norm(acc)) < 0.25:
        acc = np.array([3.5 - fi, 4.5 - rank])

    def score(sq: str) -> tuple:
        kind = piece_kind(sq, occ[sq])
        f2, r2 = parse_square(sq)
        v = np.array([float(f2 - fi), float(r2 - rank)])
        return (0 if kind == "pawn" else 1, -float(np.dot(v, acc)))

    candidates.sort(key=score)
    return candidates[0]


@register("unstack_policy", "pawns_first")
class PawnsFirst:
    """Park the blocking neighbour most aligned with the clutter, pawns first, on the emptiest rank-3..6 square."""

    max_moves = MAX_UNSTACK

    def next(self, sim, cfg: dict, square: str, kind: str, skip_squares: set[str], skip_kinds: set[str], prefer: str | None = None):
        return plan_next_unstack(sim, cfg, square, kind, skip_squares, skip_kinds, prefer=prefer)
