"""Minimal pickup scenario dicts. Adding a case should be data, not code."""

from __future__ import annotations

# Grasp XY is gripper-frame compensation (frozen queen values), not a per-square hack.
_GRASP_XY = [0.00376, 0.00866]

PIECES = {
    "pawn": {
        "kind": "pawn",
        "height": 0.037,
        "width": 0.018,
        "stem_width": 0.011,
        "preferred_grasp_height": 0.012,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.08,
    },
    "rook": {
        "kind": "rook",
        "height": 0.036,
        "width": 0.024,
        "stem_width": 0.016,
        "preferred_grasp_height": 0.016,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.04,
    },
    "bishop": {
        "kind": "bishop",
        "height": 0.044,
        "width": 0.020,
        "stem_width": 0.011,
        "preferred_grasp_height": 0.016,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.08,
    },
    "knight": {
        "kind": "knight",
        "height": 0.036,
        "width": 0.020,
        "stem_width": 0.014,
        "preferred_grasp_height": 0.014,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.05,
    },
    "queen": {
        "kind": "queen",
        "height": 0.055,
        "width": 0.022,
        "stem_width": 0.009,
        "preferred_grasp_height": 0.015,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.12,
    },
    "king": {
        "kind": "king",
        "height": 0.054,
        "width": 0.024,
        "stem_width": 0.012,
        "preferred_grasp_height": 0.018,
        "grasp_xy_offset": list(_GRASP_XY),
        "gripper_closed": -0.06,
    },
}

QUEEN_GEOM = PIECES["queen"]
PIECE_BODIES = tuple(PIECES)

# Representative workspace sample (near / middle / far).
NEAR_SQUARES = ("d1", "c2")
MIDDLE_SQUARES = ("d4", "c4", "e4")
FAR_SQUARES = ("a4", "a6", "d6", "f6", "h4", "h6")
SOURCE_SQUARES = NEAR_SQUARES + MIDDLE_SQUARES + FAR_SQUARES
DEST_SQUARES = SOURCE_SQUARES

HIDE_XYZ = [2.0, 2.0, 0.02]

OFFSET_1D_MM = (0, 1, 2, 3, 5, 7, 10)
OFFSET_2D_MM = (5, 10)
PERC_ERR_MM = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
NOM_GRASP_XY_MM = (3.76, 8.66)
COMP_DELTA_MM = (-4, -2, 0, 2, 4)
PROBE_ERR_MM = (
    (0, 0),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (2, 2),
    (2, -2),
    (-2, 2),
    (-2, -2),
    (4, 0),
    (-4, 0),
    (0, 4),
    (0, -4),
)


def scenario(**kwargs) -> dict:
    s = {
        "name": "queen_d1_d4",
        "target_id": "queen",
        "source": "d1",
        "dest": "d4",
        "offset_xy": [0.0, 0.0],
        "yaw_deg": 0.0,
        "neighbors": [],
        "geometry": QUEEN_GEOM,
        "perception": "gt",
        "pose_error_xy": [0.0, 0.0],
        "seed": 0,
        "quiet": False,
    }
    s.update(kwargs)
    return s


def _signed(values) -> list[int]:
    out = []
    seen = set()
    for v in values:
        for s in (0, 1, -1) if v == 0 else (1, -1):
            mm = s * v
            if mm in seen:
                continue
            seen.add(mm)
            out.append(mm)
    return out


def position_offset_scenarios() -> list[dict]:
    cases = []
    for mm in _signed(OFFSET_1D_MM):
        cases.append(
            scenario(
                name=f"pos_x_{mm:+d}mm",
                offset_xy=[mm / 1000.0, 0.0],
                quiet=True,
            )
        )
    for mm in _signed(OFFSET_1D_MM):
        if mm == 0:
            continue
        cases.append(
            scenario(
                name=f"pos_y_{mm:+d}mm",
                offset_xy=[0.0, mm / 1000.0],
                quiet=True,
            )
        )
    for x in _signed(OFFSET_2D_MM):
        for y in _signed(OFFSET_2D_MM):
            if x == 0 or y == 0:
                continue
            cases.append(
                scenario(
                    name=f"pos_xy_{x:+d}_{y:+d}mm",
                    offset_xy=[x / 1000.0, y / 1000.0],
                    quiet=True,
                )
            )
    return cases


def probe_error_scenarios(grasp_xy_offset=None) -> list[dict]:
    cases = []
    off = None
    if grasp_xy_offset is not None:
        off = [float(grasp_xy_offset[0]), float(grasp_xy_offset[1])]
    for x, y in PROBE_ERR_MM:
        kw = dict(
            name=f"probe_{x:+d}_{y:+d}mm",
            pose_error_xy=[x / 1000.0, y / 1000.0],
            quiet=True,
        )
        if off is not None:
            kw["grasp_xy_offset"] = off
        cases.append(scenario(**kw))
    return cases


def perception_error_scenarios(grasp_xy_offset=None) -> list[dict]:
    cases = []
    off = None
    if grasp_xy_offset is not None:
        off = [float(grasp_xy_offset[0]), float(grasp_xy_offset[1])]
    for x in PERC_ERR_MM:
        for y in PERC_ERR_MM:
            kw = dict(
                name=f"perc_{x:+d}_{y:+d}mm",
                pose_error_xy=[x / 1000.0, y / 1000.0],
                quiet=True,
            )
            if off is not None:
                kw["grasp_xy_offset"] = off
            cases.append(scenario(**kw))
    return cases


def neighbor_scenarios() -> list[dict]:
    """One pawn on each on-board square adjacent to the source. Grasp policy unchanged."""
    from squares import shift_square

    source = "d1"
    dirs = (("left", -1, 0), ("right", 1, 0), ("front", 0, 1), ("rear", 0, -1))
    cases = []
    for name, df, dr in dirs:
        sq = shift_square(source, df, dr)
        if sq is None:
            continue
        cases.append(
            scenario(
                name=f"neighbor_{name}_{sq}",
                neighbors=[{"id": "pawn", "square": sq}],
                quiet=True,
            )
        )
    return cases


def transport_scenarios() -> list[dict]:
    """Straight vs clearance carry: empty board, orthogonal, and diagonal d1 neighbors."""
    return [
        scenario(name="transport_none", neighbors=[], quiet=True),
        scenario(name="transport_c1", neighbors=[{"id": "pawn", "square": "c1"}], quiet=True),
        scenario(name="transport_e1", neighbors=[{"id": "pawn", "square": "e1"}], quiet=True),
        scenario(name="transport_d2", neighbors=[{"id": "pawn", "square": "d2"}], quiet=True),
        scenario(name="transport_c2", neighbors=[{"id": "pawn", "square": "c2"}], quiet=True),
        scenario(name="transport_e2", neighbors=[{"id": "pawn", "square": "e2"}], quiet=True),
    ]


def full_board_scenario(**kwargs) -> dict:
    """Starting-position board; pieces live in chess_scene_full.xml, not as pickup neighbors."""
    kw = dict(name="full_board_d1_d4", quiet=kwargs.pop("quiet", False))
    kw.update(kwargs)
    return scenario(**kw)


def piece_scenario(kind: str, source: str, dest: str, perception: str = "gt", **kwargs) -> dict:
    geom = dict(PIECES[kind])
    kw = dict(
        name=kwargs.pop("name", f"{kind}_{source}_{dest}_{perception}"),
        target_id=kind,
        source=source,
        dest=dest,
        geometry=geom,
        perception=perception,
        grasp_xy_offset=list(geom["grasp_xy_offset"]),
        quiet=kwargs.pop("quiet", True),
    )
    kw.update(kwargs)
    return scenario(**kw)


def _default_dest(source: str) -> str:
    return "e4" if source == "d4" else "d4"


def empty_matrix_scenarios(perception: str = "gt") -> list[dict]:
    """One dest per source so the sweep stays broad without a full cartesian product."""
    cases = []
    for kind in PIECE_BODIES:
        for src in SOURCE_SQUARES:
            cases.append(piece_scenario(kind, src, _default_dest(src), perception=perception))
    return cases


def dest_reach_scenarios(perception: str = "gt") -> list[dict]:
    """Queen from d1 onto each representative dest — workspace/placement map."""
    return [piece_scenario("queen", "d1", dest, perception=perception) for dest in DEST_SQUARES if dest != "d1"]


# Standard chess starting squares. One free body per type; other squares are proxies.
STARTING_POSITION = (
    ("rook", "a1"),
    ("knight", "b1"),
    ("bishop", "c1"),
    ("queen", "d1"),
    ("king", "e1"),
    ("bishop", "f1"),
    ("knight", "g1"),
    ("rook", "h1"),
    ("pawn", "a2"),
    ("pawn", "b2"),
    ("pawn", "c2"),
    ("pawn", "d2"),
    ("pawn", "e2"),
    ("pawn", "f2"),
    ("pawn", "g2"),
    ("pawn", "h2"),
    ("pawn", "a7"),
    ("pawn", "b7"),
    ("pawn", "c7"),
    ("pawn", "d7"),
    ("pawn", "e7"),
    ("pawn", "f7"),
    ("pawn", "g7"),
    ("pawn", "h7"),
    ("rook", "a8"),
    ("knight", "b8"),
    ("bishop", "c8"),
    ("queen", "d8"),
    ("king", "e8"),
    ("bishop", "f8"),
    ("knight", "g8"),
    ("rook", "h8"),
)
STARTING_AT = {sq: kind for kind, sq in STARTING_POSITION}
# Empty on a starting board (ranks 3–6). Near / center / far-center.
STARTING_DESTS = ("d4", "a4", "e5")


def populated_matrix_scenarios(perception: str = "gt") -> list[dict]:
    """Legacy subset. Prefer starting_pickup_scenarios for the full board map."""
    return starting_pickup_scenarios(perception, dest="d4", stop_after=None)


def starting_pickup_scenarios(perception: str = "gt", dest: str = "d4", stop_after: str | None = "lift") -> list[dict]:
    """Pickup (+ optional lift-only) every starting piece against full clutter."""
    cases = []
    for kind, src in STARTING_POSITION:
        tag = "pick" if stop_after == "lift" else "place"
        cases.append(
            piece_scenario(
                kind,
                src,
                dest,
                perception=perception,
                name=f"start_{tag}_{kind}_{src}_{dest}_{perception}",
                hide_proxy=src,
                stop_after=stop_after,
            )
        )
    return cases


def starting_place_scenario(kind: str, source: str, dest: str, perception: str = "gt") -> dict:
    return piece_scenario(
        kind,
        source,
        dest,
        perception=perception,
        name=f"start_place_{kind}_{source}_{dest}_{perception}",
        hide_proxy=source,
        stop_after=None,
    )
