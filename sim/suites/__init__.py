"""Suites for the batch runner (`sim/run.py`).

A suite is a case list plus a function that runs one case on one wall in a
fresh worker process and returns a row with at least id, wall, ok, reason,
stage and t. Register a new one by adding a `Suite` to SUITES.

`quick` names the case ids `verify --studies` runs: on `gate_wall` they must not
regress against `baselines/<suite>_<gate_wall>`; on `record_walls` they are
compared and printed but do not fail verify. `report` adds a suite-specific
view (grids, board maps) to the run summary. A `kinematic` suite has no
controller: it runs once on the pseudo-wall `none`, its rows carry `metrics`,
and its baseline is `baselines/<suite>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Suite:
    cases: Callable[[], list[dict]]
    run: Callable[[dict, str, float], dict]
    fail_row: Callable[[dict, str, str, str, float], dict]
    timeout: float = 90.0
    about: str = ""
    quick: tuple[str, ...] = ()
    gate_wall: str = "cheat"
    record_walls: tuple[str, ...] = ("real",)
    report: Callable[[list[dict]], str] | None = None
    kinematic: bool = False

    def walls(self, asked: list[str]) -> list[str]:
        return ["none"] if self.kinematic else asked


def _chess(name: str, about: str, quick: tuple[str, ...] = ()) -> Suite:
    from suites import chess_moves

    return Suite(cases=lambda: chess_moves.suite_cases(name), run=chess_moves.run_case, fail_row=chess_moves.fail_row,
                 about=about, quick=quick)


def _grasp() -> Suite:
    from suites import grasp

    return Suite(cases=grasp.suite_cases, run=grasp.run_case, fail_row=grasp.fail_row,
                 about="every kind x 12 yaws, one piece alone, d4 -> d6 (camera pose on walls without truth)",
                 quick=("pawn_y000", "pawn_y030", "knight_y000", "knight_y210", "queen_y090", "king_y330"))


def _pickups(name: str) -> Suite:
    from suites import pickups

    if name == "matrix":
        return Suite(cases=pickups.matrix_cases, run=pickups.run_matrix, fail_row=pickups.fail_row,
                     about="piece x square on an empty board (chess_scene_pieces.xml)", report=pickups.report_matrix,
                     quick=("empty_pawn_d1_d4", "empty_queen_d4_e4", "empty_knight_a4_d4", "dest_queen_d1_h6",
                            "empty_queen_d1_d4"))
    if name == "starting":
        return Suite(cases=pickups.starting_cases, run=pickups.run_starting, fail_row=pickups.fail_row, timeout=150.0,
                     about="pick-and-lift of every man on the starting board", report=pickups.report_starting,
                     quick=("pick_queen_d1", "pick_pawn_a2", "pick_king_e1"))
    cases, about, quick = {
        "offsets": (pickups.offsets_cases, "queen spawned off the d1 centre, pick-and-lift (chess_scene.xml)",
                    ("pos_x_+0mm", "pos_x_+10mm", "pos_xy_+5_+5mm")),
        "perception_error": (pickups.perception_error_cases, "perceived queen XY off by -8..+8 mm, pick-and-lift",
                             ("perc_+0_+0mm", "perc_+4_-4mm", "perc_-8_+8mm")),
        "neighbors": (pickups.neighbors_cases, "one pawn beside d1, pick-and-lift",
                      ("neighbor_left_c1", "neighbor_front_d2")),
        "transport": (pickups.transport_cases, "d1 -> d4 with one pawn near d1, full move",
                      ("transport_none", "transport_d2")),
    }[name]
    report = {"offsets": pickups.report_offsets, "perception_error": pickups.report_perception_error,
              "neighbors": pickups.report_neighbors,
              "transport": pickups.report_neighbors}[name]
    return Suite(cases=cases, run=pickups.run_queen, fail_row=pickups.fail_row, about=about, quick=quick, report=report)


def _kinematic(name: str) -> Suite:
    from suites import kinematic

    if name == "layout":
        return Suite(cases=kinematic.layout_cases, run=kinematic.run_layout, fail_row=kinematic.fail_row, timeout=300.0,
                     about="board square size x a-file gap: 64-square IK reach, open-gripper neighbour gaps (no wall)",
                     quick=("layout_config", "layout_40.0_100"), kinematic=True, gate_wall="none", record_walls=(),
                     report=kinematic.report_layout)
    return Suite(cases=kinematic.reach_cases, run=kinematic.run_reach, fail_row=kinematic.fail_row, timeout=120.0,
                 about="arm base position grid: observe IK over the starting squares (no wall)",
                 quick=("base_0.00_0.00", "base_0.08_0.03"), kinematic=True, gate_wall="none", record_walls=(),
                 report=kinematic.report_reach)


SUITES: dict[str, Suite] = {
    "opening": _chess("opening", "20 first moves from the start position"),
    "opera": _chess("opera", "every plain ply of the Opera game"),
    "immortal": _chess("immortal", "every plain ply of the Immortal game"),
    "evergreen": _chess("evergreen", "every plain ply of the Evergreen game"),
    "games": _chess("games", "opera + immortal + evergreen"),
    "core": _chess("core", "opening + opera (52 moves)", quick=("open_e2e4", "open_b8c6", "opera_07_d4e5", "opera_24_a8d8")),
    "all": _chess("all", "opening + all three games"),
    "grasp": _grasp(),
    "matrix": _pickups("matrix"),
    "starting": _pickups("starting"),
    **{n: _pickups(n) for n in ("offsets", "perception_error", "neighbors", "transport")},
    "layout": _kinematic("layout"),
    "reach": _kinematic("reach"),
}
