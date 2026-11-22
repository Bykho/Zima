"""Suites for the batch runner (`sim/run.py`).

A suite is a case list plus a function that runs one case on one wall in a
fresh worker process and returns a row with at least id, wall, ok, reason,
stage and t. Register a new one by adding a `Suite` to SUITES.

`quick` names the case ids `verify --studies` runs: on `gate_wall` they must not
regress against `baselines/<suite>_<gate_wall>`; on `record_walls` they are
compared and printed but do not fail verify. `report` adds a suite-specific
view (grids, board maps) to the run summary.
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


def _chess(name: str, about: str, quick: tuple[str, ...] = ()) -> Suite:
    from suites import chess_moves

    return Suite(cases=lambda: chess_moves.suite_cases(name), run=chess_moves.run_case, fail_row=chess_moves.fail_row,
                 about=about, quick=quick)


def _grasp() -> Suite:
    from suites import grasp

    return Suite(cases=grasp.suite_cases, run=grasp.run_case, fail_row=grasp.fail_row,
                 about="every kind x 12 yaws, one piece alone, d4 -> d6", record_walls=(),
                 quick=("pawn_y000", "pawn_y030", "knight_y000", "knight_y210", "queen_y090", "king_y330"))


def _pickups(name: str) -> Suite:
    from suites import pickups

    if name == "matrix":
        return Suite(cases=pickups.matrix_cases, run=pickups.run_matrix, fail_row=pickups.fail_row,
                     about="piece x square on an empty board (chess_scene_pieces.xml)", report=pickups.report_matrix,
                     quick=("empty_pawn_d1_d4_gt", "empty_queen_d4_e4_gt", "empty_knight_a4_d4_gt",
                            "dest_queen_d1_h6_gt", "empty_queen_d1_d4_vision"))
    if name == "starting":
        return Suite(cases=pickups.starting_cases, run=pickups.run_starting, fail_row=pickups.fail_row, timeout=150.0,
                     about="pick-and-lift of every man on the starting board, gt + vision", report=pickups.report_starting,
                     record_walls=(), quick=("pick_queen_d1_gt", "pick_pawn_a2_gt", "pick_king_e1_gt", "pick_queen_d1_vision"))
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
}
