"""Suites for the batch runner (`sim/run.py`).

A suite is a case list plus a function that runs one case on one wall in a
fresh worker process and returns a row with at least id, wall, ok, reason,
stage and t. Register a new one by adding a `Suite` to SUITES.
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


def _chess(name: str, about: str) -> Suite:
    from suites import chess_moves

    return Suite(cases=lambda: chess_moves.suite_cases(name), run=chess_moves.run_case, fail_row=chess_moves.fail_row, about=about)


SUITES: dict[str, Suite] = {
    "opening": _chess("opening", "20 first moves from the start position"),
    "opera": _chess("opera", "every plain ply of the Opera game"),
    "immortal": _chess("immortal", "every plain ply of the Immortal game"),
    "evergreen": _chess("evergreen", "every plain ply of the Evergreen game"),
    "games": _chess("games", "opera + immortal + evergreen"),
    "core": _chess("core", "opening + opera (52 moves)"),
    "all": _chess("all", "opening + all three games"),
}
