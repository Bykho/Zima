"""Types shared by the move stages and the runner (`pickup.execute_move`)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Move:
    """One move in flight: the robot, its config, the scenario, the result row, and what stages hand on.

    Stages read the world only through `sim` (the wall applies). Values one
    stage produces for a later one are plain attributes (`queen_xyz`,
    `plan_xy`, `grasp`, `hold`, ...).
    """

    sim: object
    cfg: dict
    sc: dict
    result: dict
    params: dict = field(default_factory=dict)

    def __getattr__(self, name):
        raise AttributeError(f"Move has no {name!r}: an earlier stage did not set it")


@dataclass(frozen=True)
class Fail:
    """A stage that cannot go on. `reason` is today's failure code; `retry` picks the hand-off.

    retry: "pick" (re-grasp from where the piece is), "place" (fix a bad place), None (finish now).
    """

    stage: str
    reason: str | None
    retry: str | None = None


@dataclass(frozen=True)
class Stop:
    """A stage ended the move early with success (e.g. `stop_after: lift`)."""

    stage: str
