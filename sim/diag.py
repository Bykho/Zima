"""Trace events from the controller to whoever is measuring it.

The controller only emits; it never reads what subscribers do. Runner suites
and debug tools subscribe a callback `fn(kind, sim, name, data)` for one
episode and take their world-side measurements there (through `world_of`).

Events today:
  phase       every `pickup.log_state` name (`001_initial` ... `010_final`, `fail_*`, `recover_*`); data: extra
  scan_start  / scan_end   around `perceive.perceive_board`
  look_start  / look       around `pickup.verify_piece`; data: tid, square (+ out on `look`)
"""

from __future__ import annotations

from contextlib import contextmanager

_subscribers: list = []


def emit(kind: str, sim, name: str = "", **data) -> None:
    for fn in list(_subscribers):
        fn(kind, sim, name, data)


@contextmanager
def subscribe(fn):
    _subscribers.append(fn)
    try:
        yield fn
    finally:
        _subscribers.remove(fn)
