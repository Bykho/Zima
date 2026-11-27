#!/usr/bin/env python3
"""Alias for `sim/run.py` with the old defaults (opening suite, cheat + real).

Rows land in runs/<tag>/ and are mirrored to sim/debug/harness/<tag>.{jsonl,txt}.

  python sim/harness.py                              # opening suite, cheat + real
  python sim/harness.py --suite opera --walls real
  python sim/harness.py --suite all --jobs 2 --tag baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import run  # noqa: E402
from evaluate_move import tilt_deg as _tilt_deg  # noqa: E402,F401
from runner import run_guarded  # noqa: E402,F401
from suites.chess_moves import _pads_in_piece, _setup_world, suite_cases  # noqa: E402,F401

OUT = ROOT / "sim" / "debug" / "harness"


def main() -> int:
    argv = sys.argv[1:]
    if not any(a == "--suite" or a.startswith("--suite=") for a in argv):
        argv = ["--suite", "opening", *argv]
    return run.main(argv, default_walls="cheat,real", mirror=OUT)


if __name__ == "__main__":
    sys.exit(main())
