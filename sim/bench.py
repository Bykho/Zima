#!/usr/bin/env python3
"""Alias for `sim/run.py` with the old bench defaults (cheat wall).

  python sim/bench.py                    # starting suite: pick-and-lift every man, gt + vision
  python sim/bench.py --matrix           # matrix suite: piece x square on an empty board
  python sim/bench.py --matrix --quick   # the suite's quick subset
  python sim/bench.py --walls cheat,real --tag nov22

Rows land in runs/<tag>/ and are mirrored to sim/debug/eval/<tag>.{jsonl,txt}.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import run  # noqa: E402
from suites import SUITES  # noqa: E402

OUT = ROOT / "sim" / "debug" / "eval"


def main() -> int:
    argv = sys.argv[1:]
    suite = "matrix" if "--matrix" in argv else "starting"
    argv = [a for a in argv if a != "--matrix"]
    if "--quick" in argv:
        argv.remove("--quick")
        argv += ["--only", ",".join(SUITES[suite].quick)]
    return run.main(["--suite", suite, *argv], default_walls="cheat", mirror=OUT)


if __name__ == "__main__":
    sys.exit(main())
