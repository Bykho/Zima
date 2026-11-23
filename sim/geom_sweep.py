#!/usr/bin/env python3
"""Alias for `sim/run.py --suite layout` (board square size x a-file gap; see suites/kinematic.py).

  python sim/geom_sweep.py                 # every layout
  python sim/geom_sweep.py --quick         # the configured layout and one sweep cell

Rows land in runs/<tag>/; compare with `python sim/compare.py runs/<tag>` (baseline `layout`).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import run  # noqa: E402
from suites import SUITES  # noqa: E402


def main(suite: str = "layout") -> int:
    argv = sys.argv[1:]
    if "--quick" in argv:
        argv.remove("--quick")
        argv += ["--only", ",".join(SUITES[suite].quick)]
    return run.main(["--suite", suite, "--tag", suite, *argv])


if __name__ == "__main__":
    sys.exit(main())
