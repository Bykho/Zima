#!/usr/bin/env python3
"""Alias for `sim/run.py`: the queen grasp studies on `chess_scene.xml` (cheat wall by default).

  python sim/evaluate.py --suite position_offsets      # suite `offsets`
  python sim/evaluate.py --suite perception_error
  python sim/evaluate.py --suite both                  # offsets + perception_error (default)
  python sim/evaluate.py --suite diagnostics           # perception_error; the report gives the robust radius
  python sim/evaluate.py --suite neighbors
  python sim/evaluate.py --suite transport
  python sim/evaluate.py --suite perception_error --walls cheat,real --tag nov22

Rows land in runs/<tag>/ and are mirrored to sim/debug/eval/<tag>.{jsonl,txt}.
The old grasp-compensation sweep is gone: the pickup grasp XY comes from the pinch model and
ignores `grasp_xy_offset`, so the sweep measured nothing. Nothing here writes config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import run  # noqa: E402

OUT = ROOT / "sim" / "debug" / "eval"
NAMES = {
    "position_offsets": "offsets",
    "both": "offsets,perception_error",
    "diagnostics": "perception_error",
}


def main() -> int:
    argv = sys.argv[1:]
    if "--suite" in argv:
        i = argv.index("--suite") + 1
        argv[i] = ",".join(NAMES.get(s, s) for s in argv[i].split(","))
    else:
        argv = ["--suite", NAMES["both"], *argv]
    return run.main(argv, default_walls="cheat", mirror=OUT)


if __name__ == "__main__":
    sys.exit(main())
