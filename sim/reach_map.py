#!/usr/bin/env python3
"""Alias for `sim/run.py --suite reach` (arm base grid x observe IK; see suites/kinematic.py).

  python sim/reach_map.py
  python sim/reach_map.py --quick
"""

from __future__ import annotations

import sys

from geom_sweep import main

if __name__ == "__main__":
    sys.exit(main("reach"))
