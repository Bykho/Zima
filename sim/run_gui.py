#!/usr/bin/env python3
"""Launch the interactive sim (same as ./sim/run_gui.sh)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interact import main

if __name__ == "__main__":
    main()
