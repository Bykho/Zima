#!/usr/bin/env python3
"""Agent closed-loop check: bridge if running, else direct serial."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real import bridge
from real.arm import RealArm
from real.calib import find_port

if bridge.available():
    s = bridge.status()
    print("via bridge", s["port"])
    print("ticks", s["ticks"])
    print("ok")
    sys.exit(0)

port = find_port()
if not port:
    sys.exit("no port and no bridge — run: python scripts/arm_bridge.py")
arm = RealArm(port)
arm.connect()
print("direct", port)
print("ticks", arm.read_ticks())
arm.close()
print("ok")
