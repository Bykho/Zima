#!/usr/bin/env python3
"""Closed-loop sim verification. Agent runs this after every sim change.

  python sim/verify.py          # full suite (~7s, fast headless)
  python sim/verify.py --quick  # smoke + core tests (~5s)
  python sim/verify.py --slow   # wall-clock motion + all tests (~2+ min)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python3"


def run(cmd: list[str], label: str, env: dict | None = None) -> bool:
    print(f"\n=== {label} ===")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    ok = r.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}: {label} ({time.time() - t0:.1f}s)")
    return ok


def main():
    quick = "--quick" in sys.argv
    slow = "--slow" in sys.argv
    env = os.environ.copy()
    if not slow:
        env["ZIMA_SIM_FAST"] = "1"

    test_cmd = [str(PY), "sim/test_sim.py"] + ([] if slow else ["--fast"])
    steps = [
        ([str(PY), "sim/arm.py", "--test"], "smoke", env),
        (test_cmd, "test_sim", env),
    ]
    if not quick:
        steps.append(([str(PY), "sim/demo_chess.py", "--headless"], "chess e2e4", env))

    t0 = time.time()
    results = [run(cmd, label, e) for cmd, label, e in steps]
    passed = sum(results)
    total = len(results)
    mode = "quick" if quick else ("slow" if slow else "fast")
    print(f"\n{mode} verify: {passed}/{total} in {time.time() - t0:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
