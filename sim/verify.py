#!/usr/bin/env python3
"""Closed-loop sim verification. Agent runs this after every sim change.

  python sim/verify.py            # MuJoCo tests + centered GT and vision pick-place
  python sim/verify.py --quick    # load/joints/IK only
  python sim/verify.py --studies  # plus each study suite's quick subset vs its committed baseline
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python3"


def run(cmd: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}: {label} ({time.time() - t0:.1f}s)")
    return ok


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    steps = [
        ([str(PY), "sim/mj.py"], "smoke"),
        ([str(PY), "sim/test_mj.py"] + (["--quick"] if quick else []), "test_mj"),
    ]
    if not quick:
        pick = [str(PY), "sim/pickup.py", "--headless"]
        steps.append((pick + ["--wall", "off"], "pick-place (no wall, true poses)"))
        steps.append((pick + ["--wall", "cheat"], "pick-place (robot wall, cheat: true poses)"))
        steps.append((pick, "vision pick-place (robot wall, real)"))
    if "--studies" in sys.argv:
        steps.append(([str(PY), "sim/studies.py"], "studies (quick subsets vs baselines)"))
    results = [run(cmd, label) for cmd, label in steps]
    passed = sum(results)
    total = len(results)
    mode = "quick" if quick else "full"
    print(f"\n{mode} verify: {passed}/{total} in {time.time() - t0:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
