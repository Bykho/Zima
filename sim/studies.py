#!/usr/bin/env python3
"""Quick study subsets against the committed baselines (`verify --studies`).

  python sim/studies.py                 # every suite with a quick subset
  python sim/studies.py --suite core    # one suite
  python sim/studies.py --exact         # also fail on any reason/stage change on the gate wall

Each suite's `quick` cases run on its gate wall (fails on a regression against
baselines/<suite>_<gate>) and on its record walls (compared and printed only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import compare  # noqa: E402
import runner  # noqa: E402
from suites import SUITES  # noqa: E402


def check_suite(name: str, exact: bool = False, jobs: int = runner.DEFAULT_JOBS) -> bool:
    s = SUITES[name]
    walls = [s.gate_wall, *[w for w in s.record_walls if w != s.gate_wall]]
    missing = [w for w in walls if not (compare.BASELINES / f"{name}_{w}").is_dir()]
    if s.gate_wall in missing:
        print(f"[{name}] no baseline {name}_{s.gate_wall}: save one with `compare.py runs/<tag> --save-baseline`")
        return False
    rows, run_dir = runner.run([name], walls, jobs=jobs, ids=list(s.quick), tag=f"studies_{name}")
    ok = True
    for w in walls:
        if w in missing:
            print(f"[{name} {w}] no baseline, not compared")
            continue
        new = [r for r in rows if r["wall"] == w]
        old, _, label = compare.load(f"{name}_{w}")
        old = [r for r in old if r["id"] in s.quick]
        res = compare.compare(new, old)
        diff = bool(res["flips"] or res["changed"] or res["only_old"] or res["only_new"])
        regress = any(a["ok"] for _, a, _ in res["flips"]) or bool(res["only_new"])
        if exact:
            regress = regress or diff
        gate = w == s.gate_wall
        verdict = "EXACT" if not diff else ("REGRESSION" if regress else "OK")
        print(f"\n[{name} {w}{' gate' if gate else ' record'}] {verdict}")
        if diff:
            print(compare.report(res, str(run_dir.relative_to(ROOT)), label))
        if gate and regress:
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=None, help="comma list (default: every suite with a quick subset)")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--jobs", type=int, default=runner.DEFAULT_JOBS)
    args = ap.parse_args(argv)
    names = args.suite.split(",") if args.suite else [n for n, s in SUITES.items() if s.quick]
    results = {n: check_suite(n, exact=args.exact, jobs=args.jobs) for n in names}
    print("\nstudies: " + "  ".join(f"{n} {'PASS' if ok else 'FAIL'}" for n, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
