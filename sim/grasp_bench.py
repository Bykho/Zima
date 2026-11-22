"""One move of a single piece on an empty board, swept over piece yaw and grasp height.

  python sim/grasp_bench.py                       # every kind, 12 yaws, table heights
  python sim/grasp_bench.py --kinds knight --heights 6,10,14,18,22 --tag knight_h

Cheat wall, so only the grasp varies. The default grid is the `grasp` suite
(`python sim/run.py --suite grasp --walls cheat`); this CLI adds custom heights,
close targets and squares. Cases run through `suites.grasp.run_case`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from runner import run_guarded  # noqa: E402
from suites import grasp  # noqa: E402

OUT = ROOT / "sim" / "debug" / "grasp_bench"
TIMEOUT = 90.0


def _run(case: dict) -> dict:
    return grasp.run_case(case, "cheat", TIMEOUT)


def _fail(case: dict, reason: str, stage: str, t: float) -> dict:
    return grasp.fail_row(case, "cheat", reason, stage, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(grasp.BODIES))
    ap.add_argument("--yaws", default=",".join(f"{a:g}" for a in grasp.YAWS))
    ap.add_argument("--heights", default="", help="mm, comma separated; empty = the PIECES table")
    ap.add_argument("--square", default="d4")
    ap.add_argument("--dest", default="d6")
    ap.add_argument("--closed", type=float, default=None, help="gripper close target; default = the PIECES table")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    yaws = [float(a) for a in args.yaws.split(",")]
    heights = [float(h) / 1000 for h in args.heights.split(",")] if args.heights else [None]
    work = [grasp.make_case(k, y, h, args.closed, args.square, args.dest) for k in kinds for h in heights for y in yaws]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(run_guarded(_run, work, args.jobs, lambda c: TIMEOUT + 30, _fail))
    rows.sort(key=lambda r: (r["kind"], r["height_mm"] or 0, r["yaw"]))
    (OUT / f"{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = defaultdict(list)
    for r in rows:
        table[(r["kind"], r["height_mm"])].append(r)
    for (kind, h), rs in table.items():
        marks = " ".join(("#" if r["held"] else ".") for r in rs)
        print(f"{kind:7s} h={str(h):5s} held {sum(r['held'] for r in rs):2d}/{len(rs)}  yaw 0..330: {marks}")
    fails = [r for r in rows if not r["held"]]
    for r in fails[:12]:
        print(f"  {r['kind']} yaw={r['yaw']:.0f} h={r['height_mm']} {r.get('reason')} close_tilt={r.get('close_tilt')} carry_tilt={r.get('carry_tilt')} plumb={r.get('plumb')} squeeze={r.get('squeeze_close')}/{r.get('squeeze_lift')}")


if __name__ == "__main__":
    main()
