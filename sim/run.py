#!/usr/bin/env python3
"""Run suites x walls in parallel and save the rows under runs/<tag>/. Scores from the world.

  python sim/run.py --suite core --walls cheat,real --tag nov22
  python sim/run.py --suite core --walls real --only open_e2e4,opera_05 --tag probe
  python sim/run.py --suite core --walls cheat --variants base,legacy=carry:legacy_only --tag ab
  python sim/run.py --suite core --walls real --stage carry=legacy_only --tag try
  python sim/run.py --list

Stage selection (lowest first): config.yaml `pipeline:`, `--stage stage=impl`, `--variants`.
A variant is `label=stage:impl[,stage:impl][;stage.key=value]`; `base` is the default pipeline.
Rows carry the variant label, the resolved pipeline and a hash of its params;
`compare.py runs/<tag> --by variant` puts each variant beside `base`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import runner  # noqa: E402
from stages import cli_arg as stages_cli_arg  # noqa: E402


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parser(default_walls: str = "cheat,real") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="core", help="comma list of suites (see --list)")
    ap.add_argument("--walls", default=default_walls, help="comma list of robot walls")
    ap.add_argument("--jobs", type=int, default=runner.DEFAULT_JOBS)
    ap.add_argument("--timeout", type=float, default=None, help="seconds per case (default: the suite's)")
    ap.add_argument("--only", default="", help="comma-separated substrings; keep cases whose id contains any")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--list", action="store_true", help="print suites and walls, then exit")
    ap.add_argument("--realism", action="store_true", help="rough physics + noisy depth (sim/realism.py); rows get variant 'realism'")
    stages_cli_arg(ap)
    ap.add_argument("--variants", default="", help="A/B pipelines, e.g. base,new=carry:legacy_only (see above)")
    return ap


def list_all() -> None:
    from robot import WALLS
    from suites import SUITES

    print("suites:")
    for name, s in SUITES.items():
        print(f"  {name:12s} {s.about}")
    print("walls:  " + ", ".join(WALLS))
    import stages

    print("stages (default first):")
    for stage, names in stages.registered().items():
        default = stages.DEFAULTS[stage]
        print(f"  {stage:15s} {', '.join([default] + [n for n in names if n != default])}")


def main(argv: list[str] | None = None, default_walls: str = "cheat,real", mirror: Path | None = None) -> int:
    args = parser(default_walls).parse_args(argv)
    if args.list:
        list_all()
        return 0
    from robot import WALLS
    from suites import SUITES

    suites, walls = _csv(args.suite), _csv(args.walls)
    bad = [s for s in suites if s not in SUITES] + [w for w in walls if w not in WALLS]
    if bad:
        raise SystemExit(f"unknown suite/wall: {', '.join(bad)} (see --list)")
    n_var = len(runner.parse_variants(args.variants)) if args.variants else 1
    n = sum(len(runner.select_cases(s, _csv(args.only))) * len(SUITES[s].walls(walls)) * (1 if SUITES[s].kinematic else n_var)
            for s in suites)
    print(f"{n} episodes on {args.jobs} workers")
    if args.realism:
        os.environ["ZIMA_REALISM"] = "1"
    import stages

    base_spec = stages.select(args.stage)
    variants = runner.parse_variants(args.variants, base_spec) if args.variants else None
    for _, spec in variants or ():
        stages.resolve(None, spec)
    rows, run_dir = runner.run(suites, walls, jobs=args.jobs, timeout=args.timeout, only=_csv(args.only),
                               tag=args.tag, on_row=runner.print_row, variants=variants)
    report = (run_dir / "summary.txt").read_text()
    print(report, end="")
    if mirror is not None:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / f"{args.tag}.jsonl").write_text((run_dir / "rows.jsonl").read_text())
        (mirror / f"{args.tag}.txt").write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
