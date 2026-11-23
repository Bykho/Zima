#!/usr/bin/env python3
"""Compare two batch runs move by move: totals per suite x wall, flips, reason/stage changes, phase times.

  python sim/compare.py runs/new                     # against the committed baselines/<suite>_<wall>
  python sim/compare.py runs/new runs/old            # two runs
  python sim/compare.py runs/new core_cheat          # a named baseline
  python sim/compare.py runs/new sim/debug/harness/stock_baseline.jsonl
  python sim/compare.py runs/new --exact             # exit 1 on any ok/reason/stage difference

Exit codes: 0 no regression, 1 regression (ok -> fail; with --exact any difference
or missing row), 2 the two sides ran on different config.yaml (pass --allow-config).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines"
FIELDS = ("ok", "reason", "stage")
# Kinematic suite metrics: counts must match; *_mm within MM_TOL. Higher is better unless listed.
MM_TOL = 0.5
LOWER_IS_BETTER = {"n_collide"}


def baseline_name(suite: str, wall: str) -> str:
    """`<suite>_<wall>`, or `<suite>` for a kinematic suite (wall `none`)."""
    return suite if wall == "none" else f"{suite}_{wall}"


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        r.setdefault("suite", "core")
    return rows


def load(spec: str) -> tuple[list[dict], dict, str]:
    """(rows, meta, label) from a run dir, a rows .jsonl, or a baseline name."""
    p = Path(spec)
    if not p.exists() and (BASELINES / spec).is_dir():
        p = BASELINES / spec
    if p.is_dir():
        meta_path = p / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return _read_rows(p / "rows.jsonl"), meta, str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    if p.suffix == ".jsonl" and p.exists():
        return _read_rows(p), {}, str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    raise SystemExit(f"no run, rows file or baseline named {spec}")


def load_baselines(rows: list[dict]) -> tuple[list[dict], dict, str]:
    """Every committed baselines/<suite>_<wall> that the new rows cover."""
    out: list[dict] = []
    metas: dict = {}
    names = []
    for suite, wall in sorted({(r.get("suite", ""), r["wall"]) for r in rows}):
        name = baseline_name(suite, wall)
        if not (BASELINES / name).is_dir():
            print(f"note: no baseline {name}", file=sys.stderr)
            continue
        rs, meta, _ = load(name)
        out += rs
        metas[name] = meta
        names.append(name)
    if not out:
        raise SystemExit("no baselines cover these rows")
    return out, next(iter(metas.values())) if len(metas) == 1 else {"config_sha": {m.get("config_sha") for m in metas.values()}}, "baselines " + ",".join(names)


def key(r: dict, with_suite: bool) -> tuple:
    return ((r.get("suite", ""),) if with_suite else ()) + (r["id"], r["wall"], r.get("variant", "base"))


def group(r: dict) -> tuple:
    return (r.get("suite", ""), r["wall"], r.get("variant", "base"))


def totals(rs: list[dict]) -> dict:
    if not rs:
        return {"n": 0, "ok": 0, "lying": 0, "disturbed": 0, "t_med": 0.0}
    return {
        "n": len(rs),
        "ok": sum(bool(r["ok"]) for r in rs),
        "lying": sum(1 for r in rs if r.get("claimed") and not r["ok"]),
        "disturbed": sum(len(r.get("disturbed") or []) for r in rs),
        "t_med": median(r["t"] for r in rs),
    }


def _fmt(r: dict | None) -> str:
    if r is None:
        return "missing"
    return ("ok" if r["ok"] else "FAIL") + (f" {r.get('stage') or '?'}/{r.get('reason')}" if not r["ok"] else "")


def metric_changes(a: dict, b: dict) -> list[tuple[str, object, object, bool]]:
    """(name, old, new, worse) for each kinematic metric outside tolerance."""
    out = []
    ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
    for name in sorted(set(ma) | set(mb)):
        x, y = ma.get(name), mb.get(name)
        if x is None or y is None:
            if x != y:
                out.append((name, x, y, y is None))
            continue
        tol = MM_TOL if name.endswith("_mm") else 0
        if abs(y - x) <= tol:
            continue
        out.append((name, x, y, (y > x) if name in LOWER_IS_BETTER else (y < x)))
    return out


def compare(new: list[dict], old: list[dict]) -> dict:
    with_suite = all("suite" in r for r in new + old)
    A = {key(r, with_suite): r for r in old}
    B = {key(r, with_suite): r for r in new}
    groups = sorted({group(r) for r in new} | {group(r) for r in old if key(r, with_suite) in B})
    out: dict = {"groups": [], "flips": [], "changed": [], "only_old": [], "only_new": [], "phases": [], "metrics": []}
    for g in groups:
        rb = [r for r in new if group(r) == g]
        ra = [A[key(r, with_suite)] for r in rb if key(r, with_suite) in A]
        out["groups"].append((g, totals(ra), totals(rb)))
        pa: Counter = Counter()
        pb: Counter = Counter()
        for r in ra:
            pa.update(r.get("phases") or {})
        for r in rb:
            pb.update(r.get("phases") or {})
        if ra and rb:
            out["phases"].append((g, {k: (pa[k] / len(ra), pb[k] / len(rb)) for k in sorted(set(pa) | set(pb))}))
    for k, b in B.items():
        a = A.get(k)
        if a is None:
            out["only_new"].append(k)
        elif bool(a["ok"]) != bool(b["ok"]):
            out["flips"].append((k, a, b))
        elif any(a.get(f) != b.get(f) for f in FIELDS):
            out["changed"].append((k, a, b))
        if a is not None:
            out["metrics"] += [(k, *m) for m in metric_changes(a, b)]
    out["only_old"] = [k for k in A if k not in B and group(A[k]) in {group(r) for r in new}]
    return out


def report(res: dict, new_label: str, old_label: str) -> str:
    lines = [f"new: {new_label}", f"old: {old_label}", ""]
    lines.append(f"{'suite':10s} {'wall':11s} {'variant':10s} {'ok old':>8s} {'ok new':>8s} {'delta':>6s}  {'lying':>9s}  {'disturbed':>9s}  {'t med':>11s}")
    for (s, w, v), a, b in res["groups"]:
        lines.append(f"{s:10s} {w:11s} {v:10s} {a['ok']:>4d}/{a['n']:<3d} {b['ok']:>4d}/{b['n']:<3d} {b['ok'] - a['ok']:>+6d}  "
                     f"{a['lying']:>4d}>{b['lying']:<4d}  {a['disturbed']:>4d}>{b['disturbed']:<4d}  {a['t_med']:>5.1f}>{b['t_med']:<5.1f}")
    flips = res["flips"]
    lines.append(f"\nflips: {len(flips)}  ({sum(1 for _, a, b in flips if a['ok'])} ok->fail, {sum(1 for _, a, b in flips if not a['ok'])} fail->ok)")
    for k, a, b in sorted(flips, key=lambda f: (not f[1]["ok"], f[0])):
        lines.append(f"  {'/'.join(map(str, k)):44s} {_fmt(a):40s} -> {_fmt(b)}")
    if res["changed"]:
        lines.append(f"\nsame verdict, different reason/stage: {len(res['changed'])}")
        for k, a, b in res["changed"]:
            lines.append(f"  {'/'.join(map(str, k)):44s} {_fmt(a):40s} -> {_fmt(b)}")
    if res["metrics"]:
        worse = sum(1 for m in res["metrics"] if m[-1])
        lines.append(f"\nmetrics changed: {len(res['metrics'])}  ({worse} worse)")
        for k, name, x, y, bad in res["metrics"]:
            lines.append(f"  {'/'.join(map(str, k)):44s} {name:14s} {x} -> {y}{'  WORSE' if bad else ''}")
    if res["only_old"] or res["only_new"]:
        lines.append(f"\nonly in old: {len(res['only_old'])}   only in new: {len(res['only_new'])}")
        for k in res["only_old"][:20]:
            lines.append(f"  - {'/'.join(map(str, k))}")
        for k in res["only_new"][:20]:
            lines.append(f"  + {'/'.join(map(str, k))}")
    for (s, w, v), ph in res["phases"]:
        top = sorted(ph.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:6]
        lines.append(f"\nphase time per case [{s} {w} {v}]: " + ", ".join(f"{k} {a:.1f}>{b:.1f}s" for k, (a, b) in top))
    keys = [k for k, _, _ in flips + res["changed"]]
    if keys:
        ids, walls = sorted({k[-3] for k in keys}), sorted({k[-2] for k in keys})
        suites = sorted({k[0] for k in keys if len(k) == 4}) or ["core"]
        lines.append(f"\nrerun: python sim/run.py --suite {','.join(suites)} --walls {','.join(walls)} --only {','.join(ids)} --tag recheck")
    return "\n".join(lines)


def save_baselines(run_dir: Path) -> list[str]:
    """Split a run into baselines/<suite>_<wall>/ (rows + meta)."""
    rows = _read_rows(run_dir / "rows.jsonl")
    meta = json.loads((run_dir / "meta.json").read_text())
    saved = []
    for suite, wall in sorted({(r.get("suite", ""), r["wall"]) for r in rows}):
        name = baseline_name(suite, wall)
        d = BASELINES / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        rs = [r for r in rows if r.get("suite", "") == suite and r["wall"] == wall]
        (d / "rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rs))
        (d / "meta.json").write_text(json.dumps({**meta, "baseline": name, "from_run": run_dir.name}, indent=2) + "\n")
        saved.append(f"{name} ({sum(bool(r['ok']) for r in rs)}/{len(rs)})")
    return saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", help="run dir, rows .jsonl, or baseline name")
    ap.add_argument("old", nargs="?", default=None, help="same; default: the committed baselines the new rows cover")
    ap.add_argument("--exact", action="store_true", help="fail on any ok/reason/stage difference or missing row")
    ap.add_argument("--allow-config", action="store_true", help="do not fail when config.yaml differs")
    ap.add_argument("--save-baseline", action="store_true", help="write NEW (a run dir) to baselines/<suite>_<wall>/ and exit")
    args = ap.parse_args(argv)
    if args.save_baseline:
        for s in save_baselines(Path(args.new)):
            print(f"saved baseline {s}")
        return 0
    new, new_meta, new_label = load(args.new)
    old, old_meta, old_label = load(args.old) if args.old else load_baselines(new)
    res = compare(new, old)
    print(report(res, new_label, old_label))
    regress = any(a["ok"] for _, a, _ in res["flips"]) or any(m[-1] for m in res["metrics"])
    if args.exact:
        regress = regress or bool(res["flips"] or res["changed"] or res["metrics"] or res["only_old"] or res["only_new"])
    cfg_old, cfg_new = old_meta.get("config_sha"), new_meta.get("config_sha")
    if cfg_old and cfg_new and cfg_old != cfg_new and not isinstance(cfg_old, set) and not args.allow_config:
        print(f"\nconfig.yaml differs ({cfg_old} vs {cfg_new}); pass --allow-config if intended")
        return 2
    verdict = "EXACT" if not (res["flips"] or res["changed"] or res["metrics"] or res["only_old"] or res["only_new"]) else ("REGRESSION" if regress else "OK")
    print(f"\n{verdict}")
    return 1 if regress else 0


if __name__ == "__main__":
    sys.exit(main())
