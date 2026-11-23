"""Batch runner: suites x walls, one fresh process per episode, under a memory and time guard.

Each run writes runs/<tag>/:
  meta.json    commit, dirty diff hash, config hash, suites, walls, caps, host, times
  rows.jsonl   one row per episode (suite, id, wall, ok, reason, stage, ...)
  summary.txt  totals and failure breakdown per suite x wall
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import platform
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

RUNS = ROOT / "runs"
# A healthy episode peaks near 1.6 GB; runaways have taken the machine past 60 GB.
# The parent enforces both caps with SIGKILL: a worker stuck in a C call never
# runs a Python signal handler, and RSS on macOS hides compressed pages.
# Defaults sized for the 16 GB Mac: 2 workers x 3 GB, 8 GB total.
WORKER_MEM_CAP_GB = float(os.environ.get("ZIMA_MEM_CAP_GB", 3.0))
TOTAL_MEM_CAP_GB = float(os.environ.get("ZIMA_TOTAL_MEM_CAP_GB", 8.0))
DEFAULT_JOBS = 2
_PHASES = None  # child end of the pipe to the parent, for the last phase reached before a kill


def _footprint_gb(pid: int) -> float:
    """Physical footprint (what Activity Monitor shows), 0 once the process is gone."""
    try:
        if sys.platform == "darwin":
            buf = ctypes.create_string_buffer(256)
            if _libproc.proc_pid_rusage(pid, 0, buf) != 0:
                return 0.0
            return struct.unpack_from("Q", buf, 72)[0] / 1e9  # rusage_info_v0.ri_phys_footprint
        with open(f"/proc/{pid}/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1e6
    except (OSError, StopIteration):
        return 0.0


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def send_phase(name: str) -> None:
    """From inside a worker: tell the parent the last phase reached (named in a kill row)."""
    if _PHASES is not None:
        _PHASES.send(("phase", name))


def _child(fn, args, conn) -> None:
    global _PHASES
    _PHASES = conn
    conn.send(("row", fn(args)))


def run_guarded(fn, work: list, jobs: int, timeout_of, fail_row):
    """fn(args) per item in its own process, at most `jobs` at once, killed by the parent on memory or time.

    A killed item yields fail_row(args, reason, stage, seconds).
    """
    os.environ["PYTHONHASHSEED"] = "0"  # set iteration order feeds association and planning; runs must repeat
    ctx = mp.get_context("spawn")
    queue = list(work)
    live: dict = {}  # pid -> dict(proc, conn, args, t0, stage, peak)
    while queue or live:
        total = sum(_footprint_gb(pid) for pid in live)
        while queue and len(live) < jobs and total + WORKER_MEM_CAP_GB <= TOTAL_MEM_CAP_GB:
            a = queue.pop(0)
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(fn, a, child), daemon=True)
            proc.start()
            child.close()
            live[proc.pid] = dict(proc=proc, conn=parent, args=a, t0=time.time(), stage="setup", peak=0.0)
            total += 1.0
        time.sleep(0.25)
        use = {pid: _footprint_gb(pid) for pid in live}
        over = sum(use.values()) > TOTAL_MEM_CAP_GB
        biggest = max(use, key=use.get) if use else None
        for pid, w in list(live.items()):
            w["peak"] = max(w["peak"], use[pid])
            row = None
            try:
                while w["conn"].poll():
                    kind, val = w["conn"].recv()
                    if kind == "phase":
                        w["stage"] = val
                    else:
                        row = val
            except (EOFError, OSError):
                pass
            reason = ("MEM_LIMIT" if use[pid] > WORKER_MEM_CAP_GB or (over and pid == biggest)
                      else "TIMEOUT" if time.time() - w["t0"] > timeout_of(w["args"]) else None)
            if row is None and reason:
                w["proc"].kill()
            elif row is None and w["proc"].is_alive():
                continue
            w["proc"].join(5)
            w["conn"].close()
            del live[pid]
            if row is None:
                row = fail_row(w["args"], reason or "CRASH", w["stage"], round(time.time() - w["t0"], 1))
            row["peak_gb"] = round(w["peak"], 2)
            yield row


def phase_times(phases: list[tuple[str, float]], t0: float) -> dict[str, float]:
    """Seconds spent reaching each logged phase (interval ending at that log)."""
    out: dict[str, float] = {}
    prev = t0
    for name, t in phases:
        if name == "scan_start":
            prev = t
            continue
        key = name.split("_", 1)[1] if name[:3].isdigit() else name
        out[key] = round(out.get(key, 0.0) + (t - prev), 2)
        prev = t
    return out


def group_key(r: dict) -> tuple:
    return (r.get("suite", ""), r.get("wall", ""), r.get("variant", "base"))


def group_label(key: tuple, keys: list[tuple]) -> str:
    """`wall` alone when there is one suite and one variant, else `suite wall variant`."""
    suites = {k[0] for k in keys}
    variants = {k[2] for k in keys}
    parts = ([key[0]] if len(suites) > 1 else []) + [key[1]] + ([key[2]] if len(variants) > 1 else [])
    return " ".join(parts)


def summarize(rows: list[dict]) -> str:
    lines = []
    keys = sorted({group_key(r) for r in rows})
    for key in keys:
        rs = [r for r in rows if group_key(r) == key]
        ok = sum(r["ok"] for r in rs)
        loose = sum(bool(r.get("ok_loose")) for r in rs)
        claimed = sum(r.get("claimed", False) for r in rs)
        lying = sum(1 for r in rs if r.get("claimed") and not r["ok"])
        ts = [r["t"] for r in rs]
        lines.append(
            f"\n[{group_label(key, keys)}] ok {ok}/{len(rs)} ({100 * ok / max(1, len(rs)):.0f}%)   ok_loose {loose}   claimed {claimed}   "
            f"claimed-but-wrong {lying}   time median {median(ts):.1f}s  max {max(ts):.1f}s  total {sum(ts):.0f}s"
        )
        caps = [r for r in rs if r.get("capture")]
        if caps:
            lines.append(f"  captures ok {sum(r['ok'] for r in caps)}/{len(caps)}")
        fails = Counter((r.get("stage") or "?", r.get("reason") or ("DISTURBED" if r.get("disturbed") else "?"))
                        for r in rs if not r["ok"])
        for (stage, reason), n in fails.most_common():
            lines.append(f"  fail {n:3d}  stage={stage:10s} reason={reason}")
        dist = sum(len(r.get("disturbed") or []) for r in rs)
        lines.append(f"  pieces disturbed: {dist}")
        agg: Counter = Counter()
        for r in rs:
            for k, v in (r.get("phases") or {}).items():
                agg[k] += v
        top = ", ".join(f"{k} {v / len(rs):.1f}s" for k, v in agg.most_common(8))
        lines.append(f"  mean time per case by phase: {top}")
    return "\n".join(lines)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def run_meta() -> dict:
    """What the rows were produced from, so a later compare can tell code, config and run changes apart."""
    import realism

    diff = _git("diff", "HEAD", "--", "sim", "config.yaml", "assets")
    cfg = ROOT / "config.yaml"
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "dirty": bool(diff.strip()),
        "diff_sha": _sha(diff.encode()) if diff.strip() else None,
        "config_sha": _sha(cfg.read_bytes()) if cfg.exists() else None,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mujoco": _version("mujoco"),
        "numpy": _version("numpy"),
        "variant": realism.variant(),
        "worker_mem_cap_gb": WORKER_MEM_CAP_GB,
        "total_mem_cap_gb": TOTAL_MEM_CAP_GB,
        "argv": sys.argv,
    }


def _run_item(args: tuple) -> dict:
    suite, case, wall, timeout = args
    from suites import SUITES

    import realism

    row = SUITES[suite].run(case, wall, timeout)
    return {"suite": suite, "variant": realism.variant(), **row}


def _fail_item(args: tuple, reason: str, stage: str, t: float) -> dict:
    suite, case, wall, timeout = args
    from suites import SUITES

    import realism

    return {"suite": suite, "variant": realism.variant(), **SUITES[suite].fail_row(case, wall, reason, stage, t)}


def select_cases(suite: str, only: list[str], ids: list[str] | None = None) -> list[dict]:
    """Cases whose id contains any `only` substring, or whose id is exactly in `ids`."""
    from suites import SUITES

    if ids is not None:
        want = set(ids)
        return [c for c in SUITES[suite].cases() if c["id"] in want]
    keys = only or [""]
    return [c for c in SUITES[suite].cases() if any(k in c["id"] for k in keys)]


def run(suites: list[str], walls: list[str], *, jobs: int = DEFAULT_JOBS, timeout: float | None = None,
        only: list[str] | None = None, ids: list[str] | None = None, tag: str = "latest", out: Path | None = None,
        on_row=None) -> tuple[list[dict], Path]:
    """Run every case of every suite on every wall. Returns (rows, run dir)."""
    from suites import SUITES

    work = []
    for s in suites:
        t = timeout if timeout is not None else SUITES[s].timeout
        cases = select_cases(s, only or [], ids)
        work += [(s, c, w, t) for w in SUITES[s].walls(walls) for c in cases]
    run_dir = (out or RUNS) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tag": tag, "suites": suites, "walls": walls, "jobs": jobs, "only": only or [], "ids": ids, "episodes": len(work),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **run_meta()}
    rows_path = run_dir / "rows.jsonl"
    rows_path.write_text("")
    rows: list[dict] = []
    t0 = time.time()
    for row in run_guarded(_run_item, work, jobs, lambda a: a[3] + 30, _fail_item):
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if on_row:
            on_row(row)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["wall_clock_s"] = round(time.time() - t0)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    report = summarize(rows)
    for s in suites:
        if SUITES[s].report:
            report += "\n" + SUITES[s].report([r for r in rows if r.get("suite") == s])
    report += f"\n\nwall clock {meta['wall_clock_s']}s   rows: {rows_path}"
    (run_dir / "summary.txt").write_text(report + "\n")
    return rows, run_dir


def print_row(row: dict) -> None:
    mark = "ok  " if row["ok"] else "FAIL"
    print(f"  {mark} {row.get('suite', ''):8s} {row['wall']:10s} {row['id']:28s} {row['t']:5.1f}s  {row.get('reason') or ''}"
          f"{'  disturbed=' + str(len(row['disturbed'])) if row.get('disturbed') else ''}", flush=True)
