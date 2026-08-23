#!/usr/bin/env python3
"""Broad piece × square matrix. Record failures; do not stop to fix them.

  python sim/bench.py
  python sim/bench.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from mj import DEBUG_DIR, Sim, load_config
from pickup import FULL_BOARD_XML, PIECES_XML, run_pickup
from scenario import (
    DEST_SQUARES,
    PIECE_BODIES,
    SOURCE_SQUARES,
    STARTING_AT,
    STARTING_DESTS,
    STARTING_POSITION,
    dest_reach_scenarios,
    empty_matrix_scenarios,
    populated_matrix_scenarios,
    starting_pickup_scenarios,
    starting_place_scenario,
)

OUT = DEBUG_DIR / "eval"
KEEP = (
    "scenario",
    "piece",
    "target_id",
    "source",
    "destination",
    "perception",
    "perception_success",
    "lift_success",
    "transport_success",
    "placement_success",
    "retreat_success",
    "success",
    "failure_reason",
    "failure_category",
    "ik_fail_phase",
    "neighbor_collision_phase",
    "first_collision",
    "estimated_position_error_mm",
    "board",
    "stop_after",
    "obstruction_body",
    "obstruction_square",
    "obstruction_piece",
    "obstruction_robot_part",
    "obstruction_phase",
    "approach_plan",
)


def _row(result: dict, board: str) -> dict:
    out = {k: result.get(k) for k in KEEP}
    out["board"] = board
    out["pickup_success"] = bool(result.get("lift_success"))
    out["perception_success"] = bool(result.get("perception_success", result.get("queen_detected")))
    return out


def _run_empty(sim: Sim, cfg: dict, cases: list[dict], board: str) -> list[dict]:
    rows = []
    n = len(cases)
    for i, sc in enumerate(cases, 1):
        sim.reset()
        sim.quiet = True
        result = run_pickup(sim, cfg, scenario=sc)
        row = _row(result, board)
        rows.append(row)
        cat = row.get("failure_category") or "OK"
        print(f"  [{i:3d}/{n}] {sc['name']:36s} {cat:16s} {row.get('failure_reason') or '-'}")
    return rows


def _run_populated(cfg: dict, cases: list[dict], sim: Sim | None = None) -> list[dict]:
    rows = []
    n = len(cases)
    own = sim is None
    if own:
        sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    orig = sim.model.body_pos.copy()
    try:
        for i, sc in enumerate(cases, 1):
            sim.model.body_pos[:] = orig
            sim.reset()
            sim.quiet = True
            result = run_pickup(sim, cfg, scenario=sc)
            row = _row(result, "full")
            rows.append(row)
            cat = row.get("failure_category") or "OK"
            obs = ""
            if not row.get("pickup_success") and row.get("obstruction_body"):
                obs = (
                    f"  hit={row.get('obstruction_body')}"
                    f"/{row.get('obstruction_robot_part')}"
                )
            print(f"  [{i:3d}/{n}] {sc['name']:44s} {cat:16s} {row.get('failure_reason') or '-'}{obs}")
        sim.model.body_pos[:] = orig
        sim.reset()
    finally:
        if own:
            sim.close()
    return rows


def _rate(rows: list[dict], key: str) -> str:
    if not rows:
        return "n/a"
    n = sum(1 for r in rows if r.get(key))
    return f"{n}/{len(rows)} ({100.0 * n / len(rows):.0f}%)"


def _matrix(rows: list[dict], key: str, squares: tuple[str, ...], square_field: str = "source") -> dict[str, dict[str, str]]:
    grid: dict[str, dict[str, str]] = {p: {s: "-" for s in squares} for p in PIECE_BODIES}
    for r in rows:
        piece = r.get("piece") or r.get("target_id")
        sq = r.get(square_field) or r.get("dest")
        if piece not in grid or sq not in grid[piece]:
            continue
        grid[piece][sq] = "P" if r.get(key) else "F"
    return grid


def _fmt_grid(grid: dict[str, dict[str, str]], squares: tuple[str, ...]) -> str:
    w = 8
    lines = ["        " + "".join(f"{s:>{w}s}" for s in squares)]
    for piece in PIECE_BODIES:
        cells = "".join(f"{grid[piece][s]:>{w}s}" for s in squares)
        lines.append(f"{piece:8s}{cells}")
    return "\n".join(lines)


def summarize(rows: list[dict]) -> str:
    empty = [r for r in rows if r.get("board") == "empty"]
    dest_rows = [r for r in rows if r.get("board") == "dest"]
    full = [r for r in rows if r.get("board") == "full"]
    gt = [r for r in empty if r.get("perception") == "gt"]
    vis = [r for r in empty if r.get("perception") == "vision"]
    cats = Counter(r.get("failure_category") or "SUCCESS" for r in rows)
    reasons = Counter(r.get("failure_reason") or "SUCCESS" for r in rows)
    ik = [r for r in rows if r.get("ik_fail_phase")]
    grasp = [
        r
        for r in rows
        if (r.get("failure_reason") or "").startswith("GRASP")
        or r.get("failure_reason")
        in ("GRASP_MISS", "APPROACH_COLLISION", "APPROACH_TRACK_FAIL")
    ]
    nb = [r for r in rows if r.get("neighbor_collision_phase")]
    place = [r for r in rows if r.get("failure_category") == "PLACEMENT_FAIL"]
    retreat = [r for r in rows if r.get("failure_category") == "RETREAT_FAIL"]
    pickup_grid = _matrix(gt, "pickup_success", SOURCE_SQUARES)
    trans_grid = _matrix(gt, "transport_success", SOURCE_SQUARES)
    place_grid = _matrix(gt, "placement_success", SOURCE_SQUARES)
    full_grid = _matrix(gt, "success", SOURCE_SQUARES)
    dest_grid = _matrix(dest_rows, "success", DEST_SQUARES, square_field="destination")
    bottlenecks = [k for k, _ in reasons.most_common() if k != "SUCCESS"][:3]
    lines = [
        f"PIECES TESTED: {', '.join(PIECE_BODIES)}",
        f"SOURCE SQUARES: {', '.join(SOURCE_SQUARES)}",
        f"DESTINATION SQUARES: {', '.join(DEST_SQUARES)}",
        f"TOTAL EPISODES: {len(rows)} (empty GT {len(gt)}, empty vision {len(vis)}, dest-reach {len(dest_rows)}, populated {len(full)})",
        "",
        f"PICKUP SUCCESS RATE:     {_rate(rows, 'pickup_success')}",
        f"TRANSPORT SUCCESS RATE:  {_rate(rows, 'transport_success')}",
        f"PLACEMENT SUCCESS RATE:  {_rate(rows, 'placement_success')}",
        f"FULL EPISODE SUCCESS RATE: {_rate(rows, 'success')}",
        "",
        "FAILURES BY CATEGORY:",
    ]
    for k, v in cats.most_common():
        lines.append(f"  {k}: {v}")
    lines += ["", "WORKSPACE / IK FAILURES:"]
    if not ik:
        lines.append("  none")
    else:
        for r in ik:
            lines.append(f"  {r.get('scenario')} phase={r.get('ik_fail_phase')} reason={r.get('failure_reason')}")
    lines += ["", f"GRASP FAILURES: {len(grasp)}"]
    for r in grasp[:12]:
        lines.append(f"  {r.get('scenario')} {r.get('failure_reason')}")
    lines += ["", f"NEIGHBOR COLLISIONS: {len(nb)}"]
    for r in nb[:12]:
        lines.append(f"  {r.get('scenario')} phase={r.get('neighbor_collision_phase')}")
    lines += ["", f"PLACEMENT FAILURES: {len(place)}"]
    for r in place[:12]:
        lines.append(f"  {r.get('scenario')} {r.get('failure_reason')}")
    lines += ["", f"RETREAT FAILURES: {len(retreat)}"]
    for r in retreat[:12]:
        lines.append(f"  {r.get('scenario')} {r.get('failure_reason')}")
    lines += [
        "",
        "PIECE × SQUARE PICKUP MATRIX (empty GT):",
        _fmt_grid(pickup_grid, SOURCE_SQUARES),
        "",
        "PIECE × SQUARE TRANSPORT MATRIX (empty GT):",
        _fmt_grid(trans_grid, SOURCE_SQUARES),
        "",
        "PIECE × SQUARE PLACEMENT MATRIX (empty GT):",
        _fmt_grid(place_grid, SOURCE_SQUARES),
        "",
        "PIECE × SQUARE FULL-EPISODE MATRIX (empty GT):",
        _fmt_grid(full_grid, SOURCE_SQUARES),
    ]
    if dest_rows:
        lines += [
            "",
            "QUEEN × DEST FULL-EPISODE MATRIX (d1 source, GT):",
            _fmt_grid(dest_grid, DEST_SQUARES),
        ]
    vis_pick = _matrix(vis, "pickup_success", SOURCE_SQUARES)
    if vis:
        lines += ["", "PIECE × SQUARE PICKUP MATRIX (empty vision):", _fmt_grid(vis_pick, SOURCE_SQUARES)]
        vis_full = _matrix(vis, "success", SOURCE_SQUARES)
        lines += ["", "PIECE × SQUARE FULL-EPISODE MATRIX (empty vision):", _fmt_grid(vis_full, SOURCE_SQUARES)]
    if full:
        pop_sq = tuple(sorted({r.get("source") or "" for r in full}))
        lines += ["", "POPULATED PICKUP (GT+vision):", _fmt_grid(_matrix(full, "pickup_success", pop_sq), pop_sq)]
        lines += ["", "POPULATED FULL EPISODE (GT+vision):", _fmt_grid(_matrix(full, "success", pop_sq), pop_sq)]
    lines += ["", "3 BIGGEST BOTTLENECKS:"]
    if not bottlenecks:
        lines.append("  none")
    else:
        for i, k in enumerate(bottlenecks, 1):
            lines.append(f"  {i}. {k} ({reasons[k]})")
    return "\n".join(lines)


FILES = "abcdefgh"
RANKS = "12345678"
MECHANISMS = (
    "PERCEPTION_FAIL",
    "IK_NO_SOLUTION",
    "APPROACH_COLLISION",
    "APPROACH_TRACK_FAIL",
    "NO_COLLISION_FREE_APPROACH",
    "GRASP_MISS",
    "LIFT_DROP",
    "TRANSPORT_COLLISION",
    "TRANSPORT_DROP",
    "RELEASE_FAILURE",
    "PLACEMENT_POSITION_ERROR",
    "RETREAT_COLLISION",
)


def _pick_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("stop_after") == "lift"]


def _place_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("stop_after") != "lift"]


def _board_map(rows: list[dict], perception: str) -> dict[str, dict]:
    out = {}
    for kind, sq in STARTING_POSITION:
        out[sq] = {"piece": kind, "ok": None, "reason": None, "obs": None}
    for r in _pick_rows(rows):
        if r.get("perception") != perception:
            continue
        sq = r.get("source")
        if sq not in out:
            continue
        ok = bool(r.get("pickup_success"))
        obs = None
        if r.get("obstruction_body"):
            obs = (
                f"{r.get('obstruction_piece') or r.get('obstruction_body')}"
                f"@{r.get('obstruction_square') or '?'} "
                f"{r.get('obstruction_robot_part') or r.get('obstruction_body')}"
            )
        out[sq] = {
            "piece": r.get("piece") or STARTING_AT.get(sq, "?"),
            "ok": ok,
            "reason": None if ok else (r.get("failure_reason") or "FAIL"),
            "obs": None if ok else obs,
        }
    return out


def _fmt_chess_map(board: dict[str, dict], title: str) -> str:
    lines = [title, "    " + "  ".join(f"{f:>3s}" for f in FILES)]
    for rank in reversed(RANKS):
        cells = []
        for f in FILES:
            cell = board.get(f"{f}{rank}", {})
            if cell.get("ok") is True:
                mark = " P "
            elif cell.get("ok") is False:
                mark = " F "
            else:
                mark = " - "
            cells.append(mark)
        lines.append(f" {rank}  " + "  ".join(cells))
    lines.append("    (P=pickup+lift ok  F=fail  -=not run)")
    return "\n".join(lines)


def _fmt_fail_key(board: dict[str, dict]) -> str:
    lines = []
    for rank in reversed(RANKS):
        for f in FILES:
            sq = f"{f}{rank}"
            cell = board.get(sq) or {}
            if cell.get("ok") is not False:
                continue
            piece = cell.get("piece") or STARTING_AT.get(sq, "?")
            extra = f"  obs={cell['obs']}" if cell.get("obs") else ""
            lines.append(f"  {piece:7s} {sq}  {cell.get('reason')}{extra}")
    return "\n".join(lines) if lines else "  none"


def summarize_starting(rows: list[dict]) -> str:
    picks = _pick_rows(rows)
    places = _place_rows(rows)
    gt = [r for r in picks if r.get("perception") == "gt"]
    vis = [r for r in picks if r.get("perception") == "vision"]
    gt_map = _board_map(rows, "gt")
    vis_map = _board_map(rows, "vision")
    reasons = Counter(r.get("failure_reason") or "SUCCESS" for r in picks)
    place_reasons = Counter(r.get("failure_reason") or "SUCCESS" for r in places)
    mech = Counter()
    for r in picks:
        k = r.get("failure_reason")
        if k:
            mech[k] += 1
    for r in places:
        k = r.get("failure_reason")
        if k:
            mech[k] += 1
    top = [k for k, _ in mech.most_common() if k != "SUCCESS"][:3]

    def by_kind(subset: list[dict]) -> list[str]:
        lines = []
        for kind in PIECE_BODIES:
            rs = [r for r in subset if (r.get("piece") or r.get("target_id")) == kind]
            if not rs:
                continue
            n = sum(1 for r in rs if r.get("pickup_success"))
            lines.append(f"  {kind:8s} {n}/{len(rs)}")
        return lines

    def by_rank(subset: list[dict]) -> list[str]:
        lines = []
        for rank in RANKS:
            rs = [r for r in subset if (r.get("source") or "")[1:] == rank]
            if not rs:
                continue
            n = sum(1 for r in rs if r.get("pickup_success"))
            lines.append(f"  rank {rank}  {n}/{len(rs)}")
        return lines

    parts_fail = Counter()
    for r in picks:
        if r.get("pickup_success"):
            continue
        part = r.get("obstruction_robot_part") or "none"
        parts_fail[part] += 1

    lines = [
        "STARTING-BOARD POPULATED STUDY",
        "Frozen: perception, camera, grasp compensation, aperture, transport, release.",
        f"EPISODES: {len(rows)} (pickup GT {len(gt)}, pickup vision {len(vis)}, place {len(places)})",
        f"DESTINATIONS (place follow-up): {', '.join(STARTING_DESTS)}",
        "",
        f"PICKUP+LIFT GT:     {_rate(gt, 'pickup_success')}",
        f"PICKUP+LIFT VISION: {_rate(vis, 'pickup_success')}",
        f"PLACE (among follow-ups): {_rate(places, 'placement_success')}",
        f"FULL PLACE EPISODE: {_rate(places, 'success')}",
        "",
        _fmt_chess_map(gt_map, "STARTING PICKUP MAP — GT"),
        "",
        _fmt_chess_map(vis_map, "STARTING PICKUP MAP — VISION"),
        "",
        "GT FAILURES:",
        _fmt_fail_key(gt_map),
        "",
        "VISION FAILURES:",
        _fmt_fail_key(vis_map),
        "",
        "SUCCESS BY PIECE TYPE (GT pickup):",
        *by_kind(gt),
        "",
        "SUCCESS BY PIECE TYPE (vision pickup):",
        *by_kind(vis),
        "",
        "SUCCESS BY RANK (GT pickup):",
        *by_rank(gt),
        "",
        "SUCCESS BY RANK (vision pickup):",
        *by_rank(vis),
        "",
        "PICKUP FAILURE REASONS (GT+vision):",
    ]
    for k, v in reasons.most_common():
        lines.append(f"  {k}: {v}")
    lines += ["", "FAILED-PICKUP FIRST OBSTRUCTION (robot part):"]
    if not parts_fail:
        lines.append("  none")
    else:
        for k, v in parts_fail.most_common():
            lines.append(f"  {k}: {v}")
    if places:
        lines += ["", "PLACE FOLLOW-UP REASONS:"]
        for k, v in place_reasons.most_common():
            lines.append(f"  {k}: {v}")
        lines += ["", "PLACE FOLLOW-UPS:"]
        for r in places:
            cat = r.get("failure_reason") or "OK"
            lines.append(
                f"  {r.get('piece')} {r.get('source')}->{r.get('destination')} "
                f"{r.get('perception')} {cat}"
            )
    lines += ["", "TOP 3 FAILURE MECHANISMS (all episodes):"]
    if not top:
        lines.append("  none")
    else:
        for i, k in enumerate(top, 1):
            lines.append(f"  {i}. {k} ({mech[k]})")
    listed = set(MECHANISMS)
    extra = [k for k in mech if k not in listed and k != "SUCCESS"]
    if extra:
        lines += ["", "OTHER REASONS:", *[f"  {k}: {mech[k]}" for k in extra]]
    return "\n".join(lines)


def _save(path: Path, rows: list[dict], text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    path.with_suffix(".txt").write_text(text + "\n")


def run_starting(cfg: dict, quick: bool = False, pickup_only: bool = False) -> list[dict]:
    rows: list[dict] = []
    if quick:
        from scenario import piece_scenario

        pick_gt = [
            piece_scenario("queen", "d1", "d4", "gt", hide_proxy="d1", stop_after="lift", quiet=True),
            piece_scenario("pawn", "a2", "d4", "gt", hide_proxy="a2", stop_after="lift", quiet=True),
            piece_scenario("king", "e1", "d4", "gt", hide_proxy="e1", stop_after="lift", quiet=True),
        ]
        pick_vis = [
            piece_scenario("queen", "d1", "d4", "vision", hide_proxy="d1", stop_after="lift", quiet=True),
        ]
    else:
        pick_gt = starting_pickup_scenarios("gt")
        pick_vis = starting_pickup_scenarios("vision")

    sim = Sim(xml_path=FULL_BOARD_XML, cfg=cfg)
    try:
        print(f"\n=== starting pickup GT ({len(pick_gt)}) ===")
        rows.extend(_run_populated(cfg, pick_gt, sim=sim))
        _save(OUT / "starting.json", rows, summarize_starting(rows))
        print(f"\n=== starting pickup vision ({len(pick_vis)}) ===")
        rows.extend(_run_populated(cfg, pick_vis, sim=sim))
        _save(OUT / "starting.json", rows, summarize_starting(rows))

        ok_keys = {
            (r.get("piece"), r.get("source"), r.get("perception"))
            for r in _pick_rows(rows)
            if r.get("pickup_success")
        }
        if pickup_only:
            place_cases = []
        else:
            place_cases = []
            dests = ("d4",) if quick else STARTING_DESTS
            seen = set()
            for piece, src, perc in sorted(ok_keys, key=lambda t: (t[2], t[1] or "")):
                for dest in dests:
                    key = (piece, src, perc, dest)
                    if key in seen:
                        continue
                    seen.add(key)
                    place_cases.append(starting_place_scenario(piece, src, dest, perc))
        if place_cases:
            print(f"\n=== starting place follow-up ({len(place_cases)}) ===")
            rows.extend(_run_populated(cfg, place_cases, sim=sim))
    finally:
        sim.close()
    return rows


def run_matrix(cfg: dict, quick: bool = False) -> list[dict]:
    rows: list[dict] = []
    if quick:
        from scenario import piece_scenario

        empty_gt = [
            piece_scenario("pawn", "d1", "d4", "gt"),
            piece_scenario("queen", "d4", "e4", "gt"),
            piece_scenario("knight", "a4", "d4", "gt"),
        ]
        empty_vis = []
        dest_gt = []
        pop_gt = []
        pop_vis = []
    else:
        vis_src = ("d1", "c2", "d4", "a4", "d6", "h4")
        empty_gt = empty_matrix_scenarios("gt")
        empty_vis = [s for s in empty_matrix_scenarios("vision") if s["source"] in vis_src]
        dest_gt = dest_reach_scenarios("gt")
        pop_gt = populated_matrix_scenarios("gt")
        pop_vis = [s for s in populated_matrix_scenarios("vision") if s["source"] in ("a1", "c1", "d1", "e1", "d2", "c2", "e2")]

    print(f"\n=== empty GT ({len(empty_gt)}) ===")
    sim = Sim(xml_path=PIECES_XML, cfg=cfg)
    try:
        rows.extend(_run_empty(sim, cfg, empty_gt, "empty"))
        if dest_gt:
            print(f"\n=== queen dest reach GT ({len(dest_gt)}) ===")
            rows.extend(_run_empty(sim, cfg, dest_gt, "dest"))
        if empty_vis:
            print(f"\n=== empty vision ({len(empty_vis)}) ===")
            rows.extend(_run_empty(sim, cfg, empty_vis, "empty"))
    finally:
        sim.close()
    if pop_gt:
        print(f"\n=== populated GT ({len(pop_gt)}) ===")
        rows.extend(_run_populated(cfg, pop_gt))
    if pop_vis:
        print(f"\n=== populated vision ({len(pop_vis)}) ===")
        rows.extend(_run_populated(cfg, pop_vis))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny smoke of the selected study")
    parser.add_argument("--matrix", action="store_true", help="legacy empty-board piece×square matrix")
    parser.add_argument("--pickup-only", action="store_true", help="starting study: skip place follow-ups")
    args = parser.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.matrix:
        rows = run_matrix(cfg, quick=args.quick)
        text = summarize(rows)
        _save(OUT / "matrix.json", rows, text)
        out = OUT / "matrix.txt"
    else:
        rows = run_starting(cfg, quick=args.quick, pickup_only=args.pickup_only)
        text = summarize_starting(rows)
        _save(OUT / "starting.json", rows, text)
        out = OUT / "starting.txt"
    print("\n" + text)
    print(f"\nwrote {out}  ({time.time() - t0:.1f}s, {len(rows)} episodes)")


if __name__ == "__main__":
    main()
