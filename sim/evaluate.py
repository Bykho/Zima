#!/usr/bin/env python3
"""Characterize the queen grasp. Same run_pickup, scenario dicts, a loop.

  python sim/evaluate.py --suite position_offsets
  python sim/evaluate.py --suite perception_error
  python sim/evaluate.py --suite both
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

from mj import DEBUG_DIR, Sim, load_config
from pickup import run_pickup
from scenario import (
    COMP_DELTA_MM,
    NOM_GRASP_XY_MM,
    PERC_ERR_MM,
    neighbor_scenarios,
    perception_error_scenarios,
    position_offset_scenarios,
    probe_error_scenarios,
    scenario,
    transport_scenarios,
)

OUT = DEBUG_DIR / "eval"
KEEP = (
    "scenario",
    "seed",
    "target_id",
    "source",
    "destination",
    "actual_start_pose",
    "pose_used_for_grasp",
    "perception_error",
    "offset_xy_mm",
    "pose_error_xy_mm",
    "close_shove",
    "tilt_after_close",
    "grasp_success",
    "lift_success",
    "T_gripper_object",
    "neighbor_contact",
    "neighbor_displacement",
    "placement_error",
    "failure_reason",
    "stage_flags",
    "success",
    "close_shove_xy_mm",
    "close_tilt_deg",
    "release_position_error_mm",
    "final_position_error_mm",
    "T_gripper_queen_lift",
    "ik_solved",
    "approach_reached",
    "approach_ok",
    "grasp_ok",
    "ee_requested",
    "ee_actual",
    "ee_err_mm",
    "ik_residual_mm",
    "joint_limit_margin_rad",
    "pre_close_contacts",
    "pre_close_gap_mm",
    "first_contact_jaw",
    "ik_jaw",
    "jaw_before_close",
    "grasp_xy_offset_mm",
    "neighbor_eval",
    "neighbor_tilt",
    "min_robot_neighbor_clearance_mm",
    "transport_path",
    "transport_success",
    "placement_success",
)


def _slim(result: dict) -> dict:
    out = {k: result.get(k) for k in KEEP}
    t = result.get("T_gripper_object") or result.get("T_gripper_queen_lift")
    if t:
        out["T_gripper_object"] = t
    return out


def _ok(ep: dict) -> bool:
    """Grasp primitive success: lift, ignoring later place/retreat."""
    return bool(ep.get("lift_success"))


def run_cases(sim: Sim, cfg: dict, cases: list[dict]) -> list[dict]:
    rows = []
    n = len(cases)
    for i, sc in enumerate(cases, 1):
        sim.reset()
        sim.quiet = True
        result = run_pickup(sim, cfg, scenario=sc)
        row = _slim(result)
        rows.append(row)
        flag = "LIFT" if _ok(row) else (row.get("failure_reason") or "FAIL")
        shove = row.get("close_shove")
        shove_s = f"{shove:.1f}" if isinstance(shove, (int, float)) else "-"
        print(f"  [{i:3d}/{n}] {sc['name']:24s} {flag:22s} shove={shove_s}mm")
    return rows


def _axis(rows: list[dict], axis: int, other_must_be_zero: bool = True) -> list[tuple[float, dict]]:
    out = []
    for r in rows:
        xy = r.get("offset_xy_mm") or [0.0, 0.0]
        if other_must_be_zero and abs(xy[1 - axis]) > 0.05:
            continue
        if abs(xy[axis]) < 0.05 and abs(xy[1 - axis]) > 0.05:
            continue
        out.append((xy[axis], r))
    out.sort(key=lambda t: t[0])
    return out


def _max_success_extent(pairs: list[tuple[float, dict]]) -> str:
    plus = [x for x, r in pairs if x > 0.05 and _ok(r)]
    minus = [x for x, r in pairs if x < -0.05 and _ok(r)]
    p = max(plus) if plus else 0.0
    m = min(minus) if minus else 0.0
    zero = any(abs(x) < 0.05 and _ok(r) for x, r in pairs)
    contig_p = 0.0
    for x, r in pairs:
        if x < -0.05:
            continue
        if abs(x) < 0.05:
            if not _ok(r):
                break
            continue
        if x > 0.05:
            if not _ok(r):
                break
            contig_p = x
    contig_m = 0.0
    for x, r in sorted(pairs, key=lambda t: t[0], reverse=True):
        if x > 0.05:
            continue
        if abs(x) < 0.05:
            if not _ok(r):
                break
            continue
        if not _ok(r):
            break
        contig_m = x
    return (
        f"+{p:.0f} / {m:.0f} mm (any success); "
        f"contiguous from 0: [{contig_m:.0f}, +{contig_p:.0f}] mm"
        + ("" if zero else " (baseline failed)")
    )


def _matrix(rows: list[dict], xs: list[int], ys: list[int], key: str) -> str:
    lookup = {}
    for r in rows:
        if key == "offset":
            xy = r.get("offset_xy_mm") or [0, 0]
        else:
            xy = r.get("pose_error_xy_mm") or [0, 0]
        lookup[(int(round(xy[0])), int(round(xy[1])))] = r
    header = "     " + " ".join(f"{x:+4d}" for x in xs)
    lines = [header]
    for y in reversed(ys):
        cells = []
        for x in xs:
            r = lookup.get((x, y))
            if r is None:
                cells.append("   .")
            elif _ok(r):
                cells.append("   Y")
            else:
                cells.append("   n")
        lines.append(f"{y:+4d} " + " ".join(cells))
    lines.append("(Y = lift success, n = lift fail, . = not run)")
    return "\n".join(lines)


def _stats(rows: list[dict]) -> dict:
    lifts = [r for r in rows if _ok(r)]
    shoves = [r["close_shove"] for r in lifts if isinstance(r.get("close_shove"), (int, float))]
    tilts = [r["tilt_after_close"] for r in lifts if isinstance(r.get("tilt_after_close"), (int, float))]
    pts = []
    for r in lifts:
        t = r.get("T_gripper_object") or {}
        p = t.get("p_ee_mm") if isinstance(t, dict) else None
        if p and len(p) >= 3:
            pts.append(p[:3])
    var = None
    if len(pts) >= 2:
        arr = np.asarray(pts, dtype=float)
        std = arr.std(axis=0)
        var = {
            "n": len(pts),
            "mean_mm": arr.mean(axis=0).tolist(),
            "std_mm": std.tolist(),
            "rms_std_mm": float(np.linalg.norm(std)),
        }
    reasons = Counter(r.get("failure_reason") or "SUCCESS" for r in rows)
    lift_fail = Counter(r.get("failure_reason") or "UNKNOWN" for r in rows if not _ok(r))
    return {
        "n": len(rows),
        "lift_success": len(lifts),
        "lift_rate": len(lifts) / max(1, len(rows)),
        "episode_success": sum(1 for r in rows if r.get("success")),
        "mean_shove_mm": float(np.mean(shoves)) if shoves else None,
        "max_shove_mm": float(np.max(shoves)) if shoves else None,
        "mean_tilt_deg": float(np.mean(tilts)) if tilts else None,
        "max_tilt_deg": float(np.max(tilts)) if tilts else None,
        "grasp_transform": var,
        "failure_modes": dict(reasons),
        "lift_failure_modes": dict(lift_fail),
    }


def _useful_transform(rows: list[dict]) -> dict | None:
    useful = []
    for r in rows:
        if not _ok(r):
            continue
        off = r.get("offset_xy_mm") or [0, 0]
        err = r.get("pose_error_xy_mm") or [0, 0]
        if (off[0] ** 2 + off[1] ** 2) ** 0.5 > 3.1:
            continue
        if (err[0] ** 2 + err[1] ** 2) ** 0.5 > 4.1:
            continue
        t = r.get("T_gripper_object") or {}
        p = t.get("p_ee_mm") if isinstance(t, dict) else None
        if p:
            useful.append(p[:3])
    if len(useful) < 2:
        return None
    arr = np.asarray(useful, dtype=float)
    std = arr.std(axis=0)
    return {
        "n": len(useful),
        "mean_mm": arr.mean(axis=0).tolist(),
        "std_mm": std.tolist(),
        "rms_std_mm": float(np.linalg.norm(std)),
    }


def _vision_adequate(perc_rows: list[dict]) -> str:
    inside = []
    for r in perc_rows:
        xy = r.get("pose_error_xy_mm") or [0, 0]
        rad = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
        if rad <= 4.05:
            inside.append(r)
    if not inside:
        return "no samples at r<=4 mm"
    n_ok = sum(1 for r in inside if _ok(r))
    failed = [r for r in inside if not _ok(r)]
    max_ok_r = 0.0
    for r in perc_rows:
        xy = r.get("pose_error_xy_mm") or [0, 0]
        rad = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
        if _ok(r):
            max_ok_r = max(max_ok_r, rad)
    contig = True
    for r in perc_rows:
        xy = r.get("pose_error_xy_mm") or [0, 0]
        rad = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
        if rad <= 4.05 and not _ok(r):
            contig = False
            break
    if contig and n_ok == len(inside):
        return (
            f"yes — all {n_ok} cells with r<=4 mm lifted; "
            f"max successful radial perception error {max_ok_r:.1f} mm; "
            f"current wrist error ~4 mm sits inside the measured envelope"
        )
    names = ", ".join(r["scenario"] for r in failed[:6])
    return (
        f"no — {n_ok}/{len(inside)} cells with r<=4 mm lifted; "
        f"failures: {names}; max successful radial error {max_ok_r:.1f} mm; "
        f"current wrist error ~4 mm is not safely inside the envelope"
    )


def _perc_axis(rows: list[dict], axis: int) -> list[tuple[float, dict]]:
    out = []
    for r in rows:
        xy = r.get("pose_error_xy_mm") or [0.0, 0.0]
        if abs(xy[1 - axis]) > 0.05:
            continue
        out.append((xy[axis], r))
    out.sort(key=lambda t: t[0])
    return out


def summarize(pos_rows: list[dict], perc_rows: list[dict]) -> str:
    all_rows = pos_rows + perc_rows
    st = _stats(all_rows)
    pos_st = _stats(pos_rows) if pos_rows else None
    perc_st = _stats(perc_rows) if perc_rows else None
    x_pairs = _axis(pos_rows, 0) if pos_rows else []
    y_pairs = _axis(pos_rows, 1) if pos_rows else []
    px = _perc_axis(perc_rows, 0) if perc_rows else []
    py = _perc_axis(perc_rows, 1) if perc_rows else []
    baseline = next((r for r in pos_rows if r["scenario"] == "pos_x_+0mm"), None)
    if baseline is None and pos_rows:
        baseline = next((r for r in pos_rows if r.get("offset_xy_mm") == [0.0, 0.0] or r.get("offset_xy_mm") == [0, 0]), pos_rows[0])
    b_flags = " ".join((baseline or {}).get("stage_flags") or [])
    b_lift = (baseline or {}).get("lift_success")
    b_reason = (baseline or {}).get("failure_reason")
    t_all = st.get("grasp_transform")
    t_use = _useful_transform(all_rows)
    lift_fail = st["lift_failure_modes"]
    if lift_fail:
        dom = max(lift_fail.items(), key=lambda kv: kv[1])
        dom_s = f"{dom[0]} ({dom[1]}/{st['n'] - st['lift_success']} lift fails)"
    else:
        ep_fail = {k: v for k, v in st["failure_modes"].items() if k != "SUCCESS"}
        if ep_fail:
            dom = max(ep_fail.items(), key=lambda kv: kv[1])
            dom_s = f"{dom[0]} (episode; no lift failures)"
        else:
            dom_s = "none"
    xs_pos = sorted({int(round((r.get("offset_xy_mm") or [0, 0])[0])) for r in pos_rows})
    ys_pos = sorted({int(round((r.get("offset_xy_mm") or [0, 0])[1])) for r in pos_rows})
    xs_p = list(PERC_ERR_MM)
    ys_p = list(PERC_ERR_MM)
    lines = [
        "SCENARIO HARNESS FILES: sim/scenario.py sim/evaluate.py (run_pickup in sim/pickup.py)",
        f"BASELINE RESULT: lift={'PASS' if b_lift else 'FAIL'} {b_flags} reason={b_reason}",
        f"MAX X OFFSET WITH SUCCESS: {_max_success_extent(x_pairs) if x_pairs else 'n/a'}",
        f"MAX Y OFFSET WITH SUCCESS: {_max_success_extent(y_pairs) if y_pairs else 'n/a'}",
        "2D OFFSET SUCCESS REGION:",
        _matrix(pos_rows, xs_pos, ys_pos, "offset") if pos_rows else "n/a",
        f"MAX PERCEPTION X ERROR WITH SUCCESS: {_max_success_extent(px) if px else 'n/a'}",
        f"MAX PERCEPTION Y ERROR WITH SUCCESS: {_max_success_extent(py) if py else 'n/a'}",
        "2D PERCEPTION-ERROR SUCCESS REGION:",
        _matrix(perc_rows, xs_p, ys_p, "perc") if perc_rows else "n/a",
        f"MEAN CLOSE SHOVE: {st['mean_shove_mm']:.2f} mm" if st["mean_shove_mm"] is not None else "MEAN CLOSE SHOVE: n/a",
        f"MAX CLOSE SHOVE: {st['max_shove_mm']:.2f} mm" if st["max_shove_mm"] is not None else "MAX CLOSE SHOVE: n/a",
        f"MEAN TILT: {st['mean_tilt_deg']:.2f} deg" if st["mean_tilt_deg"] is not None else "MEAN TILT: n/a",
        f"GRASP TRANSFORM VARIANCE: {json.dumps(t_use or t_all)}",
        f"CURRENT ~4 MM VISION ERROR ADEQUATE?: {_vision_adequate(perc_rows) if perc_rows else 'n/a (run perception_error)'}",
        f"DOMINANT FAILURE MODE: {dom_s}",
        "",
        f"lift success rate (all): {st['lift_success']}/{st['n']} = {100 * st['lift_rate']:.0f}%",
        f"full episode success: {st['episode_success']}/{st['n']}",
        f"failure_modes: {st['failure_modes']}",
    ]
    if pos_st:
        lines.append(f"position_offsets lift: {pos_st['lift_success']}/{pos_st['n']}")
    if perc_st:
        lines.append(f"perception_error lift: {perc_st['lift_success']}/{perc_st['n']}")
    return "\n".join(lines)


def _lookup_err(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        xy = r.get("pose_error_xy_mm") or [0, 0]
        out[(int(round(xy[0])), int(round(xy[1])))] = r
    return out


def _reach_matrix(rows: list[dict], xs: list[int], ys: list[int]) -> str:
    lookup = _lookup_err(rows)
    header = "     " + " ".join(f"{x:+4d}" for x in xs)
    lines = [header]
    for y in reversed(ys):
        cells = []
        for x in xs:
            r = lookup.get((x, y))
            if r is None:
                cells.append("   .")
            elif _ok(r) or r.get("approach_ok"):
                cells.append("   R")
            elif r.get("failure_reason") == "APPROACH_COLLISION":
                cells.append("   C")
            elif r.get("failure_reason") == "APPROACH_TRACK_FAIL":
                cells.append("   T")
            elif r.get("failure_reason") == "IK_NO_SOLUTION":
                cells.append("   N")
            elif r.get("ik_solved"):
                cells.append("   T")
            else:
                cells.append("   N")
        lines.append(f"{y:+4d} " + " ".join(cells))
    lines.append("R=clean approach  C=APPROACH_COLLISION  T=APPROACH_TRACK_FAIL  N=IK_NO_SOLUTION")
    return "\n".join(lines)


def _grasp_cond_matrix(rows: list[dict], xs: list[int], ys: list[int]) -> str:
    lookup = _lookup_err(rows)
    header = "     " + " ".join(f"{x:+4d}" for x in xs)
    lines = [header]
    n_app = 0
    n_ok = 0
    for y in reversed(ys):
        cells = []
        for x in xs:
            r = lookup.get((x, y))
            if r is None or not r.get("approach_ok"):
                cells.append("   .")
            elif _ok(r):
                cells.append("   Y")
                n_app += 1
                n_ok += 1
            else:
                cells.append("   n")
                n_app += 1
        lines.append(f"{y:+4d} " + " ".join(cells))
    lines.append(f"(Y/n only where approach was clean; {n_ok}/{n_app} lifts | . = not reachable cleanly)")
    return "\n".join(lines)


def _lift_matrix(rows: list[dict], xs: list[int], ys: list[int]) -> str:
    return _matrix(rows, xs, ys, "perc")


def _class_matrix(rows: list[dict], xs: list[int], ys: list[int]) -> str:
    lookup = _lookup_err(rows)
    header = "     " + " ".join(f"{x:+4d}" for x in xs)
    lines = [header]
    counts = Counter()
    for y in reversed(ys):
        cells = []
        for x in xs:
            r = lookup.get((x, y))
            if r is None:
                cells.append("   .")
                continue
            if _ok(r):
                tag = "Y"
            else:
                reason = r.get("failure_reason") or "?"
                tag = {
                    "GRASP_MISS": "G",
                    "APPROACH_COLLISION": "C",
                    "APPROACH_TRACK_FAIL": "T",
                    "IK_NO_SOLUTION": "N",
                }.get(reason, "?")
            counts[tag] += 1
            cells.append(f"   {tag}")
        lines.append(f"{y:+4d} " + " ".join(cells))
    lines.append("Y=lift  G=GRASP_MISS  C=APPROACH_COLLISION  T=APPROACH_TRACK_FAIL  N=IK_NO_SOLUTION")
    lines.append("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def robust_radius(rows: list[dict]) -> int:
    lookup = _lookup_err(rows)
    if not _ok(lookup.get((0, 0), {})):
        return 0
    axis2 = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    diag2 = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
    axis4 = [(4, 0), (-4, 0), (0, 4), (0, -4)]
    if not all(_ok(lookup.get(p, {})) for p in axis2):
        return 0
    if all(_ok(lookup.get(p, {})) for p in diag2) and all(_ok(lookup.get(p, {})) for p in axis4):
        return 4
    if all(_ok(lookup.get(p, {})) for p in diag2):
        return 2
    return 2


def score_probe(rows: list[dict], offset_mm) -> dict:
    lifts = [r for r in rows if _ok(r)]
    ik_fail = sum(1 for r in rows if r.get("failure_reason") == "IK_NO_SOLUTION")
    pre = sum(1 for r in rows if r.get("failure_reason") == "APPROACH_COLLISION")
    pre += sum(1 for r in rows if r.get("failure_reason") == "APPROACH_TRACK_FAIL")
    baseline = next((r for r in rows if r.get("pose_error_xy_mm") == [0.0, 0.0] or r.get("scenario") == "probe_+0_+0mm"), rows[0])
    shoves = [r["close_shove"] for r in lifts if isinstance(r.get("close_shove"), (int, float))]
    tilts = [r["tilt_after_close"] for r in lifts if isinstance(r.get("tilt_after_close"), (int, float))]
    fail_r = []
    for r in rows:
        xy = r.get("pose_error_xy_mm") or [0, 0]
        rad = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
        if not _ok(r):
            fail_r.append(rad)
    return {
        "offset_mm": [float(offset_mm[0]), float(offset_mm[1])],
        "baseline_shove_mm": baseline.get("close_shove"),
        "baseline_lift": bool(baseline.get("lift_success")),
        "successes": sum(1 for r in rows if _ok(r)),
        "n": len(rows),
        "ik_failures": ik_fail,
        "preclose_collisions": pre,
        "robust_radius_mm": robust_radius(rows),
        "min_fail_radius_mm": min(fail_r) if fail_r else None,
        "worst_successful_shove_mm": float(max(shoves)) if shoves else None,
        "worst_successful_tilt_deg": float(max(tilts)) if tilts else None,
        "mean_successful_shove_mm": float(sum(shoves) / len(shoves)) if shoves else None,
    }


def _fmt_geom(label: str, snap: dict | None) -> str:
    if not snap:
        return f"  {label}: (none)"
    def xy(k):
        v = snap.get(k)
        if not v:
            return "-"
        return f"[{v[0]:+6.1f}, {v[1]:+6.1f}]"
    return (
        f"  {label}: ee{xy('ee_xy_mm')} queen{xy('queen_xy_mm')} "
        f"fix->q{xy('fixed_to_queen_xy_mm')} mov->q{xy('moving_to_queen_xy_mm')} "
        f"pinch->q{xy('pinch_to_queen_xy_mm')} gap={snap.get('min_gap_mm')} "
        f"contact={snap.get('contact_jaws')} ncon={snap.get('n_contact')}"
    )


def explain_plus_xy(sim, cfg) -> str:
    lines = ["GEOMETRY at old compensation, GT queen, requested pose error:"]
    for err in ((0, 0), (2, 0), (0, 2), (4, 0), (0, 4)):
        sc = scenario(
            name=f"geom_{err[0]:+d}_{err[1]:+d}",
            pose_error_xy=[err[0] / 1000.0, err[1] / 1000.0],
            quiet=True,
        )
        sim.reset()
        sim.quiet = True
        r = run_pickup(sim, cfg, scenario=sc)
        xy = r.get("pose_error_xy_mm")
        lines.append(
            f"\n  err={xy} lift={r.get('lift_success')} reason={r.get('failure_reason')} "
            f"ik={r.get('ik_solved')} ee_err={r.get('ee_err_mm')} "
            f"pre_ncon={r.get('pre_close_contacts')} first={r.get('first_contact_jaw')} "
            f"ik_gap={((r.get('ik_jaw') or {}).get('min_gap_mm'))}"
        )
        lines.append(_fmt_geom("IK goal (open)", r.get("ik_jaw")))
        lines.append(_fmt_geom("after descend", r.get("jaw_before_close")))
        jlim = r.get("joint_limit_margin_rad") or {}
        lines.append(f"  joint_limit_min_rad={jlim.get('min_rad')} residual_mm={r.get('ik_residual_mm')}")
    return "\n".join(lines)


def write_grasp_offset(xy_m: list[float]):
    path = ROOT / "config.yaml"
    text = path.read_text()
    old = "grasp_xy_offset: [0.00376, 0.01266]"
    new = f"grasp_xy_offset: [{xy_m[0]:.5f}, {xy_m[1]:.5f}]"
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return
    import re
    path.write_text(re.sub(r"grasp_xy_offset:\s*\[[^\]]+\]", new, text, count=1))


def vision_error_samples(sim, cfg, n: int = 16) -> list[list[float]]:
    from dataclasses import replace

    from perception import estimate_target_pose
    from pickup import open_gripper, pickup_cfg, seed_vertical
    from sensor import capture as capture_wrist
    from squares import square_to_world

    p = pickup_cfg(cfg)
    surface = float(cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02))
    source = p.get("source", "d1")
    sim.reset()
    sim.quiet = True
    spawn = square_to_world(source, cfg, z=surface)
    sim.place_object("queen", spawn)
    seed_vertical(sim, cfg)
    open_gripper(sim, float(p.get("gripper_open", 0.45)), steps=40)
    for _ in range(40):
        sim.step()
    look = [spawn[0], spawn[1], surface + 0.08]
    from ik import move_ik
    move_ik(sim, look, duration=1.0)
    obs = capture_wrist(sim)
    gt, _ = sim.object_pose("queen")
    errs = []
    raw = estimate_target_pose(obs, cfg, source)
    if raw is not None:
        errs.append([float((raw[0] - gt[0]) * 1000), float((raw[1] - gt[1]) * 1000)])
    rng = np.random.default_rng(1)
    for _ in range(n):
        noisy = replace(obs, depth=obs.depth + rng.normal(0.0, 0.002, obs.depth.shape))
        est = estimate_target_pose(noisy, cfg, source)
        if est is None:
            continue
        errs.append([float((est[0] - gt[0]) * 1000), float((est[1] - gt[1]) * 1000)])
    return errs


def _in_success_cell(err_mm, lookup, step=2) -> bool:
    x = int(round(err_mm[0] / step) * step)
    y = int(round(err_mm[1] / step) * step)
    r = lookup.get((x, y))
    return bool(r and _ok(r))


def run_robustness(sim, cfg) -> str:
    xs = list(PERC_ERR_MM)
    ys = list(PERC_ERR_MM)
    nom = [NOM_GRASP_XY_MM[0] / 1000.0, NOM_GRASP_XY_MM[1] / 1000.0]

    print("\n=== geometry: why +X / +Y fail (old compensation) ===")
    geom = explain_plus_xy(sim, cfg)
    print(geom)

    print("\n=== old compensation perception-error grid ===")
    old_rows = run_cases(sim, cfg, perception_error_scenarios(nom))
    (OUT / "perc_old.json").write_text(json.dumps(old_rows, indent=2, default=str))
    old_probe = run_cases(sim, cfg, probe_error_scenarios(nom))
    old_score = score_probe(old_probe, NOM_GRASP_XY_MM)

    print("\n=== compensation sweep ===")
    scores = []
    for dx in COMP_DELTA_MM:
        for dy in COMP_DELTA_MM:
            ox = (NOM_GRASP_XY_MM[0] + dx) / 1000.0
            oy = (NOM_GRASP_XY_MM[1] + dy) / 1000.0
            rows = run_cases(sim, cfg, probe_error_scenarios([ox, oy]))
            sc = score_probe(rows, [NOM_GRASP_XY_MM[0] + dx, NOM_GRASP_XY_MM[1] + dy])
            sc["rows"] = rows
            scores.append(sc)
            print(
                f"  comp [{sc['offset_mm'][0]:+6.2f}, {sc['offset_mm'][1]:+6.2f}] "
                f"r={sc['robust_radius_mm']}  {sc['successes']}/{sc['n']} "
                f"shove0={sc['baseline_shove_mm']}  ik_fail={sc['ik_failures']}"
            )

    def key(s):
        return (
            s["robust_radius_mm"],
            s["successes"],
            -(s["worst_successful_shove_mm"] or 99),
        )

    best = max(scores, key=key)
    best_m = [best["offset_mm"][0] / 1000.0, best["offset_mm"][1] / 1000.0]
    print(f"\nBEST compensation mm={best['offset_mm']} robust_radius={best['robust_radius_mm']}")

    print("\n=== new compensation perception-error grid ===")
    new_rows = run_cases(sim, cfg, perception_error_scenarios(best_m))
    (OUT / "perc_new.json").write_text(json.dumps(new_rows, indent=2, default=str))
    (OUT / "comp_sweep.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "rows"} for s in scores], indent=2, default=str)
    )

    write_grasp_offset(best_m)
    cfg = load_config()

    print("\n=== wrist estimate samples (depth noise after grasp sweep) ===")
    vis = vision_error_samples(sim, cfg)
    lookup_new = _lookup_err(new_rows)
    n_in = sum(1 for e in vis if _in_success_cell(e, lookup_new))
    n_rad = sum(1 for e in vis if (e[0] ** 2 + e[1] ** 2) ** 0.5 <= best["robust_radius_mm"] + 0.05)
    vis_mean = None
    if vis:
        arr = np.asarray(vis)
        vis_mean = [float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(np.linalg.norm(arr, axis=1).mean())]

    old_app = sum(1 for r in old_rows if r.get("approach_ok"))
    old_g = sum(1 for r in old_rows if r.get("approach_ok") and _ok(r))
    new_app = sum(1 for r in new_rows if r.get("approach_ok"))
    new_g = sum(1 for r in new_rows if r.get("approach_ok") and _ok(r))
    old_lift_fail = [r.get("failure_reason") for r in new_rows if not _ok(r)]
    from collections import Counter
    dom = Counter(old_lift_fail).most_common(1)

    adequate = (
        best["robust_radius_mm"] >= 4
        and vis
        and n_in / max(1, len(vis)) >= 0.8
    )
    lines = [
        f"IK-REACHABLE ERROR REGION:\n{_reach_matrix(old_rows, xs, ys)}",
        f"GRASP REGION CONDITIONAL ON REACHABLE APPROACH:\n{_grasp_cond_matrix(old_rows, xs, ys)}",
        f"  (old: {old_g}/{old_app} lifts given clean approach)",
        "",
        f"OLD COMPENSATION: [{NOM_GRASP_XY_MM[0]:.2f}, {NOM_GRASP_XY_MM[1]:.2f}] mm",
        f"OLD ROBUST RADIUS: {old_score['robust_radius_mm']} mm",
        f"OLD BASELINE SHOVE: {old_score['baseline_shove_mm']}",
        "",
        f"BEST NEW COMPENSATION: [{best['offset_mm'][0]:.2f}, {best['offset_mm'][1]:.2f}] mm",
        f"NEW ROBUST RADIUS: {best['robust_radius_mm']} mm",
        f"NEW BASELINE SHOVE: {best['baseline_shove_mm']}",
        f"NEW WORST SUCCESSFUL SHOVE: {best['worst_successful_shove_mm']}",
        f"NEW WORST SUCCESSFUL TILT: {best['worst_successful_tilt_deg']}",
        f"NEW PROBE SCORE: {best['successes']}/{best['n']}  ik_fail={best['ik_failures']}",
        "",
        geom,
        "",
        "OLD ERROR MAP (lift):\n" + _lift_matrix(old_rows, xs, ys),
        "NEW ERROR MAP (lift):\n" + _lift_matrix(new_rows, xs, ys),
        "NEW IK-REACHABLE:\n" + _reach_matrix(new_rows, xs, ys),
        "NEW GRASP | APPROACH:\n" + _grasp_cond_matrix(new_rows, xs, ys),
        f"  (new: {new_g}/{new_app} lifts given clean approach)",
        "",
        f"VISION SAMPLES (1 raw + depth N(0,2mm)): n={len(vis)} mean_xy_r={vis_mean}",
        f"  in new success cells: {n_in}/{len(vis)}",
        f"  hypot <= new robust_radius: {n_rad}/{len(vis)}",
        f"CURRENT VISION LIKELY ADEQUATE AFTER CHANGE?: {'yes' if adequate else 'no'}",
        f"DOMINANT REMAINING GRASP FAILURE: {dom[0] if dom else 'none'}",
    ]
    text = "\n".join(lines)
    (OUT / "robustness.txt").write_text(text)
    return text


def run_diagnostics(sim, cfg) -> str:
    xs = list(PERC_ERR_MM)
    ys = list(PERC_ERR_MM)
    print("\n=== robustness probe (locked compensation) ===")
    probe = run_cases(sim, cfg, probe_error_scenarios())
    score = score_probe(probe, NOM_GRASP_XY_MM)
    print("\n=== perception-error grid ===")
    perc = run_cases(sim, cfg, perception_error_scenarios())
    (OUT / "perc_locked.json").write_text(json.dumps(perc, indent=2, default=str))
    (OUT / "probe_locked.json").write_text(json.dumps(probe, indent=2, default=str))
    reasons = Counter(r.get("failure_reason") or "SUCCESS" for r in perc)
    lift_fail = Counter(r.get("failure_reason") or "UNKNOWN" for r in perc if not _ok(r))
    text = "\n".join(
        [
            f"COMPENSATION: [{NOM_GRASP_XY_MM[0]:.2f}, {NOM_GRASP_XY_MM[1]:.2f}] mm",
            f"PROBE: {score['successes']}/{score['n']}  robust_radius={score['robust_radius_mm']} mm",
            f"PROBE BASELINE SHOVE: {score['baseline_shove_mm']}",
            "",
            "DIAGNOSTIC MAP:",
            _class_matrix(perc, xs, ys),
            "",
            "APPROACH REACHABILITY:",
            _reach_matrix(perc, xs, ys),
            "",
            "GRASP | CLEAN APPROACH:",
            _grasp_cond_matrix(perc, xs, ys),
            "",
            f"episode reasons: {dict(reasons)}",
            f"lift-fail reasons: {dict(lift_fail)}",
            f"IK_NO_SOLUTION count: {reasons.get('IK_NO_SOLUTION', 0)}",
            f"APPROACH_TRACK_FAIL count: {reasons.get('APPROACH_TRACK_FAIL', 0)}",
            f"APPROACH_COLLISION count: {reasons.get('APPROACH_COLLISION', 0)}",
            f"GRASP_MISS count: {reasons.get('GRASP_MISS', 0)}",
        ]
    )
    (OUT / "diagnostics.txt").write_text(text)
    return text


def run_neighbors(sim, cfg) -> str:
    cases = neighbor_scenarios()
    print(f"\n=== neighbor clearance ({len(cases)}) ===")
    rows = run_cases(sim, cfg, cases)
    (OUT / "neighbors.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = []
    first_fail = None
    for r in rows:
        ne = (r.get("neighbor_eval") or [{}])[0]
        lift = "LIFT" if _ok(r) else (r.get("failure_reason") or "FAIL")
        line = (
            f"{r.get('scenario')}: target={lift} "
            f"neighbor_contact={r.get('neighbor_contact')} "
            f"disp_mm={ne.get('xy_disp_mm')} tilt={ne.get('tilt_deg')} "
            f"clearance_mm={ne.get('clearance_mm')} "
            f"flags={r.get('stage_flags')}"
        )
        lines.append(line)
        print("  " + line)
        if first_fail is None and (not _ok(r) or r.get("neighbor_contact") or (ne.get("xy_disp_mm") or 0) > 2.0):
            first_fail = line
    if first_fail is None:
        first_fail = "none — all adjacent pawns left the target lift intact with <2 mm neighbor motion"
    text = "\n".join(lines + ["", f"FIRST FAILURE MODE: {first_fail}"])
    (OUT / "neighbors.txt").write_text(text)
    return text


def _episode_ok(r: dict) -> bool:
    return bool(r.get("success"))


def run_transport(sim, cfg) -> str:
    cases = transport_scenarios()
    print(f"\n=== transport clearance ({len(cases)}) ===")
    rows = run_cases(sim, cfg, cases)
    (OUT / "transport.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = []
    first_fail = None
    for r in rows:
        flags = r.get("stage_flags") or []
        lift = "PASS" if r.get("lift_success") else "FAIL"
        place = "PASS" if r.get("placement_success") else "FAIL"
        retreat = "PASS" if "RETREAT_PASS" in flags else "FAIL"
        disp = r.get("neighbor_displacement")
        tilt = r.get("neighbor_tilt")
        path = r.get("transport_path")
        line = (
            f"{r.get('scenario')}: lift={lift} place={place} retreat={retreat} "
            f"path={path} disp_mm={disp} tilt={tilt} "
            f"success={r.get('success')} reason={r.get('failure_reason')}"
        )
        lines.append(line)
        print("  " + line)
        disturbed = (disp or 0) > 2.0 or (tilt or 0) > 10.0
        if first_fail is None and (not _episode_ok(r) or disturbed):
            first_fail = line
    if first_fail is None:
        first_fail = "none — all transport cases completed with neighbors undisturbed"
    text = "\n".join(lines + ["", f"FIRST FAILURE MODE: {first_fail}"])
    (OUT / "transport.txt").write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "position_offsets",
            "perception_error",
            "both",
            "robustness",
            "diagnostics",
            "neighbors",
            "transport",
        ),
        default="both",
    )
    args = parser.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    sim = Sim(cfg=cfg)
    sim.quiet = True
    try:
        if args.suite == "robustness":
            text = run_robustness(sim, cfg)
            print("\n" + text)
            print(f"\nwrote {OUT / 'robustness.txt'}")
            return
        if args.suite == "diagnostics":
            text = run_diagnostics(sim, cfg)
            print("\n" + text)
            print(f"\nwrote {OUT / 'diagnostics.txt'}")
            return
        if args.suite == "neighbors":
            text = run_neighbors(sim, cfg)
            print("\n" + text)
            print(f"\nwrote {OUT / 'neighbors.txt'}")
            return
        if args.suite == "transport":
            text = run_transport(sim, cfg)
            print("\n" + text)
            print(f"\nwrote {OUT / 'transport.txt'}")
            return
        pos_rows: list[dict] = []
        perc_rows: list[dict] = []
        if args.suite in ("position_offsets", "both"):
            cases = position_offset_scenarios()
            print(f"\n=== position_offsets ({len(cases)}) ===")
            pos_rows = run_cases(sim, cfg, cases)
            (OUT / "position_offsets.json").write_text(json.dumps(pos_rows, indent=2, default=str))
        if args.suite in ("perception_error", "both"):
            cases = perception_error_scenarios()
            print(f"\n=== perception_error ({len(cases)}) ===")
            perc_rows = run_cases(sim, cfg, cases)
            (OUT / "perception_error.json").write_text(json.dumps(perc_rows, indent=2, default=str))
        text = summarize(pos_rows, perc_rows)
        (OUT / "summary.txt").write_text(text)
        print("\n" + text)
        print(f"\nwrote {OUT / 'summary.txt'}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
