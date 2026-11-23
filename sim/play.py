#!/usr/bin/env python3
"""Interactive persistent-board play.

  python sim/play.py              # viewer + prompt (wrist RGB-D; macOS uses mjpython)
  python sim/play.py --gt         # viewer + prompt, ground-truth pose
  python sim/play.py --gt --speed 2
  python sim/play.py --headless   # prompt only
  python sim/play.py e2 e4 g1 f3  # one-shot moves, then exit

Type moves like `e2 e4`. Same board until `reset`. No chess engine.
`--speed` is optional; omit it for real-time. Unset ZIMA_SIM_FAST if a
previous command left it in your shell.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import evaluate_move
from episode import occupancy_text, play_move
from mj import Sim, load_config
from pickup import FULL_BOARD_XML, seed_vertical
from robot import WALLS, open_robot, world_of
from squares import parse_move


def _summarize(result: dict) -> str:
    src = result.get("source")
    dst = result.get("dest")
    piece = result.get("piece")
    if result.get("success"):
        detour = result.get("transport_detour")
        tail = f"  {detour}" if detour and detour not in ("straight", "lock", None) else ""
        return f"OK  {piece} {src}->{dst}{tail}"
    stage = result.get("failure_stage") or "?"
    reason = result.get("failure_reason") or result.get("error") or "?"
    return f"FAIL  {piece or '-'} {src}->{dst}  {stage}  {reason}"


_IK_REASONS = {"IK_FAILURE", "IK_NO_SOLUTION"}
_COLLISION_REASONS = {
    "APPROACH_COLLISION",
    "NO_COLLISION_FREE_APPROACH",
    "TRANSPORT_COLLISION",
}


def _fmt_xyz(v) -> str:
    if not v or len(v) < 3:
        return "-"
    return f"[{float(v[0]):.3f} {float(v[1]):.3f} {float(v[2]):.3f}]"


def _fail_detail(result: dict) -> str | None:
    if result.get("success"):
        return None
    reason = result.get("failure_reason") or result.get("error")
    if reason in ("BAD_MOVE", "EMPTY_SRC"):
        return None
    lines: list[str] = []
    req, act = result.get("ee_requested"), result.get("ee_actual")
    if req or act:
        line = f"  ee  requested={_fmt_xyz(req)}  actual={_fmt_xyz(act)}"
        if reason in _IK_REASONS:
            res = result.get("ik_residual_mm")
            if res is not None:
                line += f"  residual={float(res):.1f}mm"
            lim = result.get("joint_limit_margin_rad") or {}
            mn = lim.get("min_rad")
            if mn is not None:
                line += f"  limit_min={float(mn):.3f}rad"
                if lim.get("tightest"):
                    line += f"({lim['tightest']})"
            if result.get("ik_lin_wp"):
                line += f"  wp={result['ik_lin_wp']}"
            if result.get("ik_fail_phase"):
                line += f"  phase={result['ik_fail_phase']}"
        lines.append(line)
    if reason in _COLLISION_REASONS or (reason not in _IK_REASONS and result.get("collision_pair")):
        pair = result.get("collision_pair") or {}
        if pair:
            lines.append(
                f"  collision  {pair.get('robot')} vs {pair.get('body')}  "
                f"clearance={pair.get('clearance_mm')}mm  "
                f"square={pair.get('square') or '-'}  phase={pair.get('phase') or '-'}"
            )
        elif reason in _COLLISION_REASONS:
            lines.append("  collision  (no pair recorded)")
    return "\n".join(lines) if lines else None


def _snapshot(sim: Sim) -> dict:
    """Physics state of the world behind the controller (operator undo, not a robot input)."""
    world = world_of(sim)
    return {
        "body_pos": np.array(world.model.body_pos),
        "qpos": world.data.qpos.copy(),
        "qvel": world.data.qvel.copy(),
        "ctrl": world.data.ctrl.copy(),
    }


def _restore(sim: Sim, cfg: dict, snap: dict) -> None:
    """Rewind the world to `snap`; the robot forgets what it saw and homes physically."""
    world = world_of(sim)
    with world.scene_setup():
        world.model.body_pos[:] = snap["body_pos"]
        mujoco.mj_resetData(world.model, world.data)
        world.data.qpos[:] = snap["qpos"]
        world.data.qvel[:] = snap["qvel"]
        world.data.ctrl[:] = snap["ctrl"]
        mujoco.mj_forward(world.model, world.data)
    if sim is not world:
        sim.data.ctrl[:] = snap["ctrl"]
        sim.forget_all()
    seed_vertical(sim, cfg)


def _under_mjpython() -> bool:
    try:
        from mujoco.viewer import _MJPYTHON, _MjPythonBase

        return isinstance(_MJPYTHON, _MjPythonBase)
    except Exception:
        return False


def _reexec_mjpython() -> None:
    """macOS MuJoCo viewer only works under mjpython."""
    if sys.platform != "darwin" or _under_mjpython():
        return
    # Don't .resolve() sys.executable: in a venv it's a symlink into the base
    # install, which has no mjpython. Look next to the venv's python first.
    candidates = [
        Path(sys.prefix) / "bin" / "mjpython",
        Path(sys.executable).parent / "mjpython",
        Path(shutil.which("mjpython") or "/nonexistent"),
    ]
    mjpython = next((p for p in candidates if p.is_file()), None)
    if mjpython is None:
        raise SystemExit(
            "On macOS the viewer needs mjpython (pip install mujoco provides it). "
            "Run:  .venv/bin/mjpython sim/play.py"
        )
    os.execv(str(mjpython), [str(mjpython), *sys.argv])


def _sync_viewer(viewer) -> None:
    if viewer is None or not viewer.is_running():
        return
    try:
        viewer.sync(state_only=True)
    except TypeError:
        viewer.sync()


def _hook_viewer(sim: Sim, viewer) -> None:
    n = [0]

    def _sync():
        if viewer is None:
            return
        n[0] += 1
        if n[0] % 8:
            return
        if not viewer.is_running():
            return
        _sync_viewer(viewer)

    sim.on_step = _sync


def _run_move(sim, cfg, text: str, use_vision: bool, viewer) -> dict:
    """play_move plus the world verdict: `ok` also needs the mover on its square and no bystander disturbed."""
    _hook_viewer(sim, viewer)
    try:
        src, dst = parse_move(text)
    except ValueError:
        src = dst = None
    occ, before = evaluate_move.board_snapshot(world_of(sim), cfg)
    result = play_move(sim, cfg, text, use_vision=use_vision)
    score = None if dst is None else evaluate_move.score_board_move(world_of(sim), cfg, before, occ.get(src), occ.get(dst), dst)
    result["world"] = score
    result["ok"] = bool(result.get("success") and score and score["ok"])
    print(_summarize(result))
    detail = _fail_detail(result)
    if detail:
        print(detail)
    if score and result.get("success") and not score["ok"]:
        what = f"disturbed {[d['body'] for d in score['disturbed']]}" if score["disturbed"] else "mover not on its square"
        print(f"FAIL  world: {what}")
    _sync_viewer(viewer)
    _hook_viewer(sim, viewer)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Play typed chess moves on a persistent sim board")
    parser.add_argument("moves", nargs="*", help="moves like e2 e4 or e2e4")
    parser.add_argument("--gt", action="store_true", help="ground-truth pose instead of wrist RGB-D")
    parser.add_argument("--view", action="store_true", help="open MuJoCo viewer (default in interactive mode)")
    parser.add_argument("--headless", action="store_true", help="no viewer")
    parser.add_argument("--speed", type=float, default=None, help="video clock (2 = 2x dt). omit for real-time")
    parser.add_argument("--wall", choices=WALLS, default="real", help="controller/world separation (robot.py)")
    args = parser.parse_args()
    interactive = not args.moves
    want_view = (not args.headless) and (args.view or interactive)
    if args.speed is not None:
        clock = float(args.speed) if args.speed >= 1.0 else 1.0 / max(0.05, float(args.speed))
        os.environ["ZIMA_SIM_FAST"] = "1"
        os.environ["ZIMA_SIM_CLOCK"] = str(clock)
    elif want_view:
        # Don't inherit a leftover FAST=1 from a previous command.
        os.environ.pop("ZIMA_SIM_FAST", None)
        os.environ.pop("ZIMA_SIM_CLOCK", None)
    if want_view:
        _reexec_mjpython()

    cfg = load_config()
    sim = open_robot(Sim(xml_path=FULL_BOARD_XML, cfg=cfg), args.wall)
    seed_vertical(sim, cfg)
    start = _snapshot(sim)
    use_vision = not args.gt

    viewer = None
    if want_view:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(world_of(sim).model, world_of(sim).data)
        _hook_viewer(sim, viewer)

    print(occupancy_text(sim, cfg))
    print("perception:", "wrist RGB-D" if use_vision else "ground truth")
    print("e2 e4  |  board  |  reset  |  quit")

    code = 0
    try:
        if args.moves:
            buf = []
            for tok in args.moves:
                buf.append(tok)
                joined = " ".join(buf)
                try:
                    from squares import parse_move

                    parse_move(joined)
                except ValueError:
                    continue
                result = _run_move(sim, cfg, joined, use_vision, viewer)
                if not result["ok"]:
                    code = 1
                buf = []
            if buf:
                print(f"FAIL  unused tokens: {' '.join(buf)}  parse")
                code = 1
        else:
            while True:
                if viewer is not None and not viewer.is_running():
                    print("viewer closed")
                    break
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                cmd = line.lower()
                if cmd in ("q", "quit", "exit"):
                    break
                if cmd in ("board", "show"):
                    print(occupancy_text(sim, cfg))
                    continue
                if cmd == "reset":
                    _restore(sim, cfg, start)
                    _hook_viewer(sim, viewer)
                    _sync_viewer(viewer)
                    print("reset")
                    print(occupancy_text(sim, cfg))
                    continue
                if cmd in ("help", "?"):
                    print("e2 e4  |  board  |  reset  |  quit")
                    continue
                result = _run_move(sim, cfg, line, use_vision, viewer)
                if not result["ok"]:
                    code = 1
    finally:
        if viewer is not None:
            viewer.close()
        sim.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
