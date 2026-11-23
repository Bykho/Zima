#!/usr/bin/env python3
"""Replay a famous game on the live stack.

  python sim/game.py                   # viewer, real-time, wrist camera (wall real)
  mjpython sim/game.py                 # macOS
  python sim/game.py --headless        # no window (1x clock; --speed 2 to hurry)
  python sim/game.py --wall cheat      # true poses, no wrist look
  python sim/game.py --game opera --speed 2
  mjpython sim/game.py --keep-going    # do not stop on a scored fail

Default is Morphy's Opera Game. Stops at the first physical failure
unless `--keep-going`.
Castling is king then rook. Promotions are not executed.
The viewer runs in real time so the arm doesn't strobe. Headless uses
the same 1x clock unless `--speed` is set. 3x+ drops grasps.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import evaluate_move
import stages
from episode import discard_square, occupancy_text, play_move
from fast import clock_scale
from mj import DEBUG_DIR, Sim, load_config
from pickup import FULL_BOARD_XML, seed_vertical
from robot import WALLS, open_robot, world_of

_GAMES = {
    "opera": """\
[Event "Paris Opera"]
[Site "Paris"]
[Date "1858.??.??"]
[White "Paul Morphy"]
[Black "Duke of Brunswick / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7
8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7
14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
""",
    "immortal": """\
[Event "London"]
[Site "London"]
[Date "1851.06.21"]
[White "Adolf Anderssen"]
[Black "Lionel Kieseritzky"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5
8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8
15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6
21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0
""",
    "evergreen": """\
[Event "Berlin"]
[Site "Berlin"]
[Date "1852.??.??"]
[White "Adolf Anderssen"]
[Black "Jean Dufresne"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3
8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6
14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3
20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0
""",
}


def _load_game(name: str, max_ply: int) -> tuple[chess.pgn.Game, chess.Board, list[chess.Move]]:
    key = name.strip().lower()
    if key not in _GAMES:
        want = ", ".join(sorted(_GAMES))
        raise SystemExit(f"Unknown game '{name}'. Choose: {want}")
    parsed = chess.pgn.read_game(io.StringIO(_GAMES[key]))
    if parsed is None:
        raise SystemExit(f"Could not parse PGN for '{key}'")
    board = parsed.board()
    moves: list[chess.Move] = []
    for move in parsed.mainline_moves():
        if len(moves) >= max_ply:
            break
        board.push(move)
        moves.append(move)
    return parsed, board, moves


def _game_label(game: chess.pgn.Game) -> str:
    white = game.headers.get("White") or "White"
    black = game.headers.get("Black") or "Black"
    event = game.headers.get("Event") or ""
    year = (game.headers.get("Date") or "")[:4]
    where = f"{event} {year}".strip()
    return f"{white} vs {black}" + (f"  ({where})" if where else "")


def _legs(board: chess.Board, move: chess.Move) -> tuple[list[tuple[str, str]], str | None]:
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)
    legs = [(src, dst)]
    hide = None
    if board.is_en_passant(move):
        hide = f"{dst[0]}{src[1]}"
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            rook = ("h1", "f1") if board.turn == chess.WHITE else ("h8", "f8")
        else:
            rook = ("a1", "d1") if board.turn == chess.WHITE else ("a8", "d8")
        legs.append(rook)
    return legs, hide


def _under_mjpython() -> bool:
    try:
        from mujoco.viewer import _MJPYTHON, _MjPythonBase

        return isinstance(_MJPYTHON, _MjPythonBase)
    except Exception:
        return False


def _reexec_mjpython() -> None:
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
            "Run:  .venv/bin/mjpython sim/game.py"
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
    every = 8

    def _sync():
        if viewer is None:
            return
        n[0] += 1
        if n[0] % every:
            return
        if not viewer.is_running():
            return
        _sync_viewer(viewer)

    sim.on_step = _sync


def main() -> int:
    parser = argparse.ArgumentParser(description="Engine game replayed on the sim board")
    parser.add_argument("--game", default="opera", help="famous game: opera, immortal, evergreen")
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--headless", action="store_true", help="no viewer")
    parser.add_argument("--speed", type=float, default=None, help="video clock (2 = 2x dt). viewer default 1; 3+ drops grasps")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="keep playing after a physical fail (collateral, drop, …)",
    )
    parser.add_argument("--wall", choices=WALLS, default="real", help="controller/world separation (robot.py)")
    stages.cli_arg(parser)
    args = parser.parse_args()
    stages.select(args.stage)
    want_view = not args.headless
    if args.speed is not None:
        clock = float(args.speed) if args.speed >= 1.0 else 1.0 / max(0.05, float(args.speed))
        os.environ["ZIMA_SIM_FAST"] = "1"
        os.environ["ZIMA_SIM_CLOCK"] = str(clock)
    elif want_view:
        os.environ.setdefault("ZIMA_SIM_CLOCK", "1")
    else:
        # Match the viewer: 1x clock, full grasp/settle. FAST shortens those
        # and flakes opening-file collateral (headless e5 / d6). Use --speed
        # when you want the faster, sloppier clock.
        os.environ.setdefault("ZIMA_SIM_CLOCK", "1")
    if want_view:
        _reexec_mjpython()

    game, final, moves = _load_game(args.game, args.max_ply)
    replay = chess.Board()
    print(f"game  {_game_label(game)}  ply={len(moves)}  result={final.result()}")
    print(replay.variation_san(moves))
    print(f"perception: {'true poses' if args.wall in ('cheat', 'off') else 'wrist RGB-D'}  wall: {args.wall}")
    print(f"clock: {clock_scale()}x  viewer: {'on' if want_view else 'off'}")
    if args.keep_going:
        print("keep-going: on  (scored fails do not stop the game)")

    cfg = load_config()
    sim = open_robot(Sim(xml_path=FULL_BOARD_XML, cfg=cfg), args.wall)
    sim.quiet = True
    seed_vertical(sim, cfg)

    viewer = None
    if want_view:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(world_of(sim).model, world_of(sim).data)
        _hook_viewer(sim, viewer)

    print(occupancy_text(sim, cfg))

    out_dir = DEBUG_DIR / "game"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "latest.jsonl"
    rows: list[dict] = []
    code = 0
    try:
        with log_path.open("w") as log:
            for i, move in enumerate(moves, start=1):
                san = replay.san(move)
                if move.promotion:
                    rec = {
                        "ply": i,
                        "san": san,
                        "uci": move.uci(),
                        "ok": False,
                        "reason": "PROMOTION",
                        "stage": "rules",
                    }
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"{i:02d}  {san}  FAIL  rules  PROMOTION")
                    code = 1
                    rows.append(rec)
                    break
                legs, hide = _legs(replay, move)
                replay.push(move)
                ok = True
                last: dict = {}
                occ0, before = evaluate_move.board_snapshot(world_of(sim), cfg)
                involved = {occ0.get(sq) for leg in legs for sq in leg} | {occ0.get(hide)}
                if hide:
                    dumped = discard_square(
                        sim,
                        cfg,
                        hide,
                        keep_squares={legs[0][0]} if legs else set(),
                    )
                    if not dumped.get("success"):
                        rec = {
                            "ply": i,
                            "san": san,
                            "uci": move.uci(),
                            "ok": False,
                            "reason": dumped.get("failure_reason") or "CAPTURE",
                            "stage": dumped.get("failure_stage") or "capture",
                            "captured": dumped.get("captured"),
                        }
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        print(
                            f"{i:02d}  {san}  FAIL  {rec['stage']}  {rec['reason']}",
                            flush=True,
                        )
                        code = 1
                        rows.append(rec)
                        if not args.keep_going:
                            break
                        continue
                for src, dst in legs:
                    _hook_viewer(sim, viewer)
                    last = play_move(sim, cfg, f"{src} {dst}")
                    _sync_viewer(viewer)
                    _hook_viewer(sim, viewer)
                    if not last.get("success"):
                        ok = False
                        break
                disturbed = evaluate_move.disturbed(world_of(sim), before, skip=involved)
                placed = {f"{src}{dst}": evaluate_move.placement(world_of(sim), cfg, occ0[src], dst) for src, dst in legs if occ0.get(src)}
                if ok and not all(p["ok"] for p in placed.values()):
                    ok = False
                    last = {**last, "failure_stage": "place", "failure_reason": "NOT_PLACED"}
                if ok and disturbed:
                    ok = False
                    last = {**last, "failure_stage": "collateral", "failure_reason": "COLLATERAL_KNOCK"}
                rec = {
                    "ply": i,
                    "san": san,
                    "uci": move.uci(),
                    "legs": [f"{a}{b}" for a, b in legs],
                    "ok": ok,
                    "piece": last.get("piece"),
                    "detour": last.get("transport_detour"),
                    "standoff": last.get("transport_standoff"),
                    "stage": last.get("failure_stage") if not ok else None,
                    "reason": last.get("failure_reason") if not ok else None,
                    "captured": last.get("captured"),
                    "disturbed": disturbed,
                    "placed": placed,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                rows.append(rec)
                if ok:
                    detour = rec.get("detour")
                    tail = f"  {detour}" if detour and detour not in ("straight", None) else ""
                    print(f"{i:02d}  {san}  OK{tail}", flush=True)
                else:
                    print(
                        f"{i:02d}  {san}  FAIL  {rec.get('stage') or '?'}  {rec.get('reason') or '?'}",
                        flush=True,
                    )
                    code = 1
                    if args.keep_going:
                        continue
                    print(occupancy_text(sim, cfg), flush=True)
                    break
            else:
                print(occupancy_text(sim, cfg))
                print(f"game over  {final.result()}  {len(rows)} ply")
    finally:
        summary = {
            "game": args.game,
            "label": _game_label(game),
            "result": final.result(),
            "planned_ply": len(moves),
            "played": len(rows),
            "ok": code == 0,
            "pgn": chess.Board().variation_san(moves),
            "log": str(log_path),
        }
        (out_dir / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
        if viewer is not None:
            viewer.close()
        sim.close()
        print(f"log  {log_path}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
