#!/usr/bin/env python3
"""Sim chess move: spawn pieces from FEN and play e2 -> e4."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm import SimArm, load_config
from board import add_board_visuals, mark_square
from pick import execute_move, piece_near_square
from pieces import START_FEN, BoardPieces, parse_fen


def main():
    parser = argparse.ArgumentParser(description="Chess sim demo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fen", default=START_FEN, help="starting FEN")
    parser.add_argument("--move", default="e2e4", help="UCI move")
    parser.add_argument("--full-board", action="store_true", help="spawn all pieces from FEN")
    args = parser.parse_args()

    cfg = load_config()
    gui = not args.headless
    if args.headless:
        os.environ.setdefault("ZIMA_SIM_FAST", "1")
    arm = SimArm(gui=gui, fast=not gui)
    arm.reset_pose(cfg["poses"]["above_board"])

    if gui and (
        cfg.get("sim", {}).get("board", {}).get("show_grid", True)
        or cfg.get("sim", {}).get("board", {}).get("show_checker", True)
    ):
        add_board_visuals(cfg)

    board_cfg = cfg.get("sim", {}).get("board", {})
    if args.full_board or (gui and board_cfg.get("show_pieces", True)):
        pieces = BoardPieces.from_fen(cfg, args.fen)
    else:
        from_sq, to_sq = args.move[:2], args.move[2:4]
        board = parse_fen(args.fen)
        pieces = BoardPieces(cfg=cfg)
        if from_sq in board:
            color, ptype = board[from_sq]
            pieces.add(from_sq, color, ptype)

    print(f"spawned {len(pieces.by_square)} piece(s)")

    from_sq, to_sq = args.move[:2], args.move[2:4]
    if gui:
        mark_square(from_sq, cfg, [0.9, 0.2, 0.2])
        mark_square(to_sq, cfg, [0.2, 0.2, 0.9])

    print(f"move {args.move}")
    ok = execute_move(arm, pieces, args.move, cfg, duration=2.0)
    body = pieces.at(to_sq)
    placed = body is not None and piece_near_square(pieces, body, to_sq, cfg)
    print(f"{'ok' if ok and placed else 'failed'} — piece on {to_sq}: {placed}")

    if gui:
        input("Press Enter...")
    arm.disconnect()
    sys.exit(0 if ok and placed else 1)


if __name__ == "__main__":
    main()
