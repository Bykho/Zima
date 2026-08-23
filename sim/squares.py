"""Board square ↔ world XY. No simulator dependency."""

from __future__ import annotations

import math

FILES = "abcdefgh"


def parse_square(name: str) -> tuple[int, int]:
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"Bad square '{name}'")
    return FILES.index(name[0]), int(name[1])


def square_to_xy(square: str, origin: list[float], square_size: float, yaw: float = 0.0) -> tuple[float, float]:
    file_i, rank = parse_square(square)
    lx = (file_i + 0.5) * square_size
    ly = (rank - 0.5) * square_size
    c, s = math.cos(yaw), math.sin(yaw)
    x = origin[0] + c * lx - s * ly
    y = origin[1] + s * lx + c * ly
    return x, y


def square_to_world(square: str, cfg: dict, z: float | None = None) -> list[float]:
    board = cfg.get("sim", {}).get("board", {})
    origin = board.get("origin", [0.22, -0.10, 0.0])
    size = board.get("square_size", 0.0375)
    yaw = board.get("yaw", 0.0)
    height = board.get("surface_z", 0.02) if z is None else z
    x, y = square_to_xy(square, origin, size, yaw)
    return [x, y, height]


def shift_square(square: str, dfile: int = 0, drank: int = 0) -> str | None:
    file_i, rank = parse_square(square)
    nf, nr = file_i + int(dfile), rank + int(drank)
    if nf < 0 or nf > 7 or nr < 1 or nr > 8:
        return None
    return f"{FILES[nf]}{nr}"


def jitter_near_square(cfg: dict, square: str, rng, radius: float = 0.010) -> list[float]:
    """On-board XY near a square, not at the square center.

    The wrist camera at the source look pose sees forward (+X). Spawns too far
    toward the robot fall out of that view, so keep x close to the square.
    """
    base = square_to_world(square, cfg)
    for _ in range(80):
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        r = radius * math.sqrt(float(rng.uniform(0.4, 1.0)))
        x = base[0] + r * math.cos(ang)
        y = base[1] + r * math.sin(ang)
        if x >= base[0] - 0.006:
            return [x, y, base[2]]
    return [base[0] + 0.008, base[1], base[2]]
