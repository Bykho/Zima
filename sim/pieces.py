"""Primitive chess pieces from FEN — spawn and track bodies in PyBullet."""

from __future__ import annotations

from dataclasses import dataclass, field

import pybullet as p

from board import FILES, square_to_world

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_TYPES = ("pawn", "rook", "knight", "bishop", "queen", "king")
FEN_CHARS = {
    "p": "pawn",
    "r": "rook",
    "n": "knight",
    "b": "bishop",
    "q": "queen",
    "k": "king",
}

# radius/height in meters — sized for 37.5 mm squares
SPECS: dict[str, tuple[str, float, float]] = {
    "pawn": ("cylinder", 0.009, 0.022),
    "rook": ("cylinder", 0.012, 0.030),
    "bishop": ("cylinder", 0.010, 0.034),
    "queen": ("cylinder", 0.013, 0.038),
    "king": ("cylinder", 0.013, 0.042),
    "knight": ("box", 0.012, 0.026),
}

WHITE = [0.95, 0.95, 0.92, 1.0]
BLACK = [0.08, 0.08, 0.10, 1.0]


def parse_fen(fen: str) -> dict[str, tuple[str, str]]:
    """Map square name -> (color, piece_type)."""
    rows = fen.split()[0].split("/")
    out: dict[str, tuple[str, str]] = {}
    for rank_i, row in enumerate(reversed(rows)):
        file_i = 0
        for ch in row:
            if ch.isdigit():
                file_i += int(ch)
                continue
            color = "white" if ch.isupper() else "black"
            out[f"{FILES[file_i]}{rank_i + 1}"] = (color, FEN_CHARS[ch.lower()])
            file_i += 1
    return out


def piece_height(piece_type: str) -> float:
    return SPECS[piece_type][2]


def piece_center_z(cfg: dict, piece_type: str) -> float:
    surface = cfg.get("sim", {}).get("board", {}).get("surface_z", 0.02)
    return surface + piece_height(piece_type) / 2


def _make_shape(piece_type: str):
    kind, radius, height = SPECS[piece_type]
    if kind == "cylinder":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height)
    else:
        half = radius
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, height / 2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, height / 2])
    return col, vis, height


def spawn_piece(square: str, color: str, piece_type: str, cfg: dict, *, settle: bool = True) -> int:
    col, vis, height = _make_shape(piece_type)
    pos = square_to_world(square, cfg, z=piece_center_z(cfg, piece_type))
    rgba = WHITE if color == "white" else BLACK
    orn = [0, 0, 0, 1]
    if piece_type == "knight":
        orn = p.getQuaternionFromEuler([0, 0, 0.35])
    body = p.createMultiBody(
        baseMass=0.02,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn,
    )
    p.changeVisualShape(body, -1, rgbaColor=rgba)
    p.changeDynamics(body, -1, lateralFriction=1.2, rollingFriction=0.01)
    if settle:
        from fast import piece_settle_steps

        for _ in range(piece_settle_steps()):
            p.stepSimulation()
    return body


@dataclass
class BoardPieces:
    cfg: dict
    by_square: dict[str, int] = field(default_factory=dict)
    meta: dict[str, tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def from_fen(cls, cfg: dict, fen: str = START_FEN) -> BoardPieces:
        board = cls(cfg=cfg)
        for square, (color, piece_type) in parse_fen(fen).items():
            board.add(square, color, piece_type, settle=False)
        from fast import piece_settle_steps

        for _ in range(piece_settle_steps()):
            p.stepSimulation()
        return board

    def add(self, square: str, color: str, piece_type: str, *, settle: bool = True) -> int:
        body = spawn_piece(square, color, piece_type, self.cfg, settle=settle)
        self.by_square[square] = body
        self.meta[square] = (color, piece_type)
        return body

    def at(self, square: str) -> int | None:
        return self.by_square.get(square)

    def square_of(self, body_id: int) -> str | None:
        for sq, bid in self.by_square.items():
            if bid == body_id:
                return sq
        return None

    def remove(self, square: str) -> int | None:
        body = self.by_square.pop(square, None)
        self.meta.pop(square, None)
        return body

    def place(self, square: str, body_id: int, color: str, piece_type: str):
        self.by_square[square] = body_id
        self.meta[square] = (color, piece_type)

    def body_position(self, body_id: int) -> list[float]:
        return list(p.getBasePositionAndOrientation(body_id)[0])

    def nearest_square(self, body_id: int) -> str:
        pos = self.body_position(body_id)
        best, best_d = "", float("inf")
        size = self.cfg.get("sim", {}).get("board", {}).get("square_size", 0.0375)
        for sq in self.by_square:
            target = square_to_world(sq, self.cfg, z=pos[2])
            d = (pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2
            if d < best_d:
                best_d = d
                best = sq
        if best_d > (size * 0.6) ** 2:
            for sq in parse_fen(START_FEN):
                target = square_to_world(sq, self.cfg, z=pos[2])
                d = (pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = sq
        return best
