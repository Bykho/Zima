"""Sim pose library — config.yaml + sim_poses.json."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POSES_FILE = ROOT / "data" / "sim_poses.json"


def load_config_poses() -> dict[str, dict[str, float]]:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f).get("poses", {})


def load_saved() -> dict[str, dict[str, float]]:
    if not POSES_FILE.exists():
        return {}
    with open(POSES_FILE) as f:
        return json.load(f)


def load_all() -> dict[str, dict[str, float]]:
    poses = dict(load_config_poses())
    poses.update(load_saved())
    return poses


def save_pose(name: str, joints: dict[str, float]):
    POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_poses = load_saved()
    all_poses[name] = {k: round(v, 4) for k, v in joints.items()}
    with open(POSES_FILE, "w") as f:
        json.dump(all_poses, f, indent=2)
    return name


def delete_pose(name: str):
    all_poses = load_saved()
    all_poses.pop(name, None)
    with open(POSES_FILE, "w") as f:
        json.dump(all_poses, f, indent=2)


def list_poses() -> list[str]:
    return list(load_all().keys())
