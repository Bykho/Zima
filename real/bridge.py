"""HTTP client for scripts/arm_bridge.py when agent sandbox blocks serial."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8765"


def available() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/status", timeout=0.3) as r:
            return r.status == 200
    except Exception:
        return False


def status() -> dict:
    with urllib.request.urlopen(f"{URL}/status", timeout=2) as r:
        return json.loads(r.read())


def read_ticks() -> dict[str, int]:
    return status()["ticks"]


def write_ticks(ticks: dict[str, int]):
    req = urllib.request.Request(
        f"{URL}/ticks",
        data=json.dumps(ticks).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())
