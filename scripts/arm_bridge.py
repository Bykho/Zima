#!/usr/bin/env python3
"""Local arm bridge — run in your terminal, agent talks over HTTP.

  python scripts/arm_bridge.py

Keeps USB open in an unsandboxed process. Agent uses http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from real.arm import RealArm  # noqa: E402
from real.calib import find_port  # noqa: E402

PORT = 8765
arm: RealArm | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[bridge] {args[0]}")

    def _json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/status":
            return self._json(404, {"error": "not found"})
        assert arm
        self._json(200, {"port": arm.port, "ticks": arm.read_ticks()})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}

        if self.path == "/ticks":
            assert arm
            arm.write_ticks({k: int(v) for k, v in body.items()})
            self._json(200, {"ok": True, "ticks": arm.read_ticks()})

        elif self.path == "/close":
            global arm
            if arm:
                arm.close()
                arm = None
            self._json(200, {"ok": True})

        else:
            self._json(404, {"error": "not found"})


def main():
    global arm
    dev = find_port()
    if not dev:
        sys.exit("no arm port — plug in USB")
    arm = RealArm(dev)
    arm.connect()
    print(f"bridge on http://127.0.0.1:{PORT}  device {dev}")
    print(f"ticks: {arm.read_ticks()}")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if arm:
            arm.close()


if __name__ == "__main__":
    main()
