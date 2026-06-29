#!/usr/bin/env bash
# Start arm bridge in background (unsandboxed terminal). Agent uses HTTP.
cd "$(dirname "$0")/.."
exec .venv/bin/python3 scripts/arm_bridge.py
