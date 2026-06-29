#!/bin/bash
# Agent closed-loop sim check — run after every sim change
cd "$(dirname "$0")/.."
exec .venv/bin/python3 sim/verify.py "$@"
