#!/usr/bin/env bash
# Run the Jarvis test harness without manually activating the venv.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/scripts/test_wakeword.py" "$@"
