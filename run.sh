#!/usr/bin/env bash
# Run Jarvis without manually activating the venv.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/venv/bin/python" -m jarvis "$@"
