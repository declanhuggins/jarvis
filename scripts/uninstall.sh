#!/usr/bin/env bash
# Jarvis Uninstall Script
# Stops the assistant, removes the LaunchAgent, and optionally cleans up.

set -euo pipefail

PLIST_NAME="com.user.jarvis.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=== Jarvis Uninstaller ==="

# 1. Stop the service
echo "[1/3] Stopping Jarvis..."
launchctl stop com.user.jarvis 2>/dev/null || true

# 2. Unload and remove the LaunchAgent
echo "[2/3] Removing LaunchAgent..."
if [ -f "$PLIST_DST" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm "$PLIST_DST"
    echo "  Removed $PLIST_DST"
else
    echo "  LaunchAgent not found (already removed?)"
fi

# 3. Ask about cleanup
echo "[3/3] Cleanup options:"
read -p "  Remove virtual environment ($VENV_DIR)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$VENV_DIR"
    echo "  Virtual environment removed."
fi

read -p "  Remove log files (~/.jarvis/)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$HOME/.jarvis"
    echo "  Log directory removed."
fi

echo ""
echo "=== Uninstall Complete ==="
echo "Your config.yaml and project files have been preserved."
