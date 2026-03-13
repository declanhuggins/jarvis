#!/usr/bin/env bash
# Jarvis Install Script
# Sets up the virtual environment, installs dependencies, downloads models,
# and registers the LaunchAgent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
PLIST_NAME="com.user.jarvis.plist"
PLIST_SRC="$PROJECT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
JARVIS_DIR="$HOME/.jarvis"

echo "=== Jarvis Installer ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# 1. Create ~/.jarvis directory for logs
echo "[1/6] Creating log directory..."
mkdir -p "$JARVIS_DIR"

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/6] Virtual environment already exists."
fi

# 3. Install dependencies
echo "[3/6] Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR"

# 4. Download wake word models
echo "[4/6] Downloading wake word models..."
"$VENV_DIR/bin/python" -c "
import openwakeword
openwakeword.utils.download_models()
print('Wake word models downloaded.')
"

# 5. Download Whisper model (pre-cache)
echo "[5/6] Pre-downloading Whisper model (this may take a minute)..."
"$VENV_DIR/bin/python" -c "
from faster_whisper import WhisperModel
model = WhisperModel('small', device='cpu', compute_type='int8')
print('Whisper small model ready.')
"

# 6. Install LaunchAgent
echo "[6/6] Installing LaunchAgent..."

PYTHON_PATH="$VENV_DIR/bin/python"

# Fill in the plist template with actual paths
sed -e "s|__VENV_PYTHON__|$PYTHON_PATH|g" \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$PLIST_SRC" > "$PLIST_DST"

# Unload if already loaded, suppress errors
launchctl unload "$PLIST_DST" 2>/dev/null || true

# Load the agent
launchctl load "$PLIST_DST"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Before starting Jarvis, you need to:"
echo "  1. Copy config.example.yaml to config.yaml"
echo "  2. Add your API keys to config.yaml"
echo "  3. Grant microphone access to Terminal/iTerm in:"
echo "     System Settings > Privacy & Security > Microphone"
echo ""
echo "Jarvis will start automatically on login."
echo "To start now:  launchctl start com.user.jarvis"
echo "To check logs: tail -f ~/.jarvis/jarvis.log"
echo "To stop:       launchctl stop com.user.jarvis"
echo "To uninstall:  ./scripts/uninstall.sh"
