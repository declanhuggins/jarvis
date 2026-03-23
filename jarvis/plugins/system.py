"""System commands plugin: open apps, volume, brightness, lock screen, shell."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


def _find_executable(name: str) -> str | None:
    """Find a Homebrew-installed CLI in shell and launchd environments."""
    candidates = [
        shutil.which(name),
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


class SystemPlugin(BasePlugin):
    """macOS system control actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "open_app",
                "description": "Open a macOS application by name",
                "parameters": {
                    "app_name": {
                        "type": "string",
                        "description": "Application name, e.g. 'Safari', 'Terminal', 'Spotify'",
                    }
                },
                "destructive": False,
                "handler": self.open_app,
            },
            {
                "name": "set_volume",
                "description": "Set the system audio output volume to a percentage (0-100)",
                "parameters": {
                    "level": {
                        "type": "integer",
                        "description": "Volume percentage from 0 (mute) to 100 (max)",
                    }
                },
                "destructive": False,
                "handler": self.set_volume,
            },
            {
                "name": "set_brightness",
                "description": "Set the screen brightness to a percentage (0-100)",
                "parameters": {
                    "level": {
                        "type": "integer",
                        "description": "Brightness percentage from 0 (darkest) to 100 (brightest)",
                    }
                },
                "destructive": False,
                "handler": self.set_brightness,
            },
            {
                "name": "lock_screen",
                "description": "Lock the screen immediately",
                "parameters": {},
                "destructive": False,
                "handler": self.lock_screen,
            },
            {
                "name": "toggle_dark_mode",
                "description": "Toggle macOS dark mode on or off",
                "parameters": {},
                "destructive": False,
                "handler": self.toggle_dark_mode,
            },
            {
                "name": "shell_command",
                "description": (
                    "Run an arbitrary shell command and return its output. "
                    "Use only when no specific action fits the request."
                ),
                "parameters": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "destructive": True,
                "handler": self.shell_command,
            },
        ]

    def open_app(self, app_name: str) -> str:
        """Open a macOS application by name."""
        logger.info("Opening application: %s", app_name)
        result = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error = result.stderr.strip()
            return f"Failed to open {app_name}: {error}"
        return f"Opened {app_name}"

    def set_volume(self, level: int) -> str:
        """Set system volume using osascript."""
        level = max(0, min(100, int(level)))
        logger.info("Setting volume to %d%%", level)
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            check=True,
        )
        return f"Volume set to {level}%"

    def set_brightness(self, level: int) -> str:
        """Set screen brightness using the `brightness` CLI."""
        level = max(0, min(100, int(level)))
        fraction = level / 100.0
        logger.info("Setting brightness to %d%% (%.2f)", level, fraction)

        brightness_cli = _find_executable("brightness")
        if brightness_cli is None:
            return (
                "Brightness control is unavailable because the `brightness` CLI "
                "is not installed. Run `brew install brightness`."
            )

        try:
            subprocess.run(
                [brightness_cli, str(fraction)],
                check=True,
                capture_output=True,
            )
            return f"Brightness set to {level}%"
        except subprocess.CalledProcessError as e:
            error = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            details = f": {error.strip()}" if error else ""
            return (
                "Could not set brightness with the `brightness` CLI"
                f"{details}"
            )

    def lock_screen(self) -> str:
        """Lock the screen using pmset."""
        logger.info("Locking screen")
        subprocess.run(
            ["pmset", "displaysleepnow"],
            check=True,
        )
        return "Screen locked"

    def toggle_dark_mode(self) -> str:
        """Toggle macOS dark mode."""
        logger.info("Toggling dark mode")
        script = """
        tell application "System Events"
            tell appearance preferences
                set dark mode to not dark mode
                if dark mode then
                    return "Dark mode enabled"
                else
                    return "Light mode enabled"
                end if
            end tell
        end tell
        """
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "Failed to toggle dark mode"

    def shell_command(self, command: str) -> str:
        """Run an arbitrary shell command. Destructive - requires confirmation."""
        logger.info("Executing shell command: %s", command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip() or result.stderr.strip()
            # Truncate long output to prevent TTS from reading too much
            if len(output) > 500:
                output = output[:500] + "... (output truncated)"
            if result.returncode != 0:
                return f"Command exited with code {result.returncode}: {output}"
            return output or "Command completed successfully"
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
