"""Screenshot plugin."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class ScreenshotPlugin(BasePlugin):
    """Screenshot actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "take_screenshot",
                "description": "Take a full-screen screenshot and save it to a screenshots folder",
                "parameters": {},
                "destructive": False,
                "handler": self.take_screenshot,
            }
        ]

    def take_screenshot(self) -> str:
        """Capture a full-screen screenshot."""
        screenshot_dir = Path.home() / "Desktop" / "Jarvis Screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = screenshot_dir / f"screenshot_{timestamp}.png"

        logger.info("Taking screenshot: %s", path)
        subprocess.run(
            ["screencapture", "-x", str(path)],
            check=True,
            capture_output=True,
        )
        return f"Screenshot saved to {path}"
