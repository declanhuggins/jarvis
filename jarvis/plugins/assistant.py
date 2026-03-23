"""Jarvis self-management actions."""

from __future__ import annotations

from jarvis.config import JarvisConfig
from jarvis.plugins.base import BasePlugin


class AssistantPlugin(BasePlugin):
    """Assistant control actions."""

    def __init__(self, config: JarvisConfig):
        self._config = config

    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "shutdown_jarvis",
                "description": (
                    "Shut down Jarvis now."
                ),
                "parameters": {},
                "destructive": True,
                "handler": self.shutdown_jarvis,
            },
            {
                "name": "disable_jarvis",
                "description": (
                    "Disable Jarvis so its LaunchAgent stays off until manually re-enabled."
                ),
                "parameters": {},
                "destructive": True,
                "handler": self.disable_jarvis,
            }
        ]

    def shutdown_jarvis(self) -> str:
        """Acknowledge a shutdown request.

        The actual shutdown behavior is handled in the main loop after
        Jarvis has spoken its final response.
        """
        return ""

    def disable_jarvis(self) -> str:
        """Acknowledge a disable request.

        The actual disable behavior is handled in the main loop after Jarvis
        has spoken its final response.
        """
        return ""
