"""Plugin discovery and registration."""

from __future__ import annotations

import logging

from jarvis.config import JarvisConfig
from jarvis.plugins.assistant import AssistantPlugin
from jarvis.plugins.files import FilesPlugin
from jarvis.plugins.music import MusicPlugin
from jarvis.plugins.screenshot import ScreenshotPlugin
from jarvis.plugins.system import SystemPlugin
from jarvis.plugins.weather import WeatherPlugin
from jarvis.router import CommandRouter

logger = logging.getLogger(__name__)


def register_all_plugins(router: CommandRouter, config: JarvisConfig) -> None:
    """Instantiate all plugins and register their actions with the router."""
    plugins = [
        AssistantPlugin(config),
        SystemPlugin(config),
        FilesPlugin(config),
        WeatherPlugin(config),
        ScreenshotPlugin(config),
        MusicPlugin(config),
    ]

    for plugin in plugins:
        plugin_name = type(plugin).__name__
        actions = plugin.get_actions()
        for action in actions:
            router.register(
                action_name=action["name"],
                handler=action["handler"],
                metadata={
                    "description": action["description"],
                    "parameters": action["parameters"],
                    "destructive": action["destructive"],
                },
            )
        logger.info(
            "Registered plugin %s with %d actions", plugin_name, len(actions)
        )

    # Register the catch-all conversational response action
    router.register(
        action_name="conversational_response",
        handler=lambda **_: "",
        metadata={
            "description": "Respond conversationally when no specific action applies",
            "parameters": {},
            "destructive": False,
        },
    )

    logger.info(
        "Total registered actions: %d",
        len(router.get_action_catalog()),
    )
