"""Command router - maps intent actions to plugin handlers."""

from __future__ import annotations

import logging
from typing import Any, Callable

from jarvis.errors import PluginError, UnknownActionError
from jarvis.intent import Intent

logger = logging.getLogger(__name__)


class CommandRouter:
    """Routes Intent actions to registered plugin handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self,
        action_name: str,
        handler: Callable,
        metadata: dict,
    ) -> None:
        """Register a handler function for an action name.

        Args:
            action_name: Unique action identifier (e.g. "open_app").
            handler: Callable that accepts **parameters and returns a result string.
            metadata: Dict with description, parameters schema, and destructive flag.
        """
        if action_name in self._handlers:
            logger.warning("Overwriting handler for action: %s", action_name)

        self._handlers[action_name] = handler
        self._metadata[action_name] = metadata
        logger.debug("Registered action: %s", action_name)

    def get_action_catalog(self) -> list[dict]:
        """Return metadata for all registered actions.

        This is fed into the LLM system prompt so it knows what actions exist.
        """
        catalog = []
        for name, meta in self._metadata.items():
            catalog.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "parameters": meta.get("parameters", {}),
                    "destructive": meta.get("destructive", False),
                }
            )
        return catalog

    def execute(self, intent: Intent) -> str:
        """Execute the handler for the given intent.

        Args:
            intent: Parsed Intent with action name and parameters.

        Returns:
            Result string from the handler (may be empty).

        Raises:
            UnknownActionError: If no handler is registered for the action.
            PluginError: If the handler raises an exception.
        """
        handler = self._handlers.get(intent.action)
        if handler is None:
            raise UnknownActionError(
                f"No handler registered for action: {intent.action!r}"
            )

        logger.info(
            "Executing action: %s with params: %s",
            intent.action,
            intent.parameters,
        )

        try:
            result = handler(**intent.parameters)
            return str(result) if result is not None else ""
        except TypeError as e:
            raise PluginError(
                f"Bad parameters for {intent.action}: {e}"
            ) from e
        except Exception as e:
            raise PluginError(
                f"Plugin {intent.action} failed: {e}"
            ) from e
