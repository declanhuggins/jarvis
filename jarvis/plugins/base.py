"""Abstract base class for Jarvis plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class BasePlugin(ABC):
    """All plugins inherit from this and declare their actions via get_actions()."""

    @abstractmethod
    def get_actions(self) -> list[dict]:
        """Return a list of action descriptors.

        Each dict must have:
            name: str           - Unique action identifier (e.g. "open_app").
            description: str    - Human-readable description for the LLM.
            parameters: dict    - JSON Schema-style parameter descriptions.
            destructive: bool   - Whether this action requires user confirmation.
            handler: Callable   - The method to call when this action is triggered.
        """
        ...
