"""Intent parsing and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jarvis.errors import IntentParseError

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Structured representation of a parsed LLM intent."""

    action: str
    parameters: dict
    confirmation_required: bool
    spoken_response: str
    reasoning: str


def parse_intent(raw: dict) -> Intent:
    """Validate and normalize LLM JSON output into an Intent.

    Args:
        raw: Dict parsed from the LLM's JSON response.

    Returns:
        Validated Intent dataclass.

    Raises:
        IntentParseError: If required fields are missing or have wrong types.
    """
    _require_field(raw, "action", str)
    _require_field(raw, "parameters", dict)
    _require_field(raw, "confirmation_required", bool)
    _require_field(raw, "spoken_response", str)

    # Reasoning is optional - some LLM responses may omit it
    reasoning = raw.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    intent = Intent(
        action=raw["action"].strip(),
        parameters=raw["parameters"],
        confirmation_required=raw["confirmation_required"],
        spoken_response=raw["spoken_response"].strip(),
        reasoning=reasoning,
    )

    if not intent.action:
        raise IntentParseError("Action field is empty")

    logger.debug(
        "Parsed intent: action=%s, confirm=%s, reasoning=%s",
        intent.action,
        intent.confirmation_required,
        intent.reasoning,
    )

    return intent


def _require_field(data: dict, field: str, expected_type: type) -> None:
    """Check that a field exists and has the expected type."""
    if field not in data:
        raise IntentParseError(f"Missing required field: {field!r}")
    if not isinstance(data[field], expected_type):
        raise IntentParseError(
            f"Field {field!r} must be {expected_type.__name__}, "
            f"got {type(data[field]).__name__}"
        )
