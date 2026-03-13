"""Tests for intent parsing."""

import pytest

from jarvis.errors import IntentParseError
from jarvis.intent import Intent, parse_intent


class TestParseIntent:
    def test_valid_intent(self):
        raw = {
            "action": "open_app",
            "parameters": {"app_name": "Safari"},
            "confirmation_required": False,
            "spoken_response": "Opening Safari",
            "reasoning": "User wants to open Safari",
        }
        intent = parse_intent(raw)
        assert intent.action == "open_app"
        assert intent.parameters == {"app_name": "Safari"}
        assert intent.confirmation_required is False
        assert intent.spoken_response == "Opening Safari"
        assert intent.reasoning == "User wants to open Safari"

    def test_valid_intent_without_reasoning(self):
        raw = {
            "action": "lock_screen",
            "parameters": {},
            "confirmation_required": False,
            "spoken_response": "Locking the screen",
        }
        intent = parse_intent(raw)
        assert intent.action == "lock_screen"
        assert intent.reasoning == ""

    def test_missing_action(self):
        raw = {
            "parameters": {},
            "confirmation_required": False,
            "spoken_response": "Hello",
        }
        with pytest.raises(IntentParseError, match="Missing required field.*action"):
            parse_intent(raw)

    def test_missing_parameters(self):
        raw = {
            "action": "open_app",
            "confirmation_required": False,
            "spoken_response": "Opening app",
        }
        with pytest.raises(IntentParseError, match="Missing required field.*parameters"):
            parse_intent(raw)

    def test_wrong_type_action(self):
        raw = {
            "action": 123,
            "parameters": {},
            "confirmation_required": False,
            "spoken_response": "test",
        }
        with pytest.raises(IntentParseError, match="must be str"):
            parse_intent(raw)

    def test_wrong_type_confirmation(self):
        raw = {
            "action": "test",
            "parameters": {},
            "confirmation_required": "yes",
            "spoken_response": "test",
        }
        with pytest.raises(IntentParseError, match="must be bool"):
            parse_intent(raw)

    def test_empty_action(self):
        raw = {
            "action": "  ",
            "parameters": {},
            "confirmation_required": False,
            "spoken_response": "test",
        }
        with pytest.raises(IntentParseError, match="Action field is empty"):
            parse_intent(raw)

    def test_whitespace_stripped(self):
        raw = {
            "action": "  open_app  ",
            "parameters": {},
            "confirmation_required": False,
            "spoken_response": "  Opening   ",
        }
        intent = parse_intent(raw)
        assert intent.action == "open_app"
        assert intent.spoken_response == "Opening"
