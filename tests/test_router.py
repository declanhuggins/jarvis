"""Tests for the command router."""

import pytest

from jarvis.errors import PluginError, UnknownActionError
from jarvis.intent import Intent
from jarvis.router import CommandRouter


def _make_intent(action: str = "test", params: dict | None = None, **kwargs) -> Intent:
    return Intent(
        action=action,
        parameters=params or {},
        confirmation_required=kwargs.get("confirm", False),
        spoken_response=kwargs.get("response", ""),
        reasoning=kwargs.get("reasoning", ""),
    )


class TestCommandRouter:
    def test_register_and_execute(self):
        router = CommandRouter()
        router.register(
            "greet",
            handler=lambda name="World": f"Hello, {name}!",
            metadata={"description": "Greet someone", "parameters": {}, "destructive": False},
        )
        intent = _make_intent("greet", {"name": "Jarvis"})
        result = router.execute(intent)
        assert result == "Hello, Jarvis!"

    def test_unknown_action(self):
        router = CommandRouter()
        intent = _make_intent("nonexistent")
        with pytest.raises(UnknownActionError, match="nonexistent"):
            router.execute(intent)

    def test_bad_parameters(self):
        router = CommandRouter()
        router.register(
            "greet",
            handler=lambda name: f"Hello, {name}!",
            metadata={"description": "Greet", "parameters": {}, "destructive": False},
        )
        intent = _make_intent("greet", {"wrong_param": "test"})
        with pytest.raises(PluginError, match="Bad parameters"):
            router.execute(intent)

    def test_handler_exception_wrapped(self):
        def bad_handler():
            raise ValueError("something broke")

        router = CommandRouter()
        router.register(
            "bad",
            handler=bad_handler,
            metadata={"description": "Broken", "parameters": {}, "destructive": False},
        )
        intent = _make_intent("bad")
        with pytest.raises(PluginError, match="something broke"):
            router.execute(intent)

    def test_action_catalog(self):
        router = CommandRouter()
        router.register(
            "open_app",
            handler=lambda: None,
            metadata={
                "description": "Open an app",
                "parameters": {"app_name": {"type": "string"}},
                "destructive": False,
            },
        )
        router.register(
            "shell_command",
            handler=lambda: None,
            metadata={
                "description": "Run a command",
                "parameters": {"command": {"type": "string"}},
                "destructive": True,
            },
        )
        catalog = router.get_action_catalog()
        assert len(catalog) == 2
        names = {a["name"] for a in catalog}
        assert names == {"open_app", "shell_command"}
        shell = next(a for a in catalog if a["name"] == "shell_command")
        assert shell["destructive"] is True

    def test_none_result_returns_empty_string(self):
        router = CommandRouter()
        router.register(
            "noop",
            handler=lambda: None,
            metadata={"description": "No-op", "parameters": {}, "destructive": False},
        )
        intent = _make_intent("noop")
        assert router.execute(intent) == ""
