"""Tests for the assistant plugin."""

from jarvis.config import JarvisConfig
from jarvis.plugins.assistant import AssistantPlugin


def test_assistant_actions_are_destructive():
    plugin = AssistantPlugin(JarvisConfig())

    actions = {action["name"]: action for action in plugin.get_actions()}

    assert actions["shutdown_jarvis"]["destructive"] is True
    assert actions["disable_jarvis"]["destructive"] is True
    assert plugin.shutdown_jarvis() == ""
    assert plugin.disable_jarvis() == ""
