"""Tests for confirmation prompt wording."""

from jarvis.confirmation import _build_confirmation_prompt


def test_build_confirmation_prompt_strips_duplicate_proceed_question():
    prompt = _build_confirmation_prompt(
        "I can run the 'whoami' command now. Want me to proceed?"
    )

    assert prompt == "I can run the 'whoami' command now. Do you want me to proceed?"


def test_build_confirmation_prompt_handles_bare_action_fragment():
    prompt = _build_confirmation_prompt("run the whoami command now")

    assert prompt == "run the whoami command now. Do you want me to proceed?"


def test_build_confirmation_prompt_falls_back_when_empty():
    assert _build_confirmation_prompt("") == "Do you want me to proceed?"


def test_build_confirmation_prompt_uses_shutdown_specific_wording():
    prompt = _build_confirmation_prompt(
        "I want to shut Jarvis down.",
        action="shutdown_jarvis",
    )

    assert prompt == "Should I shutdown Jarvis?"


def test_build_confirmation_prompt_uses_disable_specific_wording():
    prompt = _build_confirmation_prompt(
        "I want to disable Jarvis.",
        action="disable_jarvis",
    )

    assert prompt == "Disable Jarvis?"
