"""Tests for LLM provider integration."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from jarvis.config import JarvisConfig
from jarvis.llm import LLMClient, _normalize_openai_base_url


class _FakeCompletions:
    def __init__(self, outputs=None):
        self.last_kwargs = None
        self.outputs = outputs or [
            (
                '{"action":"conversational_response",'
                '"parameters":{},'
                '"confirmation_required":false,'
                '"spoken_response":"Hello",'
                '"reasoning":"test"}'
            )
        ]

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        content = self.outputs.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ]
        )


class _FakeResponses:
    def __init__(self, outputs=None):
        self.last_kwargs = None
        self.outputs = outputs or [
            (
                '{"action":"conversational_response",'
                '"parameters":{},'
                '"confirmation_required":false,'
                '"spoken_response":"Hello",'
                '"reasoning":"test"}'
            )
        ]

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        text = self.outputs.pop(0)
        return SimpleNamespace(output_text=text, output=[])


class _FakeOpenAIClient:
    def __init__(self, *, api_key: str, base_url: str | None = None, outputs=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=_FakeCompletions(outputs=outputs))
        self.responses = _FakeResponses(outputs=outputs)


def test_normalize_openai_base_url_adds_v1():
    assert _normalize_openai_base_url("http://127.0.0.1:18789") == "http://127.0.0.1:18789/v1"
    assert _normalize_openai_base_url("http://127.0.0.1:18789/v1/") == "http://127.0.0.1:18789/v1"


def test_openclaw_uses_responses_api(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    config = JarvisConfig(
        llm_provider="openclaw",
        openclaw_api_key="gateway-token",
        openclaw_base_url="http://127.0.0.1:18789",
        openclaw_model="openclaw",
        openclaw_agent_id="main",
    )

    client = LLMClient(config)
    result = client.get_intent("hello", [])

    assert client._client.api_key == "gateway-token"
    assert client._client.base_url == "http://127.0.0.1:18789/v1"
    assert client._client.responses.last_kwargs["extra_headers"] == {
        "x-openclaw-agent-id": "main"
    }
    assert client._client.responses.last_kwargs["instructions"] == client._build_system_prompt([])
    assert client._client.responses.last_kwargs["input"] == "hello"
    assert client._client.responses.last_kwargs["user"].startswith("jarvis-")
    assert result["action"] == "conversational_response"


def test_llm_client_reuses_history_and_session(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    config = JarvisConfig(
        llm_provider="openclaw",
        openclaw_api_key="gateway-token",
        openclaw_base_url="http://127.0.0.1:18789",
        openclaw_model="openclaw",
        openclaw_agent_id="main",
        llm_history_turns=2,
    )

    client = LLMClient(config)
    client.record_turn(
        "Who are you?",
        {
            "action": "conversational_response",
            "parameters": {},
            "confirmation_required": False,
            "reasoning": "identity question",
        },
        "I'm Jarvis.",
    )

    client.get_intent("And what can you do?", [])
    kwargs = client._client.responses.last_kwargs

    assert kwargs["user"] == client._session_key
    assert kwargs["input"] == "And what can you do?"
    assert "messages" not in kwargs


def test_openai_client_keeps_local_history(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    config = JarvisConfig(
        llm_provider="openai",
        openai_api_key="openai-token",
        openai_model="gpt-4o",
        llm_history_turns=2,
    )

    client = LLMClient(config)
    client.record_turn(
        "Who are you?",
        {
            "action": "conversational_response",
            "parameters": {},
            "confirmation_required": False,
            "reasoning": "identity question",
        },
        "I'm Jarvis.",
    )

    client.get_intent("And what can you do?", [])
    messages = client._client.chat.completions.last_kwargs["messages"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Who are you?"
    assert messages[2]["role"] == "assistant"
    assert "I'm Jarvis." in messages[2]["content"]


def test_llm_client_retries_invalid_json(monkeypatch):
    fake_module = ModuleType("openai")

    class RepairingClient(_FakeOpenAIClient):
        def __init__(self, *, api_key: str, base_url: str):
            super().__init__(
                api_key=api_key,
                base_url=base_url,
                outputs=[
                    "notre dame,in: cloudy and cold",
                    (
                        '{"action":"conversational_response",'
                        '"parameters":{},'
                        '"confirmation_required":false,'
                        '"spoken_response":"Which location do you mean?",'
                        '"reasoning":"need location"}'
                    ),
                ],
            )

    fake_module.OpenAI = RepairingClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    config = JarvisConfig(
        llm_provider="openclaw",
        openclaw_api_key="gateway-token",
        openclaw_base_url="http://127.0.0.1:18789",
        openclaw_model="openclaw",
    )

    client = LLMClient(config)
    result = client.get_intent("How was the weather today?", [])

    assert result["spoken_response"] == "Which location do you mean?"
