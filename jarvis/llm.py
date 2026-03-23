"""LLM client for intent resolution using Ollama, Anthropic, OpenAI, or OpenClaw."""

from __future__ import annotations

import json
import logging
import uuid

import httpx

from jarvis.config import JarvisConfig
from jarvis.errors import LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are Jarvis, a personal AI assistant running on the user's macOS computer.
You receive voice commands transcribed from speech. Your job is to understand
the user's intent and map it to one of the available actions.

AVAILABLE ACTIONS:
{actions_json}

RESPONSE FORMAT:
You MUST respond with a single JSON object and nothing else. No markdown, no
code fences, no explanation outside the JSON. The JSON must have exactly
these fields:

{{
  "action": "action_name",
  "parameters": {{ ... }},
  "confirmation_required": true or false,
  "spoken_response": "What you want to say to the user",
  "reasoning": "Brief explanation of why you chose this action"
}}

RULES:
1. Match the user's intent to the most specific available action.
2. If no action fits, use "conversational_response" with an empty parameters
   object and put your answer in "spoken_response".
3. Set "confirmation_required" to true for any action marked as destructive
   in the action catalog above.
4. The "spoken_response" should be concise and natural - this will be read
   aloud. Do not use markdown, bullet points, or long lists.
5. If the user's request is ambiguous, ask a clarifying question in
   "spoken_response" and set action to "conversational_response".
6. For shell_command, write the exact command in the parameters. Be cautious
   - prefer specific actions over shell_command when possible.
   Never use shell_command for weather, screenshots, Spotify, or Jarvis
   control if dedicated actions are available.
7. Keep "spoken_response" under 2 sentences for action confirmations.
   For conversational responses, keep it under 4 sentences.
8. Always respond with valid JSON. No trailing commas, no comments.\
"""


class LLMClient:
    """Sends transcribed commands to an LLM and gets structured intents back."""

    def __init__(self, config: JarvisConfig):
        self._provider = config.llm_provider
        self._client = None
        self._history: list[dict[str, str]] = []
        self._history_turns = max(0, config.llm_history_turns)
        self._json_retry_count = max(0, config.llm_json_retry_count)
        self._session_key = f"jarvis-{uuid.uuid4().hex}"

        if self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            self._model = config.anthropic_model
        elif self._provider == "ollama":
            # Use native Ollama API (not OpenAI-compat) so we can set think=false
            self._ollama_base = config.ollama_base_url.replace("/v1", "")
            self._model = config.ollama_model
        elif self._provider == "openclaw":
            import openai

            self._client = openai.OpenAI(
                api_key=config.openclaw_api_key,
                base_url=_normalize_openai_base_url(config.openclaw_base_url),
            )
            self._model = config.openclaw_model
            self._extra_headers = {}
            if config.openclaw_agent_id:
                self._extra_headers["x-openclaw-agent-id"] = config.openclaw_agent_id
        else:
            import openai

            self._client = openai.OpenAI(api_key=config.openai_api_key)
            self._model = config.openai_model
            self._extra_headers = {}

        logger.info("LLM client initialized: %s (%s)", self._provider, self._model)

    def get_intent(self, transcript: str, available_actions: list[dict]) -> dict:
        """Send transcript to the LLM and return the parsed JSON intent.

        Args:
            transcript: The user's spoken command as text.
            available_actions: List of action descriptors from the router.

        Returns:
            Parsed JSON dict with action, parameters, confirmation_required,
            spoken_response, and reasoning fields.

        Raises:
            LLMError: On API failure or unparseable response.
        """
        system_prompt = self._build_system_prompt(available_actions)

        try:
            if self._provider == "anthropic":
                response_text = self._call_anthropic(system_prompt, transcript)
            elif self._provider == "ollama":
                response_text = self._call_ollama(system_prompt, transcript)
            else:
                response_text = self._call_openai(system_prompt, transcript)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM API call failed: {e}") from e

        logger.debug("LLM raw response: %s", response_text)

        # Strip markdown code fences if the model wraps JSON in them
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        result = self._parse_json_response(cleaned, system_prompt, transcript, response_text)

        return result

    def _build_system_prompt(self, available_actions: list[dict]) -> str:
        """Build the system prompt with the current action catalog."""
        actions_json = json.dumps(available_actions, indent=2)
        return SYSTEM_PROMPT_TEMPLATE.format(actions_json=actions_json)

    def record_turn(
        self,
        transcript: str,
        intent: dict,
        final_response: str,
        execution_result: str = "",
    ) -> None:
        """Record a completed turn so follow-up requests can reuse context."""
        if self._history_turns <= 0:
            return

        assistant_payload = {
            "action": intent.get("action", ""),
            "parameters": intent.get("parameters", {}),
            "confirmation_required": intent.get("confirmation_required", False),
            "spoken_response": final_response,
            "reasoning": intent.get("reasoning", ""),
        }
        if execution_result:
            assistant_payload["execution_result"] = execution_result

        self._history.extend(
            [
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        )

        max_messages = self._history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _build_messages(self, system_prompt: str, user_message: str) -> list[dict[str, str]]:
        """Build chat messages including recent conversation history."""
        if self._provider == "openclaw":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        return [
            {"role": "system", "content": system_prompt},
            *self._history,
            {"role": "user", "content": user_message},
        ]

    def _parse_json_response(
        self,
        cleaned: str,
        system_prompt: str,
        transcript: str,
        response_text: str,
    ) -> dict:
        """Parse the model response, optionally retrying once to repair invalid JSON."""
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_error:
            repaired_text = response_text
            for _ in range(self._json_retry_count):
                repaired_text = self._retry_for_json(system_prompt, transcript, repaired_text)
                repaired_cleaned = repaired_text.strip()
                if repaired_cleaned.startswith("```"):
                    lines = repaired_cleaned.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    repaired_cleaned = "\n".join(lines).strip()
                try:
                    return json.loads(repaired_cleaned)
                except json.JSONDecodeError:
                    continue

            raise LLMError(
                f"LLM returned invalid JSON: {first_error}\nResponse: {response_text[:500]}"
            ) from first_error

    def _retry_for_json(self, system_prompt: str, transcript: str, invalid_response: str) -> str:
        """Ask the model to reformat its previous answer as strict JSON only."""
        repair_prompt = (
            "Your previous reply was invalid because it was not a single valid JSON object. "
            "Return only one JSON object that follows the required schema. "
            f"Original user request: {transcript}\n"
            f"Previous invalid reply: {invalid_response}"
        )

        if self._provider == "anthropic":
            return self._call_anthropic(system_prompt, repair_prompt)
        if self._provider == "ollama":
            return self._call_ollama(system_prompt, repair_prompt)
        return self._call_openai(system_prompt, repair_prompt)

    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call the Anthropic Messages API."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system_prompt,
                messages=(
                    [{"role": "user", "content": user_message}]
                    if self._provider == "openclaw"
                    else [
                        {"role": message["role"], "content": message["content"]}
                        for message in self._history
                    ]
                    + [{"role": "user", "content": user_message}]
                ),
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic API error: {e}") from e

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """Call the native Ollama API (supports think=false for Qwen3 models)."""
        try:
            payload = {
                "model": self._model,
                "stream": False,
                "think": False,
                "format": "json",
                "messages": self._build_messages(system_prompt, user_message),
            }
            resp = httpx.post(
                f"{self._ollama_base}/api/chat",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            raise LLMError(f"Ollama API error: {e}") from e

    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call an OpenAI-compatible Chat Completions API."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                messages=self._build_messages(system_prompt, user_message),
                extra_headers=self._extra_headers,
                user=self._session_key,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"{self._provider} API error: {e}") from e


def _normalize_openai_base_url(base_url: str) -> str:
    """Ensure OpenAI-compatible base URLs include the /v1 prefix expected by the SDK."""
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized
