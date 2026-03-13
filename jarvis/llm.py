"""LLM client for intent resolution using Ollama, Anthropic, or OpenAI."""

from __future__ import annotations

import json
import logging

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
7. Keep "spoken_response" under 2 sentences for action confirmations.
   For conversational responses, keep it under 4 sentences.
8. Always respond with valid JSON. No trailing commas, no comments.\
"""


class LLMClient:
    """Sends transcribed commands to an LLM and gets structured intents back."""

    def __init__(self, config: JarvisConfig):
        self._provider = config.llm_provider
        self._client = None

        if self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            self._model = config.anthropic_model
        elif self._provider == "ollama":
            # Use native Ollama API (not OpenAI-compat) so we can set think=false
            self._ollama_base = config.ollama_base_url.replace("/v1", "")
            self._model = config.ollama_model
        else:
            import openai

            self._client = openai.OpenAI(api_key=config.openai_api_key)
            self._model = config.openai_model

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

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"LLM returned invalid JSON: {e}\nResponse: {response_text[:500]}"
            ) from e

        return result

    def _build_system_prompt(self, available_actions: list[dict]) -> str:
        """Build the system prompt with the current action catalog."""
        actions_json = json.dumps(available_actions, indent=2)
        return SYSTEM_PROMPT_TEMPLATE.format(actions_json=actions_json)

    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call the Anthropic Messages API."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
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
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
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
        """Call an OpenAI-compatible Chat Completions API (OpenAI or Ollama)."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"{self._provider} API error: {e}") from e
