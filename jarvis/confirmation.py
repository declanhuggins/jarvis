"""Verbal confirmation flow for destructive operations."""

from __future__ import annotations

import logging
import re

from jarvis.audio import AudioStream
from jarvis.stt import SpeechToText
from jarvis.tts import TTSEngine

logger = logging.getLogger(__name__)

# Words/phrases that count as affirmative confirmation
_AFFIRMATIVES = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok",
    "do it", "go ahead", "proceed", "confirm", "affirmative",
    "absolutely", "please", "go for it", "approved",
})


class ConfirmationHandler:
    """Asks the user for verbal confirmation before destructive operations."""

    def __init__(
        self,
        tts: TTSEngine,
        stt: SpeechToText,
        audio_stream: AudioStream,
        timeout: float = 10.0,
    ):
        self._tts = tts
        self._stt = stt
        self._audio = audio_stream
        self._timeout = timeout

    def confirm(self, description: str, action: str | None = None) -> bool:
        """Speak a confirmation prompt and listen for yes/no.

        Args:
            description: What Jarvis is about to do (e.g. "organize your
                Downloads folder").
            action: Optional action name for action-specific confirmation wording.

        Returns:
            True if the user confirms, False otherwise.
        """
        prompt = _build_confirmation_prompt(description, action=action)
        logger.info("Asking for confirmation: %s", prompt)

        self._tts.speak(prompt)

        # Drain any audio from TTS playback before listening
        self._audio.drain()

        # Listen for the user's response
        transcript = self._stt.transcribe_stream(
            self._audio,
            max_duration_sec=self._timeout,
        )

        normalized = transcript.strip().lower()
        logger.info("Confirmation response: %r", normalized)

        if not normalized:
            logger.info("No response received, treating as declined")
            return False

        # Check if any affirmative word/phrase appears in the response
        is_confirmed = any(word in normalized for word in _AFFIRMATIVES)

        if is_confirmed:
            logger.info("User confirmed the operation")
        else:
            logger.info("User declined (response did not match affirmatives)")

        return is_confirmed


def _build_confirmation_prompt(description: str, action: str | None = None) -> str:
    """Normalize an action description into a natural confirmation prompt."""
    if action == "shutdown_jarvis":
        return "Should I shutdown Jarvis?"
    if action == "disable_jarvis":
        return "Disable Jarvis?"

    cleaned = " ".join(description.strip().split())
    if not cleaned:
        return "Do you want me to proceed?"

    # Drop any follow-up confirmation question the LLM may have already added.
    cleaned = re.sub(
        r"\s*(Want me to proceed|Should I proceed|Shall I proceed|Do you want me to proceed)\??\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if not cleaned:
        return "Do you want me to proceed?"

    # Keep the original wording if it is already a natural sentence.
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."

    return f"{cleaned} Do you want me to proceed?"
