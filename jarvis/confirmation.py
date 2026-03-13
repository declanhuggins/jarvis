"""Verbal confirmation flow for destructive operations."""

from __future__ import annotations

import logging

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

    def confirm(self, description: str) -> bool:
        """Speak a confirmation prompt and listen for yes/no.

        Args:
            description: What Jarvis is about to do (e.g. "organize your
                Downloads folder").

        Returns:
            True if the user confirms, False otherwise.
        """
        prompt = f"I'm about to {description}. Shall I proceed?"
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
