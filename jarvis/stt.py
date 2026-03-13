"""Speech-to-text using faster-whisper."""

from __future__ import annotations

import logging
import queue

import numpy as np

from jarvis.audio import AudioStream
from jarvis.errors import STTError

logger = logging.getLogger(__name__)


class SpeechToText:
    """Records speech after wake word and transcribes using faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info(
                "Whisper model loaded: %s (device=%s, compute=%s)",
                self._model_size,
                self._device,
                self._compute_type,
            )
        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {e}") from e

    def transcribe_stream(
        self,
        audio_stream: AudioStream,
        silence_threshold: float = 500.0,
        silence_chunks: int = 15,
        max_duration_sec: float = 30.0,
    ) -> str:
        """Record from the audio stream until silence, then transcribe.

        Args:
            audio_stream: Active AudioStream to read chunks from.
            silence_threshold: Mean absolute amplitude below which a chunk
                is considered silence (int16 scale, 0-32768).
            silence_chunks: Number of consecutive silent chunks before
                stopping (15 chunks * 80ms = ~1.2 seconds).
            max_duration_sec: Maximum recording duration in seconds.

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            raise STTError("Model not loaded. Call load() first.")

        chunk_duration = audio_stream.chunk_samples / audio_stream.sample_rate
        max_chunks = int(max_duration_sec / chunk_duration)

        frames: list[np.ndarray] = []
        consecutive_silent = 0
        has_speech = False

        logger.debug("Recording command (max %.1fs)...", max_duration_sec)

        for _ in range(max_chunks):
            try:
                chunk = audio_stream.read_chunk(timeout=2.0)
            except queue.Empty:
                logger.warning("Audio read timeout during recording")
                break

            frames.append(chunk)

            energy = np.abs(chunk.astype(np.float32)).mean()
            if energy < silence_threshold:
                consecutive_silent += 1
            else:
                consecutive_silent = 0
                has_speech = True

            # Only stop on silence if we've heard some speech first
            if has_speech and consecutive_silent >= silence_chunks:
                break

        if not frames:
            return ""

        # Concatenate and convert int16 -> float32 normalized [-1.0, 1.0]
        audio_int16 = np.concatenate(frames)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        duration = len(audio_float32) / audio_stream.sample_rate
        logger.info("Recorded %.1f seconds of audio, transcribing...", duration)

        try:
            segments, info = self._model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text for segment in segments).strip()
        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e

        logger.info("Transcript: %s", text)
        return text
