"""Speech-to-text using faster-whisper or MLX Whisper."""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import threading
from pathlib import Path

import numpy as np

from jarvis.audio import AudioStream
from jarvis.errors import STTError

logger = logging.getLogger(__name__)


class SpeechToText:
    """Records speech after wake word and transcribes using the configured STT backend."""

    def __init__(
        self,
        backend: str = "faster-whisper",
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self._backend = backend
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._mlx_model_repo = None

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        try:
            if self._backend == "mlx-whisper":
                import mlx.core as mx
                from mlx_whisper.transcribe import ModelHolder

                self._mlx_model_repo = _resolve_mlx_model_repo(self._model_size)
                with _suppress_output():
                    ModelHolder.get_model(self._mlx_model_repo, mx.float16)
                    _warm_mlx_whisper(self._mlx_model_repo)
                self._model = self._mlx_model_repo
                logger.info(
                    "Whisper model loaded: %s (backend=%s, repo=%s)",
                    self._model_size,
                    self._backend,
                    self._mlx_model_repo,
                )
            else:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                logger.info(
                    "Whisper model loaded: %s (backend=%s, device=%s, compute=%s)",
                    self._model_size,
                    self._backend,
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
        initial_frames: list[np.ndarray] | None = None,
        cancel_event: threading.Event | None = None,
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

        frames: list[np.ndarray] = list(initial_frames or [])
        consecutive_silent = 0
        has_speech = False

        logger.debug("Recording command (max %.1fs)...", max_duration_sec)

        for chunk in frames:
            energy = np.abs(chunk.astype(np.float32)).mean()
            if energy < silence_threshold:
                consecutive_silent += 1
            else:
                consecutive_silent = 0
                has_speech = True

        remaining_chunks = max(0, max_chunks - len(frames))
        for _ in range(remaining_chunks):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Recording canceled")
                return ""
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

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Recording canceled")
            return ""

        if not frames:
            return ""

        # Concatenate and convert int16 -> float32 normalized [-1.0, 1.0]
        audio_int16 = np.concatenate(frames)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        duration = len(audio_float32) / audio_stream.sample_rate
        logger.info("Recorded %.1f seconds of audio, transcribing...", duration)

        try:
            if self._backend == "mlx-whisper":
                import mlx_whisper

                with _suppress_output():
                    result = mlx_whisper.transcribe(
                        audio_float32,
                        path_or_hf_repo=self._mlx_model_repo,
                        verbose=None,
                        language="en",
                        condition_on_previous_text=False,
                    )
                text = result["text"].strip()
            else:
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


def _resolve_mlx_model_repo(model_size: str) -> str:
    """Map Jarvis Whisper model names to MLX Whisper repos."""
    if "/" in model_size or Path(model_size).exists():
        return model_size

    mapping = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base",
        "small": "mlx-community/whisper-small",
        "medium": "mlx-community/whisper-medium",
        "turbo": "mlx-community/whisper-turbo",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "large-v3": "mlx-community/whisper-large-v3",
    }
    return mapping.get(model_size, model_size)


def _warm_mlx_whisper(model_repo: str) -> None:
    """Warm MLX Whisper once so the first real utterance does not pay compile cost."""
    import mlx_whisper

    t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
    warm_audio = 0.02 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    mlx_whisper.transcribe(
        warm_audio,
        path_or_hf_repo=model_repo,
        verbose=None,
        language="en",
        condition_on_previous_text=False,
    )


@contextlib.contextmanager
def _suppress_output():
    """Silence noisy third-party stdout/stderr during MLX model load/transcribe."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
