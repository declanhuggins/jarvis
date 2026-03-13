"""Microphone audio stream using sounddevice."""

from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

from jarvis.errors import AudioError

logger = logging.getLogger(__name__)


class AudioStream:
    """Continuous microphone input stream yielding chunks of int16 PCM at 16kHz mono."""

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 1280):
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback - runs on a real-time audio thread.

        Must be fast: no I/O, no allocations beyond the copy.
        """
        if status:
            logger.debug("Audio callback status: %s", status)
        self._queue.put_nowait(indata.copy())

    def start(self) -> None:
        """Open the microphone stream."""
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_samples,
                callback=self._callback,
            )
            self._stream.start()
            logger.info(
                "Audio stream started: %dHz, %d samples/chunk",
                self._sample_rate,
                self._chunk_samples,
            )
        except sd.PortAudioError as e:
            raise AudioError(f"Failed to open microphone: {e}") from e

    def stop(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio stream stopped")

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray:
        """Read one audio chunk from the queue. Blocks up to timeout seconds.

        Returns:
            numpy array of shape (chunk_samples,) with dtype int16.

        Raises:
            queue.Empty: if no audio available within timeout.
        """
        chunk = self._queue.get(timeout=timeout)
        return chunk.flatten()

    def drain(self) -> None:
        """Discard all queued audio chunks.

        Call this after TTS playback to avoid processing Jarvis's own voice.
        """
        discarded = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        if discarded:
            logger.debug("Drained %d audio chunks", discarded)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples
