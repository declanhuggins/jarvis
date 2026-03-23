"""Tests for audio stream state handling."""

import numpy as np

from jarvis.audio import AudioStream


def test_audio_stream_toggle_muted():
    audio = AudioStream()

    assert audio.muted is False
    assert audio.toggle_muted() is True
    assert audio.muted is True
    assert audio.toggle_muted() is False
    assert audio.muted is False


def test_audio_callback_drops_frames_when_muted():
    audio = AudioStream()
    frame = np.ones((audio.chunk_samples, 1), dtype=np.int16)

    audio.set_muted(True)
    audio._callback(frame, audio.chunk_samples, {}, None)

    assert audio._queue.empty()
