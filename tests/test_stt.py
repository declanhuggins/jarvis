"""Tests for speech-to-text backend helpers."""

import queue
import threading
from pathlib import Path

import numpy as np

from jarvis.stt import SpeechToText, _resolve_mlx_model_repo


def test_resolve_mlx_model_repo_for_known_sizes():
    assert _resolve_mlx_model_repo("tiny") == "mlx-community/whisper-tiny"
    assert _resolve_mlx_model_repo("medium") == "mlx-community/whisper-medium"
    assert _resolve_mlx_model_repo("turbo") == "mlx-community/whisper-turbo"
    assert _resolve_mlx_model_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert _resolve_mlx_model_repo("large-v3") == "mlx-community/whisper-large-v3"


def test_resolve_mlx_model_repo_passthrough_for_repo_or_path(tmp_path: Path):
    assert _resolve_mlx_model_repo("org/custom-whisper") == "org/custom-whisper"

    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    assert _resolve_mlx_model_repo(str(model_dir)) == str(model_dir)


class _FakeAudio:
    sample_rate = 16000
    chunk_samples = 1280

    def read_chunk(self, timeout=1.0):
        raise queue.Empty


def test_transcribe_stream_returns_empty_when_canceled():
    stt = SpeechToText()
    stt._model = object()
    cancel_event = threading.Event()
    cancel_event.set()

    result = stt.transcribe_stream(_FakeAudio(), cancel_event=cancel_event)

    assert result == ""
