"""Tests for main-loop helpers."""

import queue
import numpy as np
from types import SimpleNamespace

from jarvis.config import JarvisConfig
from jarvis.main import (
    _build_response_text,
    _capture_post_wake_audio,
    _emit_listen_cue,
    _should_wait_for_followup,
    _disable_launch_agent,
)


def test_should_wait_for_followup_for_conversational_question():
    intent = SimpleNamespace(action="conversational_response")

    assert _should_wait_for_followup(intent, "Which location do you mean?") is True


def test_should_not_wait_for_followup_for_non_question_or_action():
    assert _should_wait_for_followup(SimpleNamespace(action="shell_command"), "Done?") is False
    assert _should_wait_for_followup(SimpleNamespace(action="conversational_response"), "It's sunny.") is False


class _FakeAudio:
    def __init__(self, chunks):
        self.sample_rate = 16000
        self.chunk_samples = 1280
        self._chunks = list(chunks)

    def read_chunk(self, timeout=1.0):
        if not self._chunks:
            raise queue.Empty
        return self._chunks.pop(0)


def test_capture_post_wake_audio_detects_continued_speech():
    silent = np.zeros(1280, dtype=np.int16)
    speech = np.full(1280, 3000, dtype=np.int16)
    frames, continued = _capture_post_wake_audio(
        _FakeAudio([silent, speech]),
        grace_ms=400,
    )

    assert continued is True
    assert len(frames) == 2


def test_capture_post_wake_audio_returns_false_when_only_silence():
    silent = np.zeros(1280, dtype=np.int16)
    frames, continued = _capture_post_wake_audio(
        _FakeAudio([silent, silent, silent]),
        grace_ms=200,
    )

    assert continued is False
    assert len(frames) >= 1


def test_emit_listen_cue_uses_tone_for_followup(monkeypatch):
    calls = []

    class _FakeTTS:
        def speak(self, text):
            calls.append(("speak", text))

    def fake_popen(cmd, stdout=None, stderr=None):
        calls.append(("popen", cmd))
        class _Proc:
            pass
        return _Proc()

    monkeypatch.setattr("jarvis.main.subprocess.Popen", fake_popen)

    config = JarvisConfig(
        wake_acknowledgement_mode="speech",
        wake_acknowledgement="Yes?",
        wake_acknowledgement_sound="Pop",
    )
    _emit_listen_cue(config, _FakeTTS(), tone_only=True)

    assert calls == [("popen", ["afplay", "/System/Library/Sounds/Pop.aiff"])]


def test_build_response_text_prefers_execution_result_for_actions():
    assert (
        _build_response_text(
            "set_brightness",
            "Setting your screen brightness to 75 now.",
            "Brightness set to 75%",
        )
        == "Brightness set to 75%"
    )


def test_build_response_text_keeps_conversational_response():
    assert (
        _build_response_text(
            "conversational_response",
            "Sure, here's the weather.",
            "",
        )
        == "Sure, here's the weather."
    )


def test_build_response_text_uses_fixed_shutdown_response():
    assert _build_response_text("shutdown_jarvis", "Anything", "") == "Shutting down Jarvis."


def test_build_response_text_uses_fixed_disable_response():
    assert _build_response_text("disable_jarvis", "Anything", "") == "Disabling Jarvis."


def test_disable_launch_agent_uses_plist_bootout(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        calls.append(cmd)
        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr("jarvis.main.subprocess.run", fake_run)
    monkeypatch.setattr("jarvis.main.os.getuid", lambda: 501)
    monkeypatch.setattr("jarvis.main._LAUNCH_AGENT_PLIST", tmp_path / "com.user.jarvis.plist")
    (tmp_path / "com.user.jarvis.plist").write_text("plist")

    _disable_launch_agent()

    assert calls == [
        ["launchctl", "disable", "gui/501/com.user.jarvis"],
        ["launchctl", "bootout", "gui/501", str(tmp_path / "com.user.jarvis.plist")],
    ]
