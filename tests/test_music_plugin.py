"""Tests for the music plugin."""

import subprocess

from jarvis.config import JarvisConfig
from jarvis.plugins.music import MusicPlugin, _spotify_target_for_query


def test_spotify_target_for_plain_query():
    assert _spotify_target_for_query("my daylist") == "spotify:search:my%20daylist"


def test_spotify_target_for_uri():
    assert _spotify_target_for_query("spotify:playlist:abc") == "spotify:playlist:abc"


def test_play_spotify_searches_and_plays(monkeypatch):
    plugin = MusicPlugin(JarvisConfig())
    calls = []

    def fake_run(cmd, check, capture_output, text=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = plugin.play_spotify("my daylist")

    assert result == "Playing my daylist on Spotify"
    assert calls[0] == ["open", "spotify:search:my%20daylist"]
    assert calls[1][0] == "osascript"


def test_pause_spotify(monkeypatch):
    plugin = MusicPlugin(JarvisConfig())
    calls = []

    def fake_run(cmd, check, capture_output, text=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = plugin.pause_spotify()

    assert result == "Paused Spotify"
    assert calls == [["osascript", "-e", 'tell application "Spotify" to pause']]
