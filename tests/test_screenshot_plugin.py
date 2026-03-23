"""Tests for the screenshot plugin."""

import subprocess

from jarvis.config import JarvisConfig
from jarvis.plugins.screenshot import ScreenshotPlugin


def test_take_screenshot_uses_screencapture(monkeypatch, tmp_path):
    plugin = ScreenshotPlugin(JarvisConfig())
    monkeypatch.setattr("jarvis.plugins.screenshot.Path.home", lambda: tmp_path)

    calls = []

    def fake_run(cmd, check, capture_output):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = plugin.take_screenshot()

    assert calls
    assert calls[0][0:2] == ["screencapture", "-x"]
    assert "Screenshot saved to" in result
