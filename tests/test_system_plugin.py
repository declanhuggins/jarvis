"""Tests for the system plugin."""

import subprocess

import jarvis.plugins.system as system_module
from jarvis.config import JarvisConfig
from jarvis.plugins.system import SystemPlugin


def test_set_brightness_uses_homebrew_cli(monkeypatch):
    plugin = SystemPlugin(JarvisConfig())
    monkeypatch.setattr(system_module, "_find_executable", lambda name: "/opt/homebrew/bin/brightness")

    calls = []

    def fake_run(cmd, check, capture_output):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = plugin.set_brightness(25)

    assert result == "Brightness set to 25%"
    assert calls == [["/opt/homebrew/bin/brightness", "0.25"]]


def test_set_brightness_reports_missing_cli(monkeypatch):
    plugin = SystemPlugin(JarvisConfig())
    monkeypatch.setattr(system_module, "_find_executable", lambda name: None)

    result = plugin.set_brightness(50)

    assert "brew install brightness" in result
