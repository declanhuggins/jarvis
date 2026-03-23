"""Tests for configuration validation."""

import shutil
import subprocess
from pathlib import Path

import pytest

import jarvis.config as config_module
from jarvis.config import JarvisConfig, _resolve_op_references, load_config, _validate
from jarvis.errors import ConfigError


class TestConfigValidation:
    def test_openclaw_requires_api_key(self):
        config = JarvisConfig(llm_provider="openclaw", openclaw_api_key="")

        with pytest.raises(ConfigError, match="openclaw_api_key is required"):
            _validate(config)

    def test_openclaw_is_accepted_with_api_key(self):
        config = JarvisConfig(llm_provider="openclaw", openclaw_api_key="test-token")

        _validate(config)

    def test_unknown_whisper_backend_rejected(self):
        config = JarvisConfig(whisper_backend="unknown")

        with pytest.raises(ConfigError, match="Unknown whisper_backend"):
            _validate(config)

    def test_unknown_tts_provider_rejected(self):
        config = JarvisConfig(tts_provider="unknown")

        with pytest.raises(ConfigError, match="Unknown tts_provider"):
            _validate(config)

    def test_unknown_wake_ack_mode_rejected(self):
        config = JarvisConfig(wake_acknowledgement_mode="weird")

        with pytest.raises(ConfigError, match="wake_acknowledgement_mode"):
            _validate(config)


class TestOnePasswordResolution:
    def test_resolve_op_reference_from_yaml(self, monkeypatch, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            'llm_provider: "openclaw"\n'
            'openclaw_api_key: "op://Private/OpenClaw/credential"\n'
        )

        monkeypatch.setattr(shutil, "which", lambda _: "/opt/homebrew/bin/op")

        def fake_run(cmd, check, capture_output, text):
            assert cmd == ["/opt/homebrew/bin/op", "read", "op://Private/OpenClaw/credential"]
            return subprocess.CompletedProcess(cmd, 0, stdout="secret-token\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = load_config(config_path)

        assert config.openclaw_api_key == "secret-token"

    def test_resolve_op_reference_from_env_override(self, monkeypatch, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm_provider: \"openclaw\"\nopenclaw_api_key: \"\"\n")
        monkeypatch.setenv("JARVIS_OPENCLAW_API_KEY", "op://Private/OpenClaw/credential")
        monkeypatch.setattr(shutil, "which", lambda _: "/opt/homebrew/bin/op")

        def fake_run(cmd, check, capture_output, text):
            return subprocess.CompletedProcess(cmd, 0, stdout="env-secret\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = load_config(config_path)

        assert config.openclaw_api_key == "env-secret"

    def test_missing_op_cli_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("JARVIS_OP_BIN", raising=False)
        monkeypatch.setattr(shutil, "which", lambda _: None)
        monkeypatch.setattr(config_module.Path, "exists", lambda self: False)

        with pytest.raises(ConfigError, match="1Password CLI not found"):
            _resolve_op_references({"openclaw_api_key": "op://Private/OpenClaw/credential"})


class TestLegacyTTSConfig:
    def test_legacy_chatterbox_flag_maps_to_tts_provider(self, monkeypatch, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text('chatterbox_enabled: true\n')

        config = load_config(config_path)

        assert config.tts_provider == "chatterbox"

    def test_legacy_piper_disabled_maps_to_edge_tts_without_openai_key(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text('piper_enabled: false\n')

        config = load_config(config_path)

        assert config.tts_provider == "edge-tts"
