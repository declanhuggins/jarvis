"""Configuration loading and validation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from jarvis.errors import ConfigError

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATHS = [
    Path("config.yaml"),
    Path.home() / ".jarvis" / "config.yaml",
]


@dataclass
class JarvisConfig:
    """All Jarvis configuration with sensible defaults."""

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:0.6b"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # TTS - tries in order: piper -> chatterbox -> openai -> edge-tts -> macOS say
    piper_enabled: bool = True
    piper_voice: str = "en_GB-alan-medium"
    chatterbox_enabled: bool = False
    openai_tts_voice: str = "onyx"
    edge_tts_voice: str = "en-GB-RyanNeural"
    macos_tts_voice: str = "Daniel"

    # Audio
    sample_rate: int = 16000
    chunk_ms: int = 80

    # Wake word
    wakeword_model: str = "hey_jarvis"
    wakeword_threshold: float = 0.5

    # STT
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    # Safety
    confirmation_timeout_sec: float = 10.0

    # Logging
    log_level: str = "INFO"
    log_file: str = "~/.jarvis/jarvis.log"


def load_config(path: str | Path | None = None) -> JarvisConfig:
    """Load config from YAML file, overlay onto defaults, validate."""
    config_path = _resolve_config_path(path)
    raw = {}

    if config_path and config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    elif path is not None:
        raise ConfigError(f"Config file not found: {path}")
    else:
        logger.warning("No config file found, using defaults")

    # Override with environment variables (JARVIS_ANTHROPIC_API_KEY, etc.)
    valid_fields = {f.name for f in fields(JarvisConfig)}
    for key in valid_fields:
        env_key = f"JARVIS_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            raw[key] = env_val

    # Build config, filtering unknown keys
    known = {k: v for k, v in raw.items() if k in valid_fields}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        logger.warning("Unknown config keys ignored: %s", unknown)

    config = JarvisConfig(**known)

    # Expand paths
    config.log_file = str(Path(config.log_file).expanduser())

    # Validate
    _validate(config)

    return config


def _resolve_config_path(path: str | Path | None) -> Path | None:
    """Find the config file to use."""
    if path is not None:
        return Path(path)
    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    return None


def _validate(config: JarvisConfig) -> None:
    """Validate that required fields are present for the chosen providers."""
    if config.llm_provider == "anthropic" and not config.anthropic_api_key:
        raise ConfigError(
            "anthropic_api_key is required when llm_provider is 'anthropic'. "
            "Set it in config.yaml or JARVIS_ANTHROPIC_API_KEY env var."
        )
    if config.llm_provider == "openai" and not config.openai_api_key:
        raise ConfigError(
            "openai_api_key is required when llm_provider is 'openai'. "
            "Set it in config.yaml or JARVIS_OPENAI_API_KEY env var."
        )
    if config.llm_provider not in ("ollama", "anthropic", "openai"):
        raise ConfigError(
            f"Unknown llm_provider: {config.llm_provider!r}. "
            "Must be 'ollama', 'anthropic', or 'openai'."
        )
