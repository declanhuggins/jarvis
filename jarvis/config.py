"""Configuration loading and validation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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
    openclaw_base_url: str = "http://127.0.0.1:18789/v1"
    openclaw_api_key: str = ""
    openclaw_model: str = "openclaw"
    openclaw_agent_id: str = ""
    llm_history_turns: int = 6
    llm_json_retry_count: int = 1

    # TTS
    tts_provider: str = "piper"
    piper_voice: str = "en_GB-alan-medium"
    piper_length_scale: float = 0.9
    chatterbox_voice_reference: str = ""
    chatterbox_device: str = "cpu"
    openai_tts_voice: str = "onyx"
    edge_tts_voice: str = "en-GB-RyanNeural"
    macos_tts_voice: str = "Daniel"

    # Audio
    sample_rate: int = 16000
    chunk_ms: int = 80

    # Wake word
    wakeword_model: str = "hey_jarvis"
    wakeword_threshold: float = 0.5
    wake_acknowledgement_mode: str = "tone"
    wake_acknowledgement: str = "Yes?"
    wake_acknowledgement_sound: str = "Pop"
    wake_acknowledgement_delay_ms: int = 120
    wake_barge_in_grace_ms: int = 400

    # STT
    whisper_backend: str = "faster-whisper"
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    # Safety
    confirmation_timeout_sec: float = 10.0
    followup_timeout_sec: float = 8.0
    followup_max_turns: int = 2

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

    # Legacy TTS env vars for backward compatibility.
    if "JARVIS_TTS_PROVIDER" not in os.environ:
        for legacy_key in ("piper_enabled", "chatterbox_enabled"):
            env_val = os.environ.get(f"JARVIS_{legacy_key.upper()}")
            if env_val is not None:
                raw[legacy_key] = env_val

    raw = _normalize_legacy_config(raw)

    # Build config, filtering unknown keys
    known = {k: v for k, v in raw.items() if k in valid_fields}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        logger.warning("Unknown config keys ignored: %s", unknown)

    known = _resolve_op_references(known)

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


def _resolve_op_references(values: dict) -> dict:
    """Resolve 1Password CLI references in string config values."""
    resolved = {}
    cache: dict[str, str] = {}

    for key, value in values.items():
        if isinstance(value, str) and value.startswith("op://"):
            resolved[key] = _read_op_reference(value, cache)
        else:
            resolved[key] = value

    return resolved


def _normalize_legacy_config(values: dict) -> dict:
    """Map older config keys onto the current config surface."""
    normalized = dict(values)

    if "tts_provider" not in normalized:
        chatterbox_enabled = _coerce_bool(normalized.pop("chatterbox_enabled", None))
        piper_enabled = _coerce_bool(normalized.pop("piper_enabled", None))

        if chatterbox_enabled is True:
            normalized["tts_provider"] = "chatterbox"
        elif piper_enabled is False:
            normalized["tts_provider"] = (
                "openai" if normalized.get("openai_api_key") else "edge-tts"
            )
        else:
            normalized["tts_provider"] = "piper"
    else:
        normalized.pop("chatterbox_enabled", None)
        normalized.pop("piper_enabled", None)

    return normalized


def _coerce_bool(value):
    """Parse legacy bool-like config values from YAML or env vars."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return value


def _read_op_reference(reference: str, cache: dict[str, str]) -> str:
    """Read a single 1Password secret reference with the `op` CLI."""
    cached = cache.get(reference)
    if cached is not None:
        return cached

    op_command = _find_op_command()
    if op_command is None:
        raise ConfigError(
            f"1Password CLI not found while resolving {reference!r}. "
            "Install `op` and sign in first."
        )

    try:
        result = subprocess.run(
            [op_command, "read", reference],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise ConfigError(
            f"1Password CLI not found while resolving {reference!r}. "
            "Install `op` and sign in first."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        details = f": {stderr}" if stderr else ""
        raise ConfigError(
            f"Failed to resolve 1Password reference {reference!r}{details}"
        ) from e

    value = result.stdout.rstrip("\r\n")
    cache[reference] = value
    return value


def _find_op_command() -> str | None:
    """Locate the 1Password CLI in shell and launchd environments."""
    configured = os.environ.get("JARVIS_OP_BIN")
    candidates = [
        configured,
        shutil.which("op"),
        "/opt/homebrew/bin/op",
        "/usr/local/bin/op",
        "/usr/bin/op",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

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
    if config.llm_provider == "openclaw" and not config.openclaw_api_key:
        raise ConfigError(
            "openclaw_api_key is required when llm_provider is 'openclaw'. "
            "Set it in config.yaml or JARVIS_OPENCLAW_API_KEY env var."
        )
    if config.llm_provider not in ("ollama", "anthropic", "openai", "openclaw"):
        raise ConfigError(
            f"Unknown llm_provider: {config.llm_provider!r}. "
            "Must be 'ollama', 'anthropic', 'openai', or 'openclaw'."
        )
    if config.tts_provider not in (
        "piper",
        "chatterbox",
        "openai",
        "edge-tts",
        "macos-say",
        "auto",
    ):
        raise ConfigError(
            f"Unknown tts_provider: {config.tts_provider!r}. "
            "Must be 'piper', 'chatterbox', 'openai', 'edge-tts', 'macos-say', or 'auto'."
        )
    if config.whisper_backend not in ("faster-whisper", "mlx-whisper"):
        raise ConfigError(
            f"Unknown whisper_backend: {config.whisper_backend!r}. "
            "Must be 'faster-whisper' or 'mlx-whisper'."
        )
    if config.wake_acknowledgement_mode not in ("tone", "speech", "none"):
        raise ConfigError(
            "wake_acknowledgement_mode must be 'tone', 'speech', or 'none'"
        )
    if config.wake_acknowledgement_delay_ms < 0:
        raise ConfigError("wake_acknowledgement_delay_ms must be >= 0")
    if config.wake_barge_in_grace_ms < 0:
        raise ConfigError("wake_barge_in_grace_ms must be >= 0")
