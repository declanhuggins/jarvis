"""Custom exception hierarchy for Jarvis."""


class JarvisError(Exception):
    """Base for all Jarvis errors."""


class ConfigError(JarvisError):
    """Invalid or missing configuration."""


class AudioError(JarvisError):
    """Microphone or playback failure."""


class WakeWordError(JarvisError):
    """Wake word model loading or detection failure."""


class STTError(JarvisError):
    """Transcription failure."""


class LLMError(JarvisError):
    """LLM API call failure (network, rate limit, bad response)."""


class IntentParseError(JarvisError):
    """LLM returned unparseable or invalid JSON."""


class UnknownActionError(JarvisError):
    """Intent references an action not registered in the router."""


class TTSError(JarvisError):
    """A specific TTS tier failed."""


class PluginError(JarvisError):
    """Plugin execution failure."""
