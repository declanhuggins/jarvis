"""Tiered text-to-speech engine.

Tries providers in order: Piper -> Chatterbox Turbo -> OpenAI TTS -> edge-tts -> macOS say.
Falls through to the next tier on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.errors import TTSError

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_PIPER_VOICE_DIR = _PROJECT_DIR / "assets" / "piper"
_TTS_VENV_PYTHON = _PROJECT_DIR / "venv-tts" / "bin" / "python"
_TTS_WORKER_SCRIPT = _PROJECT_DIR / "scripts" / "tts_worker.py"
_VOICE_REF = _PROJECT_DIR / "assets" / "voice_reference.wav"


class _ChatterboxWorker:
    """Manages a persistent Chatterbox Turbo TTS subprocess.

    Requires venv-tts (Python 3.12) to be set up separately.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._ready = False

    def start(self) -> None:
        logger.info("Starting Chatterbox Turbo TTS worker...")
        env = dict(os.environ)
        if _VOICE_REF.exists():
            env["CHATTERBOX_VOICE_REF"] = str(_VOICE_REF)
        self._proc = subprocess.Popen(
            [str(_TTS_VENV_PYTHON), str(_TTS_WORKER_SCRIPT), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
        )
        line = self._proc.stdout.readline()
        if not line:
            self.stop()
            raise TTSError("Chatterbox worker exited during startup")
        msg = json.loads(line)
        if msg.get("status") != "ready":
            self.stop()
            raise TTSError(f"Chatterbox worker unexpected startup: {msg}")
        logger.info("Chatterbox Turbo ready (device=%s, loaded in %ss)",
                     msg.get("device", "?"), msg.get("load_time", "?"))
        self._ready = True

    def generate(self, text: str, output_path: str) -> float:
        if not self._ready or self._proc is None or self._proc.poll() is not None:
            raise TTSError("Chatterbox worker not running")
        req = json.dumps({"text": text, "output_path": output_path})
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            self._ready = False
            raise TTSError("Chatterbox worker closed unexpectedly")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise TTSError(f"Chatterbox generation failed: {resp.get('error')}")
        return resp.get("gen_time", 0.0)

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            self._ready = False

    @property
    def alive(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None


class TTSEngine:
    """Speaks text aloud using the best available TTS provider."""

    def __init__(self, config: JarvisConfig):
        self._config = config
        self._piper_voice = None
        self._chatterbox: _ChatterboxWorker | None = None
        self._tiers: list[tuple[str, callable]] = []

        # Piper TTS — fast local neural TTS (~0.5s per utterance)
        if config.piper_enabled:
            try:
                self._piper_voice = self._load_piper()
                self._tiers.append(("Piper", self._speak_piper))
            except Exception as e:
                logger.warning("Failed to load Piper TTS: %s", e)

        # Chatterbox Turbo — local neural TTS with voice cloning (requires venv-tts)
        if config.chatterbox_enabled and _TTS_VENV_PYTHON.exists():
            worker = _ChatterboxWorker()
            try:
                worker.start()
                self._chatterbox = worker
                self._tiers.append(("Chatterbox Turbo", self._speak_chatterbox))
            except Exception as e:
                logger.warning("Failed to start Chatterbox Turbo: %s", e)
                worker.stop()

        if config.openai_api_key:
            self._tiers.append(("OpenAI TTS", self._speak_openai))
        self._tiers.append(("edge-tts", self._speak_edge_tts))
        self._tiers.append(("macOS say", self._speak_macos))

        tier_names = [name for name, _ in self._tiers]
        logger.info("TTS tiers configured: %s", " -> ".join(tier_names))

    def _load_piper(self):
        """Load the Piper voice model."""
        from piper import PiperVoice

        model_path = _PIPER_VOICE_DIR / self._config.piper_voice
        onnx = model_path.with_suffix(".onnx")
        if not onnx.exists():
            raise TTSError(f"Piper voice model not found: {onnx}")

        t0 = time.monotonic()
        voice = PiperVoice.load(str(onnx))
        elapsed = time.monotonic() - t0
        logger.info("Piper TTS loaded (%s) in %.1fs", self._config.piper_voice, elapsed)
        return voice

    def shutdown(self) -> None:
        """Clean up TTS resources."""
        if self._chatterbox is not None:
            self._chatterbox.stop()

    def speak(self, text: str) -> None:
        """Speak text aloud. Tries each tier in order until one succeeds."""
        if not text or not text.strip():
            return

        for name, tier_fn in self._tiers:
            try:
                tier_fn(text)
                logger.debug("TTS spoke via %s: %s", name, text[:80])
                return
            except Exception as e:
                logger.warning("TTS tier %s failed: %s", name, e)
                continue

        logger.error("All TTS tiers failed for text: %s", text[:80])

    def _speak_piper(self, text: str) -> None:
        """Piper TTS: fast local neural TTS via ONNX."""
        if self._piper_voice is None:
            raise TTSError("Piper voice not loaded")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f.name, "wb") as wav:
                self._piper_voice.synthesize_wav(text, wav)
            subprocess.run(["afplay", f.name], check=True, timeout=60)

    def _speak_chatterbox(self, text: str) -> None:
        """Chatterbox Turbo TTS via persistent worker subprocess."""
        if self._chatterbox is None or not self._chatterbox.alive:
            raise TTSError("Chatterbox worker not available")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            gen_time = self._chatterbox.generate(text, f.name)
            logger.debug("Chatterbox Turbo generated in %.1fs", gen_time)
            subprocess.run(["afplay", f.name], check=True, timeout=60)

    def _speak_openai(self, text: str) -> None:
        """OpenAI TTS API."""
        try:
            import openai
        except ImportError:
            raise TTSError("openai package not installed")

        try:
            client = openai.OpenAI(api_key=self._config.openai_api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice=self._config.openai_tts_voice,
                input=text,
                response_format="mp3",
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                response.stream_to_file(f.name)
                subprocess.run(["afplay", f.name], check=True, timeout=60)
        except Exception as e:
            raise TTSError(f"OpenAI TTS failed: {e}") from e

    def _speak_edge_tts(self, text: str) -> None:
        """edge-tts: free Microsoft TTS."""
        try:
            import edge_tts
        except ImportError:
            raise TTSError("edge-tts package not installed")

        async def _generate_and_play():
            communicate = edge_tts.Communicate(
                text, self._config.edge_tts_voice
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                await communicate.save(f.name)
                subprocess.run(["afplay", f.name], check=True, timeout=60)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _generate_and_play()).result()
            else:
                asyncio.run(_generate_and_play())
        except Exception as e:
            raise TTSError(f"edge-tts failed: {e}") from e

    def _speak_macos(self, text: str) -> None:
        """macOS built-in 'say' command. Always available, no network needed."""
        try:
            subprocess.run(
                ["say", "-v", self._config.macos_tts_voice, text],
                check=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise TTSError(f"macOS say failed: {e}") from e
        except FileNotFoundError:
            raise TTSError("macOS 'say' command not found")
