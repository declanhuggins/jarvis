"""Jarvis main event loop - orchestrates the full voice assistant pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import queue
import signal
import sys
from pathlib import Path

from jarvis.audio import AudioStream
from jarvis.config import JarvisConfig, load_config
from jarvis.confirmation import ConfirmationHandler
from jarvis.errors import (
    AudioError,
    ConfigError,
    IntentParseError,
    JarvisError,
    LLMError,
    PluginError,
    STTError,
    UnknownActionError,
    WakeWordError,
)
from jarvis.intent import parse_intent
from jarvis.llm import LLMClient
from jarvis.plugins import register_all_plugins
from jarvis.router import CommandRouter
from jarvis.stt import SpeechToText
from jarvis.tts import TTSEngine
from jarvis.wakeword import WakeWordDetector

logger = logging.getLogger("jarvis")


def setup_logging(config: JarvisConfig) -> None:
    """Configure logging to both stderr and a rotating log file."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger("jarvis")
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr - captured by LaunchAgent)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Rotating file handler
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def run() -> None:
    """Main entry point. Initializes all components and runs the event loop."""
    # --- Load Config ---
    try:
        config = load_config()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    logger.info("Jarvis starting up (v0.1.0)")

    # --- Initialize Components ---
    chunk_samples = config.sample_rate * config.chunk_ms // 1000

    audio = AudioStream(
        sample_rate=config.sample_rate,
        chunk_samples=chunk_samples,
    )

    wakeword = WakeWordDetector(
        model_name=config.wakeword_model,
        threshold=config.wakeword_threshold,
    )

    stt = SpeechToText(
        model_size=config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )

    llm = LLMClient(config)
    tts = TTSEngine(config)
    router = CommandRouter()
    confirmer = ConfirmationHandler(tts, stt, audio, config.confirmation_timeout_sec)

    # --- Register Plugins ---
    register_all_plugins(router, config)

    # --- Load Models ---
    logger.info("Loading models (this may take a moment on first run)...")
    try:
        wakeword.load()
        stt.load()
    except (WakeWordError, STTError) as e:
        logger.critical("Failed to load models: %s", e)
        sys.exit(1)

    # --- Graceful Shutdown ---
    running = True

    def shutdown(signum: int, frame) -> None:
        nonlocal running
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down...", sig_name)
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # --- Start Audio & Announce ---
    try:
        audio.start()
    except AudioError as e:
        logger.critical("Failed to start audio: %s", e)
        sys.exit(1)

    tts.speak("Jarvis online.")
    audio.drain()

    # --- Main Event Loop ---
    logger.info("Listening for wake word '%s'...", config.wakeword_model)

    while running:
        # Phase 1: Wake word detection
        try:
            chunk = audio.read_chunk(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Audio read error: %s", e)
            continue

        if not wakeword.process_chunk(chunk):
            continue

        # --- Wake word detected ---
        logger.info("=== Wake word detected ===")

        # Brief acknowledgment so the user knows Jarvis is listening
        tts.speak("Yes?")
        audio.drain()

        # Phase 2: Speech-to-Text
        try:
            transcript = stt.transcribe_stream(audio)
        except STTError as e:
            logger.error("STT failed: %s", e)
            tts.speak("Sorry, I didn't catch that.")
            audio.drain()
            continue

        if not transcript.strip():
            logger.info("Empty transcript, returning to listening")
            continue

        logger.info("Transcript: %s", transcript)

        # Phase 3: LLM Intent Resolution
        try:
            raw_intent = llm.get_intent(transcript, router.get_action_catalog())
            intent = parse_intent(raw_intent)
        except (LLMError, IntentParseError) as e:
            logger.error("LLM/intent parse failed: %s", e)
            tts.speak("I had trouble understanding that. Could you try again?")
            audio.drain()
            continue

        logger.info(
            "Intent: action=%s, confirm=%s, reasoning=%s",
            intent.action,
            intent.confirmation_required,
            intent.reasoning,
        )

        # Phase 4: Confirmation (if destructive)
        if intent.confirmation_required:
            audio.drain()
            confirmed = confirmer.confirm(intent.spoken_response)
            audio.drain()
            if not confirmed:
                tts.speak("Understood, I'll skip that.")
                audio.drain()
                logger.info("User declined confirmation")
                continue

        # Phase 5: Execute
        try:
            result = router.execute(intent)
            logger.info("Execution result: %s", result[:200] if result else "(empty)")
        except (UnknownActionError, PluginError) as e:
            logger.error("Execution failed: %s", e)
            tts.speak("Sorry, something went wrong executing that command.")
            audio.drain()
            continue

        # Phase 6: Respond
        response = intent.spoken_response
        if result:
            response = f"{response}. {result}" if response else result
        if response:
            tts.speak(response)
            audio.drain()

        logger.info("--- Command cycle complete ---")

    # --- Cleanup ---
    tts.shutdown()
    audio.stop()
    logger.info("Jarvis shut down cleanly")
