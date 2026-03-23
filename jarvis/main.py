"""Jarvis main event loop - orchestrates the full voice assistant pipeline."""

from __future__ import annotations

import logging
import logging.handlers
import math
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

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
from jarvis.hotkeys import GlobalHotkeys
from jarvis.intent import parse_intent
from jarvis.llm import LLMClient
from jarvis.plugins import register_all_plugins
from jarvis.router import CommandRouter
from jarvis.stt import SpeechToText
from jarvis.tts import TTSEngine
from jarvis.wakeword import WakeWordDetector

logger = logging.getLogger("jarvis")
_LAUNCH_AGENT_LABEL = "com.user.jarvis"
_LAUNCH_AGENT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCH_AGENT_LABEL}.plist"
_MUTE_SOUND = "Basso"
_UNMUTE_SOUND = "Glass"
_CANCEL_SOUND = "Frog"


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
        backend=config.whisper_backend,
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
    force_listen_event = threading.Event()
    cancel_request_event = threading.Event()
    listening_active = threading.Event()

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

    def _on_force_listen() -> None:
        if audio.muted:
            logger.info("Ignoring Ctrl+Space while microphone is muted")
            _play_system_sound("Basso")
            return
        if listening_active.is_set():
            cancel_request_event.set()
            logger.info("Canceling active command capture via hotkey")
            _play_system_sound(_CANCEL_SOUND)
            return
        force_listen_event.set()
        logger.info("Force-listen hotkey triggered")
        _emit_listen_cue(config, tts, tone_only=True)

    def _on_toggle_mute() -> None:
        muted = audio.toggle_muted()
        force_listen_event.clear()
        _play_system_sound(_MUTE_SOUND if muted else _UNMUTE_SOUND)
        logger.info("Microphone %s via hotkey", "muted" if muted else "unmuted")

    hotkeys = GlobalHotkeys(_on_force_listen, _on_toggle_mute)
    hotkeys.start()

    tts.speak("Jarvis online.")
    audio.drain()

    # --- Main Event Loop ---
    logger.info("Listening for wake word '%s'...", config.wakeword_model)

    while running:
        if force_listen_event.is_set():
            force_listen_event.clear()
            cancel_request_event.clear()
            logger.info("=== Direct listen hotkey triggered ===")
            initial_frames = None
            pending_transcript: str | None = None
            followup_turns_remaining = max(0, config.followup_max_turns)
            _discard_audio_window(audio, config.wake_acknowledgement_delay_ms)
            goto_command_loop = True
        else:
            goto_command_loop = False

        # Phase 1: Wake word detection
        if not goto_command_loop:
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
            cancel_request_event.clear()
            initial_frames, continued_speech = _capture_post_wake_audio(
                audio,
                grace_ms=config.wake_barge_in_grace_ms,
            )

            if continued_speech:
                logger.info("Skipping wake acknowledgement; user continued speaking")
            else:
                _emit_listen_cue(config, tts)
                if config.wake_acknowledgement_mode == "speech":
                    audio.drain()
                elif config.wake_acknowledgement_mode == "tone":
                    _discard_audio_window(audio, config.wake_acknowledgement_delay_ms)
                initial_frames = None

            pending_transcript = None
            followup_turns_remaining = max(0, config.followup_max_turns)

        while running:
            # Phase 2: Speech-to-Text
            if pending_transcript is None:
                try:
                    listening_active.set()
                    transcript = stt.transcribe_stream(
                        audio,
                        initial_frames=initial_frames,
                        cancel_event=cancel_request_event,
                    )
                    initial_frames = None
                except STTError as e:
                    logger.error("STT failed: %s", e)
                    tts.speak("Sorry, I didn't catch that.")
                    audio.drain()
                    break
                finally:
                    listening_active.clear()
            else:
                transcript = pending_transcript
                pending_transcript = None

            if cancel_request_event.is_set():
                cancel_request_event.clear()
                audio.drain()
                logger.info("Canceled current request")
                break

            if not transcript.strip():
                logger.info("Empty transcript, returning to listening")
                break

            logger.info("Transcript: %s", transcript)

            # Phase 3: LLM Intent Resolution
            try:
                raw_intent = llm.get_intent(transcript, router.get_action_catalog())
                intent = parse_intent(raw_intent)
            except (LLMError, IntentParseError) as e:
                logger.error("LLM/intent parse failed: %s", e)
                tts.speak("I had trouble understanding that. Could you try again?")
                audio.drain()
                break

            logger.info(
                "Intent: action=%s, confirm=%s, reasoning=%s",
                intent.action,
                intent.confirmation_required,
                intent.reasoning,
            )

            # Phase 4: Confirmation (if destructive)
            if intent.confirmation_required:
                audio.drain()
                confirmed = confirmer.confirm(intent.spoken_response, action=intent.action)
                audio.drain()
                if not confirmed:
                    declined_response = "Understood, I'll skip that."
                    tts.speak(declined_response)
                    audio.drain()
                    llm.record_turn(
                        transcript,
                        {
                            "action": intent.action,
                            "parameters": intent.parameters,
                            "confirmation_required": intent.confirmation_required,
                            "reasoning": intent.reasoning,
                        },
                        declined_response,
                    )
                    logger.info("User declined confirmation")
                    break

            # Phase 5: Execute
            try:
                result = router.execute(intent)
                logger.info("Execution result: %s", result[:200] if result else "(empty)")
            except (UnknownActionError, PluginError) as e:
                logger.error("Execution failed: %s", e)
                tts.speak("Sorry, something went wrong executing that command.")
                audio.drain()
                break

            # Phase 6: Respond
            response = _build_response_text(intent.action, intent.spoken_response, result)
            if response:
                tts.speak(response)
                audio.drain()

            llm.record_turn(
                transcript,
                {
                    "action": intent.action,
                    "parameters": intent.parameters,
                    "confirmation_required": intent.confirmation_required,
                    "reasoning": intent.reasoning,
                },
                response,
                result,
            )

            if intent.action == "shutdown_jarvis":
                logger.info("Shutdown requested by user")
                running = False
                logger.info("--- Command cycle complete ---")
                break

            if intent.action == "disable_jarvis":
                logger.info("Disable requested by user")
                running = False
                _disable_launch_agent()
                logger.info("--- Command cycle complete ---")
                break

            if (
                followup_turns_remaining > 0
                and _should_wait_for_followup(intent, response)
            ):
                logger.info("Listening for follow-up response (no wake word required)...")
                _emit_listen_cue(config, tts, tone_only=True)
                _discard_audio_window(audio, config.wake_acknowledgement_delay_ms)
                try:
                    listening_active.set()
                    followup = stt.transcribe_stream(
                        audio,
                        max_duration_sec=config.followup_timeout_sec,
                        cancel_event=cancel_request_event,
                    )
                except STTError as e:
                    logger.error("Follow-up STT failed: %s", e)
                    break
                finally:
                    listening_active.clear()

                if cancel_request_event.is_set():
                    cancel_request_event.clear()
                    audio.drain()
                    logger.info("Canceled current request")
                    break

                if not followup.strip():
                    logger.info("No follow-up response received, returning to wake-word mode")
                    break

                logger.info("Follow-up transcript: %s", followup)
                pending_transcript = followup
                followup_turns_remaining -= 1
                continue

            logger.info("--- Command cycle complete ---")
            break

    # --- Cleanup ---
    hotkeys.stop()
    tts.shutdown()
    audio.stop()
    logger.info("Jarvis shut down cleanly")


def _should_wait_for_followup(intent, response: str) -> bool:
    """Decide whether Jarvis should stay in a short follow-up listening mode."""
    if intent.action != "conversational_response":
        return False
    return response.strip().endswith("?")


def _capture_post_wake_audio(
    audio: AudioStream,
    grace_ms: int,
    silence_threshold: float = 500.0,
) -> tuple[list[np.ndarray], bool]:
    """Collect a short post-wake window to detect continuous speech."""
    if grace_ms <= 0:
        return [], False

    chunk_ms = (audio.chunk_samples / audio.sample_rate) * 1000.0
    grace_chunks = max(1, math.ceil(grace_ms / chunk_ms))
    frames: list[np.ndarray] = []

    for _ in range(grace_chunks):
        try:
            chunk = audio.read_chunk(timeout=0.25)
        except queue.Empty:
            break

        frames.append(chunk)
        energy = np.abs(chunk.astype(np.float32)).mean()
        if energy >= silence_threshold:
            return frames, True

    return frames, False


def _emit_listen_cue(
    config: JarvisConfig,
    tts: TTSEngine,
    *,
    tone_only: bool = False,
) -> None:
    """Play the configured listening cue."""
    mode = (config.wake_acknowledgement_mode or "tone").strip().lower()
    if mode == "none":
        return
    if mode == "speech" and not tone_only:
        acknowledgement = (config.wake_acknowledgement or "").strip()
        if acknowledgement:
            tts.speak(acknowledgement)
        return
    _play_system_sound(config.wake_acknowledgement_sound or "Pop")


def _discard_audio_window(audio: AudioStream, duration_ms: int) -> None:
    """Read and discard a short window of microphone input."""
    if duration_ms <= 0:
        return

    chunk_ms = (audio.chunk_samples / audio.sample_rate) * 1000.0
    discard_chunks = max(1, math.ceil(duration_ms / chunk_ms))
    for _ in range(discard_chunks):
        try:
            audio.read_chunk(timeout=0.25)
        except queue.Empty:
            break


def _play_system_sound(sound_name: str) -> None:
    """Play a short built-in macOS system sound."""
    sound_name = (sound_name or "Pop").strip() or "Pop"
    sound_path = Path("/System/Library/Sounds") / f"{sound_name}.aiff"
    if sound_path.exists():
        subprocess.Popen(
            ["afplay", str(sound_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _build_response_text(action: str, spoken_response: str, execution_result: str) -> str:
    """Build the final spoken response without redundant double-confirmation."""
    spoken_response = (spoken_response or "").strip()
    execution_result = (execution_result or "").strip()

    if action == "shutdown_jarvis":
        return "Shutting down Jarvis."
    if action == "disable_jarvis":
        return "Disabling Jarvis."

    if not execution_result:
        return spoken_response
    if not spoken_response:
        return execution_result
    if action == "conversational_response":
        return f"{spoken_response}. {execution_result}" if execution_result else spoken_response

    return execution_result


def _disable_launch_agent() -> None:
    """Disable the LaunchAgent if present so Jarvis stays off until re-enabled."""
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{_LAUNCH_AGENT_LABEL}"

    try:
        subprocess.run(
            ["launchctl", "disable", service],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        logger.warning("Failed to disable LaunchAgent %s: %s", service, e)

    try:
        if _LAUNCH_AGENT_PLIST.exists():
            cmd = ["launchctl", "bootout", domain, str(_LAUNCH_AGENT_PLIST)]
        else:
            cmd = ["launchctl", "bootout", service]
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        logger.warning("Failed to boot out LaunchAgent %s: %s", service, e)
