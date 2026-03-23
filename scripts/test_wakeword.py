#!/usr/bin/env python3
"""Interactive test harness for the Jarvis pipeline with a terminal UI.

Tests the full local pipeline: wake word -> STT -> LLM -> TTS,
with live resource monitoring (CPU, memory, GPU/ANE).

Usage:
    source venv/bin/activate
    python scripts/test_wakeword.py              # full pipeline using config.yaml + Piper TTS
    python scripts/test_wakeword.py --chatterbox # use Chatterbox Turbo TTS (requires venv-tts)
    python scripts/test_wakeword.py --say         # force macOS 'say' for TTS
    python scripts/test_wakeword.py --no-llm      # skip LLM, echo transcript only
    python scripts/test_wakeword.py --model=base  # smaller Whisper model

Press 'q' to quit.
"""

from __future__ import annotations

import contextlib
import curses
import io
import json
import math
import os
import queue
import subprocess
import sys
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_PIPER_VOICE_DIR = _PROJECT_DIR / "assets" / "piper"
_PIPER_VOICE = "en_GB-alan-medium"
_TTS_VENV_PYTHON = _PROJECT_DIR / "venv-tts" / "bin" / "python"
_TTS_WORKER_SCRIPT = _SCRIPT_DIR / "tts_worker.py"
_VOICE_REF = _PROJECT_DIR / "assets" / "voice_reference.wav"
_CONFIG_PATH = _PROJECT_DIR / "config.yaml"

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Shared mutable state for the TUI."""
    phase: str = "Starting up..."
    last_transcript: str = ""
    last_intent: str = ""
    last_response: str = ""
    last_llm_time: float = 0.0
    last_tts_time: float = 0.0
    last_stt_time: float = 0.0
    wakeword_score: float = 0.0
    tts_engine: str = "?"
    llm_engine: str = "?"
    log_lines: list[str] = field(default_factory=list)
    running: bool = True
    # Resource stats
    cpu_percent: float = 0.0
    mem_rss_mb: float = 0.0
    mem_percent: float = 0.0
    tts_worker_rss_mb: float = 0.0  # unused, kept for layout compat
    gpu_util: str = "n/a"
    total_commands: int = 0


def log(state: AppState, msg: str):
    """Append a timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    state.log_lines.append(f"[{ts}] {msg}")
    if len(state.log_lines) > 200:
        state.log_lines = state.log_lines[-200:]


@contextlib.contextmanager
def _suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _resolve_chatterbox_voice_reference(config) -> Path | None:
    raw = getattr(config, "chatterbox_voice_reference", "").strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PROJECT_DIR / path
    return path


def _resolve_chatterbox_device(config) -> str:
    device = getattr(config, "chatterbox_device", "cpu").strip().lower()
    if device not in {"cpu", "mps"}:
        return "cpu"
    return device


def _capture_post_wake_audio(
    audio_queue: queue.Queue[np.ndarray],
    sample_rate: int,
    chunk_samples: int,
    grace_ms: int,
    silence_threshold: float,
) -> tuple[list[np.ndarray], bool]:
    """Capture a short post-wake window to detect continuous speech."""
    if grace_ms <= 0:
        return [], False

    chunk_ms = (chunk_samples / sample_rate) * 1000.0
    grace_chunks = max(1, math.ceil(grace_ms / chunk_ms))
    frames: list[np.ndarray] = []

    for _ in range(grace_chunks):
        try:
            chunk = audio_queue.get(timeout=0.25).flatten()
        except queue.Empty:
            break
        frames.append(chunk)
        energy = np.abs(chunk.astype(np.float32)).mean()
        if energy >= silence_threshold:
            return frames, True

    return frames, False


def _emit_wake_acknowledgement(state: AppState, config, speak) -> None:
    """Play the configured wake acknowledgement cue."""
    mode = (getattr(config, "wake_acknowledgement_mode", "tone") or "tone").strip().lower()
    if mode == "none":
        return
    if mode == "speech":
        acknowledgement = (getattr(config, "wake_acknowledgement", "Yes?") or "").strip()
        if acknowledgement:
            speak(acknowledgement)
        return
    if mode == "tone":
        sound_name = (getattr(config, "wake_acknowledgement_sound", "Pop") or "Pop").strip() or "Pop"
        sound_path = Path("/System/Library/Sounds") / f"{sound_name}.aiff"
        if sound_path.exists():
            subprocess.Popen(
                ["afplay", str(sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log(state, f"Wake tone: {sound_name}")


def _discard_audio_window(
    audio_queue: queue.Queue[np.ndarray],
    sample_rate: int,
    chunk_samples: int,
    duration_ms: int,
) -> None:
    """Read and discard a short microphone window."""
    if duration_ms <= 0:
        return

    chunk_ms = (chunk_samples / sample_rate) * 1000.0
    discard_chunks = max(1, math.ceil(duration_ms / chunk_ms))
    for _ in range(discard_chunks):
        try:
            audio_queue.get(timeout=0.25)
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

def resource_monitor(state: AppState):
    """Background thread: update CPU / memory stats."""
    proc = psutil.Process()
    while state.running:
        try:
            state.cpu_percent = proc.cpu_percent(interval=1.0)
            mem = proc.memory_info()
            state.mem_rss_mb = mem.rss / (1024 * 1024)
            state.mem_percent = proc.memory_percent()

            # macOS memory pressure (0-100, lower = more pressure)
            try:
                r = subprocess.run(
                    ["sysctl", "-n", "kern.memorystatus_level"],
                    capture_output=True, text=True, timeout=2,
                )
                if r.returncode == 0:
                    state.gpu_util = f"mem pressure: {r.stdout.strip()}%"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        except Exception:
            pass
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# TTS client
# ---------------------------------------------------------------------------

class PiperTTSClient:
    """Piper TTS client — fast local neural TTS via ONNX."""

    def __init__(self):
        self._voice = None

    def start(self, state: AppState) -> bool:
        onnx = _PIPER_VOICE_DIR / f"{_PIPER_VOICE}.onnx"
        if not onnx.exists():
            log(state, f"Piper voice not found: {onnx}")
            return False
        try:
            from piper import PiperVoice
            log(state, "Loading Piper TTS...")
            t0 = time.monotonic()
            self._voice = PiperVoice.load(str(onnx))
            elapsed = time.monotonic() - t0
            log(state, f"Piper TTS ready ({_PIPER_VOICE}, loaded in {elapsed:.1f}s)")
            return True
        except Exception as e:
            log(state, f"Piper TTS failed: {e}")
            return False

    def speak(self, text: str, state: AppState) -> bool:
        if self._voice is None:
            return False
        try:
            import wave as wave_mod
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                t0 = time.monotonic()
                with wave_mod.open(f.name, "wb") as wav:
                    self._voice.synthesize_wav(text, wav)
                gen_time = time.monotonic() - t0
                state.last_tts_time = round(gen_time, 1)
                log(state, f"TTS generated in {gen_time:.1f}s")
                subprocess.run(["afplay", f.name], check=True, timeout=60)
                return True
        except Exception as e:
            log(state, f"TTS error: {e}")
            return False

    def stop(self):
        self._voice = None


class ChatterboxTTSClient:
    """Chatterbox Turbo TTS client — persistent worker subprocess."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None

    def start(self, state: AppState, voice_ref: Path | None = None, device: str = "cpu") -> bool:
        if not _TTS_VENV_PYTHON.exists():
            log(state, f"venv-tts not found: {_TTS_VENV_PYTHON}")
            return False
        try:
            log(state, "Starting Chatterbox Turbo worker...")
            env = dict(os.environ)
            if voice_ref is not None:
                env["CHATTERBOX_VOICE_REF"] = str(voice_ref)
            env["CHATTERBOX_DEVICE"] = device
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
                log(state, "Chatterbox worker exited during startup")
                return False
            msg = json.loads(line)
            if msg.get("status") != "ready":
                self.stop()
                log(state, f"Chatterbox unexpected startup: {msg}")
                return False
            log(state, f"Chatterbox Turbo ready (device={msg.get('device', '?')}, "
                       f"loaded in {msg.get('load_time', '?')}s, "
                       f"voice_clone={'on' if msg.get('voice_clone') else 'off'})")
            return True
        except Exception as e:
            log(state, f"Chatterbox start failed: {e}")
            self.stop()
            return False

    def speak(self, text: str, state: AppState) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                req = json.dumps({"text": text, "output_path": f.name})
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    log(state, "Chatterbox worker closed unexpectedly")
                    return False
                resp = json.loads(line)
                if not resp.get("ok"):
                    log(state, f"Chatterbox error: {resp.get('error')}")
                    return False
                gen_time = resp.get("gen_time", 0.0)
                state.last_tts_time = gen_time
                log(state, f"TTS generated in {gen_time:.1f}s")
                subprocess.run(["afplay", f.name], check=True, timeout=60)
                return True
        except Exception as e:
            log(state, f"TTS error: {e}")
            return False

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


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class SharedLLMClient:
    """Adapter over the main Jarvis LLM client used by the test harness."""

    def __init__(self, config):
        from jarvis.llm import LLMClient
        from jarvis.intent import parse_intent
        from jarvis.main import _build_response_text
        from jarvis.plugins import register_all_plugins
        from jarvis.router import CommandRouter

        self._config = config
        self._client = LLMClient(self._config)
        self._build_response_text = _build_response_text
        self._parse_intent = parse_intent
        self._router = CommandRouter()
        register_all_plugins(self._router, self._config)
        self._catalog = self._router.get_action_catalog()

    def connect(self, state: AppState) -> bool:
        try:
            provider = self._config.llm_provider
            model = getattr(self._config, f"{provider}_model", "?")
            state.llm_engine = f"{provider}/{model}"
            if provider == "openclaw":
                log(
                    state,
                    "LLM configured: "
                    f"{provider} (model={model}, base_url={self._config.openclaw_base_url})",
                )
            else:
                log(state, f"LLM configured: {provider} (model={model})")
            return True
        except Exception as e:
            log(state, f"LLM setup failed: {e}")
            return False

    def get_response(self, transcript: str, state: AppState) -> str:
        """Send transcript through the shared Jarvis LLM client and execute safe actions."""
        try:
            t0 = time.monotonic()
            data = self._client.get_intent(transcript, self._catalog)
            state.last_llm_time = time.monotonic() - t0
            intent = self._parse_intent(data)
            state.last_intent = intent.action

            response = intent.spoken_response
            result = ""
            if intent.confirmation_required:
                log(state, "Harness does not auto-confirm destructive actions")
            else:
                result = self._router.execute(intent)
                if result:
                    log(state, f"Action result: {result}")

            response = self._build_response_text(intent.action, response, result)

            self._client.record_turn(transcript, data, response, result)
            log(state, f"LLM ({state.last_llm_time:.1f}s): {intent.spoken_response}")
            return response
        except Exception as e:
            log(state, f"LLM error: {e}")
            state.last_intent = "error"
            return "Sorry, I had trouble processing that."


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------

def draw_bar(win, y: int, x: int, width: int, ratio: float, label: str):
    """Draw a horizontal bar gauge."""
    filled = int(width * min(ratio, 1.0))
    empty = width - filled
    try:
        win.addstr(y, x, label)
        win.addstr("[")
        if filled > 0:
            win.addstr("=" * filled, curses.color_pair(2))
        if empty > 0:
            win.addstr(" " * empty)
        win.addstr(f"] {ratio * 100:5.1f}%")
    except curses.error:
        pass


def safe_addstr(win, y, x, text, *args):
    """addstr that silently ignores writes beyond the screen edge."""
    try:
        win.addstr(y, x, text, *args)
    except curses.error:
        pass


def draw_tui(stdscr, state: AppState):
    """Redraw the full terminal UI."""
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    if h < 10 or w < 40:
        safe_addstr(stdscr, 0, 0, "Terminal too small — resize to at least 80x24")
        stdscr.refresh()
        return

    col_w = min(w - 1, 79)
    row = 0

    # ── Header ──
    header = " JARVIS TEST HARNESS "
    safe_addstr(stdscr, row, 0, header, curses.color_pair(1) | curses.A_BOLD)
    safe_addstr(stdscr, row, len(header) + 1, time.strftime("%H:%M:%S"), curses.A_DIM)
    safe_addstr(stdscr, row, col_w - 12, "press q quit", curses.A_DIM)
    row += 1

    # ── Status ──
    safe_addstr(stdscr, row, 0, " Status: ", curses.A_BOLD)
    safe_addstr(stdscr, row, 9, state.phase)
    row += 1

    # ── Engines ──
    engines = f" LLM: {state.llm_engine}  |  TTS: {state.tts_engine}  |  Commands: {state.total_commands}"
    safe_addstr(stdscr, row, 0, engines[:col_w], curses.A_DIM)
    row += 1

    # ── Divider ──
    safe_addstr(stdscr, row, 0, "─" * col_w, curses.A_DIM)
    row += 1

    # ── Resources ──
    safe_addstr(stdscr, row, 0, " RESOURCES", curses.A_BOLD)
    row += 1

    bar_w = min(30, w - 25)
    if bar_w > 5:
        draw_bar(stdscr, row, 1, bar_w, state.cpu_percent / 100.0, "CPU  ")
        row += 1
        draw_bar(stdscr, row, 1, bar_w, state.mem_percent / 100.0, "MEM  ")
        row += 1

    rss = f"  RSS: {state.mem_rss_mb:.0f} MB"
    safe_addstr(stdscr, row, 0, rss, curses.A_DIM)
    row += 1

    safe_addstr(stdscr, row, 1, f"System {state.gpu_util}", curses.A_DIM)
    row += 1

    # ── Divider ──
    safe_addstr(stdscr, row, 0, "─" * col_w, curses.A_DIM)
    row += 1

    # ── Last Cycle ──
    safe_addstr(stdscr, row, 0, " LAST CYCLE", curses.A_BOLD)
    row += 1
    timings = (f"  Wake: {state.wakeword_score:.3f}   "
               f"STT: {state.last_stt_time:.2f}s   "
               f"LLM: {state.last_llm_time:.2f}s   "
               f"TTS: {state.last_tts_time:.1f}s")
    safe_addstr(stdscr, row, 0, timings)
    row += 1
    safe_addstr(stdscr, row, 0, f'  Transcript: "{state.last_transcript[:col_w - 16]}"')
    row += 1
    safe_addstr(stdscr, row, 0, f"  Intent: {state.last_intent}")
    row += 1
    safe_addstr(stdscr, row, 0, f"  Response: {state.last_response[:col_w - 12]}", curses.A_DIM)
    row += 1

    # ── Divider ──
    safe_addstr(stdscr, row, 0, "─" * col_w, curses.A_DIM)
    row += 1

    # ── Log ──
    safe_addstr(stdscr, row, 0, " LOG", curses.A_BOLD)
    row += 1

    max_log = h - row - 1
    visible = state.log_lines[-max_log:] if max_log > 0 else []
    for i, line in enumerate(visible):
        safe_addstr(stdscr, row + i, 1, line[:col_w - 2], curses.A_DIM)

    stdscr.refresh()


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def pipeline_thread(
    state: AppState,
    force_say: bool,
    use_chatterbox: bool,
    no_llm: bool,
    whisper_model_override: str | None,
):
    """Run wake word -> STT -> LLM -> TTS pipeline."""
    try:
        import sounddevice as sd
    except ImportError:
        log(state, "ERROR: pip install sounddevice")
        state.running = False
        return
    try:
        import openwakeword
        from openwakeword.model import Model as WakeWordModel
    except ImportError:
        log(state, "ERROR: pip install openwakeword")
        state.running = False
        return
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        WhisperModel = None

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 1280
    WAKE_THRESHOLD = 0.5
    SILENCE_THRESHOLD = 500.0
    SILENCE_CHUNKS = 15
    MAX_RECORD_SEC = 15.0

    # Load app config so the harness matches the main app's TTS settings.
    from jarvis.config import load_config

    config = load_config(_CONFIG_PATH)
    chatterbox_voice_ref = _resolve_chatterbox_voice_reference(config)
    chatterbox_device = _resolve_chatterbox_device(config)
    whisper_model = whisper_model_override or config.whisper_model
    whisper_backend = (config.whisper_backend or "faster-whisper").strip()
    whisper_device = (config.whisper_device or "cpu").strip().lower()
    whisper_compute_type = (config.whisper_compute_type or "int8").strip()
    wake_barge_in_grace_ms = int(getattr(config, "wake_barge_in_grace_ms", 400))
    wake_ack_mode = (getattr(config, "wake_acknowledgement_mode", "tone") or "tone").strip().lower()
    wake_ack_delay_ms = int(getattr(config, "wake_acknowledgement_delay_ms", 120))

    # --- TTS ---
    selected_tts_provider = (getattr(config, "tts_provider", "piper") or "piper").strip().lower()
    if use_chatterbox:
        selected_tts_provider = "chatterbox"
    elif force_say:
        selected_tts_provider = "macos-say"

    tts_client = None
    if selected_tts_provider == "chatterbox":
        tts_client = ChatterboxTTSClient()
        if chatterbox_voice_ref is not None and chatterbox_voice_ref.exists():
            log(state, f"Chatterbox custom voice enabled: {chatterbox_voice_ref}")
        log(state, f"Chatterbox device requested: {chatterbox_device}")
        if tts_client.start(
            state,
            voice_ref=chatterbox_voice_ref if chatterbox_voice_ref and chatterbox_voice_ref.exists() else None,
            device=chatterbox_device,
        ):
            state.tts_engine = "Chatterbox Turbo"
        else:
            log(state, "Chatterbox failed, falling back to macOS 'say'")
            tts_client = None
    elif selected_tts_provider == "piper":
        tts_client = PiperTTSClient()
        if tts_client.start(state):
            state.tts_engine = "Piper"
        else:
            log(state, "Piper failed, falling back to macOS 'say'")
            tts_client = None
    else:
        tts_client = None

    if tts_client is None:
        state.tts_engine = "macOS say"

    def speak(text: str):
        if tts_client and tts_client.speak(text, state):
            return
        subprocess.run(["say", "-v", "Daniel", text], check=False)

    # --- LLM ---
    llm_client = None
    if not no_llm:
        state.phase = f"Loading config: {_CONFIG_PATH.name}..."
        try:
            llm_client = SharedLLMClient(config)
        except Exception as e:
            log(state, f"LLM setup failed: {e}")
            llm_client = None
        if llm_client and llm_client.connect(state):
            state.phase = "LLM ready"
        else:
            log(state, "LLM unavailable — will echo transcripts")
            llm_client = None
            state.llm_engine = "none (echo)"
    else:
        state.llm_engine = "disabled"

    # --- Resource monitor ---
    threading.Thread(
        target=resource_monitor, args=(state,), daemon=True
    ).start()

    # --- Load audio models ---
    state.phase = "Loading wake word model..."
    openwakeword.utils.download_models()
    log(state, "Loading wake word model: hey_jarvis")
    oww = WakeWordModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    if whisper_backend == "mlx-whisper":
        try:
            import mlx.core as mx
            import mlx_whisper
            from mlx_whisper.transcribe import ModelHolder
        except ImportError:
            log(state, "ERROR: pip install mlx-whisper")
            state.running = False
            return
        whisper = None
        if whisper_device != "cpu":
            log(state, f"Whisper backend {whisper_backend} ignores whisper_device={whisper_device!r}")
        whisper_repo = whisper_model
        if "/" not in whisper_model:
            whisper_repo = {
                "tiny": "mlx-community/whisper-tiny",
                "base": "mlx-community/whisper-base",
                "small": "mlx-community/whisper-small",
                "medium": "mlx-community/whisper-medium",
                "turbo": "mlx-community/whisper-turbo",
                "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
                "large-v3": "mlx-community/whisper-large-v3",
            }.get(whisper_model, whisper_model)
        log(state, f"Loading Whisper model: {whisper_model} (backend={whisper_backend}, repo={whisper_repo})")
        with _suppress_output():
            ModelHolder.get_model(whisper_repo, mx.float16)
            # Warm once so the first real utterance does not pay MLX compile cost.
            t = np.linspace(0.0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
            warm_audio = 0.02 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
            mlx_whisper.transcribe(
                warm_audio,
                path_or_hf_repo=whisper_repo,
                verbose=None,
                language="en",
                condition_on_previous_text=False,
            )
    else:
        if WhisperModel is None:
            log(state, "ERROR: pip install faster-whisper")
            state.running = False
            return
        if whisper_device == "mps":
            log(state, "Whisper device 'mps' is unsupported by faster-whisper here; falling back to cpu")
            whisper_device = "cpu"

        log(state, f"Loading Whisper model: {whisper_model} (backend={whisper_backend}, {whisper_compute_type}, {whisper_device})")
        whisper = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
    log(state, "All models loaded")

    # --- Audio stream ---
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        audio_queue.put_nowait(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=CHUNK_SAMPLES, callback=audio_cb,
    )

    def drain():
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

    # --- Main loop ---
    with stream:
        state.phase = "Listening for 'Hey Jarvis'..."
        log(state, "Ready — say 'Hey Jarvis' to begin")

        while state.running:
            try:
                chunk = audio_queue.get(timeout=0.5).flatten()
            except queue.Empty:
                continue

            prediction = oww.predict(chunk)
            score = prediction.get("hey_jarvis", 0.0)
            state.wakeword_score = score

            if score < WAKE_THRESHOLD:
                continue

            # ── Wake word detected ──
            state.phase = "Wake word detected!"
            state.total_commands += 1
            log(state, f"Wake word detected (score={score:.3f})")
            frames_list, continued_speech = _capture_post_wake_audio(
                audio_queue,
                sample_rate=SAMPLE_RATE,
                chunk_samples=CHUNK_SAMPLES,
                grace_ms=wake_barge_in_grace_ms,
                silence_threshold=SILENCE_THRESHOLD,
            )

            if continued_speech:
                state.phase = "Recording..."
                log(state, "Skipping wake acknowledgement; user kept speaking")
            else:
                frames_list = []
                if wake_ack_mode != "none":
                    state.phase = "Acknowledging..."
                    _emit_wake_acknowledgement(state, config, speak)
                    if wake_ack_mode == "speech":
                        drain()
                    elif wake_ack_mode == "tone":
                        _discard_audio_window(
                            audio_queue,
                            sample_rate=SAMPLE_RATE,
                            chunk_samples=CHUNK_SAMPLES,
                            duration_ms=wake_ack_delay_ms,
                        )

            # ── Record ──
            state.phase = "Recording... (speak now)"
            consecutive_silent = 0
            has_speech = False
            for existing in frames_list:
                energy = np.abs(existing.astype(np.float32)).mean()
                if energy < SILENCE_THRESHOLD:
                    consecutive_silent += 1
                else:
                    consecutive_silent = 0
                    has_speech = True
            max_chunks = int(MAX_RECORD_SEC * SAMPLE_RATE / CHUNK_SAMPLES)
            remaining_chunks = max(0, max_chunks - len(frames_list))

            for _ in range(remaining_chunks):
                if not state.running:
                    break
                try:
                    chunk = audio_queue.get(timeout=2.0).flatten()
                except queue.Empty:
                    break
                frames_list.append(chunk)
                energy = np.abs(chunk.astype(np.float32)).mean()
                if energy < SILENCE_THRESHOLD:
                    consecutive_silent += 1
                else:
                    consecutive_silent = 0
                    has_speech = True
                if has_speech and consecutive_silent >= SILENCE_CHUNKS:
                    break

            if not frames_list:
                log(state, "No audio captured")
                oww.reset()
                state.phase = "Listening for 'Hey Jarvis'..."
                continue

            # ── Transcribe ──
            state.phase = "Transcribing..."
            audio_data = np.concatenate(frames_list)
            audio_float = audio_data.astype(np.float32) / 32768.0
            duration = len(audio_float) / SAMPLE_RATE

            t0 = time.monotonic()
            if whisper_backend == "mlx-whisper":
                with _suppress_output():
                    result = mlx_whisper.transcribe(
                        audio_float,
                        path_or_hf_repo=whisper_repo,
                        verbose=None,
                        language="en",
                        condition_on_previous_text=False,
                    )
                text = result["text"].strip()
            else:
                segments, info = whisper.transcribe(
                    audio_float, language="en", beam_size=1,
                    vad_filter=True, condition_on_previous_text=False,
                )
                text = " ".join(seg.text for seg in segments).strip()
            state.last_stt_time = time.monotonic() - t0
            state.last_transcript = text
            log(state, f'Transcript ({state.last_stt_time:.2f}s): "{text}"')

            if not text:
                oww.reset()
                state.phase = "Listening for 'Hey Jarvis'..."
                continue

            # ── LLM ──
            if llm_client:
                state.phase = "Thinking (LLM)..."
                response_text = llm_client.get_response(text, state)
            else:
                state.last_intent = "echo"
                state.last_llm_time = 0.0
                response_text = f"You said: {text}"

            state.last_response = response_text

            # ── Speak ──
            if response_text:
                state.phase = "Speaking response..."
                speak(response_text)
                drain()

            oww.reset()
            state.phase = "Listening for 'Hey Jarvis'..."

    if tts_client:
        tts_client.stop()
    log(state, "Pipeline shut down")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def curses_main(stdscr):
    """Main curses loop — refresh TUI while pipeline runs in background."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(250)  # 4 fps

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_GREEN, -1)

    force_say = "--say" in sys.argv
    use_chatterbox = "--chatterbox" in sys.argv
    no_llm = "--no-llm" in sys.argv
    whisper_model = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            whisper_model = arg.split("=", 1)[1]

    state = AppState()

    t = threading.Thread(
        target=pipeline_thread,
        args=(state, force_say, use_chatterbox, no_llm, whisper_model),
        daemon=True,
    )
    t.start()

    while state.running:
        draw_tui(stdscr, state)
        key = stdscr.getch()
        if key == ord("q"):
            state.running = False
            break

    state.phase = "Shutting down..."
    draw_tui(stdscr, state)
    time.sleep(0.5)


def main():
    signal.signal(signal.SIGINT, lambda s, f: None)
    try:
        curses.wrapper(curses_main)
    except KeyboardInterrupt:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
