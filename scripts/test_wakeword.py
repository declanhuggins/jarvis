#!/usr/bin/env python3
"""Interactive test harness for the Jarvis pipeline with a terminal UI.

Tests the full local pipeline: wake word -> STT -> LLM -> TTS,
with live resource monitoring (CPU, memory, GPU/ANE).

Usage:
    source venv/bin/activate
    python scripts/test_wakeword.py              # full pipeline with Ollama + Piper TTS
    python scripts/test_wakeword.py --chatterbox # use Chatterbox Turbo TTS (requires venv-tts)
    python scripts/test_wakeword.py --say         # force macOS 'say' for TTS
    python scripts/test_wakeword.py --no-llm      # skip LLM, echo transcript only
    python scripts/test_wakeword.py --model=base  # smaller Whisper model

Press 'q' to quit.
"""

from __future__ import annotations

import curses
import json
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

    def start(self, state: AppState) -> bool:
        if not _TTS_VENV_PYTHON.exists():
            log(state, f"venv-tts not found: {_TTS_VENV_PYTHON}")
            return False
        try:
            log(state, "Starting Chatterbox Turbo worker...")
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
                log(state, "Chatterbox worker exited during startup")
                return False
            msg = json.loads(line)
            if msg.get("status") != "ready":
                self.stop()
                log(state, f"Chatterbox unexpected startup: {msg}")
                return False
            log(state, f"Chatterbox Turbo ready (device={msg.get('device', '?')}, "
                       f"loaded in {msg.get('load_time', '?')}s)")
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

class OllamaLLMClient:
    """Lightweight Ollama LLM client for the test harness."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:0.6b"):
        self._model = model
        self._base_url = base_url.rstrip("/").replace("/v1", "")
        self._connected = False

    def connect(self, state: AppState) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            self._connected = True
            log(state, f"Ollama connected (model={self._model})")
            return True
        except Exception as e:
            log(state, f"Ollama connection failed: {e}")
            return False

    def get_response(self, transcript: str, state: AppState) -> str:
        """Send transcript to Ollama, return spoken_response text."""
        if not self._connected:
            return f"You said: {transcript}"

        system_prompt = (
            "You are Jarvis, a personal AI assistant. The user gave a voice command. "
            "Respond with a single JSON object: "
            '{"action": "conversational_response", "spoken_response": "your reply here"}. '
            "Keep spoken_response under 2 sentences. No markdown."
        )

        try:
            import httpx
            t0 = time.monotonic()
            payload = {
                "model": self._model,
                "stream": False,
                "think": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
            }
            resp = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            state.last_llm_time = time.monotonic() - t0
            raw = resp.json()["message"]["content"]

            # Strip code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            try:
                data = json.loads(cleaned)
                state.last_intent = data.get("action", "?")
                spoken = data.get("spoken_response", raw)
                log(state, f"LLM ({state.last_llm_time:.1f}s): {spoken}")
                return spoken
            except json.JSONDecodeError:
                state.last_intent = "raw_text"
                log(state, f"LLM ({state.last_llm_time:.1f}s): {raw[:120]}")
                return raw
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

def pipeline_thread(state: AppState, force_say: bool, use_chatterbox: bool, no_llm: bool, whisper_model: str):
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
        log(state, "ERROR: pip install faster-whisper")
        state.running = False
        return

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 1280
    WAKE_THRESHOLD = 0.5
    SILENCE_THRESHOLD = 500.0
    SILENCE_CHUNKS = 15
    MAX_RECORD_SEC = 15.0

    # --- TTS ---
    tts_client = None
    if not force_say:
        if use_chatterbox:
            tts_client = ChatterboxTTSClient()
            if tts_client.start(state):
                state.tts_engine = "Chatterbox Turbo"
            else:
                log(state, "Chatterbox failed, falling back to Piper")
                tts_client = PiperTTSClient()
                if not tts_client.start(state):
                    log(state, "Falling back to macOS 'say'")
                    tts_client = None
                else:
                    state.tts_engine = "Piper"
        else:
            tts_client = PiperTTSClient()
            if tts_client.start(state):
                state.tts_engine = "Piper"
            else:
                log(state, "Falling back to macOS 'say'")
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
        state.phase = "Connecting to Ollama..."
        llm_client = OllamaLLMClient()
        if llm_client.connect(state):
            state.llm_engine = "Ollama/qwen3:0.6b"
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

    state.phase = f"Loading Whisper: {whisper_model}..."
    log(state, f"Loading Whisper model: {whisper_model} (int8, cpu)")
    whisper = WhisperModel(whisper_model, device="cpu", compute_type="int8")
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

            state.phase = "Acknowledging..."
            speak("Yes?")
            drain()

            # ── Record ──
            state.phase = "Recording... (speak now)"
            frames_list: list[np.ndarray] = []
            consecutive_silent = 0
            has_speech = False
            max_chunks = int(MAX_RECORD_SEC * SAMPLE_RATE / CHUNK_SAMPLES)

            for _ in range(max_chunks):
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
    whisper_model = "small"
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
