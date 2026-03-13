# J.A.R.V.I.S.

**Just A Really Very Intelligent System**

A voice-activated personal assistant for macOS. Runs continuously in the background, listening for a wake word, understanding spoken commands via a local LLM, and executing local actions.

## How It Works

```
"Hey Jarvis" ─> Wake Word ─> Speech-to-Text ─> LLM ─> Intent Parser ─> Command Router ─> Action
                (openWakeWord)  (faster-whisper)  (Ollama/qwen3:0.6b)                              ↓
                                                                                          TTS Response
```

1. **Wake word** — Listens for "Hey Jarvis" using [openWakeWord](https://github.com/dscripka/openWakeWord) (local, ONNX inference)
2. **Speech-to-text** — Transcribes voice with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local, runs on CPU with int8)
3. **LLM reasoning** — Sends transcript to [Ollama](https://ollama.com/) (local, default) or a cloud LLM to determine intent, returning structured JSON
4. **Command routing** — Maps the intent to a local plugin handler
5. **Execution** — Runs the action (open apps, manage files, control system settings, etc.)
6. **Voice response** — Speaks the result back via [Piper TTS](https://github.com/rhasspy/piper) (fast local neural TTS) with fallback tiers

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.11+
- [Ollama](https://ollama.com/) installed with the `qwen3:0.6b` model pulled
- Microphone access granted to your terminal app

## Project Structure

```
jarvis/
├── jarvis/                 # Core application
│   ├── __init__.py
│   ├── __main__.py         # python -m jarvis entry point
│   ├── main.py             # Event loop orchestrating the pipeline
│   ├── config.py           # YAML config loading + env var overlay
│   ├── audio.py            # Microphone capture via sounddevice
│   ├── wakeword.py         # openWakeWord detector
│   ├── stt.py              # faster-whisper transcription
│   ├── llm.py              # Ollama / Anthropic / OpenAI client
│   ├── intent.py           # Intent dataclass + parser
│   ├── router.py           # Action dispatcher
│   ├── tts.py              # Tiered TTS engine (Piper -> fallbacks)
│   ├── confirmation.py     # Voice confirmation for destructive ops
│   ├── errors.py           # Exception hierarchy
│   └── plugins/
│       ├── __init__.py     # Plugin registration
│       ├── base.py         # BasePlugin abstract class
│       ├── system.py       # open_app, volume, brightness, lock, dark mode, shell
│       └── files.py        # cleanup downloads/desktop, find/move/trash files
├── assets/
│   └── piper/              # Piper TTS voice models (.onnx + .json)
├── scripts/
│   ├── install.sh          # Full setup + LaunchAgent registration
│   ├── uninstall.sh        # Teardown
│   └── test_wakeword.py    # Standalone test: wake word + STT + LLM + TTS
├── tests/
│   ├── test_intent.py
│   └── test_router.py
├── config.example.yaml     # Template config (copy to config.yaml)
├── com.user.jarvis.plist   # LaunchAgent template (filled in by install.sh)
├── pyproject.toml
└── .gitignore
```

## Installation

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com/ then:
ollama pull qwen3:0.6b
```

### 2. Clone and set up

```bash
git clone <repo-url> jarvis
cd jarvis
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Piper TTS and its voice model are installed automatically with the package. No separate venv needed.

### 3. Configure

```bash
cp config.example.yaml config.yaml
```

The default config uses Ollama with `qwen3:0.6b` — no API keys needed. To use a cloud provider instead, edit `config.yaml`:

```yaml
llm_provider: "anthropic"   # or "openai"
anthropic_api_key: "sk-ant-..."
```

All settings can also be overridden with environment variables using the `JARVIS_` prefix (e.g. `JARVIS_LLM_PROVIDER=openai`).

### 4. Grant microphone access

Go to **System Settings > Privacy & Security > Microphone** and enable access for your terminal app (Terminal, iTerm2, etc.).

## Usage

### Run directly

```bash
source venv/bin/activate
python -m jarvis
```

### Run as a LaunchAgent (auto-start on login)

```bash
bash scripts/install.sh
```

This creates and registers a macOS LaunchAgent that starts J.A.R.V.I.S. on login and restarts it on crash. To manage:

```bash
launchctl start com.user.jarvis     # Start now
launchctl stop com.user.jarvis      # Stop
tail -f ~/.jarvis/jarvis.log        # View logs
bash scripts/uninstall.sh           # Remove LaunchAgent
```

### Test wake word detection (no API keys needed)

```bash
source venv/bin/activate
python scripts/test_wakeword.py              # Full pipeline with Piper TTS
python scripts/test_wakeword.py --say        # Force macOS 'say' for TTS
python scripts/test_wakeword.py --model=base # Smaller Whisper model
```

This tests the local pipeline: wake word detection, speech-to-text, LLM, and TTS without needing any cloud API keys.

## Plugins

### System Plugin

| Action | Description | Confirmation |
|---|---|---|
| `open_app` | Open any macOS application | No |
| `set_volume` | Set system volume (0-100) | No |
| `set_brightness` | Set screen brightness (0-100) | No |
| `lock_screen` | Lock the screen | No |
| `toggle_dark_mode` | Toggle macOS dark mode | No |
| `shell_command` | Run an arbitrary shell command | **Yes** |

### Files Plugin

| Action | Description | Confirmation |
|---|---|---|
| `find_files` | Search for files by name pattern | No |
| `cleanup_downloads` | Organize ~/Downloads by file type | **Yes** |
| `cleanup_desktop` | Organize ~/Desktop by file type | **Yes** |
| `move_file` | Move a file to a new location | **Yes** |
| `trash_file` | Move a file to the Trash | **Yes** |

### Adding your own plugins

Create a new module in `jarvis/plugins/` that subclasses `BasePlugin`:

```python
from jarvis.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    def get_actions(self) -> list[dict]:
        return [
            {
                "name": "my_action",
                "description": "Does something useful",
                "parameters": {"param1": "string"},
                "destructive": False,
                "handler": self.my_action,
            },
        ]

    def my_action(self, param1: str) -> str:
        # do something
        return "Done"
```

Then register it in `jarvis/plugins/__init__.py`.

## TTS Tiers

The TTS engine tries providers in order, falling through on failure:

1. **Piper** — Fast local neural TTS via ONNX. Sub-second generation, no GPU needed. Runs in the main process.
2. **Chatterbox Turbo** *(optional)* — Local neural TTS with voice cloning. Requires a separate `venv-tts` (Python 3.12). Set `chatterbox_enabled: true` to enable. Place a reference clip at `assets/voice_reference.wav` for voice cloning.
3. **OpenAI TTS** — Cloud API (requires `openai_api_key`). Fast and high quality.
4. **edge-tts** — Free Microsoft TTS via Edge browser API. No API key, internet required.
5. **macOS say** — Built-in macOS speech synthesis. Always available, offline.

## Configuration Reference

See `config.example.yaml` for all options. Key settings:

| Setting | Default | Description |
|---|---|---|
| `llm_provider` | `ollama` | `ollama`, `anthropic`, or `openai` |
| `ollama_model` | `qwen3:0.6b` | Ollama model name |
| `piper_enabled` | `true` | Use local Piper TTS |
| `piper_voice` | `en_GB-alan-medium` | Piper voice model name (in `assets/piper/`) |
| `chatterbox_enabled` | `false` | Enable Chatterbox Turbo TTS (requires `venv-tts`) |
| `whisper_model` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `wakeword_threshold` | `0.5` | Wake word confidence threshold (0-1) |
| `confirmation_timeout_sec` | `10.0` | Seconds to wait for destructive action confirmation |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Development

```bash
pip install -e ".[dev]"
pytest
```
