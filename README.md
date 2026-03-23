# J.A.R.V.I.S.

Just A Really Very Intelligent System.

Jarvis is a voice assistant for macOS. It listens for a wake word, transcribes speech locally, routes the request through an LLM, executes local actions, and speaks the result back.

## Pipeline

```text
Wake Word -> Speech-to-Text -> LLM Intent -> Action Router -> Plugin -> TTS
```

Core components:

- Wake word: `openWakeWord`
- STT: `faster-whisper` or `mlx-whisper`
- LLM: `ollama`, `openai`, `anthropic`, or `openclaw`
- TTS: `piper`, `chatterbox`, `openai`, `edge-tts`, or `macOS say`

## Requirements

- macOS
- Python 3.11+
- microphone access for the Jarvis process
- Ollama only if you want the default local LLM setup

Optional:

- 1Password CLI for `op://...` secret references
- Spotify desktop app for music actions
- `venv-tts` on Python 3.12 if you want Chatterbox Turbo

## Install

Basic setup:

```bash
git clone <repo-url> jarvis
cd jarvis
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
```

If you want the default local LLM path, install Ollama and pull:

```bash
ollama pull qwen3:0.6b
```

LaunchAgent install:

```bash
bash scripts/install.sh
```

## Permissions

Grant these as needed in macOS:

- Microphone
- Accessibility
- Input Monitoring

The last two are needed for global hotkeys.

## Run

Direct:

```bash
source venv/bin/activate
python -m jarvis
```

LaunchAgent:

```bash
launchctl start com.user.jarvis
launchctl stop com.user.jarvis
tail -f ~/.jarvis/jarvis.log
```

If you disable Jarvis via voice, re-enable it with:

```bash
launchctl enable gui/$(id -u)/com.user.jarvis
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.jarvis.plist
```

## Hotkeys

- `Ctrl+Space`: skip wake word and go straight into command listening
- `Ctrl+Shift+M`: toggle Jarvis mic mute/unmute

Jarvis plays a short system sound for both.

## Test Harness

```bash
source venv/bin/activate
python scripts/test_wakeword.py
python scripts/test_wakeword.py --say
python scripts/test_wakeword.py --chatterbox
python scripts/test_wakeword.py --model=base
```

The harness exercises wake word, STT, LLM, actions, and TTS without running the full background app.

## Providers

LLM providers:

- `ollama`
- `openai`
- `anthropic`
- `openclaw`

TTS providers:

- `piper`
- `chatterbox`
- `openai`
- `edge-tts`
- `macos-say`
- `auto`

OpenClaw notes:

- use the OpenResponses-compatible `/v1/responses` endpoint
- `gateway.http.endpoints.responses.enabled` must be enabled
- keep it on private ingress
- `openclaw_api_key` can use a 1Password reference like `op://Private/OpenClaw/credential`

## Built-in Actions

System:

- `open_app`
- `set_volume`
- `set_brightness`
- `lock_screen`
- `toggle_dark_mode`
- `shell_command`

Weather:

- `get_weather`

Screenshot:

- `take_screenshot`

Music:

- `play_spotify`
- `pause_spotify`
- `next_spotify_track`
- `previous_spotify_track`

Assistant:

- `shutdown_jarvis`
- `disable_jarvis`

Files:

- `find_files`
- `cleanup_downloads`
- `cleanup_desktop`
- `move_file`
- `trash_file`

## Configuration

See `config.example.yaml` for the full file. The main settings are:

| Setting | Purpose |
|---|---|
| `llm_provider` | Selects the backend LLM |
| `tts_provider` | Selects the primary TTS provider |
| `whisper_backend` | `faster-whisper` or `mlx-whisper` |
| `whisper_model` | STT model size |
| `wake_acknowledgement_mode` | `tone`, `speech`, or `none` |
| `wake_barge_in_grace_ms` | Lets you keep speaking after the wake word |
| `confirmation_timeout_sec` | Voice confirmation timeout |
| `followup_timeout_sec` | Timeout for no-wake follow-up listening |

Environment overrides use the `JARVIS_` prefix, for example:

```bash
export JARVIS_LLM_PROVIDER=openclaw
```

## Chatterbox

Chatterbox runs from `venv-tts`, not the main `venv`.

Typical setup:

```bash
python3.12 -m venv venv-tts
source venv-tts/bin/activate
pip install chatterbox-tts
```

Then set:

```yaml
tts_provider: "chatterbox"
chatterbox_voice_reference: "assets/voice_reference.wav"
```

## Adding Plugins

Create a plugin in `jarvis/plugins/`, subclass `BasePlugin`, return actions from `get_actions()`, then register it in `jarvis/plugins/__init__.py`.

Prefer narrow first-class actions over `shell_command`.

## Development

```bash
pip install -e ".[dev]"
pytest
```
