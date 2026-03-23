#!/usr/bin/env python3
"""Chatterbox Turbo TTS worker – persistent subprocess.

Runs in the venv-tts Python 3.12 environment. Loads the Turbo model once,
then reads JSON requests from stdin and writes JSON responses to stdout.

One-shot mode:
    venv-tts/bin/python scripts/tts_worker.py <output_path> <text>

Persistent mode:
    venv-tts/bin/python scripts/tts_worker.py --serve

    Reads newline-delimited JSON from stdin:
        {"text": "Hello", "output_path": "/tmp/out.wav"}

    Writes newline-delimited JSON to stdout:
        {"ok": true, "output_path": "/tmp/out.wav", "gen_time": 3.2}
        {"ok": false, "error": "some error message"}
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path


def _suppress_stdout():
    """Redirect stdout to a buffer to silence library prints."""
    real = sys.stdout
    sys.stdout = io.StringIO()
    return real


def _load_model():
    """Load ChatterboxTurboTTS model, returns model."""
    # Suppress any stdout printed by libraries during import/load
    # (e.g. perth prints "loaded PerthNet ..." to stdout)
    real_stdout = _suppress_stdout()
    try:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = os.environ.get("CHATTERBOX_DEVICE", "cpu").strip().lower() or "cpu"
        if device not in {"cpu", "mps"}:
            device = "cpu"
        t0 = time.monotonic()
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        elapsed = time.monotonic() - t0
    finally:
        sys.stdout = real_stdout

    voice_ref = _resolve_voice_ref(os.environ.get("CHATTERBOX_VOICE_REF", ""))
    voice_clone = False
    if voice_ref is not None:
        real_stdout = _suppress_stdout()
        try:
            model.prepare_conditionals(str(voice_ref))
            setattr(model, "_jarvis_cached_voice_ref", str(voice_ref))
            voice_clone = True
        finally:
            sys.stdout = real_stdout

    print(
        json.dumps(
            {
                "status": "ready",
                "device": device,
                "load_time": round(elapsed, 1),
                "voice_clone": voice_clone,
            }
        ),
        flush=True,
    )
    return model


def _resolve_voice_ref(raw_path: str) -> Path | None:
    raw_path = raw_path.strip()
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if not path.exists():
        return None
    return path.resolve()


def _generate(model, text: str, output_path: str, voice_ref: str | None = None) -> dict:
    """Generate audio and save to output_path. Returns result dict."""
    import torchaudio

    # Suppress stdout from tqdm progress bars and library prints
    real_stdout = _suppress_stdout()
    try:
        t0 = time.monotonic()
        kwargs = {}
        resolved_voice_ref = _resolve_voice_ref(voice_ref or "")
        cached_voice_ref = getattr(model, "_jarvis_cached_voice_ref", None)
        if resolved_voice_ref is not None and str(resolved_voice_ref) != cached_voice_ref:
            kwargs["audio_prompt_path"] = voice_ref
        wav = model.generate(text, **kwargs)
        gen_time = time.monotonic() - t0
        torchaudio.save(output_path, wav, model.sr)
    finally:
        sys.stdout = real_stdout
    return {"ok": True, "output_path": output_path, "gen_time": round(gen_time, 1)}


def serve():
    """Persistent mode: read JSON requests from stdin, respond on stdout."""
    # Suppress all warnings/logging to stderr so they don't interfere
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)

    model = _load_model()

    default_voice_ref = os.environ.get("CHATTERBOX_VOICE_REF", "")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text", "").strip()
            output_path = req.get("output_path", "")
            voice_ref = req.get("voice_ref", default_voice_ref) or None
            if not text or not output_path:
                result = {"ok": False, "error": "missing text or output_path"}
            else:
                result = _generate(model, text, output_path, voice_ref)
        except json.JSONDecodeError as e:
            result = {"ok": False, "error": f"invalid JSON: {e}"}
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        print(json.dumps(result), flush=True)


def oneshot(output_path: str, text: str) -> int:
    """One-shot mode."""
    if not text.strip():
        print("Empty text, skipping", file=sys.stderr)
        return 1

    try:
        import os
        import torch
        import torchaudio
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        t0 = time.monotonic()
        device = "cpu"
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        load_time = time.monotonic() - t0
        print(f"Model loaded in {load_time:.1f}s (device={device})", file=sys.stderr)

        t0 = time.monotonic()
        kwargs = {}
        voice_ref = os.environ.get("CHATTERBOX_VOICE_REF", "")
        if _resolve_voice_ref(voice_ref) is not None:
            kwargs["audio_prompt_path"] = voice_ref
        wav = model.generate(text, **kwargs)
        gen_time = time.monotonic() - t0
        print(f"Audio generated in {gen_time:.1f}s", file=sys.stderr)

        torchaudio.save(output_path, wav, model.sr)
        print(f"Saved to {output_path}", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"TTS worker error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve()
        return 0

    if len(sys.argv) >= 3:
        return oneshot(sys.argv[1], sys.argv[2])

    print("Usage:", file=sys.stderr)
    print("  tts_worker.py --serve              # persistent mode", file=sys.stderr)
    print("  tts_worker.py <output_path> <text>  # one-shot mode", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
