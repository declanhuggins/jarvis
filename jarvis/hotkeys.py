"""Global hotkeys for Jarvis."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)
_CTRL_MASK = 0x40000
_SHIFT_MASK = 0x20000


class GlobalHotkeys:
    """Listen for global keyboard shortcuts.

    Current bindings:
    - Ctrl+Space: force Jarvis into command listening mode
    - Ctrl+Shift+M: toggle Jarvis microphone mute
    """

    def __init__(
        self,
        on_force_listen: Callable[[], None],
        on_toggle_mute: Callable[[], None],
    ):
        self._on_force_listen = on_force_listen
        self._on_toggle_mute = on_toggle_mute
        self._listener = None
        self._pressed: set[str] = set()
        self._last_force_listen_at = 0.0
        self._last_toggle_mute_at = 0.0

    def start(self) -> bool:
        """Start the global hotkey listener."""
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning("Global hotkeys unavailable: pynput is not installed")
            return False

        self._keyboard = keyboard
        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                intercept=self._intercept,
            )
            self._listener.daemon = True
            self._listener.start()
            logger.info("Global hotkeys enabled: Ctrl+Space, Ctrl+Shift+M")
            return True
        except Exception as e:
            logger.warning("Failed to start global hotkeys: %s", e)
            return False

    def stop(self) -> None:
        """Stop the global hotkey listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> None:
        normalized = self._normalize_key(key)
        if normalized is None:
            return

        self._pressed.add(normalized)
        now = time.monotonic()

        if {"ctrl", "space"} <= self._pressed and now - self._last_force_listen_at > 0.5:
            self._last_force_listen_at = now
            self._on_force_listen()

        if {"ctrl", "shift", "m"} <= self._pressed and now - self._last_toggle_mute_at > 0.5:
            self._last_toggle_mute_at = now
            self._on_toggle_mute()

    def _on_release(self, key) -> None:
        normalized = self._normalize_key(key)
        if normalized is not None:
            self._pressed.discard(normalized)

    def _normalize_key(self, key) -> str | None:
        keyboard = self._keyboard
        if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
            return "ctrl"
        if key in {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}:
            return "shift"
        if key == keyboard.Key.space:
            return "space"
        char = getattr(key, "char", None)
        if isinstance(char, str) and char.lower() == "m":
            return "m"
        return None

    def _intercept(self, event_type, event):
        """Suppress Jarvis hotkey events so they do not reach the system."""
        try:
            key = self._listener._event_to_key(event)
        except Exception:
            return event

        normalized = self._normalize_key(key)
        ctrl_down = "ctrl" in self._pressed or self._event_has_ctrl_modifier(event)
        shift_down = "shift" in self._pressed or self._event_has_shift_modifier(event)
        if normalized == "space" and ctrl_down:
            return None
        if normalized == "m" and ctrl_down and shift_down:
            return None
        return event

    def _event_has_ctrl_modifier(self, event) -> bool:
        """Check the raw macOS event for a held Control modifier."""
        try:
            from Quartz import CGEventGetFlags, kCGEventFlagMaskControl

            return bool(CGEventGetFlags(event) & kCGEventFlagMaskControl)
        except Exception:
            return False

    def _event_has_shift_modifier(self, event) -> bool:
        """Check the raw macOS event for a held Shift modifier."""
        try:
            from Quartz import CGEventGetFlags, kCGEventFlagMaskShift

            return bool(CGEventGetFlags(event) & kCGEventFlagMaskShift)
        except Exception:
            return False
