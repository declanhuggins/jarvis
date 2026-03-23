"""Tests for global hotkey detection."""

from jarvis.hotkeys import GlobalHotkeys


class _FakeKey:
    ctrl = "ctrl"
    ctrl_l = "ctrl_l"
    ctrl_r = "ctrl_r"
    shift = "shift"
    shift_l = "shift_l"
    shift_r = "shift_r"
    space = "space"


class _FakeKeyCode:
    def __init__(self, vk=None, char=None):
        self.vk = vk
        self.char = char


class _FakeListener:
    def _event_to_key(self, event):
        return event


def test_ctrl_space_triggers_force_listen():
    events = []
    hotkeys = GlobalHotkeys(lambda: events.append("listen"), lambda: events.append("mute"))
    hotkeys._keyboard = type("Keyboard", (), {"Key": _FakeKey})

    hotkeys._on_press(_FakeKey.ctrl)
    hotkeys._on_press(_FakeKey.space)

    assert events == ["listen"]


def test_ctrl_shift_m_triggers_toggle_mute():
    events = []
    hotkeys = GlobalHotkeys(lambda: events.append("listen"), lambda: events.append("mute"))
    hotkeys._keyboard = type("Keyboard", (), {"Key": _FakeKey})

    hotkeys._on_press(_FakeKey.ctrl)
    hotkeys._on_press(_FakeKey.shift)
    hotkeys._on_press(_FakeKeyCode(char="m"))

    assert events == ["mute"]


def test_intercept_swallows_ctrl_space():
    hotkeys = GlobalHotkeys(lambda: None, lambda: None)
    hotkeys._keyboard = type("Keyboard", (), {"Key": _FakeKey})
    hotkeys._listener = _FakeListener()
    hotkeys._pressed = {"ctrl", "space"}

    assert hotkeys._intercept(None, _FakeKey.space) is None


def test_intercept_swallows_ctrl_shift_m():
    hotkeys = GlobalHotkeys(lambda: None, lambda: None)
    hotkeys._keyboard = type("Keyboard", (), {"Key": _FakeKey})
    hotkeys._listener = _FakeListener()
    hotkeys._pressed = {"ctrl", "shift", "m"}

    assert hotkeys._intercept(None, _FakeKeyCode(char="m")) is None
