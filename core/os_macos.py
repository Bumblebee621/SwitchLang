"""
os_macos.py — macOS input layer. Not implemented yet: same interface as
os_windows / os_linux, so main.py shows "not implemented" instead of crashing.
"""

__all__ = [
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
]

_TODO = 'macOS support is not implemented yet'


def start_hooks(on_key, on_click):
    """Quartz event tap (CGEventTapCreate) for keys and clicks; needs the Accessibility permission."""
    raise NotImplementedError(_TODO)


def stop_hooks():
    raise NotImplementedError(_TODO)


def is_caps_lock_on():
    """CGEventSourceFlagsState & kCGEventFlagMaskAlphaShift."""
    raise NotImplementedError(_TODO)


def is_password_field_active():
    """Secure input is on (IsSecureEventInputEnabled)."""
    raise NotImplementedError(_TODO)


def get_foreground_window():
    """NSWorkspace.frontmostApplication's pid."""
    raise NotImplementedError(_TODO)


def get_foreground_exe():
    raise NotImplementedError(_TODO)


def get_current_layout():
    """TISCopyCurrentKeyboardInputSource → 'en' / 'he'."""
    raise NotImplementedError(_TODO)


def toggle_layout(target_layout):
    """TISSelectInputSource."""
    raise NotImplementedError(_TODO)


def send_backspaces(count):
    """CGEventPost with kVK_Delete."""
    raise NotImplementedError(_TODO)


def toggle_caps_lock():
    raise NotImplementedError(_TODO)


def send_vk_key(vk, shift=False):
    raise NotImplementedError(_TODO)


def send_string_as_keys(text, layout):
    raise NotImplementedError(_TODO)


def native_key_to_vk(event):
    raise NotImplementedError(_TODO)
