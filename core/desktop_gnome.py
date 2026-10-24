"""
desktop_gnome.py — GNOME on Wayland or X11.

GNOME owns the layout: a direct XKB switch is undone, and Wayland exposes no
API. So the active layout is read from gsettings, and switching presses the
user's own "switch input source" shortcut (gswitch does the same:
runtime/gnome_wayland_layout_switch.go).
"""

import ast
import re
import subprocess
import time

from core.keymap import XKB_LAYOUT_LANG

_INPUT_SOURCES = 'org.gnome.desktop.input-sources'
_MODIFIERS = {'super': 125, 'shift': 42, 'control': 29, 'primary': 29, 'ctrl': 29, 'alt': 56}
_NAMED_KEYS = {'space': 57, 'tab': 15, 'escape': 1}


def _parse(text):
    """GVariant text from `gsettings get` as Python; strips type prefixes like '@a(ss) '."""
    return ast.literal_eval(re.sub(r'^@\S+ ', '', text.strip()))


def _gsettings(schema, key):
    # ponytail: one process per call (~5 ms) from the 100 ms poll; `gsettings monitor` could push changes.
    return _parse(subprocess.run(['gsettings', 'get', schema, key],
                                 capture_output=True, text=True, check=True).stdout)


def get_layout():
    # mru-sources is empty until the user's first switch; then sources[0] is active.
    sources = _gsettings(_INPUT_SOURCES, 'mru-sources') or _gsettings(_INPUT_SOURCES, 'sources')
    code = sources[0][1].split('+')[0] if sources else ''  # ('xkb', 'il') or ('xkb', 'us+intl')
    return XKB_LAYOUT_LANG.get(code, 'unknown')


def _shortcut_keys():
    """evdev codes of the first switch-input-source shortcut, e.g. '<Super>space' → [125, 57]."""
    from core.os_linux import _VK_TO_EVDEV  # os_linux imports this module; import at call time
    accel = _gsettings('org.gnome.desktop.wm.keybindings', 'switch-input-source')[0]
    key = re.sub(r'<\w+>', '', accel).lower()
    codes = [_MODIFIERS[mod.lower()] for mod in re.findall(r'<(\w+)>', accel)]
    codes.append(_NAMED_KEYS.get(key) or _VK_TO_EVDEV[ord(key.upper())])
    return codes


def set_layout(target):
    # ponytail: the shortcut flips to the previously used source, so this assumes
    # en/he are the two most recent; with 3+ sources it may land elsewhere.
    from core.os_linux import _emit
    keys = _shortcut_keys()
    for _ in range(2):  # gswitch also retries once; GNOME sometimes drops the first press
        if get_layout() == target:
            return
        _emit(*[(k, 1) for k in keys], *[(k, 0) for k in reversed(keys)])
        for _ in range(20):  # up to 1 s
            time.sleep(0.05)
            if get_layout() == target:
                return


def active_window():
    # ponytail: GNOME on Wayland exposes no focused window, so per-app blacklists
    # and window-change resets don't apply on GNOME; clicks and idle still reset.
    return None, 0
