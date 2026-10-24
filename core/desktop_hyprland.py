"""
desktop_hyprland.py — active layout and focused window through Hyprland's IPC socket.
"""

import json
import os
import socket
import time

from core.keymap import XKB_LAYOUT_LANG

_SOCKET = os.path.join(os.environ.get('XDG_RUNTIME_DIR', ''), 'hypr',
                       os.environ.get('HYPRLAND_INSTANCE_SIGNATURE', ''), '.socket.sock')

# Hyprland's name for os_linux's virtual keyboard. Every keystroke goes through
# it, so its layout is the one that matters.
_OWN_KEYBOARD = 'switchlang-virtual-keyboard'


def _ipc(command):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(_SOCKET)
        s.sendall(command.encode())
        reply = b''
        while chunk := s.recv(65536):
            reply += chunk
    return reply.decode()


def _keyboard():
    keyboards = json.loads(_ipc('j/devices'))['keyboards']
    own = [k for k in keyboards if k['name'] == _OWN_KEYBOARD]
    return (own or [k for k in keyboards if k['main']])[0]


def _langs(keyboard):
    return [XKB_LAYOUT_LANG.get(code) for code in keyboard['layout'].split(',')]


def get_layout():
    keyboard = _keyboard()
    return _langs(keyboard)[keyboard['active_layout_index']] or 'unknown'


def set_layout(target):
    _ipc(f'switchxkblayout all {_langs(_keyboard()).index(target)}')
    for _ in range(10):  # wait up to 100 ms for it to apply, like the Windows path
        if get_layout() == target:
            return
        time.sleep(0.01)


def prepare():
    """Raise RuntimeError at startup if the keyboard lacks the en/he layouts."""
    if not {'en', 'he'} <= set(_langs(_keyboard())):
        raise RuntimeError('SwitchLang needs both "us" and "il" in Hyprland\'s input:kb_layout.')


def active_window():
    """(window address, pid) of the focused window; (None, 0) if nothing has focus."""
    window = json.loads(_ipc('j/activewindow'))
    return window.get('address'), window.get('pid', 0)
