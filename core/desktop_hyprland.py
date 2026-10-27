"""
desktop_hyprland.py — active layout and focused window through Hyprland's IPC socket.
"""

import json
import os
import socket
import threading
import time

from core.keymap import XKB_LAYOUT_LANG

_SOCKET = os.path.join(os.environ.get('XDG_RUNTIME_DIR', ''), 'hypr',
                       os.environ.get('HYPRLAND_INSTANCE_SIGNATURE', ''), '.socket.sock')

# Hyprland's name for os_linux's virtual keyboard. Every keystroke goes through
# it, so its layout is the one that matters.
_OWN_KEYBOARD = 'switchlang-virtual-keyboard'
_focus_socket = None
_focus_pending = b''
_focus_generation = 0
_focus_lock = threading.Lock()


def _drain_focus_events():
    """Remember every observed focus transition, including A -> B -> A."""
    global _focus_socket, _focus_pending, _focus_generation
    if _focus_socket is None:
        stream = socket.socket(socket.AF_UNIX)
        try:
            stream.settimeout(0.02)
            stream.connect(os.path.join(os.path.dirname(_SOCKET), '.socket2.sock'))
            stream.setblocking(False)
        except OSError:
            stream.close()
            return
        _focus_socket = stream
        _focus_generation += 1
    try:
        while True:
            chunk = _focus_socket.recv(65536)
            if not chunk:
                raise ConnectionError('Hyprland event stream closed')
            _focus_pending += chunk
            while b'\n' in _focus_pending:
                line, _focus_pending = _focus_pending.split(b'\n', 1)
                if line.startswith(b'activewindowv2>>'):
                    _focus_generation += 1
    except BlockingIOError:
        pass
    except OSError:
        _focus_socket.close()
        _focus_socket = None
        _focus_pending = b''
        _focus_generation += 1


def input_identity():
    with _focus_lock:
        _drain_focus_events()
        if _focus_socket is None:
            return None, None
        return active_window()[0], _focus_generation


def stop_context_tracking():
    global _focus_socket, _focus_pending
    with _focus_lock:
        if _focus_socket is not None:
            _focus_socket.close()
        _focus_socket = None
        _focus_pending = b''


def _ipc(command):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(0.02)  # also queried before keys: a stalled desktop must fail open promptly
        s.connect(_SOCKET)
        s.sendall(command.encode())
        reply = b''
        while chunk := s.recv(65536):
            reply += chunk
    return reply.decode()


def _keyboard():
    keyboards = json.loads(_ipc('j/devices'))['keyboards']
    matches = [k for k in keyboards if k['name'] == _OWN_KEYBOARD] or [k for k in keyboards if k['main']]
    if not matches:
        raise RuntimeError('Hyprland reports no keyboard')
    return matches[0]


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
    raise TimeoutError(f'Hyprland layout did not switch to {target}')


def prepare():
    """Raise RuntimeError at startup if the keyboard lacks the en/he layouts."""
    if not {'en', 'he'} <= set(_langs(_keyboard())):
        raise RuntimeError('SwitchLang needs both "us" and "il" in Hyprland\'s input:kb_layout.')


def active_window():
    """(window address, pid) of the focused window; (None, 0) if nothing has focus."""
    window = json.loads(_ipc('j/activewindow'))
    return window.get('address'), window.get('pid', 0)
