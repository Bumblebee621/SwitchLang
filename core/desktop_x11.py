"""
desktop_x11.py — active layout (XKB group) and focused window (EWMH) through libX11.
"""

import ctypes
import ctypes.util
import subprocess
import time

from core.keymap import XKB_LAYOUT_LANG

_x11 = ctypes.CDLL(ctypes.util.find_library('X11') or 'libX11.so.6')
_x11.XInitThreads()  # the poll thread and the switch thread share the connection

# Xlib's default error handler exits the process. A window closing between two
# calls (BadWindow) is routine here, so ignore errors and read "no value".
_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda d, e: 0)
_x11.XSetErrorHandler(_ERROR_HANDLER)

_ulong_p = ctypes.POINTER(ctypes.c_ulong)
_x11.XkbOpenDisplay.restype = ctypes.c_void_p
_x11.XkbOpenDisplay.argtypes = [ctypes.c_char_p] + [ctypes.POINTER(ctypes.c_int)] * 5
_x11.XkbGetState.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
_x11.XkbLockGroup.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
_x11.XFlush.argtypes = [ctypes.c_void_p]
_x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
_x11.XDefaultRootWindow.restype = ctypes.c_ulong
_x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_x11.XInternAtom.restype = ctypes.c_ulong
_x11.XGetWindowProperty.argtypes = [
    ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_int,
    ctypes.c_ulong, _ulong_p, ctypes.POINTER(ctypes.c_int), _ulong_p, _ulong_p, ctypes.POINTER(_ulong_p)]
_x11.XFree.argtypes = [ctypes.c_void_p]

XKB_USE_CORE_KBD = 0x0100


class _XkbState(ctypes.Structure):
    """XkbStateRec. Only the effective group (first byte) is read; the rest is padding room."""
    _fields_ = [('group', ctypes.c_ubyte), ('rest', ctypes.c_ubyte * 31)]


_ints = [ctypes.c_int(v) for v in (0, 0, 1, 0, 0)]  # event, error, major (in/out), minor, reason
_display = _x11.XkbOpenDisplay(None, *[ctypes.byref(i) for i in _ints])
if not _display:
    raise RuntimeError('Cannot open the X display with XKB (is DISPLAY set?)')

_langs = None


def _layout_langs():
    """Languages of the configured XKB groups, e.g. ['en', 'he'] for "us,il"."""
    # ponytail: read once; after changing layouts in system settings, restart SwitchLang.
    global _langs
    if _langs is None:
        from core.os_linux import host_env  # os_linux imports this module; import at call time
        query = subprocess.run(['setxkbmap', '-query'], env=host_env(),
                               capture_output=True, text=True, check=True).stdout
        codes = next(line.split()[1] for line in query.splitlines() if line.startswith('layout:'))
        _langs = [XKB_LAYOUT_LANG.get(code) for code in codes.split(',')]
    return _langs


def get_layout():
    state = _XkbState()
    _x11.XkbGetState(_display, XKB_USE_CORE_KBD, ctypes.byref(state))
    langs = _layout_langs()
    return (langs[state.group] if state.group < len(langs) else None) or 'unknown'


def set_layout(target):
    _x11.XkbLockGroup(_display, XKB_USE_CORE_KBD, _layout_langs().index(target))
    _x11.XFlush(_display)
    for _ in range(10):  # wait up to 100 ms for it to apply, like the Windows path
        if get_layout() == target:
            return
        time.sleep(0.01)


def prepare():
    """Raise RuntimeError at startup if the XKB layouts lack en/he."""
    if not {'en', 'he'} <= set(_layout_langs()):
        raise RuntimeError('SwitchLang needs both "us" and "il" XKB layouts (setxkbmap -query shows the current ones).')


def _property(window, name):
    """First 32-bit item of a window property, or 0."""
    kind, fmt, count, after = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_ulong(), ctypes.c_ulong()
    data = _ulong_p()
    atom = _x11.XInternAtom(_display, name, False)
    status = _x11.XGetWindowProperty(
        _display, window, atom, 0, 1, False, 0,  # 0 = AnyPropertyType
        ctypes.byref(kind), ctypes.byref(fmt), ctypes.byref(count), ctypes.byref(after), ctypes.byref(data))
    value = data[0] if status == 0 and count.value else 0
    if data:
        _x11.XFree(data)
    return value


def active_window():
    """(window id, pid) of the focused window; (0, 0) if none."""
    window = _property(_x11.XDefaultRootWindow(_display), b'_NET_ACTIVE_WINDOW')
    return window, (_property(window, b'_NET_WM_PID') if window else 0)
