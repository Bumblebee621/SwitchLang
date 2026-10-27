"""Exercise the Win32 query logic without loading user32 on Linux."""
import ast
import ctypes
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


def windows_function(name, **namespace):
    tree = ast.parse((Path(__file__).parents[1] / 'core/os_windows.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(ctypes=ctypes, wintypes=wintypes)
    scope.update(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'os_windows.py', 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('enabled', [True, False])
def test_per_window_preference(enabled):
    def query(action, param, output, flags):
        assert (action, param, flags) == (0x104E, 0, 0)
        output._obj.value = enabled
        return 1
    query = windows_function('get_per_window_layout_enabled',
                             _user32=SimpleNamespace(SystemParametersInfoW=query))
    assert query() is enabled


def test_failed_preference_query_is_unknown():
    query = windows_function('get_per_window_layout_enabled',
                             _user32=SimpleNamespace(SystemParametersInfoW=mock.Mock(return_value=0)))
    assert query() is None


def test_layout_switch_timeout_is_reported():
    toggle = windows_function('toggle_layout',
        user32=SimpleNamespace(GetForegroundWindow=lambda: 1,
            PostMessageW=mock.Mock(return_value=1), GetKeyboardLayout=lambda tid: 0x0409),
        _resolve_hkls=lambda: None, _cached_hkl_en=0x0409, _cached_hkl_he=0x040D,
        _get_active_thread_id=lambda hwnd: 10, WM_INPUTLANGCHANGEREQUEST=0x50,
        time=SimpleNamespace(sleep=lambda duration: None))
    with pytest.raises(TimeoutError):
        toggle('he')


def test_focus_events_advance_revision_even_when_returning_to_original_window():
    callback = windows_function('_focus_event_callback', _focus_generation=0)
    callback(0, 3, 2, 0, 0, 0, 0)
    callback(0, 3, 1, 0, 0, 0, 0)
    assert callback.__globals__['_focus_generation'] == 2


def test_context_sampling_includes_native_focus_revision():
    identity = windows_function('_input_identity', _focus_generation=42,
        GUITHREADINFO=lambda **kw: SimpleNamespace(hwndFocus=10),
        ctypes=SimpleNamespace(sizeof=lambda t: 1, byref=lambda v: v),
        user32=SimpleNamespace(GetForegroundWindow=lambda: 1,
                              GetGUIThreadInfo=lambda *args: True))
    assert identity() == (1, (10, 42))
