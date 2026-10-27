"""Exercise X11 event parsing without opening a display."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def test_focus_property_events_preserve_intermediate_transitions():
    tree = ast.parse((Path(__file__).parents[1] / 'core/desktop_x11.py').read_text())
    names = {'_XPropertyEvent', '_XEvent', '_drain_focus_events'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    def next_event(display, output):
        output._obj.property.type = 28
        output._obj.property.atom = 42
    scope = dict(ctypes=ctypes, _focus_generation=0, _active_window_atom=42, _display=None,
                 _x11=SimpleNamespace(XPending=mock.Mock(side_effect=[1, 1, 0]), XNextEvent=next_event))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'desktop_x11.py', 'exec'), scope)
    scope['_drain_focus_events']()
    assert scope['_focus_generation'] == 2


def test_failed_xkb_query_cannot_report_first_layout_as_active():
    tree = ast.parse((Path(__file__).parents[1] / 'core/desktop_x11.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_layout')
    scope = dict(ctypes=SimpleNamespace(byref=lambda v: v), _display=None, XKB_USE_CORE_KBD=0x100,
                 _XkbState=lambda: SimpleNamespace(group=0), _layout_langs=lambda: ['en', 'he'],
                 _x11=SimpleNamespace(XkbGetState=lambda *args: 1))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'desktop_x11.py', 'exec'), scope)
    assert scope['get_layout']() == 'unknown'
