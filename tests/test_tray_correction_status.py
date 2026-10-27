"""Test status presentation without requiring a running desktop or Qt display."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def update_status(tray):
    tree = ast.parse((Path(__file__).parents[1] / 'ui/tray.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SystemTrayApp')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_correction_status')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'tray.py', 'exec'), scope)
    scope['_update_correction_status'](tray)


def test_unavailable_layout_is_visible_in_tray():
    tray = SimpleNamespace(hook_manager=SimpleNamespace(enabled=True, correction_status='Layout unavailable'),
                           correction_status_action=mock.Mock(), setToolTip=mock.Mock())
    update_status(tray)
    tray.correction_status_action.setVisible.assert_called_with(True)
    tray.correction_status_action.setText.assert_called_with('Layout unavailable')
    assert 'Layout unavailable' in tray.setToolTip.call_args.args[0]


def test_available_or_disabled_correction_hides_pause_message():
    for enabled, reason in [(True, ''), (False, 'Layout unavailable')]:
        tray = SimpleNamespace(hook_manager=SimpleNamespace(enabled=enabled, correction_status=reason),
                               correction_status_action=mock.Mock(), setToolTip=mock.Mock())
        update_status(tray)
        tray.correction_status_action.setVisible.assert_called_with(False)
