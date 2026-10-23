"""
test_osapi.py — every OS module defines the full osapi interface.
"""

import ast
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / 'core'
INTERFACE = {
    'start_hooks', 'stop_hooks', 'is_caps_lock_on', 'is_password_field_active',
    'get_foreground_window', 'get_foreground_exe', 'get_current_layout', 'toggle_layout',
    'send_backspaces', 'toggle_caps_lock', 'send_vk_key', 'send_string_as_keys',
    'native_key_to_vk',
}


class TestInterface(unittest.TestCase):

    def test_every_os_module_defines_the_interface(self):
        for name in ('os_windows', 'os_linux', 'os_macos'):
            tree = ast.parse((CORE / f'{name}.py').read_text(encoding='utf-8'))
            defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            self.assertEqual(INTERFACE - defined, set(), name)


if __name__ == '__main__':
    unittest.main()
