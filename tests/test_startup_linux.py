"""test_startup_linux.py — Linux autostart entry is written and removed."""
import os
import sys
import tempfile
import unittest
from unittest import mock

from core import startup


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux autostart branch')
class AutostartTest(unittest.TestCase):
    def test_toggle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'switchlang.desktop')
            with mock.patch.object(startup, '_AUTOSTART', path):
                self.assertFalse(startup.is_startup_enabled())
                self.assertTrue(startup.set_startup_enabled(True))
                self.assertTrue(startup.is_startup_enabled())
                with open(path, encoding='utf-8') as f:
                    self.assertIn('Exec=', f.read())
                self.assertTrue(startup.set_startup_enabled(False))
                self.assertFalse(startup.is_startup_enabled())

    def test_managed_startup_cannot_write_autostart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'switchlang.desktop')
            with mock.patch.object(startup, '_AUTOSTART', path):
                for value in ('0', '1'):
                    with mock.patch.dict(os.environ, SWITCHLANG_AUTOSTART_MANAGED=value):
                        self.assertTrue(startup.is_startup_managed())
                        self.assertEqual(startup.is_startup_enabled(), value == '1')
                        self.assertFalse(startup.set_startup_enabled(True))
                        self.assertFalse(startup.set_startup_enabled(False))
                        self.assertFalse(os.path.exists(path))

    def test_packaged_launcher_survives_store_path_changes(self):
        with mock.patch.dict(os.environ, SWITCHLANG_EXECUTABLE='switchlang'):
            self.assertEqual(startup.get_current_app_command(), 'switchlang')


if __name__ == '__main__':
    unittest.main()
