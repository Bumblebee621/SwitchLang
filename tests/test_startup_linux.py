"""test_startup_linux.py — Linux autostart entry is written and removed."""
import os
import sys
import tempfile
import unittest
from unittest import mock


@unittest.skipIf(sys.platform == 'win32', 'Linux autostart branch')
class AutostartTest(unittest.TestCase):
    def test_toggle(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {'HOME': home}):
            sys.modules.pop('core.startup', None)
            from core import startup
            self.assertFalse(startup.is_startup_enabled())
            self.assertTrue(startup.set_startup_enabled(True))
            self.assertTrue(startup.is_startup_enabled())
            with open(startup._AUTOSTART, encoding='utf-8') as f:
                self.assertIn('Exec=', f.read())
            self.assertTrue(startup.set_startup_enabled(False))
            self.assertFalse(startup.is_startup_enabled())
        sys.modules.pop('core.startup', None)


if __name__ == '__main__':
    unittest.main()
