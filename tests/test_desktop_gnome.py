"""
test_desktop_gnome.py — gsettings parsing and shortcut-driven switching on GNOME.
"""

import unittest
from unittest import mock

from core import desktop_gnome, os_linux


class TestGnome(unittest.TestCase):

    def setUp(self):
        self.settings = {
            'mru-sources': [('xkb', 'il'), ('xkb', 'us')],
            'sources': [('xkb', 'us'), ('xkb', 'il')],
            'switch-input-source': ['<Super>space', 'XF86Keyboard'],
        }
        patcher = mock.patch.object(desktop_gnome, '_gsettings', lambda schema, key: self.settings[key])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parse_gvariant(self):
        self.assertEqual(desktop_gnome._parse('@a(ss) []'), [])
        self.assertEqual(desktop_gnome._parse("[('xkb', 'us+intl')]"), [('xkb', 'us+intl')])

    def test_layout_is_most_recent_source(self):
        self.assertEqual(desktop_gnome.get_layout(), 'he')

    def test_layout_falls_back_to_sources_before_first_switch(self):
        self.settings['mru-sources'] = []
        self.assertEqual(desktop_gnome.get_layout(), 'en')

    def test_shortcut_keys(self):
        self.assertEqual(desktop_gnome._shortcut_keys(), [125, 57])  # Super, Space

    def test_set_layout_presses_shortcut_until_switched(self):
        pressed = []

        def emit(*keys):
            pressed.extend(keys)
            self.settings['mru-sources'].reverse()  # GNOME switches on the shortcut
        with mock.patch.object(os_linux, '_emit', emit):
            desktop_gnome.set_layout('en')
        self.assertEqual(pressed, [(125, 1), (57, 1), (57, 0), (125, 0)])
        self.assertEqual(desktop_gnome.get_layout(), 'en')


if __name__ == '__main__':
    unittest.main()
