"""
test_desktop_gnome.py — gsettings parsing and shortcut-driven switching on GNOME.
"""

import unittest
from unittest import mock

from core import desktop_gnome, os_linux


class TestGnome(unittest.TestCase):

    def setUp(self):
        self.settings = {
            'per-window': False,
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

    def test_per_window_restoration_does_not_trust_mru(self):
        self.settings['per-window'] = True
        self.assertIs(desktop_gnome.get_per_window_layout_enabled(), True)
        self.assertEqual(desktop_gnome.get_layout(), 'unknown')

    def test_unknown_preference_does_not_trust_mru(self):
        del self.settings['per-window']
        self.assertIsNone(desktop_gnome.get_per_window_layout_enabled())
        self.assertEqual(desktop_gnome.get_layout(), 'unknown')

    def test_parse_boolean(self):
        self.assertIs(desktop_gnome._parse('true'), True)
        self.assertIs(desktop_gnome._parse('false'), False)

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

    def test_shift_alt_accelerator(self):
        self.settings['switch-input-source'] = ['<Shift>Alt_L']
        self.assertEqual(desktop_gnome._shortcut_keys(), [42, 56])

    def test_unmappable_first_accelerator_falls_through(self):
        self.settings['switch-input-source'] = ['XF86Keyboard', '<Super>space']
        self.assertEqual(desktop_gnome._shortcut_keys(), [125, 57])

    def test_mod4_accelerator(self):
        self.settings['switch-input-source'] = ['<Mod4>space']
        self.assertEqual(desktop_gnome._shortcut_keys(), [125, 57])

    def test_no_usable_shortcut_raises_runtime_error(self):
        self.settings['switch-input-source'] = []
        with self.assertRaises(RuntimeError):
            desktop_gnome._shortcut_keys()

    def test_layout_timeout_is_reported(self):
        with mock.patch.object(os_linux, '_emit'), mock.patch.object(desktop_gnome.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                desktop_gnome.set_layout('en')

    def test_prepare_passes_with_us_and_il(self):
        desktop_gnome.prepare()

    def test_prepare_raises_without_shortcut(self):
        self.settings['switch-input-source'] = []
        with self.assertRaises(RuntimeError):
            desktop_gnome.prepare()

    def test_prepare_raises_without_hebrew_source(self):
        self.settings['sources'] = [('xkb', 'us')]
        with self.assertRaises(RuntimeError):
            desktop_gnome.prepare()


if __name__ == '__main__':
    unittest.main()
