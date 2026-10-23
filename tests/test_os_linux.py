"""
test_os_linux.py — key tables and the forward-or-block decision of the Linux input layer.
"""

import unittest
from unittest import mock

from core import os_linux
from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE


class TestKeyTables(unittest.TestCase):

    def test_every_mapped_char_is_typeable(self):
        for layout, table in (('en', CHAR_TO_VK_EN), ('he', CHAR_TO_VK_HE)):
            for ch in table:
                self.assertEqual(len(list(os_linux._key_events(ch, layout))), 1, (layout, ch))

    def test_shifted_char(self):
        self.assertEqual(list(os_linux._key_events('A', 'en')), [(30, True)])

    def test_char_missing_from_layout_uses_same_physical_key(self):
        self.assertEqual(list(os_linux._key_events('ש', 'en')), [(30, False)])

    def test_vk_map_round_trips(self):
        for code, vk in os_linux._EVDEV_TO_VK.items():
            self.assertEqual(os_linux._VK_TO_EVDEV[vk], code)

    def test_sysfs_bitmap_word_order(self):
        bits = os_linux._bitmap('1 0 40000000')  # most significant 64-bit word first
        self.assertTrue(bits >> 30 & 1)
        self.assertTrue(bits >> 128 & 1)


class TestKeyboardEvent(unittest.TestCase):

    def setUp(self):
        self.emitted = []
        patcher = mock.patch.object(os_linux, '_emit', lambda *keys: self.emitted.extend(keys))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _press(self, on_key):
        with mock.patch.object(os_linux, '_on_key', on_key):
            os_linux._on_keyboard_event(30, 1)

    def test_passed_key_is_forwarded(self):
        self._press(lambda vk, down: False)
        self.assertEqual(self.emitted, [(30, 1)])

    def test_blocked_key_is_dropped(self):
        self._press(lambda vk, down: True)
        self.assertEqual(self.emitted, [])

    def test_callback_error_fails_open(self):
        def boom(vk, down):
            raise RuntimeError('boom')
        self._press(boom)
        self.assertEqual(self.emitted, [(30, 1)])


if __name__ == '__main__':
    unittest.main()
