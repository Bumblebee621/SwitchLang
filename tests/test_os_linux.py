"""
test_os_linux.py — key tables and the forward-or-block decision of the Linux input layer.
"""

import os
import sys
import unittest
from unittest import mock

from core import os_linux
from core.keymap import CHAR_TO_VK_EN, CHAR_TO_VK_HE


class TestHostEnv(unittest.TestCase):

    def test_restores_original_library_path(self):
        with mock.patch.dict(os.environ, {'LD_LIBRARY_PATH': '/tmp/_MEI1', 'LD_LIBRARY_PATH_ORIG': '/usr/lib'}):
            env = os_linux.host_env()
        self.assertEqual(env['LD_LIBRARY_PATH'], '/usr/lib')
        self.assertNotIn('LD_LIBRARY_PATH_ORIG', env)

    def test_frozen_without_original_drops_library_path(self):
        with mock.patch.dict(os.environ, {'LD_LIBRARY_PATH': '/tmp/_MEI1'}), \
                mock.patch.object(sys, 'frozen', True, create=True):
            os.environ.pop('LD_LIBRARY_PATH_ORIG', None)
            env = os_linux.host_env()
        self.assertNotIn('LD_LIBRARY_PATH', env)


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
        for vk in os_linux._EVDEV_TO_VK.values():
            self.assertEqual(os_linux._EVDEV_TO_VK[os_linux._VK_TO_EVDEV[vk]], vk)

    def test_keypad_enter_maps_to_return_but_main_enter_wins_reverse(self):
        self.assertEqual(os_linux._EVDEV_TO_VK[96], 0x0D)
        self.assertEqual(os_linux._VK_TO_EVDEV[0x0D], 28)

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


class TestCapsLockLed(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(os_linux, '_caps_lock', False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_led_events_drive_caps_lock_state(self):
        os_linux._on_led_event(1, 1)  # LED_CAPSL on
        self.assertTrue(os_linux.is_caps_lock_on())
        os_linux._on_led_event(1, 0)  # LED_CAPSL off
        self.assertFalse(os_linux.is_caps_lock_on())

    def test_led_event_mirrored_to_each_grabbed_keyboard(self):
        with mock.patch.dict(os_linux._keyboards, {999: '/dev/input/fake'}), \
                mock.patch.object(os_linux.os, 'write') as write:
            os_linux._on_led_event(1, 1)
        write.assert_called_once_with(999, os_linux._EV.pack(0, 0, os_linux.EV_LED, 1, 1) + os_linux._SYN)


if __name__ == '__main__':
    unittest.main()
