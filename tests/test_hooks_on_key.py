"""
test_hooks_on_key.py — HookManager's OS-neutral key entry point.
"""

import unittest
from unittest import mock

from core.hooks import HookManager, VK_A, VK_LCONTROL, VK_LSHIFT
from core.sensitivity import SensitivityManager


class TestOnKey(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch('core.osapi.is_caps_lock_on', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Words here stay under 3 letters, so the engine is never consulted.
        self.hm = HookManager(engine=None, sensitivity=SensitivityManager(), blacklist=None, config={})

    def test_letters_fill_buffer_with_shift_tracking(self):
        self.hm._on_key(VK_LSHIFT, True)
        self.hm._on_key(VK_A, True)
        self.hm._on_key(VK_LSHIFT, False)
        self.hm._on_key(VK_A, True)
        self.assertEqual(self.hm.buffer_active, 'Aa')

    def test_ctrl_a_clears_buffer(self):
        self.hm._on_key(VK_A, True)
        self.hm._on_key(VK_LCONTROL, True)
        self.hm._on_key(VK_A, True)
        self.assertEqual(self.hm.buffer_active, '')

    def test_key_up_never_blocks(self):
        self.assertFalse(self.hm._on_key(VK_A, False))


if __name__ == '__main__':
    unittest.main()
