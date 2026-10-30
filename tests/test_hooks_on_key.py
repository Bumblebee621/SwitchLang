"""
test_hooks_on_key.py — HookManager's OS-neutral key entry point.
"""

import unittest
from unittest import mock

from core.hooks import HookManager, VK_A, VK_LCONTROL, VK_LSHIFT, VK_LWIN, _WordEntry
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

    def test_navigation_resets_typing_context_with_modifiers(self):
        for vk in (0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28):
            for modifiers in ((), (VK_LSHIFT,), (VK_LCONTROL,),
                              (VK_LCONTROL, VK_LSHIFT), (VK_LWIN,)):
                with self.subTest(vk=vk, modifiers=modifiers):
                    hm = HookManager(None, SensitivityManager(), None, {})
                    hm._on_key(VK_A, True)
                    hm.history_deque.append(_WordEntry('old', 'old', ' ', False, True, 4.0))
                    hm.sensitivity.on_word_complete()
                    hm._consecutive_midword_hits = 1
                    for modifier in modifiers:
                        hm._on_key(modifier, True)

                    self.assertFalse(hm._on_key(vk, True))

                    self.assertEqual(hm.buffer_active, '')
                    self.assertEqual(hm.buffer_shadow, '')
                    self.assertFalse(hm.history_deque)
                    self.assertEqual(hm._evidence(), 0.0)
                    self.assertEqual(hm.sensitivity.delta, hm.sensitivity.baseline_delta)
                    self.assertEqual(hm._consecutive_midword_hits, 0)
                    self.assertEqual(hm._letter_shift_history, [])

    def test_navigation_release_preserves_new_typing_context(self):
        self.hm._on_key(0x24, True)  # Home
        self.hm._on_key(VK_A, True)

        self.assertFalse(self.hm._on_key(0x24, False))

        self.assertEqual(self.hm.buffer_active, 'a')
        self.assertEqual(self.hm.buffer_shadow, 'ש')

    def test_navigation_invalidates_correction_and_passes_new_keys(self):
        self.hm._on_key(VK_A, True)
        self.hm.is_correcting = True
        self.hm._correction_generation = self.hm._context_generation

        self.assertFalse(self.hm._on_key(0x24, True))  # Home

        self.assertNotEqual(self.hm._context_generation, self.hm._correction_generation)
        self.assertFalse(self.hm._on_key(VK_A, True))
        self.assertFalse(self.hm.pending_queue)
        self.assertEqual(self.hm.buffer_active, '')

    def test_super_shortcut_is_not_buffered(self):
        self.hm._on_key(VK_LWIN, True)
        self.hm._on_key(VK_A, True)
        self.assertEqual(self.hm.buffer_active, '')

    def test_key_up_never_blocks(self):
        self.assertFalse(self.hm._on_key(VK_A, False))


if __name__ == '__main__':
    unittest.main()
