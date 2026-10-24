"""
test_switcher_layout_failure.py — a failed layout switch still retypes the word.
"""

import unittest
from unittest import mock

from core import switcher


class TestLayoutSwitchFailure(unittest.TestCase):

    def test_original_keys_retyped_when_toggle_fails(self):
        with mock.patch.object(switcher, 'toggle_layout', side_effect=RuntimeError('ipc')), \
                mock.patch.object(switcher, 'get_current_layout', return_value='en'), \
                mock.patch.object(switcher, 'send_backspaces'), \
                mock.patch.object(switcher, 'send_string_as_keys') as typed, \
                mock.patch.object(switcher, 'send_vk_key'), \
                mock.patch.object(switcher, 'toggle_caps_lock'), \
                mock.patch.object(switcher.time, 'sleep'), \
                self.assertLogs('switchlang.switcher', 'ERROR'):
            switcher.execute_switch('akuo', 'שלום', 'he')
        typed.assert_called_with('שלום', 'he')


if __name__ == '__main__':
    unittest.main()
