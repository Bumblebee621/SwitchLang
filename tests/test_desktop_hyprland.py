"""
test_desktop_hyprland.py — layout and window parsing of Hyprland's IPC replies.
"""

import json
import unittest
from unittest import mock

from core import desktop_hyprland

MAIN = {'name': 'keyd-virtual-keyboard', 'layout': 'us,il', 'active_layout_index': 0, 'main': True}
OWN = {'name': 'switchlang-virtual-keyboard', 'layout': 'us,il', 'active_layout_index': 1, 'main': False}


class TestHyprland(unittest.TestCase):

    def _ipc(self, keyboards, window=None):
        self.sent = []

        def fake(command):
            self.sent.append(command)
            replies = {'j/devices': {'keyboards': keyboards}, 'j/activewindow': window or {}}
            return json.dumps(replies.get(command, 'ok'))
        return mock.patch.object(desktop_hyprland, '_ipc', fake)

    def test_layout_comes_from_our_keyboard(self):
        with self._ipc([MAIN, OWN]):
            self.assertEqual(desktop_hyprland.get_layout(), 'he')

    def test_layout_falls_back_to_main_keyboard(self):
        with self._ipc([MAIN]):
            self.assertEqual(desktop_hyprland.get_layout(), 'en')

    def test_set_layout_switches_every_keyboard(self):
        with self._ipc([MAIN, OWN]):
            desktop_hyprland.set_layout('en')
        self.assertIn('switchxkblayout all 0', self.sent)

    def test_no_focused_window(self):
        with self._ipc([MAIN]):
            self.assertEqual(desktop_hyprland.active_window(), (None, 0))


if __name__ == '__main__':
    unittest.main()
