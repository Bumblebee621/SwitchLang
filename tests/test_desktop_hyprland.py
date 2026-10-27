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
        own = dict(OWN, active_layout_index=0)
        with self._ipc([MAIN, own]):
            desktop_hyprland.set_layout('en')
        self.assertIn('switchxkblayout all 0', self.sent)

    def test_layout_timeout_is_reported(self):
        with self._ipc([MAIN, OWN]), mock.patch.object(desktop_hyprland.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                desktop_hyprland.set_layout('en')

    def test_no_focused_window(self):
        with self._ipc([MAIN]):
            self.assertEqual(desktop_hyprland.active_window(), (None, 0))

    def test_prepare_raises_without_hebrew_layout(self):
        us_only = dict(MAIN, layout='us')
        with self._ipc([us_only]):
            with self.assertRaises(RuntimeError):
                desktop_hyprland.prepare()

    def test_prepare_passes_with_us_and_il(self):
        with self._ipc([MAIN]):
            desktop_hyprland.prepare()

    def test_focus_event_history_survives_return_to_original_window(self):
        stream = mock.Mock()
        stream.recv.side_effect = [b'activewindowv2>>0x2\nactivewindowv2>>0x1\n', BlockingIOError]
        with mock.patch.object(desktop_hyprland, '_focus_socket', stream, create=True), \
                mock.patch.object(desktop_hyprland, '_focus_generation', 0, create=True):
            desktop_hyprland._drain_focus_events()
            self.assertEqual(desktop_hyprland._focus_generation, 2)

    def test_partial_focus_event_waits_for_remaining_bytes(self):
        stream = mock.Mock()
        stream.recv.side_effect = [b'activewindowv2>', BlockingIOError, b'>0x2\n', BlockingIOError]
        with mock.patch.object(desktop_hyprland, '_focus_socket', stream, create=True), \
                mock.patch.object(desktop_hyprland, '_focus_pending', b'', create=True), \
                mock.patch.object(desktop_hyprland, '_focus_generation', 0, create=True):
            desktop_hyprland._drain_focus_events()
            self.assertEqual(desktop_hyprland._focus_generation, 0)
            desktop_hyprland._drain_focus_events()
            self.assertEqual(desktop_hyprland._focus_generation, 1)


if __name__ == '__main__':
    unittest.main()
