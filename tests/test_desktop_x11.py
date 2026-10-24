"""
test_desktop_x11.py — XKB group switching and EWMH lookups against a real X server.
Run under Xvfb with SWITCHLANG_X11_TEST=1 (see the plan, Task 4).
"""

import os
import unittest


@unittest.skipUnless(os.environ.get('SWITCHLANG_X11_TEST'), 'needs Xvfb with us,il layouts')
class TestX11(unittest.TestCase):

    def test_switch_round_trip(self):
        from core import desktop_x11
        desktop_x11.set_layout('he')
        self.assertEqual(desktop_x11.get_layout(), 'he')
        desktop_x11.set_layout('en')
        self.assertEqual(desktop_x11.get_layout(), 'en')

    def test_prepare_passes_with_us_and_il(self):
        from core import desktop_x11
        desktop_x11.prepare()

    def test_no_window_manager_means_no_active_window(self):
        from core import desktop_x11
        self.assertEqual(desktop_x11.active_window(), (0, 0))


if __name__ == '__main__':
    unittest.main()
