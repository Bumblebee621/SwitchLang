"""
sensitivity.py — Dynamic Δ threshold with Context Resumption Events.

Δ starts at baseline (high sensitivity) and decays (less sensitive)
as a continuous sentence lengthens. CREs reset Δ to baseline.
"""

import math
import time


class SensitivityManager:
    """Manages the dynamic decision boundary threshold Δ."""

    def __init__(self, baseline_delta=2.0, alpha=0.3):
        """Initialize sensitivity state.

        Args:
            baseline_delta: The starting Δ (maximum sensitivity).
            alpha: Decay rate — controls how fast Δ grows with word count.
        """
        self.baseline_delta = baseline_delta
        self.alpha = alpha
        self._delta = baseline_delta
        self._word_count = 0
        self._last_keystroke_time = time.time()
        self._last_fg_window = None

    @property
    def delta(self):
        """Current threshold value."""
        return self._delta

    def on_word_complete(self):
        """Called when a word delimiter is typed (space, enter, etc.).

        Increments word count and decays Δ.
        """
        self._word_count += 1
        self._delta = (
            self.baseline_delta
            + self.alpha * math.log(1 + self._word_count)
        )

    def reset(self, reason='unknown'):
        """Reset Δ to baseline on a Context Resumption Event.

        Args:
            reason: Descriptive reason for the reset (for logging).
        """
        self._word_count = 0
        self._delta = self.baseline_delta

    def check_idle_timeout(self, idle_timeout_seconds=5.0):
        """Check if enough time has passed since last keystroke.

        Args:
            idle_timeout_seconds: Threshold in seconds.

        Returns:
            True if idle timeout exceeded (CRE should trigger).
        """
        now = time.time()
        elapsed = now - self._last_keystroke_time
        return elapsed > idle_timeout_seconds

    def record_keystroke(self):
        """Record the timestamp of the current keystroke."""
        self._last_keystroke_time = time.time()

    def check_window_change(self, current_hwnd):
        """Check if the foreground window has changed.

        Args:
            current_hwnd: Handle of the current foreground window.

        Returns:
            True if the window changed since last check (CRE).
        """
        if self._last_fg_window is None:
            self._last_fg_window = current_hwnd
            return False
        if current_hwnd != self._last_fg_window:
            self._last_fg_window = current_hwnd
            return True
        return False

    def update_config(self, baseline_delta=None, alpha=None):
        """Update configuration values (from UI settings).

        Args:
            baseline_delta: New baseline Δ value.
            alpha: New decay alpha.
        """
        if baseline_delta is not None:
            self.baseline_delta = baseline_delta
        if alpha is not None:
            self.alpha = alpha
        self.reset(reason='config_update')
