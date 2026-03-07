"""
engine.py — Two-tier evaluation pipeline.

Tier 1: O(1) dictionary exclusion via shadow-collision hash set.
Tier 2: Trigram probabilistic model scoring.
"""

import csv
import json
import os
import threading
import logging

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluates whether a layout switch should occur."""

    def __init__(self, en_model, he_model, collisions_path=None):
        """Initialize with trigram models and optional collision set.

        Args:
            en_model: TrigramModel for English.
            he_model: TrigramModel for Hebrew.
            collisions_path: Path to collisions.json (shadow-collision set).
        """
        self.en_model = en_model
        self.he_model = he_model

        self.collisions = set()
        if collisions_path and os.path.exists(collisions_path):
            with open(collisions_path, 'r', encoding='utf-8') as f:
                self.collisions = set(json.load(f))

        # Setup CSV logging
        self.stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'decision_stats.csv'
        )
        self.stats_lock = threading.Lock()
        self._pending_logs = []
        self._ensure_stats_file()

    def _ensure_stats_file(self):
        """Ensure the stats CSV exists with headers."""
        if not os.path.exists(self.stats_path):
            try:
                os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
                with open(self.stats_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'time', 'active_word', 'shadow_word', 'layout', 
                        'on_delimiter', 'score_diff', 'category', 'should_switch'
                    ])
            except OSError:
                pass

    def _log_decision(self, s_active, s_shadow, layout, on_delimiter, score_diff, should_switch):
        """Log the evaluation decision to the CSV file."""
        import time
        from datetime import datetime

        if score_diff >= 3.0:
            category = ">= 3.0"
        elif score_diff >= 2.0:
            category = "2.0 - 3.0"
        elif score_diff >= 1.0:
            category = "1.0 - 2.0"
        elif score_diff >= 0.0:
            category = "0.0 - 1.0"
        else:
            category = "< 0.0"

        with self.stats_lock:
            # Queue the log
            self._pending_logs.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                s_active,
                s_shadow,
                layout,
                on_delimiter,
                f"{score_diff:.3f}",
                category,
                should_switch
            ])
            # Keep queue size bounded in case file is permanently locked
            if len(self._pending_logs) > 1000:
                self._pending_logs.pop(0)

            try:
                with open(self.stats_path, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    # Write all pending logs at once
                    for row in self._pending_logs:
                        writer.writerow(row)
                    # Clear pending if successful
                    self._pending_logs.clear()
            except OSError as e:
                logger.warning(f"Could not write to decision_stats.csv (is it open in Excel?): {e}")

    def check_collision(self, s_active, s_shadow):
        """Tier 1: Check if either string is a known shadow collision.

        Called on delimiter (space/punctuation). If the active buffer
        IS a collision word, we should NOT switch because it could be
        valid in either layout.

        Args:
            s_active: The string in the current OS layout.
            s_shadow: The string mapped to the alternate layout.

        Returns:
            True if a collision is detected (do NOT switch).
        """
        return (
            s_active.lower() in self.collisions
            or s_shadow.lower() in self.collisions
        )

    def evaluate(self, s_active, s_shadow, delta,
                 current_layout='en', on_delimiter=False):
        """Run the full evaluation pipeline.

        Args:
            s_active: Characters typed in current layout.
            s_shadow: Same keystrokes mapped to alternate layout.
            delta: Current decision boundary threshold.
            current_layout: 'en' or 'he' — the currently active layout.
            on_delimiter: Whether this evaluation is triggered by a
                          word delimiter (space, enter, etc.).

        Returns:
            Tuple (should_switch: bool, score_diff: float).
            score_diff = score_shadow - score_active.
        """
        if len(s_active) < 2:
            return False, 0.0

        if self.check_collision(s_active, s_shadow):
            return False, 0.0

        if current_layout == 'en':
            score_active = self.en_model.score(s_active)
            score_shadow = self.he_model.score(s_shadow)
        else:
            score_active = self.he_model.score(s_active)
            score_shadow = self.en_model.score(s_shadow)

        score_diff = score_shadow - score_active
        should_switch = score_diff > delta

        self._log_decision(s_active, s_shadow, current_layout, on_delimiter, score_diff, should_switch)

        return should_switch, score_diff

