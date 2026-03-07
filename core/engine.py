"""
engine.py — Two-tier evaluation pipeline.

Tier 1: O(1) dictionary exclusion via shadow-collision hash set.
Tier 2: Trigram probabilistic model scoring.
"""

import json
import os


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

        if on_delimiter and self.check_collision(s_active, s_shadow):
            return False, 0.0

        if current_layout == 'en':
            score_active = self.en_model.score(s_active)
            score_shadow = self.he_model.score(s_shadow)
        else:
            score_active = self.he_model.score(s_active)
            score_shadow = self.en_model.score(s_shadow)

        score_diff = score_shadow - score_active

        should_switch = score_diff > delta
        return should_switch, score_diff

    def evaluate_incremental(self, s_active_prev2, s_shadow_prev2,
                             en_char, he_char, delta, current_layout='en'):
        """Incremental per-keystroke evaluation for low latency.

        Instead of re-scoring the entire buffer each keystroke, this
        computes only the new trigram's contribution.

        Args:
            s_active_prev2: Last 2 chars of active buffer.
            s_shadow_prev2: Last 2 chars of shadow buffer.
            en_char: The new English character.
            he_char: The new Hebrew character.
            delta: Current threshold.
            current_layout: 'en' or 'he'.

        Returns:
            Tuple (score_increment_active, score_increment_shadow).
        """
        if current_layout == 'en':
            inc_active = self.en_model.score_incremental(
                s_active_prev2, en_char
            )
            inc_shadow = self.he_model.score_incremental(
                s_shadow_prev2, he_char
            )
        else:
            inc_active = self.he_model.score_incremental(
                s_active_prev2, he_char
            )
            inc_shadow = self.en_model.score_incremental(
                s_shadow_prev2, en_char
            )

        return inc_active, inc_shadow
