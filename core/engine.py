"""
engine.py — Two-tier evaluation pipeline.

Tier 1: O(1) dictionary exclusion via shadow-collision hash set.
Tier 2: Quadgram probabilistic model scoring.
"""

import csv
import json
import os
import threading
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluates whether a layout switch should occur."""

    MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024
    MAX_CSV_LINES = 10000

    def __init__(self, en_model, he_model, collisions_path=None, storage_dir=None, 
                 enable_logging=True, en_so_model=None, model_mode='standard'):
        """Initialize with quadgram models and optional collision set.

        Args:
            en_model: QuadgramModel for Standard English.
            he_model: QuadgramModel for Hebrew.
            collisions_path: Path to collisions.json (shadow-collision set).
            storage_dir: Base directory for stats and logs.
            enable_logging: Whether to log decisions to a CSV file.
            en_so_model: Optional QuadgramModel for Stack Overflow English.
            model_mode: 'standard' or 'technical'.
        """
        self.en_model = en_model
        self.he_model = he_model
        self.en_so_model = en_so_model
        self.model_mode = model_mode
        self.enable_logging = enable_logging

        self.collisions = set()
        if collisions_path and os.path.exists(collisions_path):
            with open(collisions_path, 'r', encoding='utf-8') as f:
                self.collisions = set(json.load(f))

        # Setup CSV logging
        if storage_dir:
            self.stats_path = os.path.join(storage_dir, 'decision_stats.csv')
        else:
            self.stats_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data',
                'decision_stats.csv'
            )
        self.stats_lock = threading.Lock()
        self._pending_logs = deque(maxlen=1000)
        self._logs_since_check = 0
        if self.enable_logging:
            self._ensure_stats_file()

    def set_model_mode(self, mode):
        """Switch between standard and technical model modes."""
        if mode in ('standard', 'technical'):
            self.model_mode = mode
            logger.info(f"Model mode switched to: {mode}")

    def set_enable_logging(self, enabled):
        """Toggle CSV decision logging at runtime."""
        self.enable_logging = enabled
        if enabled:
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
                        'on_delimiter', 'is_ambiguous', 'score_diff', 'delta', 'category', 'should_switch'
                    ])
            except OSError as e:
                logger.warning(f"Could not create stats file: {e}")
        else:
            self._rotate_stats_file()

    def _rotate_stats_file(self):
        """Truncate the CSV file if it exceeds the maximum allowed size."""
        try:
            if not os.path.exists(self.stats_path):
                return
            
            file_size = os.path.getsize(self.stats_path)
            if file_size <= self.MAX_CSV_SIZE_BYTES:
                return
                
            with open(self.stats_path, 'r', encoding='utf-8-sig', newline='') as f:
                header = f.readline()
                lines = deque(f, maxlen=self.MAX_CSV_LINES)
                
            with open(self.stats_path, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(header)
                f.writelines(lines)
        except OSError as e:
            logger.warning(f"Could not rotate decision_stats.csv: {e}")

    def _log_decision(self, s_active, s_shadow, layout, on_delimiter, is_ambiguous, score_diff, delta, should_switch):
        """Log the evaluation decision to the CSV file."""
        if not self.enable_logging:
            return
        abs_score_diff = abs(score_diff)
        if abs_score_diff > 3.0:
            category = "|x| > 3"
        elif abs_score_diff > 2.0:
            category = "3 > |x| > 2"
        elif abs_score_diff > 1.0:
            category = "2 > |x| > 1"
        else:
            category = "1 > |x|"

        with self.stats_lock:
            # Queue the log (bounds handled by deque automatically)
            self._pending_logs.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                s_active,
                s_shadow,
                layout,
                on_delimiter,
                is_ambiguous,
                f"{score_diff:.3f}",
                f"{delta:.2f}",
                category,
                should_switch
            ])

            try:
                if self._pending_logs:
                    with open(self.stats_path, 'a', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f)
                        # Write all pending logs at once
                        for row in self._pending_logs:
                            writer.writerow(row)
                    
                    self._logs_since_check += len(self._pending_logs)
                    self._pending_logs.clear()
                    
                if self._logs_since_check > 1000:
                    self._logs_since_check = 0
                    self._rotate_stats_file()
            except OSError as e:
                logger.warning(f"Could not write to decision_stats.csv (is it open in Excel?): {e}")

    def check_collision(self, s_active, s_shadow):
        """Tier 1: Check if either string is a known shadow collision."""
        return (
            s_active.lower() in self.collisions
            or s_shadow.lower() in self.collisions
        )

    def _score_text_en(self, text):
        """Score text using English model(s) based on current mode."""
        score_std = self.en_model.score(text)
        if self.model_mode == 'technical' and self.en_so_model:
            score_so = self.en_so_model.score(text)
            return max(score_std, score_so)
        return score_std

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
            Tuple (should_switch: bool, score_diff: float, is_ambiguous: bool).
            score_diff = score_shadow - score_active.
        """
        if len(s_active) < 2:
            return False, 0.0, False

        is_ambig = self.check_collision(s_active, s_shadow)
        
        # Prepend a space because buffer_active always represents a word start.
        eval_active = ' ' + s_active
        eval_shadow = ' ' + s_shadow

        if on_delimiter:
            eval_active += ' '
            eval_shadow += ' '

        if current_layout == 'en':
            # Scoring as English
            score_active = self._score_text_en(eval_active)
            # Scoring as Hebrew (Shadow)
            score_shadow = self.he_model.score(eval_shadow)
        else:
            # Scoring as Hebrew
            score_active = self.he_model.score(eval_active)
            # Scoring as English (Shadow)
            score_shadow = self._score_text_en(eval_shadow)

        score_diff = score_shadow - score_active
        
        if is_ambig:
            should_switch = False
        else:
            should_switch = score_diff > delta
            # Mark as ambiguous if it leans towards the target language 
            # (score_diff > 0) but hasn't crossed the current dynamic delta threshold.
            if not should_switch and score_diff > 0:
                is_ambig = True

        self._log_decision(s_active, s_shadow, current_layout, on_delimiter, is_ambig, score_diff, delta, should_switch)

        return should_switch, score_diff, is_ambig

