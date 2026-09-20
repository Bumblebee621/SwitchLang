"""
benchmark.py — SwitchLang evaluation test harness.

Measures false positive rate, false negative rate, and detection latency
by replaying text files through the evaluation engine — no OS hooks needed.

The simulation mirrors hooks.py behaviour: mid-word evaluation (≥3 chars),
delimiter evaluation, sensitivity decay, collision blacklist, and retroactive
lookback via history deque / correction blocks.

Usage:
    python benchmark.py                                   # en_corpus.txt, all lines
    python benchmark.py --max-lines 1000                  # first 1000 lines
    python benchmark.py data/he_corpus.txt                # Hebrew corpus
    python benchmark.py myfile.txt --lang en              # explicit language
    python benchmark.py --test fp                         # only false-positive test
    python benchmark.py --holdout-frac 0.1                # evaluate on held-out tail
    python benchmark.py --jobs 1                          # serial (default: all cores)
"""

import argparse
import atexit
import collections
import io
import os
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields

# Force UTF-8 output on Windows (cp1252 can't encode Hebrew characters)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import EvaluationEngine
from core.quadgram import load_models, QuadgramModel
from core.sensitivity import SensitivityManager
from core.keymap import shadow

# Re-use the same namedtuple that hooks.py uses for lookback history
_WordEntry = collections.namedtuple(
    '_WordEntry', ['active', 'shadow', 'delimiter', 'is_colliding', 'is_ambiguous']
)


# ═══════════════════════════════════════════════════════════════════════════
# RESULT DATA-CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WordResult:
    """Outcome of evaluating a single word through the engine."""
    word: str
    buffer_active: str
    buffer_shadow: str
    switched: bool
    switch_char_idx: int = -1   # ≥0 -> mid-word index;  -1 -> delimiter
    score_diff: float = 0.0
    is_colliding: bool = False
    is_ambiguous: bool = False


@dataclass
class FPReport:
    """False-positive test results."""
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    fp_count: int = 0
    recovery_count: int = 0
    lines_with_fp: int = 0
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fp_per_1k(self):
        return (self.fp_count / self.words_tested * 1000) if self.words_tested else 0.0


@dataclass
class FNReport:
    """False-negative test results."""
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    lines_switched: int = 0
    words_not_switched: int = 0
    total_latency_chars: int = 0
    latency_values: list = field(default_factory=list)
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fn_per_1k(self):
        return (self.words_not_switched / self.words_tested * 1000) if self.words_tested else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """Replays text through the SwitchLang engine to measure accuracy."""

    def __init__(self, data_dir, en_model_path=None, he_model_path=None,
                 mode='standard', scoring='incremental'):
        models = load_models(data_dir, load_so=(mode == 'technical'))
        # Allow overriding individual model files
        if en_model_path:
            models['en'] = QuadgramModel(en_model_path)
        if he_model_path:
            models['he'] = QuadgramModel(he_model_path)
        collisions_path = os.path.join(data_dir, 'collisions.json')
        self.engine = EvaluationEngine(
            models['en'], models['he'],
            collisions_path=collisions_path,
            enable_logging=False,
            en_so_model=models.get('so'),
            model_mode=mode,
        )
        self.scoring = scoring

    def _score_text_detailed(self, text, layout):
        """Score text and return (total_score, state_dict) for incremental tracking."""
        mode = self.engine.model_mode
        if layout == 'en':
            score_std = self.engine.en_model.score(text)
            state = {'std': score_std}
            if mode == 'technical' and self.engine.en_so_model:
                score_so = self.engine.en_so_model.score(text)
                state['so'] = score_so
                return max(score_std, score_so), state
            return score_std, state
        else:
            score = self.engine.he_model.score(text)
            return score, {'score': score}

    def _score_incremental_detailed(self, state, prev3, new_char, layout):
        """Score a single new character given previous state and return (total_score, new_state)."""
        mode = self.engine.model_mode
        if layout == 'en':
            inc_std = self.engine.en_model.score_incremental(prev3, new_char)
            new_std = state['std'] + inc_std
            new_state = {'std': new_std}
            if 'so' in state and self.engine.en_so_model and mode == 'technical':
                inc_so = self.engine.en_so_model.score_incremental(prev3, new_char)
                new_so = state['so'] + inc_so
                new_state['so'] = new_so
                return max(new_std, new_so), new_state
            return new_std, new_state
        else:
            inc = self.engine.he_model.score_incremental(prev3, new_char)
            new_score = state['score'] + inc
            return new_score, {'score': new_score}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _other(layout):
        return 'he' if layout == 'en' else 'en'

    @staticmethod
    def _shadow_dir(text_lang):
        """Return the shadow direction string for keymap.shadow()."""
        return 'en_to_he' if text_lang == 'en' else 'he_to_en'

    def _get_buffers(self, word, text_lang, current_layout):
        """Map *word* (in *text_lang*) to (buffer_active, buffer_shadow).

        The physical keys the user presses correspond to characters in
        *text_lang*.  Whether those keys produce *text_lang* chars or the
        other layout's chars depends on *current_layout*.
        """
        other_text = shadow(word, self._shadow_dir(text_lang))
        if current_layout == text_lang:
            return word, other_text          # correct layout
        else:
            return other_text, word          # wrong layout

    def _simulate_word(self, word, text_lang, current_layout, sensitivity):
        """Evaluate one word character-by-character, then on delimiter."""
        if self.scoring == 'incremental':
            return self._simulate_word_incremental(word, text_lang, current_layout, sensitivity)
        return self._simulate_word_full(word, text_lang, current_layout, sensitivity)

    def _simulate_word_full(self, word, text_lang, current_layout, sensitivity):
        """Evaluate one word character-by-character using full buffer rescoring."""
        buf_active, buf_shadow = self._get_buffers(word, text_lang, current_layout)

        # --- mid-word evaluation (after each char, starting at length 3) ---
        consecutive_hits = 0
        for i in range(2, len(buf_active)):
            partial_a = buf_active[:i + 1]
            partial_s = buf_shadow[:i + 1]
            should, diff, coll, amb = self.engine.evaluate(
                partial_a, partial_s, sensitivity.delta,
                current_layout=current_layout,
            )
            if should:
                consecutive_hits += 1
                if consecutive_hits >= 2:
                    return WordResult(
                        word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                        switched=True, switch_char_idx=i,
                        score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                    )
            else:
                consecutive_hits = 0

        # --- delimiter evaluation ---
        should, diff, coll, amb = self.engine.evaluate(
            buf_active, buf_shadow, sensitivity.delta,
            current_layout=current_layout, on_delimiter=True,
        )
        return WordResult(
            word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
            switched=should, switch_char_idx=-1,
            score_diff=diff, is_colliding=coll, is_ambiguous=amb,
        )

    def _simulate_word_incremental(self, word, text_lang, current_layout, sensitivity):
        """Evaluate one word character-by-character using incremental O(1) scoring."""
        buf_active, buf_shadow = self._get_buffers(word, text_lang, current_layout)

        target_layout = 'he' if current_layout == 'en' else 'en'
        consecutive_hits = 0
        act_state = None
        shd_state = None
        diff = 0.0
        coll = False
        amb = False

        for i in range(2, len(buf_active)):
            partial_a = buf_active[:i + 1]
            partial_s = buf_shadow[:i + 1]

            if i == 2:
                score_a, act_state = self._score_text_detailed(' ' + partial_a, current_layout)
                score_s, shd_state = self._score_text_detailed(' ' + partial_s, target_layout)
            else:
                prev3_a = (' ' + buf_active[:i])[-3:]
                score_a, act_state = self._score_incremental_detailed(
                    act_state, prev3_a, partial_a[-1], current_layout
                )
                prev3_s = (' ' + buf_shadow[:i])[-3:]
                score_s, shd_state = self._score_incremental_detailed(
                    shd_state, prev3_s, partial_s[-1], target_layout
                )

            diff = score_s - score_a
            coll = self.engine.check_collision(partial_a, partial_s)
            should = (diff > sensitivity.delta) if not coll else False
            amb = (not should and diff > 0 and not coll)

            if should:
                consecutive_hits += 1
                if consecutive_hits >= 2:
                    return WordResult(
                        word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                        switched=True, switch_char_idx=i,
                        score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                    )
            else:
                consecutive_hits = 0

        # delimiter evaluation
        if act_state is not None:
            prev3_a = (' ' + buf_active)[-3:]
            score_a_del, _ = self._score_incremental_detailed(
                act_state, prev3_a, ' ', current_layout
            )
            prev3_s = (' ' + buf_shadow)[-3:]
            score_s_del, _ = self._score_incremental_detailed(
                shd_state, prev3_s, ' ', target_layout
            )
            diff = score_s_del - score_a_del
            coll = self.engine.check_collision(buf_active, buf_shadow)
            should = (diff > sensitivity.delta) if not coll else False
            amb = (not should and diff > 0 and not coll)
        else:
            should, diff, coll, amb = self.engine.evaluate(
                buf_active, buf_shadow, sensitivity.delta,
                current_layout=current_layout, on_delimiter=True,
            )

        return WordResult(
            word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
            switched=should, switch_char_idx=-1,
            score_diff=diff, is_colliding=coll, is_ambiguous=amb,
        )

    def benchmark_scoring_comparison(self, lines, text_lang, baseline_delta=3.5, max_words=None):
        """Benchmark full rescoring vs incremental scoring head-to-head on identical keystrokes."""
        print(f"\n" + "=" * 80)
        print(f" SCORING BENCHMARK: Full Rescoring (Current) vs Incremental (score_incremental)")
        print(f"=" * 80)
        print(f"Language: {text_lang.upper()} | Model mode: {self.engine.model_mode} | Delta: {baseline_delta}")
        print(f"Extracting words from {len(lines):,} lines...")

        all_words = []
        for line in lines:
            for w in line.strip().split():
                if w:
                    all_words.append(w)
                    if max_words and len(all_words) >= max_words:
                        break
            if max_words and len(all_words) >= max_words:
                break

        print(f"Running side-by-side benchmark on {len(all_words):,} words...\n")

        sensitivity = SensitivityManager(baseline_delta=baseline_delta)
        current_layout = text_lang
        target_layout = 'he' if current_layout == 'en' else 'en'

        full_latencies_ns = []
        inc_latencies_ns = []
        bucket_full = {'short': [], 'medium': [], 'long': [], 'del': []}
        bucket_inc = {'short': [], 'medium': [], 'long': [], 'del': []}

        mismatches = 0
        max_score_err = 0.0
        words_tested = 0
        eval_count = 0

        t_full_total = 0
        t_inc_total = 0

        for word in all_words:
            words_tested += 1
            buf_active, buf_shadow = self._get_buffers(word, text_lang, current_layout)
            w_len = len(buf_active)
            b_key = 'short' if w_len <= 5 else ('medium' if w_len <= 9 else 'long')

            # ── 1. Full Rescore ──
            t_w0 = time.perf_counter_ns()
            consecutive_hits = 0
            full_switched = False
            full_switch_idx = -1
            full_diff = 0.0
            full_coll = False
            full_amb = False
            for i in range(2, len(buf_active)):
                partial_a = buf_active[:i + 1]
                partial_s = buf_shadow[:i + 1]
                t0 = time.perf_counter_ns()
                should, diff, coll, amb = self.engine.evaluate(
                    partial_a, partial_s, sensitivity.delta,
                    current_layout=current_layout,
                )
                dt = time.perf_counter_ns() - t0
                full_latencies_ns.append(dt)
                bucket_full[b_key].append(dt)
                eval_count += 1
                if should:
                    consecutive_hits += 1
                    if consecutive_hits >= 2:
                        full_switched = True
                        full_switch_idx = i
                        full_diff = diff
                        full_coll = coll
                        full_amb = amb
                        break
                else:
                    consecutive_hits = 0

            if not full_switched:
                t0 = time.perf_counter_ns()
                should, diff, coll, amb = self.engine.evaluate(
                    buf_active, buf_shadow, sensitivity.delta,
                    current_layout=current_layout, on_delimiter=True,
                )
                dt = time.perf_counter_ns() - t0
                full_latencies_ns.append(dt)
                bucket_full['del'].append(dt)
                eval_count += 1
                full_switched = should
                full_diff = diff
                full_coll = coll
                full_amb = amb
            t_full_total += (time.perf_counter_ns() - t_w0)

            # ── 2. Incremental Rescore ──
            t_w0 = time.perf_counter_ns()
            consecutive_hits = 0
            act_state = None
            shd_state = None
            inc_switched = False
            inc_switch_idx = -1
            inc_diff = 0.0
            inc_coll = False
            inc_amb = False

            for i in range(2, len(buf_active)):
                partial_a = buf_active[:i + 1]
                partial_s = buf_shadow[:i + 1]
                t0 = time.perf_counter_ns()
                if i == 2:
                    score_a, act_state = self._score_text_detailed(' ' + partial_a, current_layout)
                    score_s, shd_state = self._score_text_detailed(' ' + partial_s, target_layout)
                else:
                    prev3_a = (' ' + buf_active[:i])[-3:]
                    score_a, act_state = self._score_incremental_detailed(
                        act_state, prev3_a, partial_a[-1], current_layout
                    )
                    prev3_s = (' ' + buf_shadow[:i])[-3:]
                    score_s, shd_state = self._score_incremental_detailed(
                        shd_state, prev3_s, partial_s[-1], target_layout
                    )
                diff = score_s - score_a
                coll = self.engine.check_collision(partial_a, partial_s)
                should = (diff > sensitivity.delta) if not coll else False
                amb = (not should and diff > 0 and not coll)
                dt = time.perf_counter_ns() - t0

                inc_latencies_ns.append(dt)
                bucket_inc[b_key].append(dt)

                if should:
                    consecutive_hits += 1
                    if consecutive_hits >= 2:
                        inc_switched = True
                        inc_switch_idx = i
                        inc_diff = diff
                        inc_coll = coll
                        inc_amb = amb
                        break
                else:
                    consecutive_hits = 0

            if not inc_switched:
                t0 = time.perf_counter_ns()
                if act_state is not None:
                    prev3_a = (' ' + buf_active)[-3:]
                    score_a_del, _ = self._score_incremental_detailed(
                        act_state, prev3_a, ' ', current_layout
                    )
                    prev3_s = (' ' + buf_shadow)[-3:]
                    score_s_del, _ = self._score_incremental_detailed(
                        shd_state, prev3_s, ' ', target_layout
                    )
                    diff = score_s_del - score_a_del
                    coll = self.engine.check_collision(buf_active, buf_shadow)
                    should = (diff > sensitivity.delta) if not coll else False
                    amb = (not should and diff > 0 and not coll)
                else:
                    should, diff, coll, amb = self.engine.evaluate(
                        buf_active, buf_shadow, sensitivity.delta,
                        current_layout=current_layout, on_delimiter=True,
                    )
                dt = time.perf_counter_ns() - t0
                inc_latencies_ns.append(dt)
                bucket_inc['del'].append(dt)
                inc_switched = should
                inc_diff = diff
                inc_coll = coll
                inc_amb = amb
            t_inc_total += (time.perf_counter_ns() - t_w0)

            # Equivalence validation
            if full_switched != inc_switched or full_switch_idx != inc_switch_idx:
                mismatches += 1
            err = abs(full_diff - inc_diff)
            if err > max_score_err:
                max_score_err = err

        # Compute statistics
        def _stats(arr_ns):
            if not arr_ns:
                return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
            s = sorted(arr_ns)
            n = len(s)
            return {
                'mean': (sum(s) / n) / 1000.0,
                'median': s[n // 2] / 1000.0,
                'p95': s[int(n * 0.95)] / 1000.0,
                'p99': s[int(n * 0.99)] / 1000.0,
            }

        full_s = _stats(full_latencies_ns)
        inc_s = _stats(inc_latencies_ns)
        speedup_mean = (full_s['mean'] / inc_s['mean']) if inc_s['mean'] > 0 else 0.0
        speedup_median = (full_s['median'] / inc_s['median']) if inc_s['median'] > 0 else 0.0
        speedup_total = (t_full_total / t_inc_total) if t_inc_total > 0 else 0.0

        print("-" * 80)
        print(f"{'OVERALL SUMMARY':<30}")
        print("-" * 80)
        print(f"Words tested:                {words_tested:,}")
        print(f"Total keystroke evals:       {eval_count:,}")
        print(f"Total time (Full):           {t_full_total / 1e9:.4f} s")
        print(f"Total time (Incremental):    {t_inc_total / 1e9:.4f} s")
        print(f"Overall Speedup:             {speedup_total:.2f}x ({(1 - t_inc_total / t_full_total) * 100:.1f}% time saved)")

        print("\n" + "-" * 80)
        print(f"{'PER-KEYSTROKE LATENCY':<30} {'Full Rescore':>15} {'Incremental':>15} {'Speedup':>12}")
        print("-" * 80)
        print(f"{'Mean latency:':<30} {full_s['mean']:>12.2f} µs {inc_s['mean']:>12.2f} µs {speedup_mean:>11.2f}x")
        print(f"{'Median (p50):':<30} {full_s['median']:>12.2f} µs {inc_s['median']:>12.2f} µs {speedup_median:>11.2f}x")
        print(f"{'95th percentile (p95):':<30} {full_s['p95']:>12.2f} µs {inc_s['p95']:>12.2f} µs {(full_s['p95']/inc_s['p95'] if inc_s['p95'] else 0):>11.2f}x")
        print(f"{'99th percentile (p99):':<30} {full_s['p99']:>12.2f} µs {inc_s['p99']:>12.2f} µs {(full_s['p99']/inc_s['p99'] if inc_s['p99'] else 0):>11.2f}x")

        print("\n" + "-" * 80)
        print(f"{'LATENCY BY WORD LENGTH (mean)':<30} {'Full Rescore':>15} {'Incremental':>15} {'Speedup':>12}")
        print("-" * 80)
        labels = [
            ('short', 'Short words (3–5 chars):'),
            ('medium', 'Medium words (6–9 chars):'),
            ('long', 'Long words (10+ chars):'),
            ('del', 'Delimiter evaluations:')
        ]
        for key, lbl in labels:
            f_m = _stats(bucket_full[key])['mean']
            i_m = _stats(bucket_inc[key])['mean']
            sp = (f_m / i_m) if i_m > 0 else 0.0
            print(f"{lbl:<30} {f_m:>12.2f} µs {i_m:>12.2f} µs {sp:>11.2f}x")

        print("\n" + "-" * 80)
        print(f"{'CORRECTNESS VERIFICATION':<30}")
        print("-" * 80)
        pct_match = ((words_tested - mismatches) / words_tested * 100) if words_tested else 100.0
        print(f"Decision match rate:         {pct_match:.2f}% ({words_tested - mismatches:,}/{words_tested:,} words)")
        print(f"Max score difference:        {max_score_err:.2e} nats")
        print("=" * 80 + "\n")


    @staticmethod
    def _build_correction_block(history):
        """Contiguous correctable tail — same logic as hooks.py."""
        block = []
        for entry in reversed(history):
            if entry.is_colliding or entry.is_ambiguous:
                block.append(entry)
            else:
                break
        block.reverse()
        return block

    # ------------------------------------------------------------------
    # FALSE-POSITIVE TEST
    # ------------------------------------------------------------------

    def test_false_positives(self, lines, text_lang, baseline_delta=3.5,
                             line_offset=0, progress=True):
        """Feed *valid* text on the *correct* layout.  Any switch = FP."""
        report = FPReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = correct
            sensitivity = SensitivityManager(baseline_delta=baseline_delta)
            history = collections.deque(maxlen=50)
            line_fps = 0
            details = []

            for w_idx, word in enumerate(words):
                if not word:
                    continue
                report.words_tested += 1
                res = self._simulate_word(word, text_lang, current, sensitivity)

                if res.switched:
                    if current == correct:
                        # Switching AWAY from correct layout -> false positive
                        line_fps += 1
                        report.fp_count += 1
                        block = self._build_correction_block(history)
                        details.append(
                            f'  FP word {w_idx+1} "{word}" '
                            f'(diff={res.score_diff:+.2f} delta={sensitivity.delta:.2f} '
                            f'lookback={len(block)})'
                        )
                    else:
                        # Switching BACK to correct layout -> recovery (not punished)
                        report.recovery_count += 1
                        block = self._build_correction_block(history)
                        details.append(
                            f'  Recovery word {w_idx+1} "{word}" (lookback={len(block)})'
                        )

                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                else:
                    buf_a, buf_s = self._get_buffers(word, text_lang, current)
                    history.append(_WordEntry(
                        active=buf_a, shadow=buf_s, delimiter=' ',
                        is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                    ))
                    sensitivity.on_word_complete()

            if line_fps > 0:
                report.lines_with_fp += 1
                trunc = line.strip()[:100]
                report.flagged_lines.append(
                    f'Line {line_num}: "{trunc}"\n' + '\n'.join(details)
                )

            if progress and line_num % 5000 == 0:
                print(f'  [FP] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report

    # ------------------------------------------------------------------
    # FALSE-NEGATIVE TEST
    # ------------------------------------------------------------------

    def test_false_negatives(self, lines, text_lang, baseline_delta=3.5,
                             line_offset=0, progress=True):
        """Feed *inverted* text (wrong layout).  Failure to switch = FN.

        Latency = total chars typed on the wrong layout from the start of
        the line until the first switch fires (includes ALL wrong chars,
        not just those eventually corrected).
        """
        report = FNReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang
        wrong = self._other(text_lang)

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = wrong                # start on WRONG layout
            sensitivity = SensitivityManager(baseline_delta=baseline_delta)
            history = collections.deque(maxlen=50)

            latency_chars = 0
            first_switch_word = None
            line_words_not_switched = 0
            line_switched = False

            for w_idx, word in enumerate(words):
                if not word:
                    continue
                report.words_tested += 1

                res = self._simulate_word(word, text_lang, current, sensitivity)

                if res.switched and current != correct:
                    # ── correct switch detected ──
                    line_switched = True
                    if first_switch_word is None:
                        first_switch_word = w_idx

                    # Latency: chars typed so far in wrong layout
                    if res.switch_char_idx >= 0:
                        # mid-word switch: count chars up to and including trigger
                        latency_chars += res.switch_char_idx + 1
                    else:
                        # delimiter switch: full word + delimiter
                        latency_chars += len(word) + 1

                    # Determine uncorrected words
                    block = self._build_correction_block(history)
                    corrected = len(block) + 1          # block + trigger word
                    total_wrong = w_idx + 1             # all words so far
                    uncorrected = max(0, total_wrong - corrected)
                    line_words_not_switched += uncorrected

                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                elif res.switched:
                    # switched while already on correct layout (FP within FN test)
                    # — just track the layout flip
                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                else:
                    # no switch — accumulate into history
                    if current != correct:
                        # still on wrong layout: this word contributes to latency
                        latency_chars += len(word) + 1  # word + space
                    buf_a, buf_s = self._get_buffers(word, text_lang, current)
                    history.append(_WordEntry(
                        active=buf_a, shadow=buf_s, delimiter=' ',
                        is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                    ))
                    sensitivity.on_word_complete()

            # If engine never switched, entire line is FN
            if not line_switched:
                line_words_not_switched = len(words)
                latency_chars = sum(len(w) for w in words) + len(words)

            report.words_not_switched += line_words_not_switched
            report.latency_values.append(latency_chars)
            report.total_latency_chars += latency_chars

            if line_switched:
                report.lines_switched += 1

            # Flag lines with issues
            if not line_switched or line_words_not_switched > 0:
                trunc = line.strip()[:100]
                if first_switch_word is not None:
                    report.flagged_lines.append(
                        f'Line {line_num}: "{trunc}" -> switch at word '
                        f'{first_switch_word+1} ("{words[first_switch_word]}"), '
                        f'latency={latency_chars} chars, '
                        f'{line_words_not_switched} uncorrected'
                    )
                else:
                    report.flagged_lines.append(
                        f'Line {line_num}: "{trunc}" -> NO SWITCH '
                        f'({latency_chars} chars lost)'
                    )

            if progress and line_num % 5000 == 0:
                print(f'  [FN] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Every line resets layout, sensitivity and history, so lines are independent
# and splitting them across processes reproduces the serial result exactly.

# Size 1: a variant is scored FP then FN, so the second call reuses the models
# while memory stays at one model set per worker rather than one per variant.
_worker_models = (None, None)


def _run_chunk(job):
    global _worker_models
    test, offset, lines, lang, delta, key = job
    if _worker_models[0] != key:
        _worker_models = (key, EvaluationHarness(*key))
    harness = _worker_models[1]
    method = (harness.test_false_positives if test == 'fp'
              else harness.test_false_negatives)
    return method(lines, lang, delta, line_offset=offset, progress=False)


def _merge_reports(reports):
    """Sum counters and concatenate lists across chunk reports."""
    merged = reports[0]
    for report in reports[1:]:
        for f in fields(merged):
            value = getattr(report, f.name)
            if isinstance(value, bool) or f.name == 'lang':
                continue
            if isinstance(value, (int, float)):
                setattr(merged, f.name, getattr(merged, f.name) + value)
            elif isinstance(value, list):
                getattr(merged, f.name).extend(value)
    return merged


_pool = None
_pool_jobs = None


def _get_pool(jobs):
    """One pool for the whole process — respawning it per call costs seconds."""
    global _pool, _pool_jobs
    if _pool is None or _pool_jobs != jobs:
        shutdown_pool()
        _pool = ProcessPoolExecutor(max_workers=jobs)
        _pool_jobs = jobs
    return _pool


def shutdown_pool():
    global _pool, _pool_jobs
    if _pool is not None:
        _pool.shutdown()
        _pool, _pool_jobs = None, None


atexit.register(shutdown_pool)


def run_test(test, lines, lang, delta, data_dir, en_model_path=None,
             he_model_path=None, mode='standard', jobs=1, scoring='incremental'):
    """Run the 'fp' or 'fn' test over *lines*, optionally across processes."""
    key = (data_dir, en_model_path, he_model_path, mode, scoring)

    if jobs <= 1:
        harness = EvaluationHarness(*key)
        method = (harness.test_false_positives if test == 'fp'
                  else harness.test_false_negatives)
        return method(lines, lang, delta)

    # Several chunks per worker keeps them busy when line lengths vary.
    n_chunks = jobs * 4
    size = max(1, -(-len(lines) // n_chunks))
    chunks = [(test, i, lines[i:i + size], lang, delta, key)
              for i in range(0, len(lines), size)]

    t0 = time.time()
    reports = list(_get_pool(jobs).map(_run_chunk, chunks))
    merged = _merge_reports(reports)
    merged.elapsed_sec = time.time() - t0
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def load_corpus_lines(path, lang, cap=None):
    """Read *path*, keeping non-empty lines written purely in *lang*'s script.

    Mixed-script lines are excluded: a real layout switch fires a CRE, so each
    script run is an independent segment rather than one continuous context.
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue

            has_hebrew = bool(re.search(r'[\u0590-\u05FF]', l))
            has_english = bool(re.search(r'[a-zA-Z]', l))
            has_nikud = bool(re.search(r'[\u0591-\u05C7]', l))

            if lang == 'he' and (not has_hebrew or has_english or has_nikud):
                continue
            if lang == 'en' and (not has_english or has_hebrew):
                continue

            lines.append(l)
            if cap is not None and len(lines) >= cap:
                break
    return lines


def _pct(num, denom):
    return (num / denom * 100) if denom else 0.0


def print_fp_report(report, corpus_path, model_path, provenance='', max_flagged=30):
    print(f'\n{"=" * 65}')
    print(f' FALSE POSITIVE TEST  (valid {report.lang.upper()}, layout={report.lang})')
    print(f'{"=" * 65}')
    print(f'Corpus:           {corpus_path}')
    print(f'Split:            {provenance}')
    print(f'Model:            {model_path}')
    print(f'Lines tested:     {report.lines_tested}')
    print(f'Words tested:     {report.words_tested}')
    print(f'False positives:  {report.fp_count}  ({_pct(report.fp_count, report.words_tested):.4f}%)')
    print(f'FP per 1k words:  {report.fp_per_1k:.3f}')
    print(f'Recoveries:       {report.recovery_count}')
    print(f'Lines with FP:    {report.lines_with_fp}')
    print(f'Time:             {report.elapsed_sec:.1f}s')

    if report.flagged_lines:
        print(f'\nFlagged lines (first {min(max_flagged, len(report.flagged_lines))}):')
        for detail in report.flagged_lines[:max_flagged]:
            print(f'  {detail}')
        remaining = len(report.flagged_lines) - max_flagged
        if remaining > 0:
            print(f'  … and {remaining} more')


def print_fn_report(report, corpus_path, model_path, provenance='', max_flagged=30):
    print(f'\n{"=" * 65}')
    print(f' FALSE NEGATIVE TEST  (inverted {report.lang.upper()}, wrong layout)')
    print(f'{"=" * 65}')
    print(f'Corpus:             {corpus_path}')
    print(f'Split:              {provenance}')
    print(f'Model:              {model_path}')
    print(f'Lines tested:       {report.lines_tested}')
    print(f'Words tested:       {report.words_tested}')
    sr = _pct(report.lines_switched, report.lines_tested)
    print(f'Lines switched:     {report.lines_switched}/{report.lines_tested}  ({sr:.3f}%)')
    fnr = _pct(report.words_not_switched, report.words_tested)
    print(f'Words not switched: {report.words_not_switched}  ({fnr:.3f}%)')
    print(f'FN per 1k words:    {report.fn_per_1k:.3f}')

    if report.latency_values:
        vals = report.latency_values
        mean_l = statistics.mean(vals)
        median_l = statistics.median(vals)
        s = sorted(vals)
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        print(f'Mean latency:       {mean_l:.1f} chars')
        print(f'Median latency:     {median_l:.0f} chars')
        print(f'P95 latency:        {p95} chars')

    print(f'Time:               {report.elapsed_sec:.1f}s')

    if report.flagged_lines:
        print(f'\nFlagged lines (first {min(max_flagged, len(report.flagged_lines))}):')
        for detail in report.flagged_lines[:max_flagged]:
            print(f'  {detail}')
        remaining = len(report.flagged_lines) - max_flagged
        if remaining > 0:
            print(f'  … and {remaining} more')


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SwitchLang evaluation test harness — measures FP, FN, and latency.',
    )
    parser.add_argument(
        'text_file', nargs='?', default=None,
        help='Path to text file (default: data/en_corpus.txt)',
    )
    parser.add_argument(
        '--lang', choices=['en', 'he'], default=None,
        help='Language of the text file.  Auto-detected from filename if omitted.',
    )
    parser.add_argument(
        '--max-lines', type=int, default=50_000,
        help='Max non-empty lines to process (default: entire file).',
    )
    parser.add_argument(
        '--baseline-delta', type=float, default=3.5,
        help='Initial score delta threshold (default: 4.0).',
    )
    parser.add_argument(
        '--test', choices=['fp', 'fn', 'both'], default='both',
        help='Which test(s) to run (default: both).',
    )
    parser.add_argument(
        '--data-dir', default=None,
        help='Path to data/ directory (default: auto-detect).',
    )
    parser.add_argument(
        '--en-model', default=None, metavar='PATH',
        help='Override path to English quadgram JSON (e.g. data/backup_opus/en_quadgrams.json).',
    )
    parser.add_argument(
        '--he-model', default=None, metavar='PATH',
        help='Override path to Hebrew quadgram JSON (e.g. data/backup_opus/he_quadgrams.json).',
    )
    parser.add_argument(
        '--holdout-frac', type=float, default=None, metavar='F',
        help='Evaluate only on the last F fraction of eligible lines (e.g. 0.1). '
             'The model must have been built without them, or this proves nothing.',
    )
    parser.add_argument(
        '--mode', choices=['standard', 'technical'], default='standard',
        help='Model mode (default: standard).  technical also scores against so_quadgrams.json.',
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=os.cpu_count() or 1, metavar='N',
        help='Worker processes for scoring (default: all cores).  1 disables.',
    )
    parser.add_argument(
        '--scoring', choices=['full', 'incremental', 'compare'], default='incremental',
        help='Scoring algorithm: incremental (default, 2x faster O(1)), full, or compare (benchmarks both).',
    )
    args = parser.parse_args()

    # ── resolve data dir ──
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data')

    # ── resolve text file ──
    if args.text_file is None:
        args.text_file = os.path.join(args.data_dir, 'en_corpus.txt')

    # ── auto-detect language ──
    if args.lang is None:
        bn = os.path.basename(args.text_file).lower()
        if bn.startswith('en') or 'en_' in bn:
            args.lang = 'en'
        elif bn.startswith('he') or 'he_' in bn:
            args.lang = 'he'
        else:
            parser.error(
                'Cannot auto-detect language from filename.  Use --lang en/he.'
            )

    # ── load text ──
    print(f'Loading text from {args.text_file} …')
    # A holdout needs the whole file scanned before the tail can be sliced off.
    cap = None if args.holdout_frac else args.max_lines
    lines = load_corpus_lines(args.text_file, args.lang, cap)

    eligible = len(lines)
    if args.holdout_frac:
        split_at = int(eligible * (1.0 - args.holdout_frac))
        lines = lines[split_at:]
        if args.max_lines is not None:
            lines = lines[:args.max_lines]
        provenance = (f'held out last {args.holdout_frac:.0%} of {eligible:,} '
                      f'eligible lines (from index {split_at:,})')
    else:
        provenance = 'NO HOLDOUT — results may be inflated by train/test overlap'

    print(f'Loaded {len(lines)} non-empty pure lines (lang={args.lang})')
    print(f'Split: {provenance}')

    if args.en_model:
        print(f'  EN model override: {args.en_model}')
    if args.he_model:
        print(f'  HE model override: {args.he_model}')
    print(f'Mode={args.mode}, Scoring={args.scoring}, jobs={args.jobs}.\n')

    # ── compare scoring mode ──
    if args.scoring == 'compare':
        harness = EvaluationHarness(args.data_dir, en_model_path=args.en_model,
                                    he_model_path=args.he_model, mode=args.mode)
        harness.benchmark_scoring_comparison(lines, args.lang, args.baseline_delta)
        return

    # ── run tests ──
    model_override = args.en_model if args.lang == 'en' else args.he_model
    model_path = model_override if model_override else os.path.join(args.data_dir, f'{args.lang}_quadgrams.marisa')
    corpus_path = args.text_file

    common = dict(data_dir=args.data_dir, en_model_path=args.en_model,
                  he_model_path=args.he_model, mode=args.mode, jobs=args.jobs,
                  scoring=args.scoring)

    if args.test in ('fp', 'both'):
        fp = run_test('fp', lines, args.lang, args.baseline_delta, **common)
        print_fp_report(fp, corpus_path, model_path, provenance)

    if args.test in ('fn', 'both'):
        fn = run_test('fn', lines, args.lang, args.baseline_delta, **common)
        print_fn_report(fn, corpus_path, model_path, provenance)


if __name__ == '__main__':
    main()
