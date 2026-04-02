"""
test_evaluation.py — SwitchLang evaluation test harness.

Measures false positive rate, false negative rate, and detection latency
by replaying text files through the evaluation engine — no OS hooks needed.

The simulation mirrors hooks.py behaviour: mid-word evaluation (≥3 chars),
delimiter evaluation, sensitivity decay, collision blacklist, and retroactive
lookback via history deque / correction blocks.

Usage:
    python test_evaluation.py                             # en_corpus.txt, all lines
    python test_evaluation.py --max-lines 1000            # first 1000 lines
    python test_evaluation.py data/he_corpus.txt          # Hebrew corpus
    python test_evaluation.py myfile.txt --lang en        # explicit language
    python test_evaluation.py --test fp                   # only false-positive test
"""

import argparse
import collections
import io
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

# Force UTF-8 output on Windows (cp1252 can't encode Hebrew characters)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import EvaluationEngine
from core.quadgram import load_models
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


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """Replays text through the SwitchLang engine to measure accuracy."""

    def __init__(self, data_dir):
        models = load_models(data_dir)
        collisions_path = os.path.join(data_dir, 'collisions.json')
        self.engine = EvaluationEngine(
            models['en'], models['he'],
            collisions_path=collisions_path,
            enable_logging=False,
        )

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
        """Evaluate one word character-by-character, then on delimiter.

        Mirrors hooks.py: mid-word eval at ≥3 chars, delimiter eval at end.
        """
        buf_active, buf_shadow = self._get_buffers(word, text_lang, current_layout)

        # --- mid-word evaluation (after each char, starting at length 3) ---
        for i in range(2, len(buf_active)):
            partial_a = buf_active[:i + 1]
            partial_s = buf_shadow[:i + 1]
            should, diff, coll, amb = self.engine.evaluate(
                partial_a, partial_s, sensitivity.delta,
                current_layout=current_layout,
            )
            if should:
                return WordResult(
                    word=word, buffer_active=buf_active, buffer_shadow=buf_shadow,
                    switched=True, switch_char_idx=i,
                    score_diff=diff, is_colliding=coll, is_ambiguous=amb,
                )

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

    def test_false_positives(self, lines, text_lang, baseline_delta=4.0):
        """Feed *valid* text on the *correct* layout.  Any switch = FP."""
        report = FPReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang

        for line_num, line in enumerate(lines, 1):
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

            if line_num % 500 == 0:
                print(f'  [FP] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report

    # ------------------------------------------------------------------
    # FALSE-NEGATIVE TEST
    # ------------------------------------------------------------------

    def test_false_negatives(self, lines, text_lang, baseline_delta=4.0):
        """Feed *inverted* text (wrong layout).  Failure to switch = FN.

        Latency = total chars typed on the wrong layout from the start of
        the line until the first switch fires (includes ALL wrong chars,
        not just those eventually corrected).
        """
        report = FNReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang
        wrong = self._other(text_lang)

        for line_num, line in enumerate(lines, 1):
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

            if line_num % 500 == 0:
                print(f'  [FN] {line_num}/{len(lines)} lines …', flush=True)

        report.elapsed_sec = time.time() - t0
        return report


# ═══════════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def _pct(num, denom):
    return (num / denom * 100) if denom else 0.0


def print_fp_report(report, max_flagged=30):
    print(f'\n{"=" * 65}')
    print(f' FALSE POSITIVE TEST  (valid {report.lang.upper()}, layout={report.lang})')
    print(f'{"=" * 65}')
    print(f'Lines tested:     {report.lines_tested}')
    print(f'Words tested:     {report.words_tested}')
    print(f'False positives:  {report.fp_count}  ({_pct(report.fp_count, report.words_tested):.3f}%)')
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


def print_fn_report(report, max_flagged=30):
    print(f'\n{"=" * 65}')
    print(f' FALSE NEGATIVE TEST  (inverted {report.lang.upper()}, wrong layout)')
    print(f'{"=" * 65}')
    print(f'Lines tested:       {report.lines_tested}')
    print(f'Words tested:       {report.words_tested}')
    sr = _pct(report.lines_switched, report.lines_tested)
    print(f'Lines switched:     {report.lines_switched}/{report.lines_tested}  ({sr:.1f}%)')
    fnr = _pct(report.words_not_switched, report.words_tested)
    print(f'Words not switched: {report.words_not_switched}  ({fnr:.2f}%)')

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
        '--max-lines', type=int, default=None,
        help='Max non-empty lines to process (default: entire file).',
    )
    parser.add_argument(
        '--baseline-delta', type=float, default=4.0,
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
    with open(args.text_file, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    if args.max_lines is not None:
        lines = lines[:args.max_lines]
    print(f'Loaded {len(lines)} non-empty lines  (lang={args.lang})')

    # ── init harness ──
    print(f'Loading models from {args.data_dir} …')
    harness = EvaluationHarness(args.data_dir)
    print('Models loaded.\n')

    # ── run tests ──
    if args.test in ('fp', 'both'):
        fp = harness.test_false_positives(lines, args.lang, args.baseline_delta)
        print_fp_report(fp)

    if args.test in ('fn', 'both'):
        fn = harness.test_false_negatives(lines, args.lang, args.baseline_delta)
        print_fn_report(fn)


if __name__ == '__main__':
    main()
