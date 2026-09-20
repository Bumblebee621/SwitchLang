"""
consecutive_bench.py — Benchmark Consecutive Confirmation vs Baseline.

Compares:
  - Baseline (K=1): Switch fires immediately on the first mid-word keystroke where score_diff > delta.
  - Consecutive (K=2): Switch fires only after K consecutive mid-word keystrokes where score_diff > delta.

Sweeps delta thresholds for both K=1 and K=2 to map the Pareto frontier of:
  - False Positives (FP / 1k words)
  - False Negatives (FN / 1k words)
  - Detection Latency (mean characters typed on wrong layout before switch)

Usage:
    python evaluation/consecutive_bench.py --max-lines 500              # smoke test
    python evaluation/consecutive_bench.py --max-lines 7500             # phase 1 screening sweep
    python evaluation/consecutive_bench.py --variants k1:4.0,k2:2.5,k2:3.0 --max-lines 20000
    python evaluation/consecutive_bench.py --status-file data/consecutive_exp/status.json
"""

import argparse
import io
import json
import os
import statistics
import sys
import time
from collections import deque
from concurrent.futures import as_completed
from dataclasses import dataclass, field, fields

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import (
    EvaluationHarness, WordResult, SensitivityManager,
    load_corpus_lines, _get_pool, _pct, _WordEntry
)
from core.quadgram import load_models, QuadgramModel
from core.engine import EvaluationEngine


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConsecutiveFPReport:
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    fp_count: int = 0
    recovery_count: int = 0
    lines_with_fp: int = 0
    midword_fps: int = 0
    delimiter_fps: int = 0
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fp_per_1k(self):
        return (self.fp_count / self.words_tested * 1000) if self.words_tested else 0.0


@dataclass
class ConsecutiveFNReport:
    lang: str
    lines_tested: int = 0
    words_tested: int = 0
    lines_switched: int = 0
    words_not_switched: int = 0
    total_latency_chars: int = 0
    midword_switches: int = 0
    delimiter_switches: int = 0
    latency_values: list = field(default_factory=list)
    flagged_lines: list = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def fn_per_1k(self):
        return (self.words_not_switched / self.words_tested * 1000) if self.words_tested else 0.0

    @property
    def mean_latency(self):
        return (sum(self.latency_values) / len(self.latency_values)) if self.latency_values else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HARNESS
# ═══════════════════════════════════════════════════════════════════════════

class ConsecutiveHarness(EvaluationHarness):
    """EvaluationHarness with configurable consecutive mid-word confirmations."""

    def __init__(self, data_dir, en_model_path=None, he_model_path=None,
                 mode='standard', req_confirmations=1):
        super().__init__(data_dir, en_model_path=en_model_path,
                         he_model_path=he_model_path, mode=mode)
        self.req_confirmations = req_confirmations

    def _simulate_word(self, word, text_lang, current_layout, sensitivity):
        """Evaluate one word character-by-character, then on delimiter.

        Mid-word evaluation requires req_confirmations consecutive hits before switching.
        Delimiter evaluation evaluates the full token standalone.
        """
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
                if consecutive_hits >= self.req_confirmations:
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

    def test_false_positives(self, lines, text_lang, baseline_delta=4.0,
                             line_offset=0, progress=True):
        """Feed valid text on correct layout. Any switch = FP."""
        report = ConsecutiveFPReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = correct
            sensitivity = SensitivityManager(baseline_delta=baseline_delta)
            history = deque(maxlen=50)

            for w_idx, word in enumerate(words):
                if not word:
                    continue
                report.words_tested += 1
                res = self._simulate_word(word, text_lang, current, sensitivity)

                if res.switched:
                    if current == correct:
                        report.fp_count += 1
                        if res.switch_char_idx >= 0:
                            report.midword_fps += 1
                        else:
                            report.delimiter_fps += 1
                    else:
                        report.recovery_count += 1

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

        report.elapsed_sec = time.time() - t0
        return report

    def test_false_negatives(self, lines, text_lang, baseline_delta=4.0,
                             line_offset=0, progress=True):
        """Feed inverted text on wrong layout. Failure to switch = FN."""
        report = ConsecutiveFNReport(lang=text_lang)
        t0 = time.time()
        correct = text_lang
        wrong = self._other(text_lang)

        for line_num, line in enumerate(lines, line_offset + 1):
            words = line.strip().split()
            if not words:
                continue
            report.lines_tested += 1

            current = wrong
            sensitivity = SensitivityManager(baseline_delta=baseline_delta)
            history = deque(maxlen=50)

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
                    line_switched = True
                    if first_switch_word is None:
                        first_switch_word = w_idx

                    if res.switch_char_idx >= 0:
                        report.midword_switches += 1
                        latency_chars += res.switch_char_idx + 1
                    else:
                        report.delimiter_switches += 1
                        latency_chars += len(word) + 1

                    block = self._build_correction_block(history)
                    corrected = len(block) + 1
                    total_wrong = w_idx + 1
                    uncorrected = max(0, total_wrong - corrected)
                    line_words_not_switched += uncorrected

                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                elif res.switched:
                    current = self._other(current)
                    history.clear()
                    sensitivity.reset(reason='layout_switch')
                else:
                    if current != correct:
                        latency_chars += len(word) + 1
                    buf_a, buf_s = self._get_buffers(word, text_lang, current)
                    history.append(_WordEntry(
                        active=buf_a, shadow=buf_s, delimiter=' ',
                        is_colliding=res.is_colliding, is_ambiguous=res.is_ambiguous,
                    ))
                    sensitivity.on_word_complete()

            if not line_switched:
                line_words_not_switched = len(words)
                latency_chars = sum(len(w) for w in words) + len(words)

            report.words_not_switched += line_words_not_switched
            report.latency_values.append(latency_chars)
            report.total_latency_chars += latency_chars

        report.elapsed_sec = time.time() - t0
        return report


# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS & STATUS REPORTING
# ═══════════════════════════════════════════════════════════════════════════

class Progress:
    """Chunk-completion progress bar."""
    WIDTH = 24

    def __init__(self, label, total, stream=sys.stdout):
        self.label = label
        self.total = max(1, total)
        self.stream = stream
        self.tty = hasattr(stream, 'isatty') and stream.isatty()
        self.done = 0
        self.t0 = time.time()
        self._last_decile = -1
        self._draw()

    def advance(self, n=1):
        self.done += n
        self._draw()

    def _bar(self, frac):
        filled = int(frac * self.WIDTH)
        return '█' * filled + '░' * (self.WIDTH - filled)

    def _draw(self):
        frac = self.done / self.total
        elapsed = time.time() - self.t0
        if self.tty:
            self.stream.write(
                f'\r    {self.label:<24} {self._bar(frac)} '
                f'{self.done:>3}/{self.total} {elapsed:>5.0f}s')
            self.stream.flush()
        else:
            decile = int(frac * 10)
            if decile > self._last_decile:
                self._last_decile = decile
                self.stream.write(
                    f'    {self.label:<24} {frac:>4.0%} '
                    f'({self.done}/{self.total} chunks, {elapsed:.0f}s)\n')
                self.stream.flush()

    def close(self):
        if self.tty:
            self.stream.write('\r' + ' ' * 78 + '\r')
            self.stream.flush()


class StatusFile:
    def __init__(self, path):
        self.path = path
        self.state = {'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'pid': os.getpid(), 'results': {}}
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._flush()

    def set(self, **kw):
        if not self.path:
            return
        self.state.update(kw)
        self.state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self._flush()

    def record(self, lang, variant, row):
        if not self.path:
            return
        self.state.setdefault('results', {}).setdefault(lang, {})[variant] = row
        self.set()

    def _flush(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.path)


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL WORKERS
# ═══════════════════════════════════════════════════════════════════════════

_worker_models = (None, None)


def _run_chunk(job):
    global _worker_models
    test, offset, lines, lang, delta, key = job
    if _worker_models[0] != key:
        _worker_models = (key, ConsecutiveHarness(*key))
    harness = _worker_models[1]
    method = (harness.test_false_positives if test == 'fp'
              else harness.test_false_negatives)
    return method(lines, lang, delta, line_offset=offset, progress=False)


def _merge_consecutive_reports(reports):
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


def run_benchmark_test(test, lines, lang, delta, key, jobs=1, label=''):
    if jobs <= 1:
        harness = ConsecutiveHarness(*key)
        method = (harness.test_false_positives if test == 'fp'
                  else harness.test_false_negatives)
        return method(lines, lang, delta)

    n_chunks = jobs * 4
    size = max(1, -(-len(lines) // n_chunks))
    chunks = [(test, i, lines[i:i + size], lang, delta, key)
              for i in range(0, len(lines), size)]

    t0 = time.time()
    pool = _get_pool(jobs)
    futures = [pool.submit(_run_chunk, chunk) for chunk in chunks]
    bar = Progress(label or test, len(futures))
    for _ in as_completed(futures):
        bar.advance()
    bar.close()

    merged = _merge_consecutive_reports([f.result() for f in futures])
    merged.elapsed_sec = time.time() - t0
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT PARSING & RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def parse_variant(v_str):
    """Parse string like 'k1:4.0', 'k2:2.5', 'baseline' into (k, delta)."""
    v_str = v_str.strip().lower()
    if v_str == 'baseline':
        return 1, 4.0, 'baseline (K=1, Δ=4.0)'
    if ':' in v_str:
        k_part, d_part = v_str.split(':', 1)
        k = int(k_part.replace('k', ''))
        delta = float(d_part)
        label = f'K={k}, Δ={delta:.1f}'
        return k, delta, label
    raise ValueError(f"Unrecognized variant format: {v_str}")


DEFAULT_VARIANTS = (
    'k1:4.0',  # Current production baseline
    'k1:3.5',
    'k1:4.5',
    'k2:2.0',
    'k2:2.5',
    'k2:3.0',
    'k2:3.5',
    'k2:4.0',
)


def run_variant(lang, lines, variant_str, data_dir, jobs=1):
    k, delta, label = parse_variant(variant_str)
    key = (data_dir, None, None, 'standard', k)

    fp_rep = run_benchmark_test('fp', lines, lang, delta, key, jobs=jobs,
                                label=f'{variant_str} [FP]')
    fn_rep = run_benchmark_test('fn', lines, lang, delta, key, jobs=jobs,
                                label=f'{variant_str} [FN]')

    return {
        'variant': variant_str,
        'label': label,
        'k': k,
        'delta': delta,
        'words': fp_rep.words_tested,
        'fp_count': fp_rep.fp_count,
        'fp_1k': fp_rep.fp_per_1k,
        'midword_fps': fp_rep.midword_fps,
        'delimiter_fps': fp_rep.delimiter_fps,
        'fn_words_not_switched': fn_rep.words_not_switched,
        'fn_1k': fn_rep.fn_per_1k,
        'latency': fn_rep.mean_latency,
        'median_latency': statistics.median(fn_rep.latency_values) if fn_rep.latency_values else 0.0,
        'midword_switches': fn_rep.midword_switches,
        'delimiter_switches': fn_rep.delimiter_switches,
        'elapsed_sec': fp_rep.elapsed_sec + fn_rep.elapsed_sec,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Consecutive Confirmation vs Baseline over swept deltas.'
    )
    parser.add_argument('--max-lines', type=int, default=7500,
                        help='Number of corpus lines per language (default: 7500).')
    parser.add_argument('--languages', default='en,he',
                        help='Comma-separated languages: en, he (default: en,he).')
    parser.add_argument('--variants', default=None,
                        help='Comma-separated variants e.g. "k1:4.0,k2:2.5,k2:3.0"')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                        help='Worker processes (default: all cores).')
    parser.add_argument('--status-file', default=None,
                        help='Optional path to write live JSON status.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')

    variants = [v.strip() for v in args.variants.split(',')] if args.variants else list(DEFAULT_VARIANTS)
    languages = [l.strip() for l in args.languages.split(',')]
    status = StatusFile(args.status_file) if args.status_file else None

    print("=" * 86)
    print(f" CONSECUTIVE CONFIRMATION BENCHMARK (lines={args.max_lines:,}, jobs={args.jobs})")
    print("=" * 86)
    print(f"Variants to test: {', '.join(variants)}")
    print(f"Languages:        {', '.join(languages).upper()}")
    print("-" * 86)

    all_results = {}

    for lang in languages:
        corpus_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        if not os.path.exists(corpus_path):
            print(f"Warning: {corpus_path} not found, skipping {lang.upper()}.")
            continue

        lines = load_corpus_lines(corpus_path, lang, cap=args.max_lines)
        print(f"\nEvaluating {lang.upper()} ({len(lines):,} lines loaded)...")

        lang_results = []
        baseline_res = None

        for v_str in variants:
            t0 = time.time()
            res = run_variant(lang, lines, v_str, data_dir, jobs=args.jobs)
            lang_results.append(res)
            if v_str == 'k1:4.0':
                baseline_res = res

            if status:
                status.record(lang, v_str, res)

            # Print inline summary
            print(f"  {res['label']:<18} | FP/1k: {res['fp_1k']:>6.3f} | FN/1k: {res['fn_1k']:>6.3f} | "
                  f"Latency: {res['latency']:>5.2f} chars | Time: {res['elapsed_sec']:>4.1f}s")

        all_results[lang] = lang_results

        # Print language summary table
        print("\n" + "-" * 86)
        print(f" RESULTS SUMMARY: {lang.upper()} ({len(lines):,} lines, {lang_results[0]['words']:,} words)")
        print("-" * 86)
        print(f"{'Variant':<16} {'Δ':>4} {'FP/1k':>8} {'Δ FP%':>9} {'FN/1k':>8} {'Δ FN%':>9} {'Latency':>8} {'Δ Lat':>7} {'MidW%':>7}")
        print("-" * 86)

        base_fp = baseline_res['fp_1k'] if baseline_res else lang_results[0]['fp_1k']
        base_fn = baseline_res['fn_1k'] if baseline_res else lang_results[0]['fn_1k']
        base_lat = baseline_res['latency'] if baseline_res else lang_results[0]['latency']

        for r in lang_results:
            d_fp_pct = ((r['fp_1k'] - base_fp) / base_fp * 100) if base_fp else 0.0
            d_fn_pct = ((r['fn_1k'] - base_fn) / base_fn * 100) if base_fn else 0.0
            d_lat = r['latency'] - base_lat
            total_sw = r['midword_switches'] + r['delimiter_switches']
            midw_pct = (r['midword_switches'] / total_sw * 100) if total_sw else 0.0

            marker = " (BASE)" if r['variant'] == 'k1:4.0' else ""
            v_name = f"K={r['k']}{marker}"
            print(f"{v_name:<16} {r['delta']:>4.1f} {r['fp_1k']:>8.3f} {d_fp_pct:>+8.1f}% "
                  f"{r['fn_1k']:>8.3f} {d_fn_pct:>+8.1f}% {r['latency']:>8.2f} {d_lat:>+7.2f} {midw_pct:>6.1f}%")
        print("-" * 86)


if __name__ == '__main__':
    main()
