"""
compare_v109_current.py — Compare v1.0.9 (unpruned) models against current (pruned) models.

Reuses evaluation/benchmark.py to test FP, FN, and detection latency side-by-side
on identical slices of text.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))

from benchmark import load_corpus_lines, run_test


def get_static_stats(data_dir):
    stats = {}
    for lang in ('en', 'he', 'so'):
        path = os.path.join(data_dir, f'{lang}_quadgrams.json')
        if not os.path.exists(path):
            continue
        sz = os.path.getsize(path)
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        stats[lang] = {
            'size_mb': sz / (1024 * 1024),
            'quads': len(d.get('quadgram_counts', {})),
            'tri': len(d.get('trigram_counts', {})),
            'bi': len(d.get('bigram_counts', {})),
            'vocab': d.get('vocab_size', 0),
        }
    return stats


def print_static_table(v109_stats, cur_stats):
    print("=" * 82)
    print(" STATIC MODEL COMPARISON: v1.0.9 (commit edde5d4) vs Current (HEAD)")
    print("=" * 82)
    print(f"{'Model':<6} {'Version':<10} {'Disk (MB)':>10} {'Quadgrams':>12} {'Trigrams':>10} {'Bigrams':>9} {'Vocab':>7}")
    print("-" * 82)
    for lang in ('en', 'he', 'so'):
        if lang not in v109_stats or lang not in cur_stats:
            continue
        v = v109_stats[lang]
        c = cur_stats[lang]
        print(f"{lang.upper():<6} {'v1.0.9':<10} {v['size_mb']:>10.2f} {v['quads']:>12,} {v['tri']:>10,} {v['bi']:>9,} {v['vocab']:>7}")
        print(f"{'':<6} {'Current':<10} {c['size_mb']:>10.2f} {c['quads']:>12,} {c['tri']:>10,} {c['bi']:>9,} {c['vocab']:>7}")
        d_sz = ((c['size_mb'] - v['size_mb']) / v['size_mb']) * 100
        d_q = ((c['quads'] - v['quads']) / v['quads']) * 100
        print(f"{'':<6} {'Δ (%)':<10} {d_sz:>+9.1f}% {d_q:>+11.1f}% {'0.0%':>10} {'0.0%':>9} {'0':>7}")
        print("-" * 82)


def run_benchmark(lang, lines, delta, v109_dir, cur_dir, mode, jobs):
    print(f"\nEvaluating {lang.upper()} ({len(lines):,} lines) across {jobs} workers...", flush=True)

    # v1.0.9
    t0 = time.time()
    v_fp = run_test('fp', lines, lang, delta, v109_dir, mode=mode, jobs=jobs)
    v_fn = run_test('fn', lines, lang, delta, v109_dir, mode=mode, jobs=jobs)
    v_elapsed = time.time() - t0

    # Current
    t0 = time.time()
    c_fp = run_test('fp', lines, lang, delta, cur_dir, mode=mode, jobs=jobs)
    c_fn = run_test('fn', lines, lang, delta, cur_dir, mode=mode, jobs=jobs)
    c_elapsed = time.time() - t0

    v_lat = (sum(v_fn.latency_values) / len(v_fn.latency_values)) if v_fn.latency_values else 0.0
    c_lat = (sum(c_fn.latency_values) / len(c_fn.latency_values)) if c_fn.latency_values else 0.0

    return {
        'v109': {'fp_1k': v_fp.fp_per_1k, 'fn_1k': v_fn.fn_per_1k, 'latency': v_lat, 'words': v_fp.words_tested, 'time': v_elapsed},
        'current': {'fp_1k': c_fp.fp_per_1k, 'fn_1k': c_fn.fn_per_1k, 'latency': c_lat, 'words': c_fp.words_tested, 'time': c_elapsed},
    }


def main():
    parser = argparse.ArgumentParser(description='Compare SwitchLang v1.0.9 and current models.')
    parser.add_argument('--max-lines', type=int, default=20000, help='Max non-empty lines per language (default: 20000).')
    parser.add_argument('--delta', type=float, default=4.0, help='Baseline switch delta threshold (default: 4.0).')
    parser.add_argument('--mode', choices=['standard', 'technical'], default='standard')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1, help='Scoring worker processes.')
    parser.add_argument('--test-so', action='store_true', help='Also run test on stack_overflow_comments.txt.')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_dir = os.path.join(project_root, 'data')
    v109_dir = os.path.join(project_root, 'data', 'v1_0_9')

    if not os.path.exists(v109_dir):
        print(f"Error: {v109_dir} does not exist. Please extract v1.0.9 models first.")
        sys.exit(1)

    v109_stats = get_static_stats(v109_dir)
    cur_stats = get_static_stats(cur_dir)
    print_static_table(v109_stats, cur_stats)

    results = {}
    for lang in ('en', 'he'):
        corpus_path = os.path.join(cur_dir, f'{lang}_corpus.txt')
        if not os.path.exists(corpus_path):
            print(f"Warning: {corpus_path} not found, skipping.")
            continue
        lines = load_corpus_lines(corpus_path, lang, cap=args.max_lines)
        results[lang] = run_benchmark(lang, lines, args.delta, v109_dir, cur_dir, args.mode, args.jobs)

    if args.test_so:
        so_corpus = os.path.join(cur_dir, 'stack_overflow_comments.txt')
        if os.path.exists(so_corpus):
            so_lines = load_corpus_lines(so_corpus, 'en', cap=args.max_lines)
            results['so'] = run_benchmark('en', so_lines, args.delta, v109_dir, cur_dir, 'technical', args.jobs)

    print("\n" + "=" * 82)
    print(f" EMPIRICAL BENCHMARK RESULTS (delta={args.delta}, lines={args.max_lines:,})")
    print("=" * 82)
    print(f"{'Corpus':<8} {'Model':<10} {'Words':>10} {'FP / 1k':>11} {'FN / 1k':>11} {'Latency':>9} {'Time(s)':>8}")
    print("-" * 82)

    for corpus, res in results.items():
        v = res['v109']
        c = res['current']
        c_label = corpus.upper() if corpus != 'so' else 'SO(tech)'
        print(f"{c_label:<8} {'v1.0.9':<10} {v['words']:>10,} {v['fp_1k']:>11.3f} {v['fn_1k']:>11.3f} {v['latency']:>9.2f} {v['time']:>8.1f}")
        print(f"{'':<8} {'Current':<10} {c['words']:>10,} {c['fp_1k']:>11.3f} {c['fn_1k']:>11.3f} {c['latency']:>9.2f} {c['time']:>8.1f}")
        dfp = c['fp_1k'] - v['fp_1k']
        dfn = c['fn_1k'] - v['fn_1k']
        dlat = c['latency'] - v['latency']
        print(f"{'':<8} {'Δ (Cur-v1)':<10} {'':>10} {dfp:>+11.3f} {dfn:>+11.3f} {dlat:>+9.2f}")
        print("-" * 82)


if __name__ == '__main__':
    main()
