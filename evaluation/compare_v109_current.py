"""
compare_v109_current.py — Compare shipped v1.0.9 against the current engine.

Each side runs with its own models AND its own decision rule, so the table
answers "is the app better now", not just "are the models better":

    v1.0.9   unpruned JSON models, K=1, Δ=2.0, α=0.3, no evidence carried
    current  pruned MARISA models, K=2, Δ=6.0, α=0.1, evidence carried (CUSUM)

v1.0.9's scoring math is identical to today's; only the storage changed, so its
JSON models are converted once to .marisa and replayed through benchmark.py.

DATA LEAKAGE: both versions' models were built from the whole of
data/{lang}_corpus.txt, and this script scores lines from that same file, so
every test line was in both training sets.  The bias is not symmetric: v1.0.9's
unpruned models memorised the count-1/2 quadgrams that pruning drops from the
current models, so leakage flatters v1.0.9 more.
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from benchmark import load_corpus_lines, run_test
from build_quadgrams import save_model_data_to_trie

VERSIONS = {
    'v1.0.9': dict(subdir=os.path.join('data', 'v1_0_9'),
                   req_confirmations=1, delta=2.0, alpha=0.3, accumulate=False),
    'current': dict(subdir='data',
                    req_confirmations=2, delta=6.0, alpha=0.1, accumulate=True),
}
MIXED_PREFIX = 3   # correct words typed before the user forgets to switch


def ensure_tries(data_dir):
    """Convert v1.0.9's JSON models to .marisa + .meta.json (once)."""
    for lang in ('en', 'he', 'so'):
        src = os.path.join(data_dir, f'{lang}_quadgrams.json')
        trie = os.path.join(data_dir, f'{lang}_quadgrams.marisa')
        if os.path.exists(src) and not os.path.exists(trie):
            with open(src, 'r', encoding='utf-8') as f:
                save_model_data_to_trie(json.load(f), trie,
                                        os.path.join(data_dir, f'{lang}_quadgrams.meta.json'))


def get_static_stats(data_dir):
    """N-gram counts from the meta file; disk size of what the version shipped."""
    stats = {}
    for lang in ('en', 'he', 'so'):
        meta_path = os.path.join(data_dir, f'{lang}_quadgrams.meta.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        shipped = os.path.join(data_dir, f'{lang}_quadgrams.json')
        if not os.path.exists(shipped):    # current ships the trie + meta
            shipped = os.path.join(data_dir, f'{lang}_quadgrams.marisa')
        size = os.path.getsize(shipped) + (0 if shipped.endswith('.json') else os.path.getsize(meta_path))
        stats[lang] = dict(size_mb=size / (1024 * 1024), vocab=meta['vocab_size'],
                           quads=meta['counts']['quadgrams'], tri=meta['counts']['trigrams'],
                           bi=meta['counts']['bigrams'])
    return stats


def _pct(new, old):
    return f"{(new - old) / old * 100:+.1f}%" if old else '—'


def print_static_table(stats):
    v, c = stats['v1.0.9'], stats['current']
    print("=" * 82)
    print(" STATIC MODEL COMPARISON: v1.0.9 (commit edde5d4) vs Current")
    print("=" * 82)
    print(f"{'Model':<6} {'Version':<10} {'Disk (MB)':>10} {'Quadgrams':>12} {'Trigrams':>10} {'Bigrams':>9} {'Vocab':>7}")
    print("-" * 82)
    for lang in ('en', 'he', 'so'):
        if lang not in v or lang not in c:
            continue
        a, b = v[lang], c[lang]
        print(f"{lang.upper():<6} {'v1.0.9':<10} {a['size_mb']:>10.2f} {a['quads']:>12,} {a['tri']:>10,} {a['bi']:>9,} {a['vocab']:>7}")
        print(f"{'':<6} {'Current':<10} {b['size_mb']:>10.2f} {b['quads']:>12,} {b['tri']:>10,} {b['bi']:>9,} {b['vocab']:>7}")
        print(f"{'':<6} {'Δ (%)':<10} {_pct(b['size_mb'], a['size_mb']):>10} {_pct(b['quads'], a['quads']):>12} "
              f"{_pct(b['tri'], a['tri']):>10} {_pct(b['bi'], a['bi']):>9} {b['vocab'] - a['vocab']:>+7}")
        print("-" * 82)


def run_version(cfg, lang, lines, pairs, mode, jobs):
    common = dict(data_dir=os.path.join(ROOT, cfg['subdir']), mode=mode, jobs=jobs,
                  req_confirmations=cfg['req_confirmations'],
                  alpha=cfg['alpha'], accumulate=cfg['accumulate'])
    t0 = time.time()
    fp = run_test('fp', lines, lang, cfg['delta'], **common)
    fn = run_test('fn', lines, lang, cfg['delta'], **common)
    mixed = (run_test('mixed', pairs, lang, cfg['delta'], n_prefix=MIXED_PREFIX, **common)
             if pairs else None)
    return {
        'words': fp.words_tested, 'fp_1k': fp.fp_per_1k, 'fn_1k': fn.fn_per_1k,
        'latency': statistics.mean(fn.latency_values) if fn.latency_values else 0.0,
        'mixed_1k': mixed.fn_per_1k if mixed else None,
        'time': time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare SwitchLang v1.0.9 and the current engine.')
    parser.add_argument('--max-lines', type=int, default=20000, help='Max non-empty lines per language (default: 20000).')
    parser.add_argument('--mode', choices=['standard', 'technical'], default='standard')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1, help='Scoring worker processes.')
    parser.add_argument('--test-so', action='store_true', help='Also run test on stack_overflow_comments.txt.')
    args = parser.parse_args()

    v109_dir = os.path.join(ROOT, VERSIONS['v1.0.9']['subdir'])
    if not os.path.exists(v109_dir):
        print(f"Error: {v109_dir} does not exist. Please extract v1.0.9 models first.")
        sys.exit(1)
    ensure_tries(v109_dir)

    print_static_table({name: get_static_stats(os.path.join(ROOT, cfg['subdir']))
                        for name, cfg in VERSIONS.items()})

    cur_dir = os.path.join(ROOT, 'data')
    corpora = {lang: load_corpus_lines(os.path.join(cur_dir, f'{lang}_corpus.txt'), lang, cap=args.max_lines)
               for lang in ('en', 'he')}
    results = {}
    for lang in ('en', 'he'):
        lines = corpora[lang]
        other = [l for l in corpora['he' if lang == 'en' else 'en'] if len(l.split()) >= MIXED_PREFIX]
        pairs = [(l, other[i % len(other)]) for i, l in enumerate(lines)]
        print(f"\nEvaluating {lang.upper()} ({len(lines):,} lines) across {args.jobs} workers...", flush=True)
        results[lang] = {name: run_version(cfg, lang, lines, pairs, args.mode, args.jobs)
                         for name, cfg in VERSIONS.items()}

    if args.test_so:
        so_corpus = os.path.join(cur_dir, 'stack_overflow_comments.txt')
        if os.path.exists(so_corpus):
            so_lines = load_corpus_lines(so_corpus, 'en', cap=args.max_lines)
            results['so'] = {name: run_version(cfg, 'en', so_lines, None, 'technical', args.jobs)
                             for name, cfg in VERSIONS.items()}

    print("\n" + "=" * 92)
    print(f" EMPIRICAL BENCHMARK RESULTS (lines={args.max_lines:,}; each version with its own K/Δ/α)")
    print("=" * 92)
    print(f"{'Corpus':<8} {'Version':<11} {'Words':>10} {'FP / 1k':>9} {'FN / 1k':>9} "
          f"{'Latency':>8} {f'Forgot@{MIXED_PREFIX} /1k':>15} {'Time(s)':>8}")
    print("-" * 92)
    for corpus, res in results.items():
        label = corpus.upper() if corpus != 'so' else 'SO(tech)'
        for name in VERSIONS:
            r = res[name]
            mixed = f"{r['mixed_1k']:.3f}" if r['mixed_1k'] is not None else '—'
            print(f"{label if name == 'v1.0.9' else '':<8} {name:<11} {r['words']:>10,} {r['fp_1k']:>9.3f} "
                  f"{r['fn_1k']:>9.3f} {r['latency']:>7.2f}c {mixed:>15} {r['time']:>8.1f}")
        v, c = res['v1.0.9'], res['current']
        mixed = (_pct(c['mixed_1k'], v['mixed_1k']) if v['mixed_1k'] is not None else '—')
        print(f"{'':<8} {'Δ (%)':<11} {'':>10} {_pct(c['fp_1k'], v['fp_1k']):>9} {_pct(c['fn_1k'], v['fn_1k']):>9} "
              f"{c['latency'] - v['latency']:>+7.2f}c {mixed:>15}")
        print("-" * 92)


if __name__ == '__main__':
    main()
