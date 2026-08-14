"""
kfold.py — K-fold cross-validated comparison of quadgram model variants.

Each fold's models are built from the other K-1 folds and evaluated on the
held-out fold, so a variant carrying more parameters cannot win by memorising
the lines it is scored on.

Variants are compared per fold rather than on pooled averages: with K=5 the
fold-to-fold spread is often larger than the effect under test, and only the
paired differences separate a real improvement from that noise.

Usage:
    python evaluation/kfold.py                              # all variants, K=5
    python evaluation/kfold.py --k 5 --variants unpruned,prune3
    python evaluation/kfold.py --max-lines 200000           # quick smoke run
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from benchmark import load_corpus_lines, run_test
from build_quadgrams import ALLOWED_EN, ALLOWED_HE, build_quadgrams_from_lines

ALLOWED = {'en': ALLOWED_EN, 'he': ALLOWED_HE}


# ═══════════════════════════════════════════════════════════════════════════
# VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

def _prune(min_count):
    def transform(model):
        model = dict(model)
        model['quadgram_counts'] = {
            k: c for k, c in model['quadgram_counts'].items() if c > min_count
        }
        return model
    return transform


# name -> callable(raw_model_dict) -> model_dict.
# Scoring-time arms (interpolated smoothing, per-model calibration) will need a
# second hook here for a model class; see evaluation/EXPERIMENTS.md.
VARIANTS = {
    'unpruned': lambda m: m,
    'prune2': _prune(2),
    'prune3': _prune(3),   # what build_quadgrams.py currently ships
    'prune5': _prune(5),
    'prune7': _prune(7),
}


# ═══════════════════════════════════════════════════════════════════════════
# FOLD BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def fold_of(idx, k):
    return idx % k


def build_fold_parts(lines, lang, k, cache_dir):
    """Count each fold's slice once — one corpus pass whatever K is.

    Cached unpruned so every variant derives without rebuilding.
    """
    os.makedirs(cache_dir, exist_ok=True)
    parts = []

    for part in range(k):
        cache_path = os.path.join(cache_dir, f'{lang}_k{k}_p{part}.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                parts.append(json.load(f))
            continue

        slice_ = [l for i, l in enumerate(lines) if fold_of(i, k) == part]
        print(f'  counting {lang} part {part} ({len(slice_):,} lines) …', flush=True)
        model = build_quadgrams_from_lines(slice_, ALLOWED[lang], min_count=0)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False)
        parts.append(model)

    return parts


def merge_parts(parts, skip):
    """Sum every part except *skip* — the training model for that fold.

    Exact: an n-gram occurrence belongs to one line and so to one part, which
    makes summing the others identical to building from those lines directly.
    """
    merged = {}
    for name in ('quadgram_counts', 'trigram_counts', 'bigram_counts'):
        total = Counter()
        for i, part in enumerate(parts):
            if i != skip:
                total.update(part[name])
        merged[name] = dict(total)

    # Alphabet size rather than a count, so it is maxed, not summed.  Every
    # slice saturates ALLOWED_*, so the parts agree and the max is the union.
    merged['vocab_size'] = max(p['vocab_size']
                               for i, p in enumerate(parts) if i != skip)
    return merged


def write_variant(model, transform, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(transform(model), f, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_fold(parts, lines, lang, fold, k, variants, cache_dir, data_dir, delta,
             jobs, max_test_lines):
    """Return {variant: (fp_per_1k, fn_per_1k, mean_latency)} for one fold."""
    raw = merge_parts(parts, fold)
    test = [l for i, l in enumerate(lines) if fold_of(i, k) == fold]
    if max_test_lines:
        test = test[:max_test_lines]

    results = {}
    for name in variants:
        var_path = os.path.join(cache_dir, f'{lang}_k{k}_f{fold}_{name}.json')
        write_variant(raw, VARIANTS[name], var_path)

        kwargs = {'en_model_path': var_path} if lang == 'en' else {'he_model_path': var_path}
        fp = run_test('fp', test, lang, delta, data_dir, jobs=jobs, **kwargs)
        fn = run_test('fn', test, lang, delta, data_dir, jobs=jobs, **kwargs)
        latency = statistics.mean(fn.latency_values) if fn.latency_values else 0.0
        results[name] = (fp.fp_per_1k, fn.fn_per_1k, latency)

        os.remove(var_path)

    return results


def print_table(lang, per_fold, variants, baseline):
    print(f'\n{"=" * 78}')
    print(f' {lang.upper()}  —  per-fold results ({len(per_fold)} folds)')
    print(f'{"=" * 78}')
    print(f'{"variant":<12} {"fold":>5} {"FP/1k":>9} {"FN/1k":>9} {"latency":>9} '
          f'{"ΔFP":>9} {"ΔFN":>9}')
    print('-' * 78)

    for name in variants:
        for fold, res in enumerate(per_fold):
            fp, fn, lat = res[name]
            bfp, bfn, _ = res[baseline]
            d_fp = '' if name == baseline else f'{fp - bfp:+9.3f}'
            d_fn = '' if name == baseline else f'{fn - bfn:+9.3f}'
            print(f'{name:<12} {fold:>5} {fp:>9.3f} {fn:>9.3f} {lat:>9.1f} {d_fp:>9} {d_fn:>9}')
        print('-' * 78)

    print(f'\n{"variant":<12} {"mean FP/1k":>12} {"mean FN/1k":>12} '
          f'{"folds better":>14} {"folds worse":>13}')
    for name in variants:
        fps = [r[name][0] for r in per_fold]
        fns = [r[name][1] for r in per_fold]
        better = sum(1 for r in per_fold if r[name][0] < r[baseline][0])
        worse = sum(1 for r in per_fold if r[name][0] > r[baseline][0])
        tag = '  (baseline)' if name == baseline else ''
        print(f'{name:<12} {statistics.mean(fps):>12.3f} {statistics.mean(fns):>12.3f} '
              f'{better:>14} {worse:>13}{tag}')


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='K-fold cross-validated comparison of quadgram model variants.')
    parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5).')
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Cap eligible lines per language (default: whole corpus).')
    parser.add_argument('--max-test-lines', type=int, default=75_000,
                        help='Cap lines scored per fold (default: 75000).  Training '
                             'still uses every line outside the fold.')
    parser.add_argument('--baseline-delta', type=float, default=4.0)
    parser.add_argument('--data-dir', default=default_data,
                        help='Directory holding the reference models and collisions.json.')
    parser.add_argument('--corpus-dir', default=None,
                        help='Directory holding {lang}_corpus.txt (default: --data-dir).')
    parser.add_argument('--cache-dir', default=os.path.join(default_data, 'kfold_cache'),
                        help='Where per-fold models are cached between runs.')
    parser.add_argument('--variants', default=','.join(VARIANTS),
                        help=f'Comma-separated. Available: {",".join(VARIANTS)}')
    parser.add_argument('--baseline-variant', default='unpruned',
                        help='Variant that deltas are measured against.')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1,
                        metavar='N', help='Worker processes for scoring.')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f'Unknown variant(s): {", ".join(unknown)}')
    if args.baseline_variant not in variants:
        variants.insert(0, args.baseline_variant)

    langs = ['en', 'he'] if args.lang == 'both' else [args.lang]
    corpus_dir = args.corpus_dir or args.data_dir

    for lang in langs:
        corpus = os.path.join(corpus_dir, f'{lang}_corpus.txt')
        if not os.path.exists(corpus):
            print(f'SKIP {lang}: {corpus} not found — run scripts/download_corpora.py')
            continue

        print(f'\nLoading {lang} corpus …', flush=True)
        lines = load_corpus_lines(corpus, lang, args.max_lines)
        per_fold_lines = len(lines) // args.k
        print(f'{len(lines):,} eligible lines, {args.k} folds '
              f'(~{per_fold_lines:,} per fold, scoring '
              f'{min(per_fold_lines, args.max_test_lines):,})')

        t0 = time.time()
        parts = build_fold_parts(lines, lang, args.k, args.cache_dir)
        per_fold = [
            run_fold(parts, lines, lang, fold, args.k, variants, args.cache_dir,
                     args.data_dir, args.baseline_delta, args.jobs,
                     args.max_test_lines)
            for fold in range(args.k)
        ]
        print_table(lang, per_fold, variants, args.baseline_variant)
        print(f'\n{lang}: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
