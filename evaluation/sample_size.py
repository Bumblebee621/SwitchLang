"""
sample_size.py — how many test lines the combination comparison actually needs.

combination_bench.py scores 75,000 lines per arm per variant, which is most of
the wall time of a sweep.  Whether that buys anything is answerable rather than
guessable: the binding constraint is not the number of lines but the number of
false positives they contain.  The so arm at 75,000 lines yields ~2.1M words and
176 FPs under max(); a count that small carries a relative standard error near
1/sqrt(176) = 7.5% no matter how many words it was divided by.

So the estimate's precision improves as 1/sqrt(N) forever and never hits a wall.
"Diminishing returns" only means something against a decision: this branch has
to separate variants whose FP/1k differ by ~0.02 out of ~0.10.  This script
measures the confidence interval as a function of N and reports the N at which
that separation becomes resolvable — and, where it is not resolvable at 75,000,
says so instead of letting the ranking be read off noise.

Method — block bootstrap.  Lines are scored once, in blocks, and every
resampling result is computed from those cached per-block counts, so the cost is
one sweep rather than one per sample size.  Blocks are the resampling unit
because a line is the independent unit in the harness (each line resets layout,
sensitivity, and history) and blocks of lines inherit that independence while
keeping the bootstrap cheap.

Comparisons against the baseline are *paired* on blocks — the same lines score
both variants, so the shared line-to-line difficulty cancels and the interval on
the difference is far tighter than the two separate intervals would suggest.

Usage:
    python evaluation/sample_size.py --arm so
    python evaluation/sample_size.py --arm he --variants standard,max,mixture:0.5
"""

import argparse
import os
import random
import statistics
import sys
import time
from concurrent.futures import as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import _get_pool
from combination_bench import _run_chunk, resolve_variant
from technical_mode import build_so_arm, build_he_arm, build_en_arm

DEFAULT_VARIANTS = ('standard', 'max', 'mixture:0.35', 'mixture:0.5', 'merged:18')


def score_blocks(lines, block_size, lang, delta, key, jobs):
    """Score every block once, returning per-block counts.

    Returns a list of (fp_count, fp_words, fn_missed, fn_words) — the raw
    numerators and denominators, kept separate so a resample can pool them
    correctly.  Averaging per-block rates instead would weight a short block the
    same as a long one.
    """
    blocks = [lines[i:i + block_size] for i in range(0, len(lines), block_size)]
    pool = _get_pool(jobs)

    fp_jobs = {pool.submit(_run_chunk, ('fp', i, b, lang, delta, key)): i
               for i, b in enumerate(blocks)}
    fn_jobs = {pool.submit(_run_chunk, ('fn', i, b, lang, delta, key)): i
               for i, b in enumerate(blocks)}

    fp = [None] * len(blocks)
    fn = [None] * len(blocks)
    for fut in as_completed(list(fp_jobs) + list(fn_jobs)):
        if fut in fp_jobs:
            r = fut.result()
            fp[fp_jobs[fut]] = (r.fp_count, r.words_tested)
        else:
            r = fut.result()
            fn[fn_jobs[fut]] = (r.words_not_switched, r.words_tested)

    return [(fp[i][0], fp[i][1], fn[i][0], fn[i][1]) for i in range(len(blocks))]


def _rate(blocks, idx, num_i, den_i):
    """Pooled per-1k rate over the blocks named by idx."""
    num = sum(blocks[i][num_i] for i in idx)
    den = sum(blocks[i][den_i] for i in idx)
    return (num / den * 1000) if den else 0.0


def bootstrap_ci(blocks, n_blocks, num_i, den_i, reps, rng):
    """95% CI for the pooled rate over a resample of n_blocks blocks."""
    n = len(blocks)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n_blocks)]
        vals.append(_rate(blocks, idx, num_i, den_i))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return statistics.mean(vals), lo, hi


def bootstrap_paired(blocks_a, blocks_b, n_blocks, num_i, den_i, reps, rng):
    """95% CI for the % change from b (baseline) to a, paired on blocks."""
    n = len(blocks_a)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n_blocks)]
        base = _rate(blocks_b, idx, num_i, den_i)
        new = _rate(blocks_a, idx, num_i, den_i)
        vals.append(((new - base) / base * 100) if base else 0.0)
    vals.sort()
    return (statistics.mean(vals),
            vals[int(0.025 * len(vals))],
            vals[min(int(0.975 * len(vals)), len(vals) - 1)])


def print_convergence(name, blocks, block_size, fractions, reps, rng):
    """CI half-width against N, for FP/1k and FN/1k, for one variant."""
    n = len(blocks)
    total_fp = sum(b[0] for b in blocks)
    print(f'\n{name}: {total_fp} FPs over {n} blocks '
          f'({n * block_size:,} lines)')
    print(f'  {"lines":>8} {"FPs":>6} {"FP/1k":>8} {"95% CI":>17} {"±%":>7} '
          f'{"FN/1k":>8} {"95% CI":>17} {"±%":>7}')
    for frac in fractions:
        nb = max(1, int(n * frac))
        fp_m, fp_lo, fp_hi = bootstrap_ci(blocks, nb, 0, 1, reps, rng)
        fn_m, fn_lo, fn_hi = bootstrap_ci(blocks, nb, 2, 3, reps, rng)
        fp_half = (fp_hi - fp_lo) / 2
        fn_half = (fn_hi - fn_lo) / 2
        print(f'  {nb * block_size:>8,} {int(total_fp * frac):>6} '
              f'{fp_m:>8.3f} [{fp_lo:>6.3f},{fp_hi:>6.3f}] '
              f'{fp_half / fp_m * 100 if fp_m else 0:>6.0f}% '
              f'{fn_m:>8.3f} [{fn_lo:>6.3f},{fn_hi:>6.3f}] '
              f'{fn_half / fn_m * 100 if fn_m else 0:>6.0f}%')


def print_pairwise(scored, baseline, block_size, fractions, reps, rng):
    """Paired % change vs baseline with CIs — the decision-relevant quantity.

    A CI that straddles zero means the variant is not distinguishable from the
    baseline at that sample size, whatever the point estimate suggests.
    """
    base = scored[baseline]
    n = len(base)
    for name, blocks in scored.items():
        if name == baseline:
            continue
        print(f'\n{name} vs {baseline} (paired on blocks)')
        print(f'  {"lines":>8} {"FP/1k change":>26} {"FN/1k change":>26}')
        for frac in fractions:
            nb = max(1, int(n * frac))
            fp_m, fp_lo, fp_hi = bootstrap_paired(blocks, base, nb, 0, 1, reps, rng)
            fn_m, fn_lo, fn_hi = bootstrap_paired(blocks, base, nb, 2, 3, reps, rng)
            fp_sig = ' ' if fp_lo <= 0 <= fp_hi else '*'
            fn_sig = ' ' if fn_lo <= 0 <= fn_hi else '*'
            print(f'  {nb * block_size:>8,} '
                  f'{fp_m:>+8.0f}% [{fp_lo:>+6.0f}%,{fp_hi:>+6.0f}%]{fp_sig} '
                  f'{fn_m:>+8.0f}% [{fn_lo:>+6.0f}%,{fn_hi:>+6.0f}%]{fn_sig}')
    print('\n* = 95% CI excludes zero (difference resolvable at that N)')


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='Find where extra test lines stop buying precision.')
    parser.add_argument('--arm', default='so', choices=['so', 'en', 'he'])
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS))
    parser.add_argument('--baseline', default='standard')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max-test-lines', type=int, default=75_000)
    parser.add_argument('--block-size', type=int, default=500,
                        help='Lines per bootstrap block (default: 500).')
    parser.add_argument('--reps', type=int, default=2000,
                        help='Bootstrap resamples (default: 2000).')
    parser.add_argument('--baseline-delta', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-dir', default=default_data)
    parser.add_argument('--cache-dir', default=os.path.join(default_data, 'fold_counts'))
    parser.add_argument('--work-dir', default=os.path.join(default_data, 'technical_mode'))
    parser.add_argument('--merged-dir', default=os.path.join(default_data, 'combination_exp'))
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    rng = random.Random(args.seed)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    if args.arm == 'so':
        kwargs, test, eligible = build_so_arm(
            args.data_dir, args.work_dir, args.k, args.fold)
    elif args.arm == 'he':
        kwargs, test, eligible = build_he_arm(
            args.data_dir, args.cache_dir, args.work_dir, args.k, args.fold)
    else:
        kwargs, test, eligible = build_en_arm(args.data_dir, args.max_test_lines)

    test = test[:args.max_test_lines]
    lang = kwargs.pop('lang')
    data_dir = kwargs.pop('data_dir')
    so_flavour = 'heldout' if args.arm == 'so' else 'shipped'
    print(f'{args.arm}: {eligible:,} eligible, scoring {len(test):,} '
          f'in blocks of {args.block_size}')

    scored = {}
    for spec in variants:
        key = resolve_variant(spec, kwargs, data_dir, so_flavour, args.merged_dir)
        t0 = time.time()
        scored[spec] = score_blocks(test, args.block_size, lang,
                                    args.baseline_delta, key, args.jobs)
        print(f'  scored {spec:<14} ({time.time() - t0:.0f}s)', flush=True)

    fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f'\n{"=" * 78}\nPRECISION vs SAMPLE SIZE\n{"=" * 78}')
    for spec in variants:
        print_convergence(spec, scored[spec], args.block_size, fractions,
                          args.reps, rng)

    print(f'\n{"=" * 78}\nPAIRWISE, vs {args.baseline}\n{"=" * 78}')
    print_pairwise(scored, args.baseline, args.block_size, fractions,
                   args.reps, rng)


if __name__ == '__main__':
    main()
