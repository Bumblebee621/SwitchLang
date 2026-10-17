"""Seed variance of the baseline Char-GRU under the tuner's weighted cost.

Same training setup as the 2026-10-09 tuning run (EN+HE at BASELINE_PARAMS, 300k words, 3 epochs),
but K=3 so the held-out test set is ~150k words per language. Every seed is scored on the same
test lines, so the spread across seeds is model variance, not test-sampling noise. Seed 0 is
trained twice to check that training is deterministic for a fixed seed.

Usage:
    .venv/bin/python evaluation/seed_variance.py OUT.json [--epochs N] [--cosine]

Measured 2026-10-09: cost sd 0.49 across 6 seeds of one config (range 3.20-4.63), larger than the
spread across all 45 configs of the tuning run; see evaluation/EXPERIMENTS.md.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.tune_char_gru as tune
from core.engine import FAST_SWITCH_MARGIN
from core.neural_model import UNIGRAM_FLOOR

SEEDS = [0, 1, 2, 3, 4, 5, 0]   # trailing 0 = determinism check
tune.K = 3                       # test ≈ train / 2 = 150k words


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('out')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cosine', action='store_true', help="cosine-decay the lr to 0 (tests whether longer, converged training shrinks seed variance)")
    args = parser.parse_args()
    data_dir = os.path.join(ROOT, 'data')
    work = tempfile.mkdtemp(prefix='switchlang_seedvar_')
    try:
        split = tune.prepare_split(data_dir, 150_000, 300_000, os.cpu_count(), work)
        split['epochs'] = args.epochs
        rows = []
        for i, seed in enumerate(SEEDS):
            run_dir = os.path.join(work, f'run{i}')
            os.makedirs(run_dir)
            pair, val = tune.train_pair({**tune.BASELINE_PARAMS, 'cosine': args.cosine}, ('en',), split, seed, run_dir, {})
            res = tune.score([pair], split, UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, os.cpu_count())
            rows.append({'seed': seed, 'val_loss': val, **res})
            print(f"seed {seed}: {tune._fmt(res)} | val en {val['en']:.4f} he {val['he']:.4f}", flush=True)

        distinct = rows[:-1]
        costs = [r['score'] for r in distinct]
        print(f"\nreference: {tune._fmt(split['reference'])}")
        print(f"cost across {len(distinct)} seeds: mean {statistics.mean(costs):.3f} sd {statistics.stdev(costs):.3f} "
              f"min {min(costs):.3f} max {max(costs):.3f}; mean Poisson ± {statistics.mean(r['ci'] for r in distinct):.2f}")
        for key in ('fp', 'fn'):
            for lang in tune.LANGS:
                v = [r[key][lang] for r in distinct]
                print(f"{key.upper()} {lang}: mean {statistics.mean(v):.3f} sd {statistics.stdev(v):.3f} "
                      f"range {min(v):.3f}-{max(v):.3f}")
        # Compare the switching results, not val loss: GPU float rounding moves val loss in the 8th digit.
        same = all(rows[0][k] == rows[-1][k] for k in ('score', 'fp', 'fn', 'delta'))
        print(f"seed 0 repeated identically: {same} "
              f"(cost {rows[0]['score']:.4f} vs {rows[-1]['score']:.4f}, val en {rows[0]['val_loss']['en']:.6f} vs {rows[-1]['val_loss']['en']:.6f})")
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'epochs': args.epochs, 'cosine': args.cosine, 'reference': split['reference'], 'rows': rows}, fh, indent=2)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':   # forkserver workers re-import this file
    main()
