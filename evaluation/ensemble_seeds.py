"""Does averaging the best 5 of 20 Char-GRU seeds beat the quadgram, and is the gain from picking or from averaging?

Seeds share hyperparameters and validation loss but differ a lot in switching cost. Each seed is scored on a
selection split; the best --top form the "best ensemble" (probabilities averaged). Seeds 0..top-1 form a
"random ensemble" control. Both, the best single seed and the quadgram are scored on a disjoint test split.

Usage:
    .venv/bin/python evaluation/ensemble_seeds.py OUT.json [--seeds 20] [--top 5] [--epochs 15] [--train-words 300000] [--jobs N]
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.tune_char_gru as tune
from core.engine import FAST_SWITCH_MARGIN
from core.neural_model import UNIGRAM_FLOOR

MAX_LINES = 150_000   # corpus read cap; prepare_split trims to the word budget


def halve(lines):
    """(even-index lines, odd-index lines)."""
    return lines[0::2], lines[1::2]


def ensemble_pair(pairs):
    """Model-path dicts -> one dict whose paths are tuples (an ensemble), in input order."""
    return {key: tuple(p[key] for p in pairs) for key in ('en_model_path', 'he_model_path')}


def rank_seeds(rows):
    """Seeds from cheapest to costliest selection score."""
    return [r['seed'] for r in sorted(rows, key=lambda r: r['result']['score'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--train-words', type=int, default=300_000)
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    args = parser.parse_args()
    if not 1 <= args.top <= args.seeds:
        parser.error('--top must be between 1 and --seeds')
    data_dir = os.path.join(ROOT, 'data')
    hp = {**tune.BASELINE_PARAMS, 'cosine': True}
    tune.K = 2   # held-out side ≈ train_words, halved below into selection and test
    work = tempfile.mkdtemp(prefix='switchlang_ensemble_')
    try:
        split = tune.prepare_split(data_dir, MAX_LINES, args.train_words, args.jobs, work)
        split['epochs'] = args.epochs
        halves = {lang: halve(split['test'][lang]) for lang in tune.LANGS}
        sel = {**split, 'test': {lang: h[0] for lang, h in halves.items()}}
        tst = {**split, 'test': {lang: h[1] for lang, h in halves.items()}}

        memo, pairs, rows = {}, {}, []

        def train(seed):
            d = os.path.join(work, f'seed{seed}')
            os.makedirs(d)
            return tune.train_pair(hp, tune.LANGS, split, seed, d, memo)

        # ponytail: trains one seed ahead; a deeper queue only helps if training outlasts scoring
        with ThreadPoolExecutor(max_workers=1) as trainer:
            pending = trainer.submit(train, 0)
            for seed in range(args.seeds):
                pair, val = pending.result()
                if seed + 1 < args.seeds:
                    pending = trainer.submit(train, seed + 1)
                res = tune.score([pair], sel, UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, args.jobs)
                pairs[seed] = pair
                rows.append({'seed': seed, 'val_loss': val, 'result': res})
                print(f"seed {seed:>2}: {tune._fmt(res)} | val en {val['en']:.4f} he {val['he']:.4f}", flush=True)

        selected = rank_seeds(rows)[:args.top]
        random_seeds = list(range(args.top))
        tests = {
            'quadgram': tune.score([split['quad']], tst, 0.0, math.inf, data_dir, args.jobs, model_type='quadgram'),
            'best single': tune.score([pairs[selected[0]]], tst, UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, args.jobs),
            'random ensemble': tune.score([ensemble_pair([pairs[s] for s in random_seeds])], tst,
                                          UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, args.jobs),
            'best ensemble': tune.score([ensemble_pair([pairs[s] for s in selected])], tst,
                                        UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, args.jobs),
        }
        print(f"\nTest split (cost = {tune.W_FP}·FP + FN per 1k words); selected seeds {selected}, random seeds {random_seeds}")
        for name, res in tests.items():
            print(f"{name:<16} {tune._fmt(res)}")
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'seeds': args.seeds, 'top': args.top, 'epochs': args.epochs, 'train_words': args.train_words,
                       'selection': rows, 'selected': selected,
                       'test': [{'name': name, 'result': res} for name, res in tests.items()]}, fh, indent=2)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':   # forkserver workers re-import this file
    main()
