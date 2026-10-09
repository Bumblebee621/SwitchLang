"""
tune_char_gru.py — Char-GRU hyperparameter search ranked by switching quality.

Objective: an EN+HE GRU pair is scored by weighted cost, W_FP·(FP_en + FP_he) + (FN_en + FN_he),
per 1k words, with W_FP = 10. Every delta in DELTAS is evaluated and the cheapest one is kept.
The quadgram pair (fast switch off, as it ships) is scored the same way as a reference row.
Validation loss is reported, not ranked on.

--lang picks whose GRU takes the candidate hyperparameters; the other language trains with
BASELINE_PARAMS. Both languages are always scored: the app has one delta for both.

Phase 1 searches the training hyperparameters at the current UNIGRAM_FLOOR / FAST_SWITCH_MARGIN,
by random search (--trials) or --coarse-to-fine (shape and lr, then regularization).
Phase 2 sweeps floor x margin on the phase-1 winner and reports the tuned setting against
three ablations (no floor, no fast switch, neither), each at its own best setting.
FP and FN are means over --seeds training seeds.

Usage:
    .venv/bin/python scripts/tune_char_gru.py --lang en --coarse-to-fine --seeds 0 1 --out evaluation/tune_char_gru_en.json
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import FAST_SWITCH_MARGIN
from core.neural_model import UNIGRAM_FLOOR
from evaluation.benchmark import run_test
from evaluation.kfold_cv import ALLOWED, CHARS, extract_words_from_lines, load_pool
from scripts.build_quadgrams import build_quadgrams_from_lines, save_model_data_to_trie
from scripts.train_char_gru import GRU_HPARAMS, export_numpy_weights, train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('tune_char_gru')

K = 5   # held-out test = every K-th line, as fold 1 of kfold_cv

# Candidate #0: the shipped training setup. Epochs are not searched.
BASELINE_PARAMS = {k: v for k, v in GRU_HPARAMS.items() if k != 'epochs'}

LANGS = ('en', 'he')
DELTAS = (6.0, 7.5, 9.0, 10.5, 12.0, 13.5)
W_FP = 10   # a wrong switch costs as much as ten missed ones
FLOORS = (0.0, 0.02, 0.05, 0.1)
MARGINS = (math.inf, 2.5, 3.5, 5.0)                  # inf = fast switch off


def weighted_cost(fp_at, fn_at, deltas=DELTAS, w=W_FP):
    """Lowest w·FP + FN (each summed over languages, per 1k words) across *deltas*.

    fp_at(delta) -> {lang: FP/1k}; fn_at(delta) -> {lang: {'fn': FN/1k, 'med': median latency}}.
    Returns the result at the cheapest delta.
    """
    best_res = None
    for d in sorted(deltas):
        fp, fn = fp_at(d), fn_at(d)
        cost = w * sum(fp.values()) + sum(v['fn'] for v in fn.values())
        if best_res is None or cost < best_res['score']:
            best_res = {'delta': d, 'fp': fp,
                        'fn': {lang: v['fn'] for lang, v in fn.items()},
                        'med': {lang: v['med'] for lang, v in fn.items()},
                        'score': cost}
    return best_res


def rank_key(row):
    """Rows by score; rows without a result (e.g. an empty ablation) last."""
    return (row['result'] is None, row['result']['score'] if row['result'] else 0.0)


def best(rows, keep=lambda r: True):
    """Best-ranked row passing *keep*, or None."""
    return min((r for r in rows if keep(r)), key=rank_key, default=None)


def ablations(rows):
    """The tuned floor/margin and the three ablations, each at its own best phase-2 setting."""
    return {
        'tuned': best(rows),
        'no floor, fast switch': best(rows, lambda r: r['floor'] == 0 and r['margin'] != math.inf),
        'floor, no fast switch': best(rows, lambda r: r['floor'] > 0 and r['margin'] == math.inf),
        'neither': best(rows, lambda r: r['floor'] == 0 and r['margin'] == math.inf),
    }


# Default search space expanded toward higher capacity and higher learning rate
DEFAULT_SEARCH_SPACE = {
    'emb_dim': [32, 48, 64],
    'hidden_dim': [80, 96, 128, 140],
    'lr': [0.007, 0.01, 0.02, 0.03],
    'dropout': [0.0, 0.05, 0.10, 0.4, 0.7],
    'weight_decay': [0.0, 1e-5, 1e-4],
    'batch_size': [128, 256, 512],
}


def compute_param_count(vocab_size, emb_dim, hidden_dim):
    """Calculate total trainable parameter count and uncompressed size in KB."""
    n_params = (
        vocab_size * emb_dim
        + 3 * (hidden_dim * emb_dim + hidden_dim * hidden_dim + 2 * hidden_dim)
        + vocab_size * hidden_dim
        + vocab_size
    )
    size_kb = (n_params * 4) / 1024.0
    return n_params, size_kb


def generate_candidates(trials, seed=42, search_space=None):
    """Random search: baseline first, then unique random draws from the search space."""
    space = search_space or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)
    candidates = [dict(BASELINE_PARAMS)]
    seen = {tuple(sorted(BASELINE_PARAMS.items()))}
    for _ in range(trials * 50):
        if len(candidates) >= trials:
            break
        cand = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(cand)
    return candidates


def coarse_candidates():
    """Stage 1: shape and learning rate, regularization held at the baseline (27 configs, baseline first)."""
    grid = [{**BASELINE_PARAMS, 'emb_dim': e, 'hidden_dim': h, 'lr': lr}
            for e, h, lr in itertools.product([32, 48, 64], [64, 80, 96], [0.004, 0.006, 0.008])]
    return [dict(BASELINE_PARAMS)] + [c for c in grid if c != BASELINE_PARAMS]


def fine_candidates(arch):
    """Stage 2: regularization and batch size around the stage-1 winner *arch* (18 configs)."""
    return [{**arch, 'dropout': d, 'weight_decay': wd, 'batch_size': bs}
            for d, wd, bs in itertools.product([0.0, 0.05, 0.10], [0.0, 1e-5, 1e-4], [128, 256])]


def prepare_split(data_dir, max_lines, train_words, jobs, work_dir):
    """Hold out every K-th line per language and build quadgrams on the rest; score them as the reference."""
    split = {'test': {}, 'words': {}, 'quad': {}}
    for lang in LANGS:
        lines = load_pool(lang, data_dir, K, max_lines, train_words)
        train = [l for i, l in enumerate(lines) if i % K]
        split['test'][lang] = [l for i, l in enumerate(lines) if not i % K]
        split['words'][lang] = extract_words_from_lines(train, CHARS[lang], max_words=None)
        base = os.path.join(work_dir, f'quad_{lang}')
        counts = build_quadgrams_from_lines(train, ALLOWED[lang], min_count=2, num_workers=jobs)
        save_model_data_to_trie(counts, f'{base}.marisa', f'{base}.meta.json')
        split['quad'][f'{lang}_model_path'] = f'{base}.marisa'
    # Reference: the quadgram pair (fast switch off, as it ships) under the same weighted cost.
    split['reference'] = score([split['quad']], split, 0.0, math.inf, data_dir, jobs, model_type='quadgram')
    logger.info("Quadgram reference: %s", _fmt(split['reference']))
    return split


def train_one(lang, hp, seed, split, work_dir, memo):
    """Train one GRU on the split's words, memoized on (lang, hp, seed); return (npz path, val loss)."""
    key = (lang, tuple(sorted(hp.items())), seed)
    if key not in memo:
        words = split['words'][lang]
        n_val = len(words) // 20   # corpus-tail 5% held out, as train_char_gru does
        model, info = train_model(words[:-n_val], lang=lang, val_words=words[-n_val:], seed=seed, **hp)
        base = os.path.join(work_dir, f'gru{len(memo)}_{lang}')
        export_numpy_weights(model, info, base)
        memo[key] = (f'{base}.npz', info['val_loss'])
    return memo[key]


def train_pair(hp, searched, split, seed, work_dir, memo):
    """EN+HE GRUs: languages in *searched* take *hp*, the others BASELINE_PARAMS.

    Returns (run_test model-path kwargs, {lang: val loss}).
    """
    paths, val = {}, {}
    for lang in LANGS:
        paths[f'{lang}_model_path'], val[lang] = train_one(
            lang, hp if lang in searched else BASELINE_PARAMS, seed, split, work_dir, memo)
    return paths, val


def score(pairs, split, floor, margin, data_dir, jobs, model_type='neural'):
    """Weighted cost for one floor/margin, FP and FN averaged over the seed model *pairs*."""
    def run(test, lang, delta):
        return [run_test(test, split['test'][lang], lang, delta, data_dir, jobs=jobs,
                         model_type=model_type, floor=floor, fast_margin=margin, **p) for p in pairs]

    def fp_at(delta):
        return {lang: statistics.mean(r.fp_per_1k for r in run('fp', lang, delta)) for lang in LANGS}

    def fn_at(delta):
        out = {}
        for lang in LANGS:
            reports = run('fn', lang, delta)
            out[lang] = {'fn': statistics.mean(r.fn_per_1k for r in reports),
                         'med': statistics.mean(statistics.median(r.latency_values) if r.latency_values else 0.0
                                                for r in reports)}
        return out

    res = weighted_cost(fp_at, fn_at)
    if res['delta'] in (min(DELTAS), max(DELTAS)):
        logger.warning("best delta %g is at the edge of DELTAS %s; widen the grid", res['delta'], DELTAS)
    return res


def phase1(candidates, desc, searched, split, seeds, data_dir, jobs, work_dir, memo):
    """Score training-hyperparameter *candidates* at the current floor and margin, best first.

    The next candidate trains on the GPU in a background thread while the CPU pool scores this one.
    """
    def train_all(hp):
        return [train_pair(hp, searched, split, s, work_dir, memo) for s in seeds]

    rows = []
    # ponytail: trains one candidate ahead; a deeper queue only helps if training outlasts scoring
    with ThreadPoolExecutor(max_workers=1) as trainer:
        pending = trainer.submit(train_all, candidates[0])
        for i, hp in enumerate(tqdm(candidates, desc=desc, unit='config')):
            trained = pending.result()
            if i + 1 < len(candidates):
                pending = trainer.submit(train_all, candidates[i + 1])
            pairs = [p for p, _ in trained]
            rows.append({'params': hp, 'is_baseline': hp == BASELINE_PARAMS, 'pairs': pairs,
                         'val_loss': {lang: statistics.mean(v[lang] for _, v in trained) for lang in searched},
                         'result': score(pairs, split, UNIGRAM_FLOOR, FAST_SWITCH_MARGIN, data_dir, jobs)})
            logger.info("%s %d/%d %s -> %s", desc, i + 1, len(candidates), hp, _fmt(rows[-1]['result']))
    return sorted(rows, key=rank_key)


def phase2(pairs, split, data_dir, jobs):
    """Every floor x margin on fixed models. Floor is the outer loop: margins reuse the score cache."""
    settings = list(itertools.product(FLOORS, MARGINS))
    return [{'floor': f, 'margin': m, 'result': score(pairs, split, f, m, data_dir, jobs)}
            for f, m in tqdm(settings, desc='phase 2', unit='setting')]


def _fmt(res):
    if res is None:
        return "n/a"
    return (f"Δ{res['delta']:<5g} FP en {res['fp']['en']:.3f} he {res['fp']['he']:.3f} | "
            f"FN en {res['fn']['en']:.3f} he {res['fn']['he']:.3f} cost {res['score']:.3f} | "
            f"med lat en {res['med']['en']:.1f} he {res['med']['he']:.1f}")


def print_report(reference, searched, p1, comparison):
    print(f"\nQuadgram reference (fast switch off), cost = {W_FP}·FP + FN per 1k words: {_fmt(reference)}")
    print(f"\nPHASE 1 — training hyperparameters of {'+'.join(searched)} "
          f"(floor {UNIGRAM_FLOOR:g}, margin {FAST_SWITCH_MARGIN:g}), best first")
    for r in p1:
        p = r['params']
        val = ' '.join(f"{lang} {v:.4f}" for lang, v in r['val_loss'].items())
        tag = ' (baseline)' if r['is_baseline'] else ''
        print(f"  emb {p['emb_dim']:>3} hid {p['hidden_dim']:>3} drop {p['dropout']:.2f} lr {p['lr']:.4f} "
              f"wd {p['weight_decay']:.0e} bs {p['batch_size']:>3} | val {val} | {_fmt(r['result'])}{tag}")
    print("\nPHASE 2 — floor / fast-switch on the phase-1 winner")
    for name, r in comparison.items():
        knobs = '' if r is None else f"floor {r['floor']:g}, margin {r['margin']:g}"
        print(f"  {name:<22} {knobs:<24} {_fmt(r and r['result'])}")


def main():
    parser = argparse.ArgumentParser(description="Tune Char-GRU hyperparameters by weighted cost (W_FP·FP + FN)")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both',
                        help="Whose GRU takes the candidate hyperparameters; the other uses the baseline. "
                             "Both languages are always scored (default: both)")
    parser.add_argument('--coarse-to-fine', action='store_true',
                        help="Stage 1 shape x lr (27 configs), stage 2 regularization x batch (18) "
                             "instead of random search")
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--max-lines', type=int, default=150_000, help="Corpus lines read per language (default: 150000)")
    parser.add_argument('--train-words', type=int, default=1_000_000, help="Training words per language (default: 1000000)")
    parser.add_argument('--trials', type=int, default=12, help="Random-search candidates, baseline included (default: 12)")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1], help="Training seeds per candidate (default: 0 1)")
    parser.add_argument('--sample-seed', type=int, default=42, help="Seed for drawing random candidates (default: 42)")
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 4, help="Evaluation worker processes")
    parser.add_argument('--out', default=None, help="Write reference, both phases and the comparison to this JSON file")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    searched = LANGS if args.lang == 'both' else (args.lang,)
    work_dir = tempfile.mkdtemp(prefix='switchlang_tune_')
    memo = {}
    try:
        with logging_redirect_tqdm():
            split = prepare_split(data_dir, args.max_lines, args.train_words, args.jobs, work_dir)

            def run(candidates, desc):
                return phase1(candidates, desc, searched, split, args.seeds, data_dir, args.jobs, work_dir, memo)

            if args.coarse_to_fine:
                stage1 = run(coarse_candidates(), 'stage 1')
                p1 = sorted(stage1 + run(fine_candidates(stage1[0]['params']), 'stage 2'), key=rank_key)
            else:
                p1 = run(generate_candidates(args.trials, args.sample_seed), 'phase 1')
            p2 = phase2(p1[0]['pairs'], split, data_dir, args.jobs)
        comparison = ablations(p2)
        print_report(split['reference'], searched, p1, comparison)
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump({'lang': args.lang, 'mode': 'coarse-to-fine' if args.coarse_to_fine else 'random',
                           'reference': split['reference'], 'w_fp': W_FP, 'seeds': args.seeds,
                           'phase1': [{k: v for k, v in r.items() if k != 'pairs'} for r in p1],
                           'phase2': p2, 'comparison': comparison}, fh, indent=2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
