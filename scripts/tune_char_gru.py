"""
tune_char_gru.py — Reusable Hyperparameter Optimization for SwitchLang Char-GRU models.

Features:
- Multi-core parallel trial evaluation across CPU cores (--jobs N).
- Coarse-to-fine search mode (--coarse-to-fine) scanning architecture first, then regularization.
- Unbiased held-out validation loss & perplexity ranking with zero data leakage.
- Direct comparison scoreboard against the production baseline.
- Automatic weight export for winning configurations.

Usage:
    # Coarse-to-fine optimization using 4 CPU workers:
    python scripts/tune_char_gru.py --lang en --coarse-to-fine --jobs 4 --max-words 50000 --epochs 3

    # Parallel 20-trial random search:
    python scripts/tune_char_gru.py --lang en --trials 20 --jobs 4 --max-words 50000

    # Fast self-test:
    python scripts/tune_char_gru.py --test
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import logging
import math
import os
import random
import sys
import time


# Ensure switchlang modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_char_gru import (
    train_model,
    export_numpy_weights,
    load_words_from_corpus,
    EN_CHARS,
    HE_CHARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('tune_char_gru')

# Production baseline hyperparameters
BASELINE_PARAMS = {
    'emb_dim': 64,
    'hidden_dim': 128,
    'lr': 0.01,
    'dropout': 0.1,
    'weight_decay': 1e-4,
    'batch_size': 512,
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


def generate_candidates(trials=10, grid=False, search_space=None, seed=42):
    """Generate candidate hyperparameter configurations with baseline as candidate #0."""
    space = search_space or DEFAULT_SEARCH_SPACE
    candidates = [dict(BASELINE_PARAMS)]

    if grid:
        keys = list(space.keys())
        combos = list(itertools.product(*(space[k] for k in keys)))
        for c in combos:
            item = dict(zip(keys, c))
            if item != BASELINE_PARAMS:
                candidates.append(item)
            if len(candidates) >= trials:
                break
    else:
        rng = random.Random(seed)
        seen = {tuple(sorted(BASELINE_PARAMS.items()))}
        attempts = 0
        while len(candidates) < trials and attempts < trials * 50:
            attempts += 1
            cand = {k: rng.choice(v) for k, v in space.items()}
            key = tuple(sorted(cand.items()))
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

    return candidates[:trials]


def generate_coarse_candidates():
    """Stage 1: Architecture & LR grid expanded toward higher hidden_dim & lr (27 combinations)."""
    combos = itertools.product([32, 48, 64], [64, 80, 96], [0.004, 0.006, 0.008])
    return [
        {
            'emb_dim': emb,
            'hidden_dim': hid,
            'lr': lr,
            'dropout': 0.05,
            'weight_decay': 0.0,
            'batch_size': 256,
        }
        for emb, hid, lr in combos
    ]


def generate_fine_candidates(best_arch):
    """Stage 2: Regularization & Batch size grid around best architecture (18 combinations)."""
    combos = itertools.product([0.0, 0.05, 0.10], [0.0, 1e-5, 1e-4], [128, 256])
    return [
        {
            'emb_dim': best_arch['emb_dim'],
            'hidden_dim': best_arch['hidden_dim'],
            'lr': best_arch['lr'],
            'dropout': drop,
            'weight_decay': wd,
            'batch_size': bs,
        }
        for drop, wd, bs in combos
    ]


def _eval_candidate(job):
    """Worker function for single candidate evaluation."""
    cand, train_words, val_words, lang, epochs = job
    import torch
    torch.set_num_threads(1)

    t0 = time.time()
    _, vocab_info = train_model(
        train_words,
        lang=lang,
        emb_dim=cand['emb_dim'],
        hidden_dim=cand['hidden_dim'],
        epochs=epochs,
        batch_size=cand['batch_size'],
        lr=cand['lr'],
        dropout=cand['dropout'],
        weight_decay=cand['weight_decay'],
        val_words=val_words,
    )
    elapsed = time.time() - t0

    val_loss = vocab_info.get('val_loss', float('inf'))
    train_loss = vocab_info.get('train_loss', float('inf'))
    perp = math.exp(min(val_loss, 20.0))
    n_params, size_kb = compute_param_count(vocab_info['vocab_size'], cand['emb_dim'], cand['hidden_dim'])

    return {
        'params': cand,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'perplexity': perp,
        'params_count': n_params,
        'size_kb': size_kb,
        'time_sec': elapsed,
    }


def _run_batch_eval(candidates, train_words, val_words, lang, epochs, jobs=1):
    """Evaluate a batch of candidates serially or in parallel across processes."""
    job_payloads = [(c, train_words, val_words, lang, epochs) for c in candidates]

    if jobs > 1 and len(candidates) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(_eval_candidate, job_payloads))
    else:
        return [_eval_candidate(p) for p in job_payloads]


def tune_hyperparameters(
    words,
    lang='en',
    trials=10,
    epochs=3,
    val_ratio=0.15,
    search_space=None,
    grid=False,
    coarse_to_fine=False,
    jobs=1,
    seed=42,
    export_best=None,
    verbose=True,
):
    """Run reusable hyperparameter optimization sweep for Char-GRU model.

    Args:
        words: List of words for training and validation.
        lang: Language code ('en' or 'he').
        trials: Number of configurations to evaluate (ignored if coarse_to_fine=True).
        epochs: Number of training epochs per candidate.
        val_ratio: Fraction of words held out for validation (default: 0.15).
        search_space: Optional dict mapping parameter names to lists of candidate values.
        grid: If True, sweeps grid combinations; otherwise random search.
        coarse_to_fine: If True, performs 2-stage coarse-to-fine search (Stage 1: Arch, Stage 2: Reg).
        jobs: Number of parallel worker processes (default: 1).
        seed: Random seed for dataset split and candidate sampling.
        export_best: Optional path base (e.g. 'data/best_model') to save winning model weights.
        verbose: If True, prints formatted summary table to stdout.

    Returns:
        tuple: (best_params, all_results_sorted_by_val_loss)
    """
    if len(words) < 20:
        raise ValueError(f"Too few words ({len(words)}) for tuning; need at least 20.")

    # Deterministic train/validation split without data leakage
    rng = random.Random(seed)
    shuffled_words = list(words)
    rng.shuffle(shuffled_words)

    n_val = max(5, int(len(shuffled_words) * val_ratio))
    val_words = shuffled_words[:n_val]
    train_words = shuffled_words[n_val:]

    raw_results = []
    baseline_loss = None

    if coarse_to_fine:
        logger.info(
            "Starting Coarse-to-Fine search across %d workers (%d train words, %d val words)...",
            jobs, len(train_words), len(val_words)
        )
        # Stage 1: Coarse architecture search (27 trials + baseline)
        stage1_cands = [dict(BASELINE_PARAMS)] + [
            c for c in generate_coarse_candidates() if c != BASELINE_PARAMS
        ]
        logger.info("── Stage 1: Evaluating %d Architecture & LR configurations ──", len(stage1_cands))
        stage1_res = _run_batch_eval(stage1_cands, train_words, val_words, lang, epochs, jobs=jobs)
        raw_results.extend(stage1_res)

        # Baseline loss is trial #0
        baseline_loss = stage1_res[0]['val_loss']

        # Find best architecture from Stage 1
        best_s1 = min(stage1_res, key=lambda r: r['val_loss'])
        best_arch = best_s1['params']
        logger.info(
            "Stage 1 Best: emb=%d, hidden=%d, lr=%.4f (Val Loss: %.4f)",
            best_arch['emb_dim'], best_arch['hidden_dim'], best_arch['lr'], best_s1['val_loss']
        )

        # Stage 2: Regularization fine-tuning (24 trials)
        stage2_cands = generate_fine_candidates(best_arch)
        logger.info("── Stage 2: Fine-tuning Regularization & Batch size (%d combinations) ──", len(stage2_cands))
        stage2_res = _run_batch_eval(stage2_cands, train_words, val_words, lang, epochs, jobs=jobs)
        raw_results.extend(stage2_res)

    else:
        candidates = generate_candidates(trials=trials, grid=grid, search_space=search_space, seed=seed)
        logger.info(
            "Starting hyperparameter sweep (%d trials, %d workers, %d train words, %d val words)...",
            len(candidates), jobs, len(train_words), len(val_words)
        )
        raw_results = _run_batch_eval(candidates, train_words, val_words, lang, epochs, jobs=jobs)
        baseline_loss = raw_results[0]['val_loss']

    # Process and rank results
    results = []
    seen_params = set()

    for idx, r in enumerate(raw_results):
        param_key = tuple(sorted(r['params'].items()))
        if param_key in seen_params:
            continue
        seen_params.add(param_key)

        is_baseline = (r['params'] == BASELINE_PARAMS)
        val_loss = r['val_loss']
        d_loss = val_loss - (baseline_loss if baseline_loss is not None else val_loss)

        record = dict(r)
        record['trial'] = len(results) + 1
        record['is_baseline'] = is_baseline
        record['delta_val_loss'] = d_loss
        results.append(record)

    results.sort(key=lambda r: r['val_loss'])

    if verbose:
        print_scoreboard(lang, results, baseline_loss)

    best_record = results[0]
    best_cand = best_record['params']

    # Export best model if requested
    if export_best:
        logger.info("Training final winning model for export...")
        best_model, best_vocab_info = train_model(
            train_words + val_words,
            lang=lang,
            emb_dim=best_cand['emb_dim'],
            hidden_dim=best_cand['hidden_dim'],
            epochs=epochs,
            batch_size=best_cand['batch_size'],
            lr=best_cand['lr'],
            dropout=best_cand['dropout'],
            weight_decay=best_cand['weight_decay'],
        )
        export_numpy_weights(best_model, best_vocab_info, export_best)
        config_path = f"{export_best}_hyperparams.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': best_cand,
                'val_loss': best_record['val_loss'],
                'baseline_loss': baseline_loss,
                'delta_loss': best_record['val_loss'] - (baseline_loss or best_record['val_loss']),
            }, f, indent=2)
        logger.info("Saved best hyperparameters config: %s", config_path)

    return best_cand, results


def print_scoreboard(lang, results, baseline_loss):
    """Print formatted hyperparameter comparison scoreboard."""
    print(f"\n{'=' * 105}")
    print(f" {lang.upper()} CHAR-GRU HYPERPARAMETER OPTIMIZATION SCOREBOARD ({len(results)} UNIQUE TRIALS)")
    print(f"{'=' * 105}")
    print(
        f"{'Trial':<8} {'emb':>4} {'hid':>4} {'drop':>5} {'lr':>7} {'wd':>8} {'batch':>5} | "
        f"{'Val Loss':>8} {'Perp':>7} {'ΔLoss':>8} | {'Params':>8} {'Size':>7} {'Time':>6}"
    )
    print("-" * 105)

    for r in results:
        p = r['params']
        b_mark = " (B)" if r['is_baseline'] else ""
        trial_str = f"#{r['trial']}{b_mark}"
        d_str = f"{r['delta_val_loss']:>+8.4f}" if baseline_loss is not None else "     0.0"

        print(
            f"{trial_str:<8} {p['emb_dim']:>4} {p['hidden_dim']:>4} {p['dropout']:>5.2f} "
            f"{p['lr']:>7.4f} {p['weight_decay']:>8.1e} {p['batch_size']:>5} | "
            f"{r['val_loss']:>8.4f} {r['perplexity']:>7.2f} {d_str} | "
            f"{r['params_count']:>8,} {r['size_kb']:>6.1f}K {r['time_sec']:>5.1f}s"
        )

    print("-" * 105)
    best = results[0]
    p_best = best['params']
    print(f"OPTIMAL: #{best['trial']} (Val Loss: {best['val_loss']:.4f}, Perplexity: {best['perplexity']:.2f})")
    print(
        f"  emb_dim={p_best['emb_dim']}, hidden_dim={p_best['hidden_dim']}, "
        f"lr={p_best['lr']}, dropout={p_best['dropout']}, "
        f"weight_decay={p_best['weight_decay']}, batch_size={p_best['batch_size']}"
    )
    print(f"{'=' * 105}\n")


def main():
    default_jobs = min(4, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for SwitchLang Char-GRU models")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='en', help="Language to tune (default: en)")
    parser.add_argument('--data-dir', default='data', help="Data directory containing corpora")
    parser.add_argument('--max-words', type=int, default=50_000, help="Number of corpus words to load (default: 50000)")
    parser.add_argument('--trials', type=int, default=10, help="Number of hyperparameter trials (default: 10)")
    parser.add_argument('--epochs', type=int, default=3, help="Training epochs per trial (default: 3)")
    parser.add_argument('--coarse-to-fine', action='store_true', help="Run 2-stage coarse-to-fine grid search")
    parser.add_argument('--grid', action='store_true', help="Run standard Cartesian grid sweep")
    parser.add_argument('-j', '--jobs', type=int, default=default_jobs, help=f"Parallel worker processes (default: {default_jobs})")
    parser.add_argument('--export-best', metavar='PATH_BASE', help="Export winning weights to .npz/.meta.json")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--test', action='store_true', help="Run fast self-test")
    args = parser.parse_args()

    if args.test:
        test_words = ['hello', 'world', 'system', 'language', 'switch', 'keyboard', 'testing'] * 30
        print("[SELF-TEST] Running fast hyperparameter optimization check...")
        best_p, res = tune_hyperparameters(test_words, lang='en', trials=2, epochs=1, jobs=1, verbose=True)
        print(f"[SELF-TEST] Success! Best params: {best_p}")
        return

    data_dir = os.path.abspath(args.data_dir)
    langs = ['en', 'he'] if args.lang == 'both' else [args.lang]

    for lang in langs:
        corpus_path = os.path.join(data_dir, f"{lang}_corpus.txt")
        if not os.path.exists(corpus_path):
            logger.error("Corpus not found: %s", corpus_path)
            continue

        raw_chars = EN_CHARS if lang == 'en' else HE_CHARS
        words = load_words_from_corpus(corpus_path, max_words=args.max_words, allowed_chars=raw_chars)

        tune_hyperparameters(
            words,
            lang=lang,
            trials=args.trials,
            epochs=args.epochs,
            grid=args.grid,
            coarse_to_fine=args.coarse_to_fine,
            jobs=args.jobs,
            seed=args.seed,
            export_best=args.export_best,
            verbose=True,
        )


if __name__ == '__main__':
    main()
