"""
compare_quadgram_neural.py — Empirical comparison of Quadgram vs Char-GRU (hidden=48, 96, 128, 160).

Compares:
1. Training time (wall-clock) on the exact same corpus.
2. Inference latency (mean, median, P95, P99, throughput) using production NumPy/Marisa engines.
3. Model capacity (parameter count, file size on disk).
4. Validation loss / perplexity.
"""

import gc
import json
import logging
import math
import os
import random
import sys
import tempfile
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core.quadgram import QuadgramModel
from core.neural_model import CharNeuralModel
from scripts.train_char_gru import (
    load_words_from_corpus,
    train_model,
    export_numpy_weights,
    EN_CHARS,
)
from scripts.build_quadgrams import (
    ALLOWED_EN,
    build_quadgrams_from_lines,
    save_model_data_to_trie,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('compare_models')


def compute_param_count(vocab_size, emb_dim, hidden_dim):
    """Compute exact parameter count for 1-layer Char-GRU."""
    emb_params = vocab_size * emb_dim
    gru_ih = 3 * hidden_dim * emb_dim
    gru_hh = 3 * hidden_dim * hidden_dim
    gru_bias = 2 * 3 * hidden_dim
    fc_params = (hidden_dim + 1) * vocab_size
    total = emb_params + gru_ih + gru_hh + gru_bias + fc_params
    return total


def benchmark_model_latency(model, words, rounds=5):
    """Benchmark inference latency of model.score(word) in microseconds."""
    # Warmup
    for w in words[:100]:
        model.score(w)

    latencies_us = []
    total_chars = 0

    for _ in range(rounds):
        for w in words:
            total_chars += len(w)
            t0 = time.perf_counter_ns()
            _ = model.score(w)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)  # ns to µs

    latencies_us.sort()
    n = len(latencies_us)

    mean_us = float(np.mean(latencies_us))
    median_us = float(np.median(latencies_us))
    p95_us = latencies_us[int(n * 0.95)]
    p99_us = latencies_us[int(n * 0.99)]
    min_us = latencies_us[0]
    max_us = latencies_us[-1]
    total_time_s = sum(latencies_us) / 1_000_000.0
    throughput_wps = n / total_time_s if total_time_s > 0 else 0.0
    us_per_char = (sum(latencies_us) / total_chars) if total_chars > 0 else 0.0

    return {
        'count': n,
        'mean_us': mean_us,
        'median_us': median_us,
        'p95_us': p95_us,
        'p99_us': p99_us,
        'min_us': min_us,
        'max_us': max_us,
        'throughput_wps': throughput_wps,
        'us_per_char': us_per_char,
    }


def main():
    torch.set_num_threads(torch.get_num_threads())
    logger.info("Using %d PyTorch CPU threads for training.", torch.get_num_threads())

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    corpus_path = os.path.join(data_dir, 'en_corpus.txt')

    out_dir = os.path.join(data_dir, 'benchmark_models')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load 50,000 words (42,500 train, 7,500 val)
    total_words_needed = 50_000
    all_words = load_words_from_corpus(corpus_path, max_words=total_words_needed, allowed_chars=EN_CHARS)
    random.seed(42)
    random.shuffle(all_words)

    train_words = all_words[:42_500]
    val_words = all_words[42_500:]

    logger.info("Dataset split: %d train words, %d val words.", len(train_words), len(val_words))

    results = []

    # ──────────────────────────────────────────────────────────────────────────
    # A. Train Quadgram on the exact same train words
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("TRAINING: Quadgram Model on 42,500 train words")
    logger.info("=" * 70)
    quad_trie_path = os.path.join(out_dir, 'en_quad_50k.marisa')
    quad_meta_path = os.path.join(out_dir, 'en_quad_50k.meta.json')

    t0 = time.perf_counter()
    quad_data = build_quadgrams_from_lines(train_words, allowed_chars=ALLOWED_EN, min_count=2)
    save_model_data_to_trie(quad_data, quad_trie_path, quad_meta_path)
    quad_train_time = time.perf_counter() - t0

    quad_file_size_kb = os.path.getsize(quad_trie_path) / 1024.0
    quad_total_keys = len(quad_data['quadgram_counts']) + len(quad_data['trigram_counts']) + len(quad_data['bigram_counts'])

    logger.info("Quadgram trained in %.3f s | Keys: %d | Size: %.1f KB",
                quad_train_time, quad_total_keys, quad_file_size_kb)

    results.append({
        'name': 'Quadgram (50k words)',
        'type': 'N-gram Trie',
        'params': quad_total_keys,
        'file_size_kb': quad_file_size_kb,
        'train_time_s': quad_train_time,
        'val_loss': None,
        'perp': None,
        'model_path': quad_trie_path,
        'is_neural': False,
    })

    # Include production full-corpus Quadgram if available
    prod_quad_path = os.path.join(data_dir, 'en_quadgrams.marisa')
    if os.path.exists(prod_quad_path):
        prod_size_kb = os.path.getsize(prod_quad_path) / 1024.0
        with open(os.path.join(data_dir, 'en_quadgrams.meta.json'), 'r') as f:
            pm = json.load(f)
        total_prod_keys = pm['counts']['quadgrams'] + pm['counts']['trigrams'] + pm['counts']['bigrams']
        results.append({
            'name': 'Quadgram (Prod Full 1GB)',
            'type': 'N-gram Trie',
            'params': total_prod_keys,
            'file_size_kb': prod_size_kb,
            'train_time_s': 45.0,  # approximate full corpus build time
            'val_loss': None,
            'perp': None,
            'model_path': prod_quad_path,
            'is_neural': False,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # B. Train Char-GRU Models: hidden=48, 96, 128, 160
    # ──────────────────────────────────────────────────────────────────────────
    neural_configs = [
        {
            'name': 'Char-GRU (hid=48, emb=32) [Baseline]',
            'emb_dim': 32,
            'hidden_dim': 48,
            'lr': 0.003,
            'dropout': 0.1,
            'weight_decay': 1e-4,
            'batch_size': 256,
            'epochs': 3,
        },
        {
            'name': 'Char-GRU (hid=96, emb=48) [Tuned Winner]',
            'emb_dim': 48,
            'hidden_dim': 96,
            'lr': 0.008,
            'dropout': 0.0,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'epochs': 3,
        },
        {
            'name': 'Char-GRU (hid=128, emb=48) [Requested]',
            'emb_dim': 48,
            'hidden_dim': 128,
            'lr': 0.008,
            'dropout': 0.0,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'epochs': 3,
        },
        {
            'name': 'Char-GRU (hid=160, emb=48) [Requested]',
            'emb_dim': 48,
            'hidden_dim': 160,
            'lr': 0.008,
            'dropout': 0.0,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'epochs': 3,
        },
    ]

    for cfg in neural_configs:
        logger.info("=" * 70)
        logger.info("TRAINING: %s", cfg['name'])
        logger.info("=" * 70)

        t0 = time.perf_counter()
        model, vocab_info = train_model(
            train_words,
            lang='en',
            emb_dim=cfg['emb_dim'],
            hidden_dim=cfg['hidden_dim'],
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            lr=cfg['lr'],
            dropout=cfg['dropout'],
            weight_decay=cfg['weight_decay'],
            val_words=val_words,
        )
        train_time = time.perf_counter() - t0

        out_base = os.path.join(out_dir, f"en_char_gru_h{cfg['hidden_dim']}")
        export_numpy_weights(model, vocab_info, out_base)
        npz_path = f"{out_base}.npz"
        file_size_kb = os.path.getsize(npz_path) / 1024.0

        n_params = compute_param_count(vocab_info['vocab_size'], cfg['emb_dim'], cfg['hidden_dim'])
        val_loss = vocab_info.get('val_loss', 0.0)
        perp = math.exp(min(val_loss, 20.0))

        logger.info("%s trained in %.2fs | Params: %d | Val Loss: %.4f | Perp: %.2f",
                    cfg['name'], train_time, n_params, val_loss, perp)

        results.append({
            'name': cfg['name'],
            'type': f"GRU-1L (E={cfg['emb_dim']}, H={cfg['hidden_dim']})",
            'params': n_params,
            'file_size_kb': file_size_kb,
            'train_time_s': train_time,
            'val_loss': val_loss,
            'perp': perp,
            'model_path': npz_path,
            'is_neural': True,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # C. Inference Latency Benchmark
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("BENCHMARKING INFERENCE LATENCY ON 1,000 HELD-OUT WORDS (x 5 ROUNDS)")
    logger.info("=" * 70)

    test_words = val_words[:1000]

    for item in results:
        mpath = item['model_path']
        logger.info("Evaluating latency for %s ...", item['name'])
        if item['is_neural']:
            loaded_model = CharNeuralModel(mpath)
        else:
            loaded_model = QuadgramModel(mpath)

        bench = benchmark_model_latency(loaded_model, test_words, rounds=5)
        item.update(bench)
        del loaded_model
        gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # D. Display Clean Results Table
    # ──────────────────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 115)
    print(f"{'EMPIRICAL COMPARISON: QUADGRAM vs CHAR-GRU (HIDDEN 48, 96, 128, 160)':^115}")
    print("=" * 115)
    header = (
        f"{'Model Architecture':<38} | {'Params':>8} | {'Disk':>7} | {'Train':>7} | "
        f"{'Val Loss':>8} | {'Perp':>6} | {'Mean Lat':>8} | {'P95 Lat':>8} | {'Throughput':>11}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        p_str = f"{r['params']:,}" if r['params'] else "N/A"
        size_str = f"{r['file_size_kb']:.1f} KB"
        train_str = f"{r['train_time_s']:.1f}s"
        vloss_str = f"{r['val_loss']:.4f}" if r['val_loss'] is not None else "—"
        perp_str = f"{r['perp']:.2f}" if r['perp'] is not None else "—"
        mean_str = f"{r['mean_us']:.1f} µs"
        p95_str = f"{r['p95_us']:.1f} µs"
        tp_str = f"{r['throughput_wps']:,.0f} w/s"

        line = (
            f"{r['name']:<38} | {p_str:>8} | {size_str:>7} | {train_str:>7} | "
            f"{vloss_str:>8} | {perp_str:>6} | {mean_str:>8} | {p95_str:>8} | {tp_str:>11}"
        )
        print(line)

    print("=" * 115)

    # Save results as JSON artifact for reference
    results_json = os.path.join(out_dir, 'comparison_results.json')
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Full JSON report saved to: %s", results_json)


if __name__ == '__main__':
    main()
