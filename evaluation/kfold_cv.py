"""
kfold_cv.py — Strict K-Fold Cross-Validation: Neural Char-GRU vs Quadgram.

Guarantees 100% data leakage prevention:
For each fold f in [0, K):
  - Training set = lines where (i % K != f)
  - Test set     = lines where (i % K == f)
  - Quadgram trie is built STRICTLY from training lines.
  - Neural Char-GRU is trained STRICTLY on words from training lines.
  - Both models are evaluated on the exact same unseen held-out test lines.
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from evaluation.benchmark import load_corpus_lines, run_test, shutdown_pool
from scripts.build_quadgrams import ALLOWED_EN, ALLOWED_HE, build_quadgrams_from_lines, save_model_data_to_trie
from scripts.train_char_gru import train_model, export_numpy_weights, EN_CHARS, HE_CHARS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('kfold_cv')

ALLOWED = {'en': ALLOWED_EN, 'he': ALLOWED_HE}
CHARS = {'en': EN_CHARS, 'he': HE_CHARS}


def extract_words_from_lines(lines, allowed_chars=None, max_words=300_000):
    """Extract filtered word tokens from a list of lines."""
    allowed_set = set(allowed_chars) if allowed_chars else None
    words = []
    for l in lines:
        l = l.strip().lower()
        if not l:
            continue
        for w in l.split():
            if len(w) < 2 or len(w) > 14:
                continue
            if any('\u0591' <= ch <= '\u05C7' for ch in w):
                continue
            if allowed_set and any(ch not in allowed_set for ch in w):
                continue
            words.append(w)
            if len(words) >= max_words:
                return words
    return words


def run_kfold_evaluation(lang, lines, k=5, delta=3.5, epochs=3, jobs=8, max_words=300_000):
    """Run strict K-fold CV comparing Quadgram vs Neural on the same held-out folds."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    temp_dir = tempfile.mkdtemp(prefix=f'switchlang_kfold_{lang}_')

    logger.info("Running %d-fold Cross-Validation for %s on %d lines (Temp dir: %s)...",
                k, lang.upper(), len(lines), temp_dir)

    fold_results = []

    try:
        for f in range(k):
            logger.info("=" * 60)
            logger.info("STARTING FOLD %d/%d for %s", f + 1, k, lang.upper())
            logger.info("=" * 60)

            train_lines = [l for i, l in enumerate(lines) if (i % k) != f]
            test_lines = [l for i, l in enumerate(lines) if (i % k) == f]

            logger.info("Fold %d: %d train lines, %d test lines (Zero overlap)",
                        f + 1, len(train_lines), len(test_lines))

            # ── 1. Train Quadgram model on train_lines only ──
            t0 = time.time()
            quad_dict = build_quadgrams_from_lines(train_lines, ALLOWED[lang], min_count=2)
            quad_trie_path = os.path.join(temp_dir, f'quad_{lang}_f{f}.marisa')
            quad_meta_path = os.path.join(temp_dir, f'quad_{lang}_f{f}.meta.json')
            save_model_data_to_trie(quad_dict, quad_trie_path, quad_meta_path)
            logger.info("Quadgram built for fold %d in %.1fs", f + 1, time.time() - t0)

            # ── 2. Train Neural Char-GRU model on train_lines only ──
            t0 = time.time()
            train_words = extract_words_from_lines(train_lines, allowed_chars=CHARS[lang], max_words=max_words)
            logger.info("Extracted %d train words for fold %d neural training", len(train_words), f + 1)
            neural_model, vocab_info = train_model(train_words, lang=lang, epochs=epochs)
            neural_base = os.path.join(temp_dir, f'neural_{lang}_f{f}')
            export_numpy_weights(neural_model, vocab_info, neural_base)
            neural_npz_path = f"{neural_base}.npz"
            logger.info("Neural model trained for fold %d in %.1fs", f + 1, time.time() - t0)

            # ── 3. Evaluate Quadgram on held-out test_lines ──
            quad_kwargs = {'en_model_path': quad_trie_path} if lang == 'en' else {'he_model_path': quad_trie_path}
            q_fp = run_test('fp', test_lines, lang, delta, data_dir, jobs=jobs, req_confirmations=2,
                            model_type='quadgram', **quad_kwargs)
            q_fn = run_test('fn', test_lines, lang, delta, data_dir, jobs=jobs, req_confirmations=2,
                            model_type='quadgram', **quad_kwargs)
            q_lat = statistics.mean(q_fn.latency_values) if q_fn.latency_values else 0.0
            q_med = statistics.median(q_fn.latency_values) if q_fn.latency_values else 0.0

            # ── 4. Evaluate Neural on the EXACT SAME held-out test_lines ──
            neu_kwargs = {'en_model_path': neural_npz_path} if lang == 'en' else {'he_model_path': neural_npz_path}
            n_fp = run_test('fp', test_lines, lang, delta, data_dir, jobs=jobs, req_confirmations=2,
                            model_type='neural', **neu_kwargs)
            n_fn = run_test('fn', test_lines, lang, delta, data_dir, jobs=jobs, req_confirmations=2,
                            model_type='neural', **neu_kwargs)
            n_lat = statistics.mean(n_fn.latency_values) if n_fn.latency_values else 0.0
            n_med = statistics.median(n_fn.latency_values) if n_fn.latency_values else 0.0

            shutdown_pool()

            logger.info("Fold %d Results:", f + 1)
            logger.info("  Quadgram: FP/1k=%.3f, FN/1k=%.3f, Latency=%.1f (med=%.0f)",
                        q_fp.fp_per_1k, q_fn.fn_per_1k, q_lat, q_med)
            logger.info("  Neural:   FP/1k=%.3f, FN/1k=%.3f, Latency=%.1f (med=%.0f)",
                        n_fp.fp_per_1k, n_fn.fn_per_1k, n_lat, n_med)

            fold_results.append({
                'fold': f + 1,
                'words': q_fp.words_tested,
                'q_fp': q_fp.fp_per_1k,
                'q_fn': q_fn.fn_per_1k,
                'q_lat': q_lat,
                'q_med': q_med,
                'n_fp': n_fp.fp_per_1k,
                'n_fn': n_fn.fn_per_1k,
                'n_lat': n_lat,
                'n_med': n_med,
            })

    finally:
        shutdown_pool()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return fold_results


def print_cv_table(lang, results):
    """Print clean comparison table across folds."""
    print(f"\n{'=' * 95}")
    print(f" {lang.upper()} K-FOLD CROSS-VALIDATION RESULTS ({len(results)} FOLDS, ZERO DATA LEAKAGE)")
    print(f"{'=' * 95}")
    print(f"{'Fold':<6} {'Words':>8} | {'Quad FP':>8} {'Neur FP':>8} {'ΔFP':>8} | {'Quad FN':>8} {'Neur FN':>8} {'ΔFN%':>8} | {'Quad Lat':>8} {'Neur Lat':>8}")
    print("-" * 95)

    for r in results:
        dfp = r['n_fp'] - r['q_fp']
        dfn_pct = ((r['n_fn'] - r['q_fn']) / r['q_fn'] * 100) if r['q_fn'] else 0.0
        print(f"{r['fold']:<6} {r['words']:>8,} | {r['q_fp']:>8.3f} {r['n_fp']:>8.3f} {dfp:>+8.3f} | "
              f"{r['q_fn']:>8.3f} {r['n_fn']:>8.3f} {dfn_pct:>+7.1f}% | {r['q_lat']:>7.2f}c {r['n_lat']:>7.2f}c")

    print("-" * 95)
    mean_q_fp = statistics.mean(r['q_fp'] for r in results)
    mean_n_fp = statistics.mean(r['n_fp'] for r in results)
    mean_q_fn = statistics.mean(r['q_fn'] for r in results)
    mean_n_fn = statistics.mean(r['n_fn'] for r in results)
    mean_q_lat = statistics.mean(r['q_lat'] for r in results)
    mean_n_lat = statistics.mean(r['n_lat'] for r in results)

    mean_dfn_pct = ((mean_n_fn - mean_q_fn) / mean_q_fn * 100) if mean_q_fn else 0.0
    print(f"{'MEAN':<6} {'-':>8} | {mean_q_fp:>8.3f} {mean_n_fp:>8.3f} {mean_n_fp - mean_q_fp:>+8.3f} | "
          f"{mean_q_fn:>8.3f} {mean_n_fn:>8.3f} {mean_dfn_pct:>+7.1f}% | {mean_q_lat:>7.2f}c {mean_n_lat:>7.2f}c")
    print(f"{'=' * 95}\n")


def main():
    parser = argparse.ArgumentParser(description="Strict K-Fold Cross-Validation for SwitchLang")
    parser.add_argument('--k', type=int, default=3, help="Number of folds (default: 3)")
    parser.add_argument('--max-lines', type=int, default=3000, help="Total corpus lines to use (default: 3000)")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both', help="Language to evaluate")
    parser.add_argument('--epochs', type=int, default=3, help="Neural training epochs per fold")
    parser.add_argument('--jobs', type=int, default=8, help="Parallel evaluation jobs")
    parser.add_argument('--delta', type=float, default=3.5, help="Delta threshold")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')

    langs = ['he', 'en'] if args.lang == 'both' else [args.lang]

    for lang in langs:
        corpus_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        lines = load_corpus_lines(corpus_path, lang, cap=args.max_lines)
        results = run_kfold_evaluation(
            lang=lang,
            lines=lines,
            k=args.k,
            delta=args.delta,
            epochs=args.epochs,
            jobs=args.jobs
        )
        print_cv_table(lang, results)


if __name__ == '__main__':
    main()
