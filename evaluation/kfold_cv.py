"""
kfold_cv.py — Strict K-Fold Cross-Validation: Neural Char-GRU vs Quadgram.

Guarantees 100% data leakage prevention:
For each fold f in [0, K):
  - Training set = lines where (i % K != f)
  - Test set     = lines where (i % K == f)
  - Quadgram trie is built STRICTLY from training lines.
  - Neural Char-GRU is trained STRICTLY on words from training lines.
  - Both models are evaluated on the exact same unseen held-out test lines.
  - Optionally (symmetric mode), opposing language models are also trained
    on matching fold splits to ensure symmetric model scale.
"""

import argparse
import copy
from dataclasses import fields
import io
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time


# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from core.engine import FAST_SWITCH_MARGIN
from evaluation.benchmark import (
    load_corpus_lines, run_test, shutdown_pool,
    print_fp_report, print_fn_report,
)
from scripts.build_quadgrams import ALLOWED_EN, ALLOWED_HE, build_quadgrams_from_lines, save_model_data_to_trie
from scripts.train_char_gru import GRU_HPARAMS, train_model, export_numpy_weights, EN_CHARS, HE_CHARS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger('kfold_cv')

ALLOWED = {'en': ALLOWED_EN, 'he': ALLOWED_HE}
CHARS = {'en': EN_CHARS, 'he': HE_CHARS}


def compute_p95(vals):
    """Compute 95th percentile value from a list of numbers."""
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(int(len(s) * 0.95), len(s) - 1)])


def merge_reports(reports):
    """Safely sum counters and concatenate lists across reports without mutating inputs."""
    if not reports:
        return None
    merged = copy.deepcopy(reports[0])
    for report in reports[1:]:
        for f in fields(merged):
            value = getattr(report, f.name)
            if isinstance(value, bool) or f.name == 'lang':
                continue
            if isinstance(value, (int, float)):
                setattr(merged, f.name, getattr(merged, f.name) + value)
            elif isinstance(value, list):
                getattr(merged, f.name).extend(copy.deepcopy(value))
    return merged


def extract_words_from_lines(lines, allowed_chars=None, max_words=300_000):
    """Extract filtered word tokens from a list of lines."""
    allowed_set = set(allowed_chars) if allowed_chars else None
    words = []
    for l in lines:
        l = l.strip().lower()
        if not l:
            continue
        for w in l.split():
            if len(w) > 12:  # same cap as build_quadgrams
                continue
            if any('\u0591' <= ch <= '\u05C7' for ch in w):
                continue
            if allowed_set and any(ch not in allowed_set for ch in w):
                continue
            words.append(w)
            if max_words and len(words) >= max_words:
                return words
    return words


def load_pool(lang, data_dir, k, max_lines, train_words):
    """Unique lines of *lang*'s corpus, trimmed so the K-1 training folds hold ~*train_words* words."""
    lines = list(dict.fromkeys(load_corpus_lines(os.path.join(data_dir, f'{lang}_corpus.txt'), lang, cap=max_lines)))
    if not train_words:
        return lines
    budget = train_words * k / (k - 1)
    total = 0
    for i, l in enumerate(lines):
        total += len(l.split())
        if total >= budget:
            return lines[:i + 1]
    logger.warning("%s: only %d words in %d unique lines, short of %d", lang.upper(), total, len(lines), int(budget))
    return lines


def run_kfold_evaluation(lang, lines, k=5, deltas=(6.0,), epochs=GRU_HPARAMS['epochs'], jobs=8,
                         max_words=300_000, mode='technical', confirmations=2,
                         symmetric=False, other_lines=None, max_test_lines=None, folds=None,
                         fast_margin=FAST_SWITCH_MARGIN):
    """Run strict K-fold CV comparing Quadgram vs Neural on the same held-out folds.

    Returns {delta: (fold_results, merged_reports)}; models are trained once per fold
    and scored at every delta.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    temp_dir = tempfile.mkdtemp(prefix=f'switchlang_kfold_{lang}_')
    other_lang = 'he' if lang == 'en' else 'en'

    # Folds split by index, so a repeated line could land in both train and test.
    lines = list(dict.fromkeys(lines))
    if other_lines:
        other_lines = list(dict.fromkeys(other_lines))

    logger.info("Running %d-fold Cross-Validation for %s on %d unique lines (mode=%s, K=%d, symmetric=%s)...",
                k, lang.upper(), len(lines), mode, confirmations, symmetric)

    fold_results = {d: [] for d in deltas}
    reports = {d: {'q_fp': [], 'q_fn': [], 'n_fp': [], 'n_fn': []} for d in deltas}

    try:
        for f in (folds if folds is not None else range(k)):
            logger.info("=" * 60)
            logger.info("STARTING FOLD %d/%d for %s", f + 1, k, lang.upper())
            logger.info("=" * 60)

            train_lines_prim = [l for i, l in enumerate(lines) if (i % k) != f]
            test_lines_prim = [l for i, l in enumerate(lines) if (i % k) == f][:max_test_lines]
            assert not set(train_lines_prim) & set(test_lines_prim), "test line leaked into training"

            logger.info("Fold %d: %d train lines, %d test lines (Zero overlap)",
                        f + 1, len(train_lines_prim), len(test_lines_prim))

            # ── 1. Train Quadgram models ──
            t0 = time.time()
            quad_dict_prim = build_quadgrams_from_lines(train_lines_prim, ALLOWED[lang], min_count=2, num_workers=jobs)
            quad_prim_path = os.path.join(temp_dir, f'quad_{lang}_f{f}.marisa')
            quad_prim_meta = os.path.join(temp_dir, f'quad_{lang}_f{f}.meta.json')
            save_model_data_to_trie(quad_dict_prim, quad_prim_path, quad_prim_meta)

            quad_other_path = None
            if symmetric and other_lines:
                train_lines_other = [l for i, l in enumerate(other_lines) if (i % k) != f]
                quad_dict_other = build_quadgrams_from_lines(train_lines_other, ALLOWED[other_lang], min_count=2, num_workers=jobs)
                quad_other_path = os.path.join(temp_dir, f'quad_{other_lang}_f{f}.marisa')
                quad_other_meta = os.path.join(temp_dir, f'quad_{other_lang}_f{f}.meta.json')
                save_model_data_to_trie(quad_dict_other, quad_other_path, quad_other_meta)

            logger.info("Quadgram models built for fold %d in %.1fs", f + 1, time.time() - t0)

            # ── 2. Train Neural Char-GRU models ──
            t0 = time.time()
            train_words_prim = extract_words_from_lines(train_lines_prim, allowed_chars=CHARS[lang], max_words=max_words)
            logger.info("Extracted %d train words for fold %d %s neural training", len(train_words_prim), f + 1, lang.upper())
            neural_prim, vocab_prim = train_model(train_words_prim, lang=lang, epochs=epochs)
            neural_prim_base = os.path.join(temp_dir, f'neural_{lang}_f{f}')
            export_numpy_weights(neural_prim, vocab_prim, neural_prim_base)
            neural_prim_path = f"{neural_prim_base}.npz"

            neural_other_path = None
            if symmetric and other_lines:
                train_words_other = extract_words_from_lines(train_lines_other, allowed_chars=CHARS[other_lang], max_words=max_words)
                logger.info("Extracted %d train words for fold %d %s shadow neural training", len(train_words_other), f + 1, other_lang.upper())
                neural_other, vocab_other = train_model(train_words_other, lang=other_lang, epochs=epochs)
                neural_other_base = os.path.join(temp_dir, f'neural_{other_lang}_f{f}')
                export_numpy_weights(neural_other, vocab_other, neural_other_base)
                neural_other_path = f"{neural_other_base}.npz"

            logger.info("Neural models trained for fold %d in %.1fs", f + 1, time.time() - t0)

            if symmetric and quad_other_path:
                quad_kwargs = {
                    'en_model_path': quad_prim_path if lang == 'en' else quad_other_path,
                    'he_model_path': quad_other_path if lang == 'en' else quad_prim_path,
                }
            else:
                quad_kwargs = {'en_model_path': quad_prim_path} if lang == 'en' else {'he_model_path': quad_prim_path}

            if symmetric and neural_other_path:
                neu_kwargs = {
                    'en_model_path': neural_prim_path if lang == 'en' else neural_other_path,
                    'he_model_path': neural_other_path if lang == 'en' else neural_prim_path,
                }
            else:
                neu_kwargs = {'en_model_path': neural_prim_path} if lang == 'en' else {'he_model_path': neural_prim_path}

            for delta in deltas:
                # ── 3. Evaluate Quadgram on held-out test_lines ──
                q_fp = run_test('fp', test_lines_prim, lang, delta, data_dir, mode=mode, jobs=jobs,
                                req_confirmations=confirmations, fast_margin=fast_margin, model_type='quadgram', **quad_kwargs)
                q_fn = run_test('fn', test_lines_prim, lang, delta, data_dir, mode=mode, jobs=jobs,
                                req_confirmations=confirmations, fast_margin=fast_margin, model_type='quadgram', **quad_kwargs)

                q_lat = statistics.mean(q_fn.latency_values) if q_fn.latency_values else 0.0
                q_med = statistics.median(q_fn.latency_values) if q_fn.latency_values else 0.0
                q_p95 = compute_p95(q_fn.latency_values)
                q_sw_pct = (q_fn.lines_switched / q_fn.lines_tested * 100) if q_fn.lines_tested else 0.0

                # ── 4. Evaluate Neural on the EXACT SAME held-out test_lines ──
                n_fp = run_test('fp', test_lines_prim, lang, delta, data_dir, mode=mode, jobs=jobs,
                                req_confirmations=confirmations, fast_margin=fast_margin, model_type='neural', **neu_kwargs)
                n_fn = run_test('fn', test_lines_prim, lang, delta, data_dir, mode=mode, jobs=jobs,
                                req_confirmations=confirmations, fast_margin=fast_margin, model_type='neural', **neu_kwargs)

                n_lat = statistics.mean(n_fn.latency_values) if n_fn.latency_values else 0.0
                n_med = statistics.median(n_fn.latency_values) if n_fn.latency_values else 0.0
                n_p95 = compute_p95(n_fn.latency_values)
                n_sw_pct = (n_fn.lines_switched / n_fn.lines_tested * 100) if n_fn.lines_tested else 0.0

                logger.info("Fold %d Results (delta=%.2f):", f + 1, delta)
                logger.info("  Quadgram: FP/1k=%.3f, FN/1k=%.3f, Latency=%.1f (med=%.0f, p95=%.0f), Switched=%.1f%%, Rec=%d",
                            q_fp.fp_per_1k, q_fn.fn_per_1k, q_lat, q_med, q_p95, q_sw_pct, q_fp.recovery_count)
                logger.info("  Neural:   FP/1k=%.3f, FN/1k=%.3f, Latency=%.1f (med=%.0f, p95=%.0f), Switched=%.1f%%, Rec=%d",
                            n_fp.fp_per_1k, n_fn.fn_per_1k, n_lat, n_med, n_p95, n_sw_pct, n_fp.recovery_count)

                fold_results[delta].append({
                    'fold': f + 1,
                    'lines': len(test_lines_prim),
                    'words': q_fp.words_tested,
                    'q_fp': q_fp.fp_per_1k,
                    'q_fn': q_fn.fn_per_1k,
                    'q_lat': q_lat,
                    'q_med': q_med,
                    'q_p95': q_p95,
                    'q_rec': q_fp.recovery_count,
                    'q_sw_pct': q_sw_pct,
                    'n_fp': n_fp.fp_per_1k,
                    'n_fn': n_fn.fn_per_1k,
                    'n_lat': n_lat,
                    'n_med': n_med,
                    'n_p95': n_p95,
                    'n_rec': n_fp.recovery_count,
                    'n_sw_pct': n_sw_pct,
                })

                # Line numbers restart in each fold's test set; tag them so merged reports stay unambiguous.
                for report in (q_fp, q_fn, n_fp, n_fn):
                    for r in report.flagged_lines:
                        r['line_num'] = f"fold {f + 1}:{r['line_num']}"

                for name, report in (('q_fp', q_fp), ('q_fn', q_fn), ('n_fp', n_fp), ('n_fn', n_fn)):
                    reports[delta][name].append(report)

            shutdown_pool()

    finally:
        shutdown_pool()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return {d: (fold_results[d], {name: merge_reports(rs) for name, rs in reports[d].items()})
            for d in deltas}


def print_cv_table(lang, results, delta):
    """Print clean comparison table across folds with comprehensive metrics."""
    print(f"\n{'=' * 115}")
    print(f" {lang.upper()} K-FOLD CROSS-VALIDATION RESULTS ({len(results)} FOLDS, DELTA {delta:g}, ZERO DATA LEAKAGE)")
    print(f"{'=' * 115}")
    print(f"{'Fold':<5} {'Lines':>6} {'Words':>8} | {'Quad FP':>7} {'Neur FP':>7} {'ΔFP':>7} | {'Quad FN':>7} {'Neur FN':>7} {'ΔFN%':>7} | "
          f"{'Med (c)':>9} | {'Mean (c)':>11} | {'P95 (c)':>9} | {'Recov':>7} | {'Switched%':>11}")
    print(f"{'':<5} {'':>6} {'':>8} | {'/1k':>7} {'/1k':>7} {'':>7} | {'/1k':>7} {'/1k':>7} {'':>7} | "
          f"{'Q   N':>9} | {'Q     N':>11} | {'Q   N':>9} | {'Q   N':>7} | {'Q      N':>11}")
    print("-" * 115)

    for r in results:
        dfp = r['n_fp'] - r['q_fp']
        dfn_pct = ((r['n_fn'] - r['q_fn']) / r['q_fn'] * 100) if r['q_fn'] else 0.0
        med_str = f"{r['q_med']:>3.0f} {r['n_med']:>3.0f}"
        mean_str = f"{r['q_lat']:>5.1f} {r['n_lat']:>5.1f}"
        p95_str = f"{r['q_p95']:>3.0f} {r['n_p95']:>3.0f}"
        rec_str = f"{r['q_rec']:>3d} {r['n_rec']:>3d}"
        sw_str = f"{r['q_sw_pct']:>4.1f}% {r['n_sw_pct']:>4.1f}%"
        print(f"{r['fold']:<5} {r['lines']:>6,} {r['words']:>8,} | {r['q_fp']:>7.3f} {r['n_fp']:>7.3f} {dfp:>+7.3f} | "
              f"{r['q_fn']:>7.3f} {r['n_fn']:>7.3f} {dfn_pct:>+6.1f}% | {med_str:>9} | {mean_str:>11} | {p95_str:>9} | {rec_str:>7} | {sw_str:>11}")

    print("-" * 115)
    mean_lines = int(round(statistics.mean(r['lines'] for r in results)))
    mean_words = int(round(statistics.mean(r['words'] for r in results)))
    mean_q_fp = statistics.mean(r['q_fp'] for r in results)
    mean_n_fp = statistics.mean(r['n_fp'] for r in results)
    mean_q_fn = statistics.mean(r['q_fn'] for r in results)
    mean_n_fn = statistics.mean(r['n_fn'] for r in results)
    mean_q_lat = statistics.mean(r['q_lat'] for r in results)
    mean_n_lat = statistics.mean(r['n_lat'] for r in results)
    mean_q_med = statistics.mean(r['q_med'] for r in results)
    mean_n_med = statistics.mean(r['n_med'] for r in results)
    mean_q_p95 = statistics.mean(r['q_p95'] for r in results)
    mean_n_p95 = statistics.mean(r['n_p95'] for r in results)
    mean_q_rec = statistics.mean(r['q_rec'] for r in results)
    mean_n_rec = statistics.mean(r['n_rec'] for r in results)
    mean_q_sw = statistics.mean(r['q_sw_pct'] for r in results)
    mean_n_sw = statistics.mean(r['n_sw_pct'] for r in results)
    mean_dfn_pct = ((mean_n_fn - mean_q_fn) / mean_q_fn * 100) if mean_q_fn else 0.0

    med_str = f"{mean_q_med:>3.0f} {mean_n_med:>3.0f}"
    mean_str = f"{mean_q_lat:>5.1f} {mean_n_lat:>5.1f}"
    p95_str = f"{mean_q_p95:>3.0f} {mean_n_p95:>3.0f}"
    rec_str = f"{mean_q_rec:>3.1f} {mean_n_rec:>3.1f}"
    sw_str = f"{mean_q_sw:>4.1f}% {mean_n_sw:>4.1f}%"

    print(f"{'MEAN':<5} {mean_lines:>6,} {mean_words:>8,} | {mean_q_fp:>7.3f} {mean_n_fp:>7.3f} {mean_n_fp - mean_q_fp:>+7.3f} | "
          f"{mean_q_fn:>7.3f} {mean_n_fn:>7.3f} {mean_dfn_pct:>+6.1f}% | {med_str:>9} | {mean_str:>11} | {p95_str:>9} | {rec_str:>7} | {sw_str:>11}")
    print(f"{'=' * 115}\n")


def print_aggregated_reports(lang, merged_reports, delta=6.0, confirmations=2,
                             mode='standard', jobs=8, max_flagged=25):
    """Print full benchmark FP and FN reports for both Quadgram and Neural models."""
    prov = f"K-Fold CV Aggregated ({lang.upper()})"
    print(f"\n{'#' * 70}")
    print(f" QUADGRAM BENCHMARK REPORTS (ALL FOLDS COMBINED)")
    print(f"{'#' * 70}")
    print_fp_report(merged_reports['q_fp'], corpus_path=f"data/{lang}_corpus.txt",
                    model_path="Fold-specific Quadgram MARISA", provenance=prov,
                    max_flagged=max_flagged, delta=delta, req_confirmations=confirmations,
                    mode=mode, jobs=jobs)
    print_fn_report(merged_reports['q_fn'], corpus_path=f"data/{lang}_corpus.txt",
                    model_path="Fold-specific Quadgram MARISA", provenance=prov,
                    max_flagged=max_flagged, delta=delta, req_confirmations=confirmations,
                    mode=mode, jobs=jobs)

    print(f"\n{'#' * 70}")
    print(f" NEURAL CHAR-GRU BENCHMARK REPORTS (ALL FOLDS COMBINED)")
    print(f"{'#' * 70}")
    print_fp_report(merged_reports['n_fp'], corpus_path=f"data/{lang}_corpus.txt",
                    model_path="Fold-specific Char-GRU NPZ", provenance=prov,
                    max_flagged=max_flagged, delta=delta, req_confirmations=confirmations,
                    mode=mode, jobs=jobs)
    print_fn_report(merged_reports['n_fn'], corpus_path=f"data/{lang}_corpus.txt",
                    model_path="Fold-specific Char-GRU NPZ", provenance=prov,
                    max_flagged=max_flagged, delta=delta, req_confirmations=confirmations,
                    mode=mode, jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description="Strict K-Fold Cross-Validation for SwitchLang")
    parser.add_argument('--k', type=int, default=3, help="Number of folds (default: 3)")
    parser.add_argument('--max-lines', type=int, default=150_000, help="Corpus lines read before dedupe and --train-words trimming (default: 150000)")
    parser.add_argument('--train-words', type=int, default=1_000_000, help="Words in each fold's training share, per language; 0 keeps every line read (default: 1000000)")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both', help="Language to evaluate")
    parser.add_argument('--epochs', type=int, default=GRU_HPARAMS['epochs'], help="Neural training epochs per fold (default: GRU_HPARAMS)")
    parser.add_argument('--jobs', type=int, default=8, help="Parallel evaluation jobs")
    parser.add_argument('--deltas', default='6.0', help="Comma-separated delta thresholds to score each fold at (default: 6.0)")
    parser.add_argument('--confirmations', type=int, default=2, help="Consecutive confirmations (default: 2)")
    parser.add_argument('--mode', choices=['standard', 'technical'], default='standard', help="Model mode (default: standard)")
    parser.add_argument('--symmetric', action='store_true', default=False, help="Train both language models on fold data to prevent capacity mismatch")
    parser.add_argument('--max-flagged', type=int, default=25, help="Worst flagged lines to print per report (default: 25)")
    parser.add_argument('--no-reports', action='store_true', default=False, help="Skip printing detailed benchmark FP/FN reports after fold table")
    parser.add_argument('--fast-margin', type=float, default=FAST_SWITCH_MARGIN, help=f"Mid-word margin past delta that switches on one keystroke (default: {FAST_SWITCH_MARGIN})")
    parser.add_argument('--folds', default=None, help="Comma-separated 1-based folds to run (default: all)")
    parser.add_argument('--out', default=None, help="Write per-fold results for every language and delta to this JSON file")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    deltas = [float(d) for d in args.deltas.split(',')]

    langs = ['he', 'en'] if args.lang == 'both' else [args.lang]
    out = {}

    for lang in langs:
        lines = load_pool(lang, data_dir, args.k, args.max_lines, args.train_words)

        other_lines = None
        if args.symmetric:
            other_lang = 'he' if lang == 'en' else 'en'
            other_lines = load_pool(other_lang, data_dir, args.k, args.max_lines, args.train_words)

        sweep = run_kfold_evaluation(
            lang=lang,
            lines=lines,
            k=args.k,
            deltas=deltas,
            epochs=args.epochs,
            jobs=args.jobs,
            max_words=None,
            mode=args.mode,
            confirmations=args.confirmations,
            symmetric=args.symmetric,
            other_lines=other_lines,
            folds=[int(f) - 1 for f in args.folds.split(',')] if args.folds else None,
            fast_margin=args.fast_margin,
        )

        out[lang] = {str(d): results for d, (results, _) in sweep.items()}
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump(out, fh, indent=2)

        for delta, (results, merged) in sweep.items():
            print_cv_table(lang, results, delta)

            if not args.no_reports and merged:
                print_aggregated_reports(
                    lang=lang,
                    merged_reports=merged,
                    delta=delta,
                    confirmations=args.confirmations,
                    mode=args.mode,
                    jobs=args.jobs,
                    max_flagged=args.max_flagged,
                )


if __name__ == '__main__':
    main()
