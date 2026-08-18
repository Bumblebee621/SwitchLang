"""
combination_bench.py — how technical mode should fold the SO score into English.

A dedicated harness for one question: `_score_text_en` reconciles two English
models with max(), and max() is an upper envelope — it can only raise the
English side.  The SO model is ~18x smaller under the same add-1 smoothing, so
it is flatter, its penalty for unseen n-grams is milder, and it wins on junk as
well as on genuine technical text.  EXPERIMENTS.md measures the bill: the SO arm
improves 77%/64% while the Hebrew arm regresses 85%/89%.

This file leaves benchmark.py and technical_mode.py untouched.  It reuses their
machinery — EvaluationHarness for the replay itself, the arm builders for the
held-out fold models — and adds only what the comparison needs: an engine
constructed with the combination parameters, and a worker cache keyed on them.

Four families are compared per arm:

  standard    — technical mode off.  The reference both directions are measured
                against, since the SO gain and the HE cost are both relative to
                it.
  max         — what ships today.
  mixture:W   — log((1-W)*P_en + W*P_so).  A real mixture LM, of which max is
                the degenerate limit.  The weight charges the SO branch a flat
                log(W) nats, damping the junk bonus while leaving large true
                technical gains intact.
  merged:S    — one model built by summing the count tables, SO counts scaled by
                S (see scripts/merge_en_so_counts.py).  Interpolation moves down
                to the quadgram context, where each context is weighted by its
                own evidence, and runtime drops to one score() call — so this
                family runs in mode='standard' with the merged JSON passed as
                the English model.
  calibrated  — rescale the SO score onto the English model's nats-per-char
                scale before taking the max, so the flatness bias drops out
                (constants from evaluation/calibrate_models.py).

Each arm supplies its own SO flavour.  The so arm must use the held-out fold-0
SO model and the merged/calibration artifacts derived from it, or it measures
its own training data; the en and he arms never train on SO text, so they use
the shipped one.

Read-only with respect to data/: models are read, nothing is written.

Usage:
    python evaluation/combination_bench.py --arms so
    python evaluation/combination_bench.py --arms so,en,he --variants standard,max,mixture:0.2
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import EvaluationHarness, _merge_reports, _get_pool
from core.engine import EvaluationEngine
from core.quadgram import load_models, QuadgramModel
from technical_mode import build_so_arm, build_he_arm, build_en_arm, _pct

# From evaluation/calibrate_models.py, 200k words per model:
#   en          mu -1.4789  sigma 0.7544
#   so_shipped  mu -1.2898  sigma 0.5668
#   so_heldout  mu -1.2924  sigma 0.5702
# The SO model really is flatter and more generous on its own held-out text,
# which is the bias 'calibrated' removes.
CALIBRATION = {
    'shipped': (-1.4789, 0.7544, -1.2898, 0.5668),
    'heldout': (-1.4789, 0.7544, -1.2924, 0.5702),
}

DEFAULT_VARIANTS = (
    'standard', 'max',
    'mixture:0.1', 'mixture:0.2', 'mixture:0.35', 'mixture:0.5',
    'merged:1', 'merged:5', 'merged:18',
    'calibrated',
)


# ═══════════════════════════════════════════════════════════════════════════
# HARNESS
# ═══════════════════════════════════════════════════════════════════════════

class CombinationHarness(EvaluationHarness):
    """EvaluationHarness that also passes the combination knobs to the engine.

    benchmark.EvaluationHarness hard-codes en_combine at its default, so the
    constructor is re-implemented rather than extended; everything below it —
    _simulate_word, the FP and FN tests — is inherited unchanged, which is what
    keeps these numbers comparable to the EXPERIMENTS.md table.
    """

    def __init__(self, data_dir, en_model_path=None, he_model_path=None,
                 mode='standard', en_combine='max', so_weight=0.5,
                 so_calibration=None):
        models = load_models(data_dir, load_so=(mode == 'technical'))
        if en_model_path:
            models['en'] = QuadgramModel(en_model_path)
        if he_model_path:
            models['he'] = QuadgramModel(he_model_path)
        self.engine = EvaluationEngine(
            models['en'], models['he'],
            collisions_path=os.path.join(data_dir, 'collisions.json'),
            enable_logging=False,
            en_so_model=models.get('so'),
            model_mode=mode,
            en_combine=en_combine,
            so_weight=so_weight,
            so_calibration=so_calibration,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
#
# benchmark.run_test keys its worker cache on (data_dir, en_model, he_model,
# mode) alone.  Every variant here shares those four values, so reusing it would
# hand each variant whichever engine the worker happened to build first and
# report identical numbers for all of them.  The key below carries the
# combination parameters too.

_worker_models = (None, None)


def _run_chunk(job):
    global _worker_models
    test, offset, lines, lang, delta, key = job
    if _worker_models[0] != key:
        _worker_models = (key, CombinationHarness(*key))
    harness = _worker_models[1]
    method = (harness.test_false_positives if test == 'fp'
              else harness.test_false_negatives)
    return method(lines, lang, delta, line_offset=offset, progress=False)


def run_test(test, lines, lang, delta, key, jobs=1):
    """Run the 'fp' or 'fn' test over *lines* with the engine described by key."""
    if jobs <= 1:
        harness = CombinationHarness(*key)
        method = (harness.test_false_positives if test == 'fp'
                  else harness.test_false_negatives)
        return method(lines, lang, delta)

    n_chunks = jobs * 4
    size = max(1, -(-len(lines) // n_chunks))
    chunks = [(test, i, lines[i:i + size], lang, delta, key)
              for i in range(0, len(lines), size)]

    t0 = time.time()
    reports = list(_get_pool(jobs).map(_run_chunk, chunks))
    merged = _merge_reports(reports)
    merged.elapsed_sec = time.time() - t0
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

def resolve_variant(spec, arm_kwargs, data_dir, so_flavour, merged_dir):
    """Turn a variant name into a CombinationHarness constructor key.

    arm_kwargs carries the arm's own model overrides (the he arm pins a held-out
    Hebrew model); the variant only ever replaces the English one.
    """
    en_model = arm_kwargs.get('en_model_path')
    he_model = arm_kwargs.get('he_model_path')
    base = (data_dir, en_model, he_model)

    if spec == 'standard':
        return base + ('standard', 'max', 0.5, None)
    if spec == 'max':
        return base + ('technical', 'max', 0.5, None)
    if spec == 'calibrated':
        return base + ('technical', 'calibrated', 0.5, CALIBRATION[so_flavour])
    if spec.startswith('mixture:'):
        return base + ('technical', 'mixture', float(spec.split(':', 1)[1]), None)
    if spec.startswith('merged:'):
        scale = spec.split(':', 1)[1]
        path = os.path.join(merged_dir, f'merged_{so_flavour}_s{scale}.json')
        if not os.path.exists(path):
            raise SystemExit(
                f'{path} missing — build it with:\n'
                f'  python scripts/merge_en_so_counts.py --out {path} '
                f'--so-scale {scale}')
        # One model, so technical mode has nothing left to combine.
        return (data_dir, path, he_model, 'standard', 'max', 0.5, None)
    raise SystemExit(f'unknown variant: {spec}')


def run_arm(name, arm_kwargs, test, eligible, variants, args):
    """Score every variant over one arm's held-out lines."""
    if args.max_test_lines:
        test = test[:args.max_test_lines]
    print(f'\n{name}: {eligible:,} eligible, scoring {len(test):,}', flush=True)

    lang = arm_kwargs.pop('lang')
    data_dir = arm_kwargs.pop('data_dir')
    # The so arm is scored with a fold-0 SO model, so its merged and calibrated
    # artifacts must come from that same held-out model.
    so_flavour = 'heldout' if name == 'so' else 'shipped'

    rows = {}
    for spec in variants:
        key = resolve_variant(spec, arm_kwargs, data_dir, so_flavour,
                              args.merged_dir)
        t0 = time.time()
        fp = run_test('fp', test, lang, args.baseline_delta, key, jobs=args.jobs)
        fn = run_test('fn', test, lang, args.baseline_delta, key, jobs=args.jobs)
        rows[spec] = {
            'words': fp.words_tested,
            'fp': fp.fp_count,
            'fp1k': fp.fp_per_1k,
            'fn1k': fn.fn_per_1k,
            'lat': statistics.mean(fn.latency_values) if fn.latency_values else 0.0,
            'sec': time.time() - t0,
        }
        r = rows[spec]
        print(f'  {spec:<14} FP/1k {r["fp1k"]:.3f}  FN/1k {r["fn1k"]:.3f}  '
              f'({r["sec"]:.0f}s)', flush=True)
    return rows


def print_table(results, variants):
    """Per arm, each variant against that arm's own standard-mode row.

    Percentages are signed the way EXPERIMENTS.md signs them: negative is
    better, and on the he arm the baseline to beat is max's +85%/+89%.
    """
    print(f'\n{"=" * 88}')
    print(f'{"arm":<5} {"variant":<14} {"words":>12} {"FP":>7} {"FP/1k":>8} '
          f'{"vs std":>8} {"FN/1k":>8} {"vs std":>8} {"lat":>7} {"sec":>7}')
    print('-' * 88)
    for arm, rows in results.items():
        base = rows.get('standard')
        for spec in variants:
            if spec not in rows:
                continue
            r = rows[spec]
            fp_d = _pct(r['fp1k'], base['fp1k']) if base else 'n/a'
            fn_d = _pct(r['fn1k'], base['fn1k']) if base else 'n/a'
            print(f'{arm:<5} {spec:<14} {r["words"]:>12,} {r["fp"]:>7} '
                  f'{r["fp1k"]:>8.3f} {fp_d:>8} {r["fn1k"]:>8.3f} {fn_d:>8} '
                  f'{r["lat"]:>7.1f} {r["sec"]:>7.0f}')
        print('-' * 88)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='Compare ways of combining the English and SO models.')
    parser.add_argument('--arms', default='so,en,he',
                        help='Comma-separated subset of so,en,he (default: all).')
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS),
                        help='Comma-separated: standard, max, calibrated, '
                             'mixture:W, merged:S.')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max-test-lines', type=int, default=75_000)
    parser.add_argument('--baseline-delta', type=float, default=4.0)
    parser.add_argument('--data-dir', default=default_data)
    parser.add_argument('--cache-dir', default=os.path.join(default_data, 'fold_counts'))
    parser.add_argument('--work-dir', default=os.path.join(default_data, 'technical_mode'),
                        help='Where the held-out fold models are staged.')
    parser.add_argument('--merged-dir', default=os.path.join(default_data, 'combination_exp'),
                        help='Where merge_en_so_counts.py wrote the merged models.')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1, metavar='N')
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    results = {}
    for arm in arms:
        if arm == 'so':
            kwargs, test, eligible = build_so_arm(
                args.data_dir, args.work_dir, args.k, args.fold)
        elif arm == 'he':
            kwargs, test, eligible = build_he_arm(
                args.data_dir, args.cache_dir, args.work_dir, args.k, args.fold)
        elif arm == 'en':
            kwargs, test, eligible = build_en_arm(args.data_dir, args.max_test_lines)
        else:
            parser.error(f'Unknown arm: {arm}')
        results[arm] = run_arm(arm, kwargs, test, eligible, variants, args)

    print_table(results, variants)


if __name__ == '__main__':
    main()
