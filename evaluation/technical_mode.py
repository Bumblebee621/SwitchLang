"""
technical_mode.py — what turning on technical mode costs and buys.

Technical mode scores English as max(en, so), so it can only raise the English
side.  That fixes the sign of the effect per corpus before any measurement:
English text gains (the raised score is the correct one, active in the FP test
and shadow in the FN test), Hebrew text loses (the raised score is the wrong one
in both), and technical text is where the gain is supposed to be worth it.

Each arm uses a model that never saw its test lines:
  so  — rebuilt from K-1 folds of stack_overflow_comments.txt
  he  — merged from the fold parts compare_variants.py caches
  en  — CulturaX-trained, so SO text is already held out; the plain en arm
        reuses the shipped model and is a floor check, not a clean measurement

Usage:
    python evaluation/technical_mode.py                 # all three arms
    python evaluation/technical_mode.py --arms so,he
"""

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from benchmark import load_corpus_lines, run_test
from build_quadgrams import ALLOWED_EN, MIN_QUADGRAM_COUNT, build_quadgrams_from_lines
from compare_variants import VARIANTS, fold_of, merge_parts


def load_so_lines(path):
    """Eligibility filter matching load_corpus_lines(lang='en').

    Read with errors='ignore' — the scraped corpus carries invalid UTF-8.
    """
    lines = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for l in f:
            l = l.strip()
            if l and re.search(r'[a-zA-Z]', l) and not re.search(r'[֐-׿]', l):
                lines.append(l)
    return lines


def build_so_arm(data_dir, work_dir, k, fold):
    """Held-out SO model, plus the SO lines to score it on.

    The harness reads so_quadgrams.json from its data dir, so the held-out model
    is staged in a private dir alongside symlinks to the shipped en/he models.
    """
    for name in ('en_quadgrams.json', 'he_quadgrams.json', 'collisions.json'):
        link = os.path.join(work_dir, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(data_dir, name), link)

    lines = load_so_lines(os.path.join(data_dir, 'stack_overflow_comments.txt'))
    test = [l for i, l in enumerate(lines) if fold_of(i, k) == fold]

    model_path = os.path.join(work_dir, 'so_quadgrams.json')
    if not os.path.exists(model_path):
        train = [l for i, l in enumerate(lines) if fold_of(i, k) != fold]
        print(f'  building SO model from {len(train):,} lines …', flush=True)
        model = build_quadgrams_from_lines(train, ALLOWED_EN, MIN_QUADGRAM_COUNT)
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False)

    return {'lang': 'en', 'data_dir': work_dir}, test, len(lines)


def build_he_arm(data_dir, cache_dir, work_dir, k, fold):
    """HE model for one fold, pruned to match what build_quadgrams.py ships."""
    model_path = os.path.join(work_dir, f'he_k{k}_f{fold}.json')
    if not os.path.exists(model_path):
        parts = []
        for p in range(k):
            part_path = os.path.join(cache_dir, f'he_k{k}_p{p}.json')
            if not os.path.exists(part_path):
                raise SystemExit(
                    f'{part_path} missing — run compare_variants.py first to '
                    f'populate the fold cache.')
            with open(part_path, encoding='utf-8') as f:
                parts.append(json.load(f))
        model = VARIANTS['prune2'](merge_parts(parts, fold))
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False)

    lines = load_corpus_lines(os.path.join(data_dir, 'he_corpus.txt'), 'he')
    test = [l for i, l in enumerate(lines) if fold_of(i, k) == fold]
    return {'lang': 'he', 'data_dir': data_dir, 'he_model_path': model_path}, test, len(lines)


def build_en_arm(data_dir, max_lines):
    """Shipped models on plain English — a floor check, not leakage-free."""
    lines = load_corpus_lines(os.path.join(data_dir, 'en_corpus.txt'), 'en', max_lines)
    return {'lang': 'en', 'data_dir': data_dir}, lines, len(lines)


def run_arm(name, kwargs, test, eligible, delta, jobs, max_test_lines):
    if max_test_lines:
        test = test[:max_test_lines]
    print(f'\n{name}: {eligible:,} eligible, scoring {len(test):,}', flush=True)

    lang, data_dir = kwargs.pop('lang'), kwargs.pop('data_dir')
    rows = {}
    for mode in ('standard', 'technical'):
        fp = run_test('fp', test, lang, delta, data_dir, mode=mode, jobs=jobs, **kwargs)
        fn = run_test('fn', test, lang, delta, data_dir, mode=mode, jobs=jobs, **kwargs)
        rows[mode] = (fp.words_tested, fp.fp_count, fp.fp_per_1k, fn.fn_per_1k,
                      statistics.mean(fn.latency_values) if fn.latency_values else 0.0)
    return rows


def print_table(results):
    print(f'\n{"=" * 74}')
    print(f'{"arm":<6} {"mode":<10} {"words":>12} {"FP":>7} {"FP/1k":>8} '
          f'{"FN/1k":>8} {"latency":>8}')
    print('-' * 74)
    for name, rows in results.items():
        for mode, (words, fp, fp1k, fn1k, lat) in rows.items():
            print(f'{name:<6} {mode:<10} {words:>12,} {fp:>7} {fp1k:>8.3f} '
                  f'{fn1k:>8.3f} {lat:>8.1f}')
        base, tech = rows['standard'], rows['technical']
        print(f'{"":<6} {"change":<10} {"":>12} {"":>7} '
              f'{_pct(tech[2], base[2]):>8} {_pct(tech[3], base[3]):>8}')
        print('-' * 74)


def _pct(new, old):
    return f'{(new - old) / old * 100:+.0f}%' if old else 'n/a'


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='Measure technical mode against standard on each corpus.')
    parser.add_argument('--arms', default='so,en,he',
                        help='Comma-separated subset of so,en,he (default: all).')
    parser.add_argument('--k', type=int, default=5, help='Folds for the held-out arms.')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to score.')
    parser.add_argument('--max-test-lines', type=int, default=75_000)
    parser.add_argument('--baseline-delta', type=float, default=4.0)
    parser.add_argument('--data-dir', default=default_data)
    parser.add_argument('--cache-dir', default=os.path.join(default_data, 'fold_counts'))
    parser.add_argument('--work-dir', default=os.path.join(default_data, 'technical_mode'),
                        help='Where held-out models are staged between runs.')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 1, metavar='N')
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]

    results = {}
    for arm in arms:
        if arm == 'so':
            spec = build_so_arm(args.data_dir, args.work_dir, args.k, args.fold)
        elif arm == 'he':
            spec = build_he_arm(args.data_dir, args.cache_dir, args.work_dir,
                                args.k, args.fold)
        elif arm == 'en':
            spec = build_en_arm(args.data_dir, args.max_test_lines)
        else:
            parser.error(f'Unknown arm: {arm}')
        results[arm] = run_arm(arm, *spec, args.baseline_delta, args.jobs,
                               args.max_test_lines)

    print_table(results)


if __name__ == '__main__':
    main()
