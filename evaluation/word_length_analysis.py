"""
word_length_analysis.py — measures how score gap and switch rates scale with word length.

Investigates Arm C from EXPERIMENTS.md: does the English/Hebrew score gap grow with
word length, and does it penalize long Hebrew words?

Three views of the run:
  1. Mean score gap by word length, with a straight-line fit per direction.
     The slope is Arm C's per-character effect; the intercept is a flat offset.
  2. Switch rate per 1,000 evaluations by prefix length.
  3. First word of each line only, where delta is pinned at baseline.

Supports either:
  - Shipped models: --use-shipped (evaluates data/*_quadgrams.marisa directly)
  - Held-out K-fold models: --k 5 --fold 0 (built from compare_variants.py fold caches)

Usage:
    python evaluation/word_length_analysis.py --use-shipped --max-test-lines 2000
    python evaluation/word_length_analysis.py --k 5 --fold 0
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from benchmark import EvaluationHarness, load_corpus_lines
from build_quadgrams import save_model_data_to_trie
from compare_variants import VARIANTS, fold_of, merge_parts
from core.sensitivity import SensitivityManager

# Words longer than this never contributed n-grams during training,
# so their interiors are under-attested in both models.
TRAIN_MAX_WORD = 12


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

def build_fold_model(lang, cache_dir, work_dir, k, fold):
    """Model for one fold, pruned to prune2, packed directly into .marisa."""
    model_path = os.path.join(work_dir, f'{lang}_k{k}_f{fold}.marisa')
    meta_path = os.path.join(work_dir, f'{lang}_k{k}_f{fold}.meta.json')
    if os.path.exists(model_path) and os.path.exists(meta_path):
        return model_path

    parts = []
    for p in range(k):
        part_path = os.path.join(cache_dir, f'{lang}_k{k}_p{p}.json')
        if not os.path.exists(part_path):
            raise SystemExit(
                f'{part_path} missing — run compare_variants.py first to '
                f'populate the fold cache, or pass --use-shipped.')
        with open(part_path, encoding='utf-8') as f:
            parts.append(json.load(f))

    print(f'  merging {lang} fold {fold} into marisa trie …', flush=True)
    model = VARIANTS['prune2'](merge_parts(parts, fold))
    save_model_data_to_trie(model, model_path, meta_path)
    return model_path


def table_mean_logp(model):
    """Count-weighted mean of log P(c4|c1c2c3) over the model's own trie."""
    v = model.vocab_size
    total = weighted = 0
    for key, val in model._trie.items():
        if len(key) == 4:
            c = val[0]
            tri = model.count(key[:3])
            weighted += c * math.log((c + 1) / (tri + v))
            total += c
    return weighted / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# WALKING A WORD
# ═══════════════════════════════════════════════════════════════════════════

def walk_word(engine, buf_active, buf_shadow, delta, layout, req_confirmations=2):
    """Every evaluation one word triggers, in order, stopping at a switch.

    Mirrors EvaluationHarness._simulate_word — mid-word from 3 characters on,
    then once more on the delimiter — reporting each evaluation.
    Yields (chars_typed, score_diff, is_colliding, fired, on_delimiter).
    """
    consec = 0
    for i in range(2, len(buf_active)):
        should, diff, coll, _ = engine.evaluate(
            buf_active[:i + 1], buf_shadow[:i + 1], delta, current_layout=layout)
        if should:
            consec += 1
            fired = (consec >= req_confirmations)
        else:
            consec = 0
            fired = False
        yield i + 1, diff, coll, fired, False
        if fired:
            return

    should, diff, coll, _ = engine.evaluate(
        buf_active, buf_shadow, delta, current_layout=layout, on_delimiter=True)
    yield len(buf_active), diff, coll, should, True


def check_faithful(harness, lines, lang, delta, req_confirmations=2, sample=2000):
    """walk_word must decide every word exactly as _simulate_word does."""
    words = [w for line in lines for w in line.split()][:sample]
    sens = SensitivityManager(baseline_delta=delta)
    for word in words:
        buf_a, buf_s = harness._get_buffers(word, lang, lang)
        fired = False
        fired_at = -1
        for n, _diff, _coll, fired, on_delim in walk_word(
                harness.engine, buf_a, buf_s, delta, lang, req_confirmations=req_confirmations):
            if fired:
                fired_at = -1 if on_delim else n - 1
                break

        res = harness._simulate_word(word, lang, lang, sens)
        assert res.switched == fired, f'{word!r}: switched {res.switched} vs {fired}'
        if fired:
            assert res.switch_char_idx == fired_at, (
                f'{word!r}: fired at {res.switch_char_idx} vs {fired_at}')
    return len(words)


# ═══════════════════════════════════════════════════════════════════════════
# STATS & MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════

class Stats:
    """Everything the three tables need, accumulated in one pass."""

    def __init__(self):
        self.by_len = {}        # word length -> [n, sum diff, over delta, collisions, sum diff^2]
        self.first_word = {}    # same, position 0 only
        self.by_prefix = {}     # chars typed -> [evaluations, switches]
        self.fit = [0, 0.0, 0.0, 0.0, 0.0, 0.0]   # n, sx, sy, sxy, sxx, syy
        self.words = self.evals = self.switches = self.skipped = 0
        self.total_words = 0

    def add_word(self, length, pos, diff, over, colliding):
        for table in ([self.by_len] + ([self.first_word] if pos == 0 else [])):
            row = table.setdefault(length, [0, 0.0, 0, 0, 0.0])
            row[0] += 1
            row[1] += diff
            row[2] += over
            row[3] += colliding
            row[4] += diff * diff
        if length <= TRAIN_MAX_WORD:
            f = self.fit
            f[0] += 1
            f[1] += length
            f[2] += diff
            f[3] += length * diff
            f[4] += length * length
            f[5] += diff * diff

    def add_eval(self, chars, fired):
        row = self.by_prefix.setdefault(chars, [0, 0])
        row[0] += 1
        row[1] += fired

    def line(self):
        """Least-squares fit of score gap against word length."""
        n, sx, sy, sxy, sxx, syy = self.fit
        if n < 3:
            return 0.0, 0.0, 0.0
        cxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        cyy = syy - sy * sy / n
        if cxx == 0:
            return 0.0, 0.0, 0.0
        slope = cxy / cxx
        sse = max(cyy - slope * cxy, 0.0)
        return slope, (sy - slope * sx) / n, math.sqrt(sse / (n - 2) / cxx)


def measure(harness, lines, lang, delta, req_confirmations=2):
    """Walk the corpus recording every evaluation on the correct layout."""
    stats = Stats()
    engine = harness.engine

    for line in lines:
        words = line.strip().split()
        if not words:
            continue

        current = lang
        sens = SensitivityManager(baseline_delta=delta)

        for pos, word in enumerate(words):
            stats.total_words += 1
            buf_a, buf_s = harness._get_buffers(word, lang, current)
            on_correct = current == lang
            switched = False
            last = None

            for chars, diff, coll, fired, on_delim in walk_word(
                    engine, buf_a, buf_s, sens.delta, current,
                    req_confirmations=req_confirmations):
                if on_correct:
                    stats.evals += 1
                    stats.add_eval(chars, fired)
                last = (diff, coll, on_delim)
                if fired:
                    switched = True
                    break

            if on_correct:
                stats.words += 1
                if last is None or not last[2]:
                    stats.skipped += 1
                else:
                    diff, coll, _ = last
                    stats.add_word(len(word), pos, diff,
                                   1 if diff > delta else 0, 1 if coll else 0)

            if switched:
                if on_correct:
                    stats.switches += 1
                current = harness._other(current)
                sens.reset(reason='layout_switch')
            else:
                sens.on_word_complete()

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def print_gap_table(title, en, he, delta):
    print(f'\n{title}')
    print(f'{"len":>4} | {"EN words":>9} {"mean":>7} {"sd":>6} {"σ to Δ":>7} '
          f'{"over Δ":>7} | {"HE words":>9} {"mean":>7} {"sd":>6} {"σ to Δ":>7} '
          f'{"over Δ":>7} | {"HE-EN":>7}')
    print('-' * 88)

    for length in range(2, TRAIN_MAX_WORD + 1):
        e = en.get(length, [0, 0.0, 0, 0, 0.0])
        h = he.get(length, [0, 0.0, 0, 0, 0.0])

        em = e[1] / e[0] if e[0] else 0.0
        ev = max(e[4] / e[0] - em * em, 0.0) if e[0] > 1 else 0.0
        esd = math.sqrt(ev)
        esig = (delta - em) / esd if esd else 0.0
        eo = e[2] / e[0] * 1000 if e[0] else 0.0

        hm = h[1] / h[0] if h[0] else 0.0
        hv = max(h[4] / h[0] - hm * hm, 0.0) if h[0] > 1 else 0.0
        hsd = math.sqrt(hv)
        hsig = (delta - hm) / hsd if hsd else 0.0
        ho = h[2] / h[0] * 1000 if h[0] else 0.0

        gap = hm - em if (e[0] and h[0]) else 0.0
        print(f'{length:>4} | {e[0]:>9,} {em:>7.2f} {esd:>6.2f} {esig:>7.2f} {eo:>7.2f} '
              f'| {h[0]:>9,} {hm:>7.2f} {hsd:>6.2f} {hsig:>7.2f} {ho:>7.2f} | {gap:>7.2f}')


def print_prefix_table(en, he):
    print('\nTABLE 2 — switch rate per 1,000 evaluations, by characters typed')
    print(f'{"chars":>6} | {"EN evals":>11} {"EN sw":>6} {"per 1k":>7} '
          f'| {"HE evals":>11} {"HE sw":>6} {"per 1k":>7}')
    print('-' * 68)
    for chars in range(3, TRAIN_MAX_WORD + 1):
        e = en.get(chars, [0, 0])
        h = he.get(chars, [0, 0])
        er = e[1] / e[0] * 1000 if e[0] else 0.0
        hr = h[1] / h[0] * 1000 if h[0] else 0.0
        print(f'{chars:>6} | {e[0]:>11,} {e[1]:>6,} {er:>7.2f} '
              f'| {h[0]:>11,} {h[1]:>6,} {hr:>7.2f}')


def print_fits(en_stats, he_stats, en_h, he_h):
    en_slope, en_int, en_se = en_stats.line()
    he_slope, he_int, he_se = he_stats.line()
    diff_se = math.sqrt(en_se ** 2 + he_se ** 2)
    print('\nFIT — score gap against word length (length <= '
          f'{TRAIN_MAX_WORD}, delimiter evaluations)')
    print(f'  EN layout   slope {en_slope:+.4f} ± {en_se:.4f} nats/char   '
          f'intercept {en_int:+.3f}')
    print(f'  HE layout   slope {he_slope:+.4f} ± {he_se:.4f} nats/char   '
          f'intercept {he_int:+.3f}')
    print(f'  HE - EN     slope {he_slope - en_slope:+.4f} ± {diff_se:.4f} nats/char   '
          f'intercept {he_int - en_int:+.3f}')
    print(f'\n  Table-mean log P(c4|c1c2c3):')
    print(f'    en {en_h:+.4f} nats/char   he {he_h:+.4f}   offset {en_h - he_h:+.4f}')


def print_batches(harness, test_lines, delta, batches, req_confirmations=2):
    """Fit the same slope on N disjoint batches."""
    size = min(len(test_lines['en']), len(test_lines['he'])) // batches
    print(f'\nBATCH SPREAD — {batches} disjoint batches of {size:,} lines each')
    print(f'{"batch":>6} | {"EN words":>10} {"EN slope":>10} '
          f'| {"HE words":>10} {"HE slope":>10} | {"HE-EN":>9}')
    print('-' * 68)

    diffs = []
    for b in range(batches):
        cut = slice(b * size, (b + 1) * size)
        row = {}
        for lang in ('en', 'he'):
            st = measure(harness, test_lines[lang][cut], lang, delta,
                         req_confirmations=req_confirmations)
            row[lang] = (st.fit[0], st.line()[0])
        diff = row['he'][1] - row['en'][1]
        diffs.append(diff)
        print(f'{b:>6} | {row["en"][0]:>10,} {row["en"][1]:>+10.4f} '
              f'| {row["he"][0]:>10,} {row["he"][1]:>+10.4f} | {diff:>+9.4f}',
              flush=True)

    mean = sum(diffs) / len(diffs)
    spread = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    print('-' * 68)
    print(f'  HE-EN slope: mean {mean:+.4f}, sd {spread:.4f}, '
          f'range {min(diffs):+.4f} to {max(diffs):+.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5).')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to hold out (default: 0).')
    parser.add_argument('--max-test-lines', type=int, default=20000,
                        help='Max lines to test per language (default: 20000).')
    parser.add_argument('--batches', type=int, default=1,
                        help='Split test lines into N disjoint batches.')
    parser.add_argument('--baseline-delta', type=float, default=3.5,
                        help='Delta threshold (default: 3.5).')
    parser.add_argument('--confirmations', type=int, default=2,
                        help='Consecutive confirmations required (default: 2).')
    parser.add_argument('--use-shipped', action='store_true',
                        help='Use data/*_quadgrams.marisa shipped models directly.')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--cache-dir', default='data/fold_counts')
    parser.add_argument('--work-dir', default='data/word_length_analysis')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    cache_dir = os.path.join(project_root, args.cache_dir)
    work_dir = os.path.join(project_root, args.work_dir)

    os.makedirs(work_dir, exist_ok=True)
    delta = args.baseline_delta

    if args.use_shipped:
        print('Using shipped models from data/ …', flush=True)
        harness = EvaluationHarness(data_dir, req_confirmations=args.confirmations)
    else:
        print('Staging held-out models …', flush=True)
        paths = {lang: build_fold_model(lang, cache_dir, work_dir, args.k, args.fold)
                 for lang in ('en', 'he')}
        harness = EvaluationHarness(data_dir,
                                    en_model_path=paths['en'],
                                    he_model_path=paths['he'],
                                    req_confirmations=args.confirmations)

    test_lines = {}
    for lang in ('en', 'he'):
        cap = args.max_test_lines if args.use_shipped else None
        lines = load_corpus_lines(os.path.join(data_dir, f'{lang}_corpus.txt'), lang, cap=cap)
        if args.use_shipped:
            test_lines[lang] = lines
        else:
            test_lines[lang] = [l for i, l in enumerate(lines) if fold_of(i, args.k) == args.fold]
            if args.max_test_lines:
                test_lines[lang] = test_lines[lang][:args.max_test_lines]

    if args.batches > 1:
        print_batches(harness, test_lines, delta, args.batches,
                      req_confirmations=args.confirmations)
        return

    stats = {}
    for lang in ('en', 'he'):
        test = test_lines[lang]

        n = check_faithful(harness, test, lang, delta, req_confirmations=args.confirmations)
        print(f'  {lang}: walk_word agrees with _simulate_word on {n:,} words')

        print(f'  {lang}: scoring {len(test):,} lines …', flush=True)
        stats[lang] = measure(harness, test, lang, delta,
                              req_confirmations=args.confirmations)

        expected = sum(1 for l in test for _ in l.strip().split())
        assert stats[lang].total_words == expected, (
            f'walked {stats[lang].total_words} words, corpus has {expected}')
        print(f'     {stats[lang].words:,} words on correct layout '
              f'of {expected:,} total, {stats[lang].evals:,} evaluations, '
              f'{stats[lang].switches:,} switches')

    en_h = table_mean_logp(harness.engine.en_model)
    he_h = table_mean_logp(harness.engine.he_model)

    print(f'\n{"=" * 88}')
    model_src = "shipped" if args.use_shipped else f"k={args.k} fold={args.fold}"
    print(f'WORD LENGTH ANALYSIS — score gap vs word length ({model_src}, Δ={delta}, K={args.confirmations})')
    print('=' * 88)
    print_gap_table('TABLE 1 — all words on the correct layout',
                    stats['en'].by_len, stats['he'].by_len, delta)
    print(f'\n  mean = mean score_diff on delimiter evaluation. "σ to Δ" is how\n'
          f'  many standard deviations the threshold sits above that mean.')
    print_prefix_table(stats['en'].by_prefix, stats['he'].by_prefix)
    print_gap_table('TABLE 3 — first word of each line only (Δ pinned at baseline)',
                    stats['en'].first_word, stats['he'].first_word, delta)
    print_fits(stats['en'], stats['he'], en_h, he_h)

    skipped = stats['en'].skipped + stats['he'].skipped
    print(f'\n  {skipped:,} words fired mid-word and have no delimiter score; '
          f'they are in table 2 only.')


if __name__ == '__main__':
    main()
