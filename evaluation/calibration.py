"""
calibration.py — does the English/Hebrew score gap really grow with word length?

Arm C in EXPERIMENTS.md claims it does.  The two models charge a different mean
log-probability per character (en -1.372, he -1.649), and because the decision
subtracts one score from the other, that 0.277 offset should compound with word
length and push Hebrew typists out of Hebrew on long words.

Part of the offset is genuine — unvocalized Hebrew carries more information per
character — so this measures before it fixes.  Three views of the same run:

  1. mean score gap by word length, with a straight-line fit per direction.
     The slope is Arm C's per-character effect; the intercept is a flat offset
     that no length term would correct.
  2. switch rate per 1,000 *evaluations* by prefix length.  Per evaluation, not
     per word: a 10-char word gets 9 chances to fire and a 3-char word gets 2,
     so a per-word rate would rise on arithmetic alone.
  3. table 1 again, first word of each line only, where delta is pinned at
     baseline instead of climbing quadratically.

Both models are merged from the fold parts compare_variants.py caches, so
neither has seen the lines it is scored on.

Usage:
    python evaluation/calibration.py                        # k=5, fold 0
    python evaluation/calibration.py --max-test-lines 5000  # quick look
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import EvaluationHarness, load_corpus_lines
from compare_variants import VARIANTS, fold_of, merge_parts
from core.sensitivity import SensitivityManager

# Words longer than this never contributed n-grams (build_quadgrams.py:70), so
# their interiors are under-attested in both models.  Kept out of the headline.
TRAIN_MAX_WORD = 12


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

def build_fold_model(lang, cache_dir, work_dir, k, fold):
    """Model for one fold, pruned to match what build_quadgrams.py ships.

    Generalises technical_mode.build_he_arm to either language — the en fold
    parts are already cached alongside the he ones.
    """
    model_path = os.path.join(work_dir, f'{lang}_k{k}_f{fold}.json')
    if os.path.exists(model_path):
        return model_path

    parts = []
    for p in range(k):
        part_path = os.path.join(cache_dir, f'{lang}_k{k}_p{p}.json')
        if not os.path.exists(part_path):
            raise SystemExit(
                f'{part_path} missing — run compare_variants.py first to '
                f'populate the fold cache.')
        with open(part_path, encoding='utf-8') as f:
            parts.append(json.load(f))

    print(f'  merging {lang} fold {fold} …', flush=True)
    model = VARIANTS['prune2'](merge_parts(parts, fold))
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)
    return model_path


def table_mean_logp(model):
    """Count-weighted mean of log P(c4|c1c2c3) over the model's own table.

    This is where EXPERIMENTS.md's -1.372 / -1.649 come from; the provenance
    was never recorded.  Reproduced here so the claim is checkable.
    """
    v = model.vocab_size
    total = weighted = 0
    for quad, c in model.quadgram_counts.items():
        tri = model.trigram_counts.get(quad[:3], 0)
        weighted += c * math.log((c + 1) / (tri + v))
        total += c
    return weighted / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# WALKING A WORD
# ═══════════════════════════════════════════════════════════════════════════

def walk_word(engine, buf_active, buf_shadow, delta, layout):
    """Every evaluation one word triggers, in order, stopping at a switch.

    Mirrors EvaluationHarness._simulate_word (benchmark.py:155) — mid-word from
    3 characters on, then once more on the delimiter — but reports each
    evaluation instead of only the one that decided the word.  check_faithful()
    holds it to that.

    Yields (chars_typed, score_diff, is_colliding, fired, on_delimiter).
    """
    for i in range(2, len(buf_active)):
        should, diff, coll, _ = engine.evaluate(
            buf_active[:i + 1], buf_shadow[:i + 1], delta, current_layout=layout)
        yield i + 1, diff, coll, should, False
        if should:
            return

    should, diff, coll, _ = engine.evaluate(
        buf_active, buf_shadow, delta, current_layout=layout, on_delimiter=True)
    yield len(buf_active), diff, coll, should, True


def check_faithful(harness, lines, lang, delta, sample=2000):
    """walk_word must decide every word exactly as _simulate_word does."""
    words = [w for line in lines for w in line.split()][:sample]
    sens = SensitivityManager(baseline_delta=delta)
    for word in words:
        buf_a, buf_s = harness._get_buffers(word, lang, lang)
        fired = False
        fired_at = -1
        for n, _diff, _coll, fired, on_delim in walk_word(
                harness.engine, buf_a, buf_s, delta, lang):
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
# THE RUN
# ═══════════════════════════════════════════════════════════════════════════

class Stats:
    """Everything the three tables need, accumulated in one pass."""

    def __init__(self):
        self.by_len = {}        # word length -> [n, sum diff, over delta, collisions]
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
        """Least-squares fit of score gap against word length.

        Returns (slope, intercept, standard error of the slope).  The standard
        error assumes independent words, which they are not — words share a
        line, a topic and an author — so read it as a floor on the uncertainty.
        The batch-to-batch spread from --batches is the honest number.
        """
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


def measure(harness, lines, lang, delta):
    """Walk the corpus the way test_false_positives does, recording every eval.

    Only words typed on the *correct* layout are recorded.  After a switch the
    harness flips layout and the next words sit in the recovery regime, which
    is a different question; they still advance sensitivity, they just are not
    counted.
    """
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
                    engine, buf_a, buf_s, sens.delta, current):
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
                    # Fired mid-word, so no delimiter evaluation happened and
                    # the length axis has no comparable score for this word.
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
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

EMPTY_ROW = [0, 0.0, 0, 0, 0.0]


def _summary(row, delta):
    """(mean, sd, over-delta per 1k, how many sd the threshold sits away)."""
    n = row[0]
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    mean = row[1] / n
    var = max(row[4] / n - mean * mean, 0.0)
    sd = math.sqrt(var)
    sigmas = (delta - mean) / sd if sd else 0.0
    return mean, sd, row[2] / n * 1000, sigmas


def print_gap_table(title, en, he, delta):
    print(f'\n{title}')
    print(f'{"len":>4} | {"EN n":>9} {"mean":>7} {"sd":>6} {"σ to Δ":>7} {">Δ/1k":>7} '
          f'| {"HE n":>9} {"mean":>7} {"sd":>6} {"σ to Δ":>7} {">Δ/1k":>7} | {"HE-EN":>7}')
    print('-' * 104)
    for length in range(2, TRAIN_MAX_WORD + 1):
        e = en.get(length, EMPTY_ROW)
        h = he.get(length, EMPTY_ROW)
        em, esd, eo, esig = _summary(e, delta)
        hm, hsd, ho, hsig = _summary(h, delta)
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
    diff_se = math.sqrt(en_se ** 2 + he_se ** 2)   # separate corpora, independent
    print('\nFIT — score gap against word length (length <= '
          f'{TRAIN_MAX_WORD}, delimiter evaluations)')
    print(f'  EN layout   slope {en_slope:+.4f} ± {en_se:.4f} nats/char   '
          f'intercept {en_int:+.3f}')
    print(f'  HE layout   slope {he_slope:+.4f} ± {he_se:.4f} nats/char   '
          f'intercept {he_int:+.3f}')
    print(f'  HE - EN     slope {he_slope - en_slope:+.4f} ± {diff_se:.4f} nats/char   '
          f'intercept {he_int - en_int:+.3f}')
    print(f'  (± is one standard error assuming independent words, so it is a '
          f'floor;\n   --batches gives the real spread.)')
    print('\n  Arm C predicts the HE-EN slope is about +0.277: Hebrew drifting toward')
    print('  a switch as words lengthen.  A flat HE-EN slope with a positive')
    print('  intercept means the exposure is constant and no length term fixes it.')
    print(f'\n  Table-mean log P(c4|c1c2c3), the provenance of EXPERIMENTS.md:130-133:')
    print(f'    en {en_h:+.4f} nats/char   he {he_h:+.4f}   offset {en_h - he_h:+.4f}')


def print_batches(harness, test_lines, delta, batches):
    """Fit the same slope on N disjoint batches, to show how much it moves.

    A slope that is really zero wanders either side of zero from batch to
    batch; a slope that is really +0.277 does not.  This is the check that
    tells a small-sample fluke from an effect.
    """
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
            st = measure(harness, test_lines[lang][cut], lang, delta)
            row[lang] = (st.fit[0], st.line()[0])
        diff = row['he'][1] - row['en'][1]
        diffs.append(diff)
        print(f'{b:>6} | {row["en"][0]:>10,} {row["en"][1]:>+10.4f} '
              f'| {row["he"][0]:>10,} {row["he"][1]:>+10.4f} | {diff:>+9.4f}',
              flush=True)

    mean = sum(diffs) / len(diffs)
    spread = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    print('-' * 68)
    print(f'  HE-EN slope: mean {mean:+.4f}, sd {spread:.4f}, '
          f'range {min(diffs):+.4f} to {max(diffs):+.4f}')
    print(f'  Arm C predicts +0.277.  Sign flips across batches: '
          f'{sum(1 for d in diffs if d < 0)}/{len(diffs)} negative.')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max-test-lines', type=int, default=20000)
    parser.add_argument('--batches', type=int, default=1,
                        help='split the test lines into N disjoint batches and '
                             'report the fit per batch, to show how much the '
                             'answer moves with sample size')
    parser.add_argument('--baseline-delta', type=float, default=4.0)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--cache-dir', default='data/fold_counts')
    parser.add_argument('--work-dir', default='data/calibration')
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    delta = args.baseline_delta

    print('Staging held-out models …', flush=True)
    paths = {lang: build_fold_model(lang, args.cache_dir, args.work_dir,
                                    args.k, args.fold)
             for lang in ('en', 'he')}

    harness = EvaluationHarness(args.data_dir,
                                en_model_path=paths['en'],
                                he_model_path=paths['he'])

    test_lines = {}
    for lang in ('en', 'he'):
        lines = load_corpus_lines(os.path.join(args.data_dir, f'{lang}_corpus.txt'), lang)
        test_lines[lang] = [l for i, l in enumerate(lines)
                            if fold_of(i, args.k) == args.fold]
        if args.max_test_lines:
            test_lines[lang] = test_lines[lang][:args.max_test_lines]

    if args.batches > 1:
        print_batches(harness, test_lines, delta, args.batches)
        return

    stats = {}
    for lang in ('en', 'he'):
        test = test_lines[lang]

        n = check_faithful(harness, test, lang, delta)
        print(f'  {lang}: walk_word agrees with _simulate_word on {n:,} words')

        print(f'  {lang}: scoring {len(test):,} held-out lines …', flush=True)
        stats[lang] = measure(harness, test, lang, delta)

        # Every word is walked exactly once; str.split() yields no empties.
        expected = sum(1 for l in test for _ in l.strip().split())
        assert stats[lang].total_words == expected, (
            f'walked {stats[lang].total_words} words, corpus has {expected}')
        print(f'     {stats[lang].words:,} words on the correct layout '
              f'of {expected:,} total, {stats[lang].evals:,} evaluations, '
              f'{stats[lang].switches:,} switches')

    en_h = table_mean_logp(harness.engine.en_model)
    he_h = table_mean_logp(harness.engine.he_model)

    print(f'\n{"=" * 88}')
    print(f'ARM C — score gap vs word length, k={args.k} fold={args.fold} Δ={delta}')
    print('=' * 88)
    print_gap_table('TABLE 1 — all words on the correct layout',
                    stats['en'].by_len, stats['he'].by_len, delta)
    print(f'\n  mean = mean score_diff on the delimiter evaluation.  "σ to Δ" is how\n'
          f'  many standard deviations the threshold sits above that mean — the\n'
          f'  distance that actually decides whether a word switches.  A mean-level\n'
          f'  offset only matters if it moves this.')
    print_prefix_table(stats['en'].by_prefix, stats['he'].by_prefix)
    print_gap_table('TABLE 3 — first word of each line only (Δ pinned at baseline)',
                    stats['en'].first_word, stats['he'].first_word, delta)
    print_fits(stats['en'], stats['he'], en_h, he_h)

    skipped = stats['en'].skipped + stats['he'].skipped
    print(f'\n  {skipped:,} words fired mid-word and have no delimiter score; '
          f'they are in table 2 only.')


if __name__ == '__main__':
    main()
