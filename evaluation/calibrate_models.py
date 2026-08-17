"""
calibrate_models.py — per-model nats-per-char constants for the 'calibrated' arm.

EXPERIMENTS.md measures EN at -1.372 nats/char against HE at -1.649 and notes
the offset compounds linearly with word length.  The same argument applies
between the two English models, and more sharply: the SO model is trained on
~18x less text with the same add-1 smoothing, so it is flatter, and a flatter
model hands out milder penalties for anything it has not seen.  Under max()
that makes it win on junk.

Each model is measured on held-out text from its own domain, so the constants
describe how the model behaves on text it considers normal rather than mixing
in a domain shift.  Words are scored the way the engine scores them, wrapped in
spaces, and normalised by length.

Read-only: touches corpora and model JSONs, writes nothing.

Usage:
    python evaluation/calibrate_models.py
    python evaluation/calibrate_models.py --max-words 400000
"""

import argparse
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quadgram import QuadgramModel
from evaluation.technical_mode import load_so_lines
from evaluation.compare_variants import fold_of
from benchmark import load_corpus_lines

WORD_RE = re.compile(r'\S+')


def per_char_stats(model, lines, max_words):
    """Mean and sd of per-character log-probability, measured per word."""
    values = []
    for line in lines:
        for word in WORD_RE.findall(line):
            word = word[:12]
            text = ' ' + word + ' '
            if len(text) < 2:
                continue
            values.append(model.score(text) / len(text))
            if len(values) >= max_words:
                return statistics.mean(values), statistics.stdev(values), len(values)
    if len(values) < 2:
        raise SystemExit('not enough words to calibrate')
    return statistics.mean(values), statistics.stdev(values), len(values)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='Measure per-model nats-per-char constants.')
    parser.add_argument('--data-dir', default=default_data)
    parser.add_argument('--work-dir', default=os.path.join(default_data, 'technical_mode'))
    parser.add_argument('--max-words', type=int, default=200_000)
    parser.add_argument('--max-lines', type=int, default=50_000)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    en_lines = load_corpus_lines(os.path.join(args.data_dir, 'en_corpus.txt'),
                                 'en', args.max_lines)
    so_lines = load_so_lines(os.path.join(args.data_dir, 'stack_overflow_comments.txt'))
    so_test = [l for i, l in enumerate(so_lines) if fold_of(i, args.k) == args.fold]

    targets = [
        ('en', QuadgramModel(os.path.join(args.data_dir, 'en_quadgrams.json')), en_lines),
        ('so_shipped', QuadgramModel(os.path.join(args.data_dir, 'so_quadgrams.json')), so_test),
        ('so_heldout', QuadgramModel(os.path.join(args.work_dir, 'so_quadgrams.json')), so_test),
    ]

    print(f'{"model":<12} {"words":>10} {"mu (nats/char)":>16} {"sigma":>10}')
    print('-' * 52)
    results = {}
    for name, model, lines in targets:
        mu, sigma, n = per_char_stats(model, lines, args.max_words)
        results[name] = (mu, sigma)
        print(f'{name:<12} {n:>10,} {mu:>16.4f} {sigma:>10.4f}')

    mu_en, sigma_en = results['en']
    print('\n--so-calibration values (mu_en,sigma_en,mu_so,sigma_so):')
    for so_name in ('so_shipped', 'so_heldout'):
        mu_so, sigma_so = results[so_name]
        print(f'  {so_name:<11} {mu_en:.4f},{sigma_en:.4f},{mu_so:.4f},{sigma_so:.4f}')


if __name__ == '__main__':
    main()
