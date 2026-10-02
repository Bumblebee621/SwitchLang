"""
learning_curve.py — FP/FN of Quadgram vs Neural as a function of training words.

Sweeps corpus size (nested line prefixes), benchmarks each size with the strict
k-fold harness in kfold_cv.py, and plots mean FP/1k and FN/1k per model against
training words per fold, to find where more data stops paying off.

Results are appended to <out>.json after every point, so a crashed run keeps
what finished; --plot-only redraws <out>.png from that JSON.
"""

import argparse
import json
import logging
import os
import statistics
import sys

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import load_corpus_lines
from evaluation.kfold_cv import run_kfold_evaluation

logger = logging.getLogger('learning_curve')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [  # (key prefix, label, colour, marker) — validated categorical slots 1-2
    ('q', 'Quadgram', '#2a78d6', 'o'),
    ('n', 'Neural (Char-GRU)', '#eb6834', 's'),
]


def summarize(lang, n_lines, lines, k, fold_results):
    """Collapse per-fold kfold results into one curve point."""
    point = {
        'lang': lang,
        'lines': n_lines,
        'train_words': round(sum(len(l.split()) for l in lines) * (k - 1) / k),
    }
    for key in ('q_fp', 'q_fn', 'n_fp', 'n_fn'):
        vals = [r[key] for r in fold_results]
        point[key] = statistics.mean(vals)
        point[key + '_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return point


def plot(points, png_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter, NullFormatter

    langs = [l for l in ('he', 'en') if any(p['lang'] == l for p in points)]
    fig, axes = plt.subplots(len(langs), 2, figsize=(12, 4.5 * len(langs)), squeeze=False)
    for row, lang in enumerate(langs):
        pts = sorted((p for p in points if p['lang'] == lang), key=lambda p: p['train_words'])
        xs = [p['train_words'] for p in pts]
        for col, (metric, title) in enumerate((('fp', 'False positives'), ('fn', 'False negatives'))):
            ax = axes[row][col]
            for prefix, label, colour, marker in MODELS:
                key = f'{prefix}_{metric}'
                ax.errorbar(xs, [p[key] for p in pts], yerr=[p[key + '_std'] for p in pts],
                            label=label, color=colour, marker=marker, markersize=6,
                            linewidth=2, capsize=3)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(EngFormatter(sep=''))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f'{lang.upper()} — {title}', loc='left')
            ax.set_xlabel('Training words per fold (log scale)')
            ax.set_ylabel(f'{metric.upper()} per 1k words')
            ax.set_ylim(bottom=0)
            ax.grid(True, which='major', color='#e5e5e3', linewidth=0.8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", png_path)


def main():
    parser = argparse.ArgumentParser(description="FP/FN learning curve: Quadgram vs Neural")
    parser.add_argument('--sizes', default='1000,5000,20000,50000,100000,300000,800000',
                        help="Comma-separated corpus line counts to sweep")
    parser.add_argument('--k', type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both')
    parser.add_argument('--mode', choices=['standard', 'technical'], default='technical')
    parser.add_argument('--epochs', type=int, default=3, help="Neural training epochs per fold")
    parser.add_argument('--delta', type=float, default=6.0)
    parser.add_argument('--confirmations', type=int, default=2)
    parser.add_argument('--max-test-lines', type=int, default=10000,
                        help="Cap on held-out lines per fold, keeps eval cost flat (default: 10000)")
    parser.add_argument('--symmetric', action='store_true', help="Also fold-train the opposing language")
    parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'data', 'learning_curve'),
                        help="Output path without extension; writes .json and .png")
    parser.add_argument('--plot-only', action='store_true', help="Redraw the PNG from existing JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
    json_path, png_path = args.out + '.json', args.out + '.png'

    if args.plot_only:
        with open(json_path, encoding='utf-8') as f:
            plot(json.load(f), png_path)
        return

    jobs = max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(jobs)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f'CPU ({jobs} threads)'
    logger.info("Neural training device: %s | CPU workers: %d", device, jobs)

    sizes = sorted(int(s) for s in args.sizes.split(','))
    langs = ['he', 'en'] if args.lang == 'both' else [args.lang]
    points = []

    with logging_redirect_tqdm(), tqdm(total=len(langs) * len(sizes), unit='size') as bar:
        for lang in langs:
            corpus = os.path.join(PROJECT_ROOT, 'data', f'{lang}_corpus.txt')
            all_lines = load_corpus_lines(corpus, lang, cap=sizes[-1])
            if len(all_lines) < sizes[-1]:
                logger.warning("%s corpus has only %d usable lines", lang.upper(), len(all_lines))
            other_lines = None
            if args.symmetric:
                other = 'en' if lang == 'he' else 'he'
                other_lines = load_corpus_lines(
                    os.path.join(PROJECT_ROOT, 'data', f'{other}_corpus.txt'), other, cap=sizes[-1])

            for n in sizes:
                bar.set_description(f'{lang.upper()} {n:,} lines')
                lines = all_lines[:n]
                fold_results, _ = run_kfold_evaluation(
                    lang=lang, lines=lines, k=args.k, deltas=[args.delta], epochs=args.epochs,
                    jobs=jobs, max_words=None, mode=args.mode,
                    confirmations=args.confirmations, symmetric=args.symmetric,
                    other_lines=other_lines[:n] if other_lines else None,
                    max_test_lines=args.max_test_lines,
                )[args.delta]
                points.append(summarize(lang, len(lines), lines, args.k, fold_results))
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(points, f, indent=2)
                plot(points, png_path)
                bar.update()

    print(f"\nResults: {json_path}\nPlot:    {png_path}")


if __name__ == '__main__':
    main()
