"""
corpus_ab.py — Build a quadgram model from a corpus with test lines held out.

Any training line whose dedup key (after normalization) appears in one of the
--exclude files is dropped, so different corpora can be compared on the same
test sets without train/test overlap.  Kept lines reach the builder as-is,
so a raw corpus trains exactly as it does in production.

With --model gru it trains a Char-GRU the way scripts/train_char_gru.py does
(first 200k words, 3 epochs, shipped hyperparameters, last 5% held out for
validation) once per --seed, writing PREFIX_s{seed}.npz instead.

Usage:
    python evaluation/corpus_ab.py --lang he --train data/he_corpus.txt \\
        --exclude data/corpus_ab/test/he_v1.txt data/corpus_ab/test/he_chat.txt \\
        --out data/corpus_ab/models/he_v1 [--model gru --seed 0 1 2]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_quadgrams import (ALLOWED_EN, ALLOWED_HE, build_quadgrams_from_lines,
                             save_model_data_to_trie)
from clean_corpus import line_key, normalize_line


def held_out_keys(paths):
    """Dedup keys of every line in *paths*."""
    keys = set()
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as f:
            keys.update(line_key(normalize_line(line)) for line in f)
    keys.discard(None)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lang', choices=['en', 'he'], required=True)
    parser.add_argument('--train', required=True, help='Training corpus.')
    parser.add_argument('--exclude', nargs='+', required=True, help='Test files to hold out.')
    parser.add_argument('--out', required=True, help='Output prefix for .marisa/.npz + .meta.json.')
    parser.add_argument('--model', choices=['quadgram', 'gru'], default='quadgram')
    parser.add_argument('--seed', type=int, nargs='+', default=[0], help='GRU training seeds.')
    args = parser.parse_args()

    held_out = held_out_keys(args.exclude)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.model == 'gru':
        from evaluation.kfold_cv import CHARS, extract_words_from_lines, shipped_hparams
        from scripts.train_char_gru import export_numpy_weights, train_model
        with open(args.train, encoding='utf-8', errors='replace') as f:
            kept = (line for line in f if line_key(normalize_line(line)) not in held_out)
            words = extract_words_from_lines(kept, CHARS[args.lang], max_words=200_000)
        n_val = len(words) // 20
        for seed in args.seed:
            model, info = train_model(words[:-n_val], lang=args.lang, epochs=3, val_words=words[-n_val:],
                                      seed=seed, **shipped_hparams(args.lang, os.path.join(ROOT, 'data')))
            export_numpy_weights(model, info, f'{args.out}_s{seed}')
            print(f'{args.out}_s{seed}: {len(words) - n_val:,} training words, val loss {info["val_loss"]:.4f}')
        return
    with open(args.train, encoding='utf-8', errors='replace') as f:
        train = [line for line in f if line_key(normalize_line(line)) not in held_out]
    data = build_quadgrams_from_lines(train, ALLOWED_HE if args.lang == 'he' else ALLOWED_EN)
    save_model_data_to_trie(data, args.out + '.marisa', args.out + '.meta.json')
    print(f'{args.out}: {len(train):,} training lines, '
          f'{len(data["quadgram_counts"]):,} quadgrams')


if __name__ == '__main__':
    main()
