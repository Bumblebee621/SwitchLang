"""
merge_en_so_counts.py — fold Stack Overflow counts into the English model.

Technical mode currently keeps two English models and reconciles them per
string with max().  Merging the counts instead moves the interpolation down to
the quadgram context, where each context is weighted by its own evidence rather
than one model having to explain the whole word, and leaves one model to score
at runtime instead of two.

The SO corpus is ~18x smaller than the English one (56.6M bigram tokens vs
1.02B), so a raw sum barely moves the English model.  --so-scale multiplies the
SO counts before summing; a scale near the corpus-size ratio makes the two
contribute comparably per context.

Caveat worth remembering when reading results: both inputs were already pruned
at count > 2 by build_quadgrams.py, so an n-gram that fell below the threshold
in each source separately stays lost here.  Merging raw corpora would recover
those, at the cost of a full pass over en_corpus.txt per scale setting.

Usage:
    python scripts/merge_en_so_counts.py --out data/combination_exp/merged_s18.json --so-scale 18
"""

import argparse
import json
import os
import sys

# Never write over a model the repo ships and tracks in git.
TRACKED_MODELS = {
    'en_quadgrams.json',
    'he_quadgrams.json',
    'so_quadgrams.json',
    'collisions.json',
}

LEVELS = ('quadgram_counts', 'trigram_counts', 'bigram_counts')


def merge_counts(en_model, so_model, so_scale):
    """Sum the two count tables, with SO counts scaled by so_scale.

    All three levels are scaled by the same factor.  Scaling only the
    numerators would distort every conditional probability, since trigram and
    bigram counts are the denominators score() divides by.
    """
    merged = {}
    for level in LEVELS:
        out = dict(en_model.get(level, {}))
        for gram, count in so_model.get(level, {}).items():
            out[gram] = out.get(gram, 0) + int(round(count * so_scale))
        merged[level] = out

    merged['vocab_size'] = max(en_model.get('vocab_size', 0),
                               so_model.get('vocab_size', 0))
    return merged


def resolve_out_path(out):
    """Reject anything that would land on a tracked model file.

    Checks the realpath, so a symlink named merged.json pointing at
    data/en_quadgrams.json is caught too — data/technical_mode/ is full of
    symlinks back to the production models.
    """
    real = os.path.realpath(out)
    if os.path.basename(real) in TRACKED_MODELS:
        sys.exit(f'refusing to write {out} — resolves to tracked model {real}')
    return real


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')

    parser = argparse.ArgumentParser(
        description='Merge English and Stack Overflow quadgram counts.')
    parser.add_argument('--out', required=True, help='Destination JSON path.')
    parser.add_argument('--en-model', default=os.path.join(data_dir, 'en_quadgrams.json'))
    parser.add_argument('--so-model', default=os.path.join(data_dir, 'so_quadgrams.json'))
    parser.add_argument('--so-scale', type=float, default=1.0,
                        help='Multiply SO counts by this before summing.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite --out if it already exists.')
    args = parser.parse_args()

    out_path = resolve_out_path(args.out)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'{args.out} exists — pass --force to overwrite')

    with open(args.en_model, encoding='utf-8') as f:
        en_model = json.load(f)
    with open(args.so_model, encoding='utf-8') as f:
        so_model = json.load(f)

    merged = merge_counts(en_model, so_model, args.so_scale)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f'scale {args.so_scale}: '
          f'{len(merged["quadgram_counts"]):,} quadgrams, '
          f'{len(merged["trigram_counts"]):,} trigrams, '
          f'{len(merged["bigram_counts"]):,} bigrams, '
          f'vocab {merged["vocab_size"]} -> {args.out}')


if __name__ == '__main__':
    main()
