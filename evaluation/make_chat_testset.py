"""
make_chat_testset.py — Build a conversational benchmark file from OpenSubtitles.

Some subtitle files store Hebrew in visual (reversed) order; each line is kept
in whichever orientation has more words found in a reference corpus.  Lines
are normalized like the training corpora and joined in groups, so each test
line carries chat-length context instead of a ~5-word subtitle.

Usage:
    python evaluation/make_chat_testset.py he "testing quadgram distances/he_corpus_test.txt" \\
        data/corpus_ab/he_v1c.txt data/corpus_ab/test/he_chat.txt
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from clean_corpus import normalize_line

_PUNCT = '.,!?:;"\'()-'


def orient(line, vocab):
    """*line* or its reverse, whichever has more words in *vocab* (ties keep *line*).

    Visual-order Hebrew reverses letters and word order alike, so reversing
    the whole line restores it.
    """
    def hits(s):
        return sum(w.strip(_PUNCT) in vocab for w in s.split())
    rev = line[::-1]
    return rev if hits(rev) > hits(line) else line


def load_vocab(path, max_lines=1_000_000, min_count=5):
    """Words seen at least *min_count* times in the first *max_lines* lines of *path*."""
    counts = collections.Counter()
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            counts.update(w.strip(_PUNCT) for w in normalize_line(line).lower().split())
    return {w for w, c in counts.items() if c >= min_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lang', choices=['en', 'he'])
    parser.add_argument('src', help='OpenSubtitles text, one subtitle per line.')
    parser.add_argument('vocab_corpus', help='Corpus whose frequent words decide orientation.')
    parser.add_argument('out')
    parser.add_argument('--group', type=int, default=6, help='Subtitle lines per test line.')
    parser.add_argument('--max-out', type=int, default=250_000, help='Test lines to write.')
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_corpus)
    written = flipped = 0
    group = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.src, encoding='utf-8', errors='replace') as f_in, \
            open(args.out, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = normalize_line(line)
            if not line:
                continue
            fixed = orient(line, vocab)
            flipped += fixed is not line
            group.append(fixed)
            if len(group) == args.group:
                f_out.write(' '.join(group) + '\n')
                group = []
                written += 1
                if written >= args.max_out:
                    break
    print(f'{args.out}: {written:,} lines, {flipped:,} source lines reversed')


if __name__ == '__main__':
    main()
