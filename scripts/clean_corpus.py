"""
clean_corpus.py — Drop near-duplicate and empty lines from a text corpus.

Web corpora repeat page furniture ("Reply", "קרא עוד", share widgets) thousands
of times; counted as-is it inflates n-grams that real typing never produces.
Two lines are duplicates when they match after lowercasing and collapsing
digits and punctuation, so date-stamped templates collapse too.  Lines with
no letters at all ("0", "*", a lone zero-width space) are dropped.

Usage:
    python scripts/clean_corpus.py data/he_corpus.txt data/he_corpus.clean.txt
"""
import argparse
import hashlib
import os
import re

_NON_LETTERS = re.compile(r'[\W\d_]+')


def dedup_lines(lines, min_words=0):
    """Yield stripped lines whose normalized form hasn't been seen before."""
    seen = set()
    for line in lines:
        line = line.strip()
        key = _NON_LETTERS.sub(' ', line.lower()).strip()
        if not key or len(line.split()) < min_words:
            continue
        # 8-byte digests keep a 5M-line corpus at ~40 MB of set entries.
        h = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        if h not in seen:
            seen.add(h)
            yield line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--min-words', type=int, default=0,
                        help='Also drop lines with fewer words than this (default: keep all).')
    args = parser.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        parser.error('dst must differ from src — the raw corpus is not re-downloadable.')

    total = kept = 0
    tmp = args.dst + '.tmp'
    with open(args.src, encoding='utf-8', errors='replace') as f_in, \
            open(tmp, 'w', encoding='utf-8') as f_out:
        def counted(f):
            nonlocal total
            for line in f:
                total += 1
                yield line
        for line in dedup_lines(counted(f_in), args.min_words):
            f_out.write(line + '\n')
            kept += 1
    os.replace(tmp, args.dst)
    print(f'{args.src}: kept {kept:,} of {total:,} lines ({total - kept:,} dropped) -> {args.dst}')


if __name__ == '__main__':
    main()
