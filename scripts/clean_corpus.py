"""
clean_corpus.py — Normalize a text corpus to typeable text and drop duplicates.

Each line is first rewritten to what a keyboard can produce: typographic
punctuation becomes its ASCII counterpart, accents and nikud are stripped,
words with untypeable letters are dropped and untypeable symbols become
spaces (see normalize_line).

Web corpora also repeat page furniture ("Reply", "קרא עוד", share widgets)
thousands of times; counted as-is it inflates n-grams that real typing never
produces.  Two lines are duplicates when they match after lowercasing and
collapsing digits and punctuation, so date-stamped templates collapse too.
Lines with no letters left ("0", "*", a lone zero-width space) are dropped.

Usage:
    python scripts/clean_corpus.py data/he_corpus.txt data/he_corpus.clean.txt
"""
import argparse
import hashlib
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.keymap import EN_TO_HE_FULL

# What the EN (US QWERTY) and HE (SI-1452) layouts can type.
TYPEABLE = frozenset(map(chr, range(0x20, 0x7F))) | frozenset(EN_TO_HE_FULL.values())
_UNTYPEABLE = re.compile('[^' + ''.join(map(re.escape, sorted(TYPEABLE))) + ']')

# Typographic characters with a keyboard counterpart that NFKD leaves alone.
_COUNTERPARTS = str.maketrans({
    '‘': "'", '’': "'", '‚': "'", '‛': "'",
    '“': '"', '”': '"', '„': '"', '‟': '"',
    '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '―': '-', '−': '-',
    '…': '...',
    '′': "'", '″': '"',   # prime marks (8′ = 8 feet)
    '×': 'x',             # 2×4
    '־': '-',   # maqaf
    '׳': "'",   # geresh
    '״': '"',   # gershayim
})

_NON_LETTERS = re.compile(r'[\W\d_]+')


def _typeable_token(token):
    """*token* with untypeable symbols as spaces, or '' if it holds an untypeable letter."""
    if any(unicodedata.category(c).startswith('L') for c in _UNTYPEABLE.findall(token)):
        return ''  # dropping one letter would leave a fake word (straße -> strae)
    return _UNTYPEABLE.sub(' ', token)  # spaces keep a•b from fusing into ab


def normalize_line(line):
    """Rewrite *line* as a keyboard would type it.

    Typographic punctuation maps to ASCII, NFKD splits accents, ligatures and
    fullwidth forms, combining marks (accents, nikud) and invisible format
    characters are deleted, then untypeable letters drop their word and
    untypeable symbols become spaces.
    """
    line = line.translate(_COUNTERPARTS)
    if _UNTYPEABLE.search(line):
        line = ''.join(c for c in unicodedata.normalize('NFKD', line)
                       if unicodedata.category(c) not in ('Mn', 'Cf'))
        line = ' '.join(_typeable_token(t) for t in line.split())
    return ' '.join(line.split())


def line_key(line):
    """8-byte dedup key of *line*'s lowercase letters, or None if it has none."""
    key = _NON_LETTERS.sub(' ', line.lower()).strip()
    # 8-byte digests keep a 5M-line corpus at ~40 MB of set entries.
    return hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest() if key else None


def dedup_lines(lines, min_words=0):
    """Yield normalized lines whose dedup key hasn't been seen before."""
    seen = set()
    for line in lines:
        line = normalize_line(line)
        h = line_key(line)
        if h is None or h in seen or len(line.split()) < min_words:
            continue
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
