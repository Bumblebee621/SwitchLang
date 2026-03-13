"""
build_collisions.py — Generate data/collisions.json

Fetches curated word lists from the internet:
  - English: Google's top 10,000 common English words (with swears)
  - Hebrew:  hspell spell-checker dictionary

A collision pair is a Hebrew word whose physical key shadow (interpreted
on an English QWERTY layout) is a valid English word, and vice versa.
Both sides are written into collisions.json as a flat sorted list.

JSON file will contain only lower case words.

Run from the project root:
    python scripts/build_collisions.py
"""

import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

ENG_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english"
    "/master/google-10000-english.txt"
)
HEB_URL = (
    "https://raw.githubusercontent.com/eyaler/hebrew_wordlists"
    "/main/hspell_simple.txt"
)

# Physical key map: Hebrew letter → English character it shares a key with
HE_TO_EN = {
    'ק': 'e', 'ר': 'r', 'א': 't', 'ט': 'y', 'ו': 'u',
    'ן': 'i', 'ם': 'o', 'פ': 'p', 'ש': 'a', 'ד': 's',
    'ג': 'd', 'כ': 'f', 'ע': 'g', 'י': 'h', 'ח': 'j',
    'ל': 'k', 'ך': 'l', 'ז': 'z', 'ס': 'x', 'ב': 'c',
    'ה': 'v', 'נ': 'b', 'מ': 'n', 'צ': 'm',
    # ת → ',' and ץ → '.' are excluded: they map to punctuation,
    # so any Hebrew word containing them can never shadow an alpha-only
    # English word and will be skipped automatically.
}


def fetch_lines(url):
    """Fetch a URL and return decoded lines."""
    print(f'  Fetching {url} ...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return [line.decode('utf-8').strip() for line in resp]


def load_english_words(url):
    """Top-10k English words, alpha-only, length 2-6."""
    words = set()
    for word in fetch_lines(url):
        w = word.lower()
        if 2 <= len(w) <= 6 and w.isalpha():
            words.add(w)
    print(f'  {len(words):,} valid English words (2-6 chars, alpha-only)')
    return words


def load_hebrew_words(url):
    """hspell dictionary words, Hebrew alpha-only, length 2-6."""
    # Hebrew base letters: U+05D0–U+05EA
    HE_ALPHA = set(chr(c) for c in range(0x05D0, 0x05EB))
    words = set()
    for word in fetch_lines(url):
        w = word.strip()
        if 2 <= len(w) <= 6 and all(ch in HE_ALPHA for ch in w):
            words.add(w)
    print(f'  {len(words):,} valid Hebrew words (2-6 chars, Hebrew alpha-only)')
    return words


def shadow_he_to_en(word):
    """Map a Hebrew word to its English physical shadow.
    Returns '' if any character has no mapping (e.g. ת, ץ).
    """
    result = []
    for ch in word:
        en = HE_TO_EN.get(ch)
        if en is None:
            return ''
        result.append(en)
    return ''.join(result)


def build_collisions():
    print('Loading word lists...')
    en_words = load_english_words(ENG_URL)
    he_words = load_hebrew_words(HEB_URL)

    print('\nFinding collision pairs...')
    collisions = set()

    for he_word in sorted(he_words):
        en_shadow = shadow_he_to_en(he_word)
        if en_shadow and en_shadow in en_words:
            collisions.add(he_word)
            collisions.add(en_shadow)
            print(f'  "{he_word}" <-> "{en_shadow}"')

    print(f'\nTotal collision words: {len(collisions)} ({len(collisions) // 2} pairs)')

    out_path = os.path.join(DATA_DIR, 'collisions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(collisions), f, ensure_ascii=False, indent=2)

    print(f'Written to {out_path}')


if __name__ == '__main__':
    build_collisions()
