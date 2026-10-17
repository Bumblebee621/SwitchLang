"""
build_v1_blacklist.py — Rebuild the v1 HE corpus with and without download_corpora.BLACKLIST.

v1 is how the HE corpus was first made (commit 7699e29): CulturaX HE read in
order from shard 0, every stripped non-empty line kept, first 5M lines.  One
pass writes both corpora, so they differ only by the blacklisted documents.
OUT_DIR/he_base.txt is the original v1 corpus; OUT_DIR/he_bl.txt is the
shipped data/he_corpus.txt (since 2026-10-17).
OUT_DIR/he_test.txt is the next TEST_LINES non-blacklisted lines after he_bl.txt.

Usage:
    python evaluation/build_v1_blacklist.py OUT_DIR
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from download_corpora import blacklisted, shard_docs, shard_files, site_of

MAX_LINES = 5_000_000
TEST_LINES = 300_000


def v1_lines(text):
    return [s for s in (line.strip() for line in (text or '').split('\n')) if s]


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    counts = {'base': 0, 'bl': 0, 'test': 0}
    dropped_docs = 0
    files = {k: open(os.path.join(out_dir, f'he_{k}.txt'), 'w', encoding='utf-8') for k in counts}
    docs = (doc for shard in sorted(shard_files('he')) for doc in shard_docs(shard))
    for doc in docs:
        lines = v1_lines(doc['text'])
        if blacklisted(site_of(doc['url'])):
            dropped_docs += counts['base'] < MAX_LINES
            targets = ['base']
        else:
            # Documents after the blacklisted corpus is full are unseen by both: the test set.
            targets = ['test'] if counts['bl'] >= MAX_LINES else ['base', 'bl']
        for k in targets:
            limit = TEST_LINES if k == 'test' else MAX_LINES
            for line in lines[:limit - counts[k]]:
                files[k].write(line + '\n')
            counts[k] = min(limit, counts[k] + len(lines))
        if counts['test'] >= TEST_LINES:
            break
    for f in files.values():
        f.close()
    print(f'lines: {counts}; blacklisted docs within base: {dropped_docs:,}')


if __name__ == '__main__':
    main()
