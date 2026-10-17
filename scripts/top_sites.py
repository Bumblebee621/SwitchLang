"""
top_sites.py — Rank Hebrew CulturaX websites by word count, for building a site blacklist.

Reads every STEP-th row group of each HE shard (a ~1/STEP sample: download
speed, not counting, is the cost) and writes the TOP sites, each followed by
its first SAMPLES lines longer than MIN_WORDS words.

Usage:
    python scripts/top_sites.py [--out data/he_top_sites.txt]
"""

import argparse
import collections
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_corpora import CULTURAX_REVISION, shard_files, site_of

STEP = 10
TOP = 100
SAMPLES = 5
MIN_WORDS = 8


def read_row_group(path, i):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f'datasets/uonlp/CulturaX@{CULTURAX_REVISION}/{path}') as f:
        return pq.ParquetFile(f, pre_buffer=False).read_row_group(i, columns=['text', 'url']).to_pylist()


def num_row_groups(path):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f'datasets/uonlp/CulturaX@{CULTURAX_REVISION}/{path}') as f:
        return pq.ParquetFile(f).metadata.num_row_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'he_top_sites.txt'))
    args = parser.parse_args()

    jobs = [(p, i) for p in shard_files('he') for i in range(0, num_row_groups(p), STEP)]
    words = collections.Counter()
    samples = collections.defaultdict(list)
    with ThreadPoolExecutor(8) as pool:
        for n, docs in enumerate(pool.map(lambda j: read_row_group(*j), jobs), 1):
            for doc in docs:
                site = site_of(doc['url']) or '(no url)'
                for line in (doc['text'] or '').split('\n'):
                    n_words = len(line.split())
                    words[site] += n_words
                    if n_words > MIN_WORDS and len(samples[site]) < SAMPLES:
                        samples[site].append(line.strip())
            print(f'\r{n}/{len(jobs)} row groups', end='', flush=True)
    print()

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'Top {TOP} of {len(words):,} sites by words, from a 1/{STEP} sample of HE CulturaX\n\n')
        for rank, (site, n) in enumerate(words.most_common(TOP), 1):
            f.write(f'{rank}. {site} — {n:,} words in sample\n')
            for line in samples[site] or [f'(no lines longer than {MIN_WORDS} words)']:
                f.write(f'    {line}\n')
            f.write('\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
