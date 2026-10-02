"""
download_corpora.py — Download a sampled corpus from uonlp/CulturaX.

Per language: SHARDS_PER_LANG parquet files chosen with a fixed seed, read in
turn, each document kept with probability DOC_KEEP_PROB, lines normalized and
deduplicated (clean_corpus), and no website contributing more than SITE_CAP
lines.  Everything is pinned, so a re-run reproduces the same corpus.
Existing corpora are never overwritten without --force: a different corpus
makes every earlier measurement incomparable.

Usage:
    python scripts/download_corpora.py [--force] [--out-dir DIR]
"""

import argparse
import collections
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import date
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_corpus import line_key, normalize_line

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)  # one INFO line per HTTP range request

# Number of unique lines to keep per language
MAX_LINES_PER_LANG = 5_000_000

# Pinned so the same stream comes back every time (dataset last changed 2024-12-16).
CULTURAX_REVISION = '6a8734bc69fefcbb7735f4f9250f43e4cd7a442e'
# Fixes both the shard choice and the per-document sampling.
SEED = 2026
# HE has exactly 6 shards (all taken); EN has 3,072, so this samples 6 of them.
SHARDS_PER_LANG = 6
# Low enough that 5M lines need most of the chosen shards, so the sample
# spreads across each file instead of reading its start.  At 0.08 all six
# HE shards ran out at 4.62M unique lines.
DOC_KEEP_PROB = 0.12
# 0.5% of a 5M-line corpus; stops a few big sites setting the corpus's register.
SITE_CAP = 25_000


def site_of(url):
    """Host of *url* without 'www'/'m' labels, so mobile and desktop count as one site."""
    host = urlparse(url or '').netloc.lower()
    return '.'.join(label for label in host.split('.') if label not in ('www', 'm'))


def round_robin(iterables):
    """Yield one item from each iterable in turn until all are exhausted."""
    iters = [iter(i) for i in iterables]
    while iters:
        for it in list(iters):
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)


def sample_lines(docs, keep_prob, seed, site_cap):
    """Yield normalized, unique lines from a seeded sample of *docs*, capped per site.

    Records without a URL are exempt from the cap: they bundle many unrelated
    sites (5.8% of EN lines).
    """
    rng = random.Random(seed)
    seen = set()
    per_site = collections.Counter()
    for doc in docs:
        if rng.random() >= keep_prob:
            continue
        site = site_of(doc.get('url'))
        # Documents can contain multiple lines
        for line in (doc.get('text') or '').split('\n'):
            if site and per_site[site] >= site_cap:
                break
            line = normalize_line(line)
            h = line_key(line)
            if h is None or h in seen:
                continue
            seen.add(h)
            per_site[site] += 1
            yield line


def shard_files(lang):
    """SHARDS_PER_LANG of *lang*'s parquet files at CULTURAX_REVISION, chosen with SEED."""
    from huggingface_hub import HfApi
    files = sorted(f for f in HfApi().list_repo_files('uonlp/CulturaX', repo_type='dataset',
                                                      revision=CULTURAX_REVISION)
                   if f.startswith(f'{lang}/') and f.endswith('.parquet'))
    return sorted(random.Random(SEED).sample(files, min(SHARDS_PER_LANG, len(files))))


def shard_docs(path):
    """Yield {'text', 'url'} records from one CulturaX parquet file, one row group at a time."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f'datasets/uonlp/CulturaX@{CULTURAX_REVISION}/{path}') as f:
        # pre_buffer keeps every row group already read (~130 MB each in EN):
        # six open shards reached 19 GB within minutes.
        for batch in pq.ParquetFile(f, pre_buffer=False).iter_batches(
                batch_size=1000, columns=['text', 'url']):
            yield from batch.to_pylist()


def stream_corpus(lang: str, out_txt_path: str, max_lines: int) -> None:
    """Write a sampled, site-capped corpus for *lang* to out_txt_path.

    Writes to a .tmp file and renames only on success, so a dropped
    connection can't leave a truncated corpus behind.
    """
    shards = shard_files(lang)
    logger.info(f"[{lang.upper()}] sampling from {shards}")
    streams = [shard_docs(s) for s in shards]

    lines_written = 0
    start_time = time.time()
    sha = hashlib.sha256()
    tmp_path = out_txt_path + '.tmp'

    with open(tmp_path, 'wb') as f_out:
        for line in sample_lines(round_robin(streams), DOC_KEEP_PROB, SEED, SITE_CAP):
            data = (line + '\n').encode('utf-8')
            f_out.write(data)
            sha.update(data)
            lines_written += 1

            if lines_written % 100_000 == 0:
                print(f"\r[{lang.upper()}] Wrote {lines_written:,} lines ...", end="", flush=True)

            if lines_written >= max_lines:
                break

    print()
    if lines_written < max_lines:
        logger.warning(f"[{lang.upper()}] shards exhausted at {lines_written:,} lines "
                       f"— raise DOC_KEEP_PROB.")
    os.replace(tmp_path, out_txt_path)
    with open(os.path.splitext(out_txt_path)[0] + '.provenance.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': 'uonlp/CulturaX', 'revision': CULTURAX_REVISION,
                   'shards': shards, 'seed': SEED, 'doc_keep_prob': DOC_KEEP_PROB,
                   'site_cap': SITE_CAP, 'downloaded': date.today().isoformat(),
                   'lines': lines_written, 'sha256': sha.hexdigest()}, f, indent=2)
    elapsed = time.time() - start_time
    logger.info(f"Wrote {lines_written:,} lines for '{lang}' in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite corpora that already exist.')
    parser.add_argument('--out-dir', default=None,
                        help='Directory for {lang}_corpus.txt (default: data/).')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(project_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)

    for lang in ['en', 'he']:
        txt_path = os.path.join(out_dir, f'{lang}_corpus.txt')
        if os.path.exists(txt_path) and not args.force:
            logger.info(f"{txt_path} exists — skipping (use --force to replace it).")
            continue
        logger.info(f"Downloading CulturaX corpus for {lang.upper()}...")
        stream_corpus(lang, txt_path, MAX_LINES_PER_LANG)

    logger.info("All corpora ready.")


if __name__ == '__main__':
    main()
