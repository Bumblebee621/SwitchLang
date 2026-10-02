"""
download_corpora.py — Download raw text from uonlp/CulturaX.

Fetches text from the Hugging Face dataset to build n-gram models.
Existing corpora are never overwritten without --force: a re-download is a
different corpus, so every earlier measurement stops being comparable.

Usage:
    python scripts/download_corpora.py [--force]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_corpus import dedup_lines

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Number of unique lines to keep per language
MAX_LINES_PER_LANG = 5_000_000

# Pinned so the same stream comes back every time (dataset last changed 2024-12-16).
CULTURAX_REVISION = '6a8734bc69fefcbb7735f4f9250f43e4cd7a442e'


def _record_lines(ds):
    # Documents can contain multiple lines
    for record in ds:
        yield from (record.get('text') or '').split('\n')


def stream_corpus(lang: str, out_txt_path: str, max_lines: int) -> None:
    """Stream unique lines from uonlp/CulturaX into out_txt_path.

    Writes to a .tmp file and renames only on success, so a dropped
    connection can't leave a truncated corpus behind.
    """
    from datasets import load_dataset  # heavy; only needed for a download

    logger.info(f"Connecting to uonlp/CulturaX for language '{lang}' ...")
    ds = load_dataset('uonlp/CulturaX', lang, split='train', streaming=True,
                      revision=CULTURAX_REVISION)

    lines_written = 0
    start_time = time.time()
    sha = hashlib.sha256()
    tmp_path = out_txt_path + '.tmp'

    with open(tmp_path, 'wb') as f_out:
        for line in dedup_lines(_record_lines(ds)):
            data = (line + '\n').encode('utf-8')
            f_out.write(data)
            sha.update(data)
            lines_written += 1

            if lines_written % 100_000 == 0:
                print(f"\r[{lang.upper()}] Wrote {lines_written:,} lines ...", end="")

            if lines_written >= max_lines:
                break

    print()
    os.replace(tmp_path, out_txt_path)
    with open(os.path.splitext(out_txt_path)[0] + '.provenance.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': 'uonlp/CulturaX', 'revision': CULTURAX_REVISION,
                   'downloaded': date.today().isoformat(), 'lines': lines_written,
                   'sha256': sha.hexdigest()}, f, indent=2)
    elapsed = time.time() - start_time
    logger.info(f"Wrote {lines_written:,} lines for '{lang}' in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite corpora that already exist.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    for lang in ['en', 'he']:
        txt_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        if os.path.exists(txt_path) and not args.force:
            logger.info(f"{txt_path} exists — skipping (use --force to replace it).")
            continue
        logger.info(f"Downloading CulturaX corpus for {lang.upper()}...")
        stream_corpus(lang, txt_path, MAX_LINES_PER_LANG)

    logger.info("All corpora ready.")


if __name__ == '__main__':
    main()
