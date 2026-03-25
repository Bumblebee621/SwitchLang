"""
download_corpora.py — Download raw conversational text from OpenSubtitles.

Fetches the plain text gzip archives for both English and Hebrew
from the OPUS OpenSubtitles corpus. This provides a massive, highly conversational
dataset that accurately mimics everyday typing and slang.

Usage:
    python scripts/download_corpora.py
"""

import gzip
import logging
import os
import time
import urllib.request
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Number of lines to read from the corpus (5,000,000 lines should be plenty for strong quadgrams)
MAX_LINES_PER_LANG = 5_000_000

# Base URL to the OPUS OpenSubtitles raw mono text files
OPUS_URL_TEMPLATE = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/{lang}.txt.gz"

def stream_corpus(lang: str, out_txt_path: str, max_lines: int) -> None:
    """Stream a .txt.gz file from OPUS, deduplicate lines, and write to out_txt_path."""
    url = OPUS_URL_TEMPLATE.format(lang=lang)
    logger.info(f"Connecting to {url} ...")
    
    headers = {'User-Agent': 'SwitchLang/1.0 (Real-time keyboard switcher; NLP training)'}
    req = urllib.request.Request(url, headers=headers)
    
    lines_written = 0
    lines_seen_total = 0
    seen = set()
    start_time = time.time()
    
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                with open(out_txt_path, 'wb') as f_out:
                    for line in gz:
                        lines_seen_total += 1
                        if line not in seen:
                            seen.add(line)
                            f_out.write(line)
                            lines_written += 1
                        
                        if lines_seen_total % 100_000 == 0:
                            print(f"\r[{lang.upper()}] Scanned {lines_seen_total:,} / wrote {lines_written:,} unique ...", end="")
                            
                        if lines_written >= max_lines:
                            break
                            
        print()
        elapsed = time.time() - start_time
        dupes = lines_seen_total - lines_written
        logger.info(f"Wrote {lines_written:,} unique lines for '{lang}' (scanned {lines_seen_total:,}, removed {dupes:,} duplicates) in {elapsed:.1f}s.")
        
    except Exception as e:
        logger.error(f"Error downloading or processing '{lang}' corpus: {e}")


def deduplicate_file(path: str) -> None:
    """Deduplicate an existing corpus file in-place, preserving order."""
    logger.info(f"Deduplicating '{path}' ...")
    start_time = time.time()
    
    seen = set()
    total = 0
    unique = 0
    tmp_path = path + '.dedup.tmp'
    
    with open(path, 'rb') as f_in, open(tmp_path, 'wb') as f_out:
        for line in f_in:
            total += 1
            if line not in seen:
                seen.add(line)
                f_out.write(line)
                unique += 1
            if total % 500_000 == 0:
                print(f"\r  Scanned {total:,} / kept {unique:,} ...", end="")
    
    print()
    os.replace(tmp_path, path)
    elapsed = time.time() - start_time
    logger.info(f"Deduplication done: {total:,} -> {unique:,} lines (removed {total - unique:,}) in {elapsed:.1f}s.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    for lang in ['en', 'he']:
        txt_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        logger.info(f"Downloading OpenSubtitles corpus for {lang.upper()}...")
        stream_corpus(lang, txt_path, MAX_LINES_PER_LANG)

    logger.info("All corpora ready.")


if __name__ == '__main__':
    main()
