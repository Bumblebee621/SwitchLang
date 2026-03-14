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

# Number of lines to read from the corpus (1,000,000 lines should be plenty for strong quadgrams)
MAX_LINES_PER_LANG = 1_000_000

# Base URL to the OPUS OpenSubtitles raw mono text files
OPUS_URL_TEMPLATE = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/{lang}.txt.gz"

def stream_corpus(lang: str, out_txt_path: str, max_lines: int) -> None:
    """Stream a .txt.gz file from OPUS and write a specific number of lines to out_txt_path."""
    url = OPUS_URL_TEMPLATE.format(lang=lang)
    logger.info(f"Connecting to {url} ...")
    
    headers = {'User-Agent': 'SwitchLang/1.0 (Real-time keyboard switcher; NLP training)'}
    req = urllib.request.Request(url, headers=headers)
    
    lines_written = 0
    start_time = time.time()
    
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                with open(out_txt_path, 'wb') as f_out:
                    for line in gz:
                        # Optional: Add any filtering for extremely short or noisy subtitle lines here if needed.
                        # For quadgrams, raw text is perfectly fine as is.
                        f_out.write(line)
                        lines_written += 1
                        
                        if lines_written % 100_000 == 0:
                            print(f"\r[{lang.upper()}] Read {lines_written:,} lines...", end="")
                            
                        if lines_written >= max_lines:
                            break
                            
        print()
        elapsed = time.time() - start_time
        logger.info(f"Successfully wrote {lines_written:,} lines for '{lang}' to {out_txt_path} in {elapsed:.1f}s.")
        
    except Exception as e:
        logger.error(f"Error downloading or processing '{lang}' corpus: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    for lang in ['en', 'he']:
        txt_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        
        # Check if we already have a reasonably sized text file
        # We delete it if it is the old Wikipedia text (~17MB max text size)
        if os.path.exists(txt_path):
            if os.path.getsize(txt_path) < 20_000_000:
                logger.info(f"Removing old small corpus '{txt_path}' to replace with OpenSubtitles...")
                os.remove(txt_path)
            else:
                logger.info(f"Corpus for '{lang}' already exists ({os.path.getsize(txt_path) // 1000000} MB). Skipping.")
                continue
            
        logger.info(f"Downloading OpenSubtitles corpus for {lang.upper()}...")
        stream_corpus(lang, txt_path, MAX_LINES_PER_LANG)

    logger.info("All corpora ready.")


if __name__ == '__main__':
    main()
