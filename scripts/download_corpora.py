"""
download_corpora.py — Download raw text from uonlp/CulturaX.

Fetches text from the Hugging Face dataset to build n-gram models.
Usage:
    python scripts/download_corpora.py
"""

import logging
import os
import time
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Number of lines to read from the corpus (5,000,000 lines should be plenty for strong quadgrams)
MAX_LINES_PER_LANG = 5_000_000

def stream_corpus(lang: str, out_txt_path: str, max_lines: int) -> None:
    """Stream text from uonlp/CulturaX and write to out_txt_path."""
    logger.info(f"Connecting to uonlp/CulturaX for language '{lang}' ...")
    
    ds = load_dataset('uonlp/CulturaX', lang, split='train', streaming=True)
    
    lines_written = 0
    start_time = time.time()
    
    try:
        with open(out_txt_path, 'wb') as f_out:
            for record in ds:
                text = record.get('text', '')
                if not text:
                    continue
                
                # Documents can contain multiple lines
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    f_out.write((line + '\n').encode('utf-8'))
                    lines_written += 1
                    
                    if lines_written % 100_000 == 0:
                        print(f"\r[{lang.upper()}] Wrote {lines_written:,} lines ...", end="")
                        
                    if lines_written >= max_lines:
                        break
                        
                if lines_written >= max_lines:
                    break
                        
        print()
        elapsed = time.time() - start_time
        logger.info(f"Wrote {lines_written:,} lines for '{lang}' in {elapsed:.1f}s.")
        
    except Exception as e:
        logger.error(f"Error downloading or processing '{lang}' corpus: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    for lang in ['en', 'he']:
        txt_path = os.path.join(data_dir, f'{lang}_corpus.txt')
        logger.info(f"Downloading CulturaX corpus for {lang.upper()}...")
        stream_corpus(lang, txt_path, MAX_LINES_PER_LANG)

    logger.info("All corpora ready.")


if __name__ == '__main__':
    main()
