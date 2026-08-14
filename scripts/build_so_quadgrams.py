import json
import os
import sys
import time
import logging

# Ensure we can use the same processing logic as build_quadgrams.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_quadgrams import (ALLOWED_EN, MIN_QUADGRAM_COUNT,
                             build_quadgrams_from_lines)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def build_so_quadgrams(file_path, min_count=MIN_QUADGRAM_COUNT):
    """Build the Stack Overflow model.

    Shares build_quadgrams_from_lines so pruning stays in step: technical mode
    takes max(en, so), and pruning one model but not the other would tilt that
    choice.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    # errors='ignore' — the scraped corpus carries some invalid UTF-8.
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return build_quadgrams_from_lines(f, ALLOWED_EN, min_count)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    so_txt_path = os.path.join(data_dir, 'stack_overflow_comments.txt')
    output_path = os.path.join(data_dir, 'so_quadgrams.json')
    
    start_time = time.time()
    
    logger.info("Processing Stack Overflow corpus...")
    so_data = build_so_quadgrams(so_txt_path)
    
    if so_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(so_data, f, ensure_ascii=False, indent=2)
        logger.info(f"SO model saved to: {output_path} (Vocab: {so_data['vocab_size']})")
        
        elapsed = time.time() - start_time
        logger.info(f"Done! SO Model built in {elapsed:.2f}s.")
    else:
        logger.error("Failed to build SO model.")

if __name__ == '__main__':
    main()
