import json
import os
import sys
import time
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure we can use the same processing logic as build_quadgrams.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_quadgrams import _process_chunk, ALLOWED_EN, CHUNK_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def build_so_quadgrams(file_path):
    """Build quadgram model for the Stack Overflow corpus."""
    total_quads = Counter()
    total_tris = Counter()
    total_bis = Counter()
    total_chars = set()

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    num_workers = os.cpu_count() or 4
    logger.info(f"Building SO model using {num_workers} processes from {file_path}...")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk = []
            
            for line in f:
                chunk.append(line)
                if len(chunk) >= CHUNK_SIZE:
                    futures.append(executor.submit(_process_chunk, chunk, ALLOWED_EN))
                    chunk = []
            
            if chunk:
                futures.append(executor.submit(_process_chunk, chunk, ALLOWED_EN))

            total_chunks = len(futures)
            for i, future in enumerate(as_completed(futures)):
                try:
                    quads, tris, bis, chars = future.result()
                    total_quads.update(quads)
                    total_tris.update(tris)
                    total_bis.update(bis)
                    total_chars.update(chars)
                    
                    if (i + 1) % 5 == 0 or (i + 1) == total_chunks:
                        print(f"\r  Progress: {i + 1}/{total_chunks} chunks merged...", end="", flush=True)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
            print()

    return {
        'quadgram_counts': dict(total_quads),
        'trigram_counts': dict(total_tris),
        'bigram_counts': dict(total_bis),
        'vocab_size': len(total_chars) + 1
    }

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
