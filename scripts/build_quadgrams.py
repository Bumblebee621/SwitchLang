import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Ensure we can import download_corpora if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import download_corpora

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 100_000  # Lines per process

def _process_chunk(lines, allowed_chars=None):
    """Worker function to process a list of lines and return n-gram counts."""
    quadgram_counts = Counter()
    trigram_counts = Counter()
    bigram_counts = Counter()
    chars = set()

    for line in lines:
        line = line.lower().strip()
        # Remove Right-to-Left and Left-to-Right Marks
        line = line.replace('\u200f', '').replace('\u200e', '')
        
        if not line:
            continue

        words = line.split()
        for word in words:
            # Word-level purity check:
            # If a word contains characters outside the allowed set, discard it.
            # This ensures the model isn't poisoned by foreign scripts or noise.
            if allowed_chars is not None:
                if any(ch not in allowed_chars for ch in word):
                    continue
                
                # Nikud (vocalization) filter for Hebrew:
                # Discard entire words containing Hebrew nikud/vocalization marks 
                # (Unicode range \u0591 to \u05C7).
                if any('\u0591' <= ch <= '\u05C7' for ch in word):
                    continue

            if len(word) > 12:
                continue

            for ch in word:
                chars.add(ch)

            word = ' ' + word + ' '
            n = len(word)
            # Extract quadgrams, trigrams, and bigrams in a single pass
            for i in range(n - 1):
                # Bigrams (length 2)
                bigram_counts[word[i:i + 2]] += 1
                
                # Trigrams (length 3)
                if i < n - 2:
                    trigram_counts[word[i:i + 3]] += 1
                
                # Quadgrams (length 4)
                if i < n - 3:
                    quadgram_counts[word[i:i + 4]] += 1

    return quadgram_counts, trigram_counts, bigram_counts, chars

def build_quadgrams_from_file_parallel(file_path, allowed_chars=None):
    """Build n-gram models using multiple CPU cores."""
    total_quads = Counter()
    total_tris = Counter()
    total_bis = Counter()
    total_chars = set()

    num_workers = os.cpu_count() or 4
    logger.info(f"Building models using {num_workers} processes...")

    with open(file_path, 'r', encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk = []
            
            # Submit chunks to the pool
            for line in f:
                chunk.append(line)
                if len(chunk) >= CHUNK_SIZE:
                    futures.append(executor.submit(_process_chunk, chunk, allowed_chars))
                    chunk = []
            
            if chunk:
                futures.append(executor.submit(_process_chunk, chunk, allowed_chars))

            # Collect and merge results as they become available for better throughput
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
        'vocab_size': len(total_chars) + 1  # +1 for space
    }

# Allowed character sets for each language to ensure model purity.
# We include standard English/Hebrew letters and common punctuation.
# We explicitly EXCLUDE numbers and accented characters (like è, é) to 
# keep the models focused on the primary layout scripts.
ALLOWED_EN = set("abcdefghijklmnopqrstuvwxyz `~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")
ALLOWED_HE = set("אבגדהוזחטיכלמנסעפצקרשתםןץףך `~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    en_txt_path = os.path.join(data_dir, 'en_corpus.txt')
    he_txt_path = os.path.join(data_dir, 'he_corpus.txt')
    
    if not os.path.exists(en_txt_path) or not os.path.exists(he_txt_path):
        logger.error("Corpora text files not found!")
        logger.info("Downloading corpora...")
        download_corpora.main()

    start_time = time.time()

    # English
    logger.info("Processing English corpus...")
    en_data = build_quadgrams_from_file_parallel(en_txt_path, allowed_chars=ALLOWED_EN)
    en_path = os.path.join(data_dir, 'en_quadgrams.json')
    with open(en_path, 'w', encoding='utf-8') as f:
        json.dump(en_data, f, ensure_ascii=False, indent=2)
    logger.info(f"English model saved to: {en_path} (Vocab: {en_data['vocab_size']})")

    # Hebrew
    logger.info("Processing Hebrew corpus...")
    he_data = build_quadgrams_from_file_parallel(he_txt_path, allowed_chars=ALLOWED_HE)
    he_path = os.path.join(data_dir, 'he_quadgrams.json')
    with open(he_path, 'w', encoding='utf-8') as f:
        json.dump(he_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Hebrew model saved to: {he_path} (Vocab: {he_data['vocab_size']})")

    # Reset placeholder collisions file
    collision_path = os.path.join(data_dir, 'collisions.json')
    if not os.path.exists(collision_path):
        with open(collision_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Created placeholder collision set at: {collision_path}")

    elapsed = time.time() - start_time
    logger.info(f"Done! Models built in {elapsed:.2f}s.")

if __name__ == '__main__':
    main()
