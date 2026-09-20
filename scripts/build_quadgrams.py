"""
Script to build quadgram language models from text corpora.
Processes large text files in parallel, extracting bigram, trigram, and quadgram
frequencies, and saves them as JSON models used by the SwitchLang engine.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Ensure we can import download_corpora if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 100_000  # Lines per process

# Drops 43% of keys for +3.5% false positives (evaluation/compare_variants.py).
# The rare tail grows with the corpus, so revisit this if the corpus does.
MIN_QUADGRAM_COUNT = 2

def _process_chunk(lines, allowed_chars=None):
    """
    Worker function to process a list of text lines and return n-gram frequencies.
    
    This function cleans the input text by removing unicode directional marks,
    filters out words containing characters outside the allowed set,
    and then extracts bigrams, trigrams, and quadgrams from valid words.
    
    Args:
        lines (list[str]): A list of string lines to process.
        allowed_chars (set, optional): A set of characters allowed in valid words.
            Words containing unallowed characters are discarded.
            
    Returns:
        tuple: (quadgram_counts, trigram_counts, bigram_counts, unique_chars_set)
    """
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

def build_quadgrams_from_lines(lines, allowed_chars=None, min_count=MIN_QUADGRAM_COUNT):
    """
    Builds n-gram models via parallel processing from an iterable of lines.

    Groups lines into chunks and processes them concurrently using a
    ProcessPoolExecutor to maximize CPU utilization.

    Args:
        lines (iterable[str]): The text lines to process.
        allowed_chars (set, optional): Set of allowed characters for filtering words.
        min_count (int): Quadgrams at or below this count are dropped.
            Pass 0 to keep the full tail (used by the k-fold harness).

    Returns:
        dict: A dictionary containing the aggregated 'quadgram_counts',
              'trigram_counts', 'bigram_counts', and the resulting 'vocab_size'.
    """
    total_quads = Counter()
    total_tris = Counter()
    total_bis = Counter()
    total_chars = set()

    num_workers = os.cpu_count() or 4
    logger.info(f"Building models using {num_workers} processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        chunk = []

        # Submit chunks to the pool
        for line in lines:
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

    # Must run after the merge: a quadgram seen once per chunk still adds up.
    # Trigrams and bigrams are left intact — they are the denominators.
    pruned = {k: c for k, c in total_quads.items() if c > min_count}
    if min_count:
        logger.info(f"Pruned quadgrams at count<={min_count}: "
                    f"{len(total_quads):,} -> {len(pruned):,} keys")

    return {
        'quadgram_counts': pruned,
        'trigram_counts': dict(total_tris),
        'bigram_counts': dict(total_bis),
        'vocab_size': len(total_chars) + 1  # +1 for space
    }


def build_quadgrams_from_file_parallel(file_path, allowed_chars=None,
                                       min_count=MIN_QUADGRAM_COUNT):
    """Build n-gram models from a text corpus file.  See build_quadgrams_from_lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return build_quadgrams_from_lines(f, allowed_chars, min_count)

def save_model_data_to_trie(data, output_trie_path, output_meta_path):
    """Save in-memory quadgram model data dict directly to .marisa and .meta.json."""
    import marisa_trie

    quadgram_counts = data.get('quadgram_counts', {})
    trigram_counts = data.get('trigram_counts', {})
    bigram_counts = data.get('bigram_counts', {})
    vocab_size = data.get('vocab_size', 30)

    # Compute total bigrams and per-first-character totals
    total_bigrams = sum(bigram_counts.values())
    bigram_first_totals = {}
    for k, c in bigram_counts.items():
        if k:
            first_char = k[0]
            bigram_first_totals[first_char] = bigram_first_totals.get(first_char, 0) + c
    bigram_first_totals = dict(sorted(bigram_first_totals.items()))

    # Build list of (ngram, (count,)) for RecordTrie
    logger.info(f"Packing {len(quadgram_counts)} quads, {len(trigram_counts)} tris, {len(bigram_counts)} bis...")
    items = []
    for k, v in quadgram_counts.items():
        items.append((k, (int(v),)))
    for k, v in trigram_counts.items():
        items.append((k, (int(v),)))
    for k, v in bigram_counts.items():
        items.append((k, (int(v),)))

    logger.info(f"Building RecordTrie with {len(items)} total entries...")
    trie = marisa_trie.RecordTrie("<I", items)

    logger.info(f"Saving trie to {output_trie_path}...")
    trie.save(output_trie_path)

    metadata = {
        'vocab_size': vocab_size,
        'total_bigrams': total_bigrams,
        'bigram_first_totals': bigram_first_totals,
        'counts': {
            'quadgrams': len(quadgram_counts),
            'trigrams': len(trigram_counts),
            'bigrams': len(bigram_counts)
        }
    }

    logger.info(f"Saving metadata to {output_meta_path}...")
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    trie_size = os.path.getsize(output_trie_path) / (1024 * 1024)
    meta_size = os.path.getsize(output_meta_path) / 1024
    logger.info(f"Done! Trie: {trie_size:.2f} MB, Meta: {meta_size:.1f} KB")
    return True


# Allowed character sets for each language to ensure model purity.
# We include standard English/Hebrew letters and common punctuation.
# We explicitly EXCLUDE numbers and accented characters (like è, é) to 
# keep the models focused on the primary layout scripts.
ALLOWED_EN = set("abcdefghijklmnopqrstuvwxyz `~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")
ALLOWED_HE = set("אבגדהוזחטיכלמנסעפצקרשתםןץףך `~!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?")

def main():
    """
    Main entry point for the quadgram building script.
    
    The built models ship with the repository, so this is a no-op unless they
    are missing or --force is given. When a build is needed, the corpora are
    downloaded first if absent, then processed in parallel for both English
    and Hebrew and saved directly to .marisa.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--force', action='store_true',
        help='Rebuild the models even if they already exist.'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    en_trie_path = os.path.join(data_dir, 'en_quadgrams.marisa')
    en_meta_path = os.path.join(data_dir, 'en_quadgrams.meta.json')
    he_trie_path = os.path.join(data_dir, 'he_quadgrams.marisa')
    he_meta_path = os.path.join(data_dir, 'he_quadgrams.meta.json')

    # The models are committed, so a fresh clone has nothing to build and no
    # reason to pull down several GB of corpora.
    if not args.force and all(os.path.exists(p) for p in
                              (en_trie_path, en_meta_path, he_trie_path, he_meta_path)):
        logger.info(f"Models already present in {data_dir} — nothing to build.")
        logger.info("Re-run with --force to rebuild them from the corpora.")
        return

    en_txt_path = os.path.join(data_dir, 'en_corpus.txt')
    he_txt_path = os.path.join(data_dir, 'he_corpus.txt')

    if not os.path.exists(en_txt_path) or not os.path.exists(he_txt_path):
        logger.info("Corpora text files not found — downloading...")
        import download_corpora  # imports `datasets`, only needed for a download
        download_corpora.main()

    start_time = time.time()

    # English
    logger.info("Processing English corpus...")
    en_data = build_quadgrams_from_file_parallel(en_txt_path, allowed_chars=ALLOWED_EN)
    save_model_data_to_trie(en_data, en_trie_path, en_meta_path)
    logger.info(f"English model saved to: {en_trie_path} (Vocab: {en_data['vocab_size']})")

    # Hebrew
    logger.info("Processing Hebrew corpus...")
    he_data = build_quadgrams_from_file_parallel(he_txt_path, allowed_chars=ALLOWED_HE)
    save_model_data_to_trie(he_data, he_trie_path, he_meta_path)
    logger.info(f"Hebrew model saved to: {he_trie_path} (Vocab: {he_data['vocab_size']})")

    elapsed = time.time() - start_time
    logger.info(f"Done! Models built in {elapsed:.2f}s.")

if __name__ == '__main__':
    main()
