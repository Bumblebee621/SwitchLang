"""
convert_models_to_trie.py — Convert quadgram JSON models into compact binary RecordTries.

Generates:
  <model_name>.marisa     — Memory-mapped binary trie of all n-gram counts
  <model_name>.meta.json  — Scalar metadata (vocab_size, total_bigrams, _bigram_first_totals)
"""

import os
import sys
import json
import logging
import argparse

try:
    import marisa_trie
except ImportError:
    print("Error: marisa-trie is required. Install it via 'pip install marisa-trie'", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def convert_json_to_trie(json_path, output_trie_path=None, output_meta_path=None):
    """Convert a single quadgram JSON file to .marisa and .meta.json files."""
    if not os.path.exists(json_path):
        logger.error(f"Input file not found: {json_path}")
        return False

    base, _ = os.path.splitext(json_path)
    output_trie_path = output_trie_path or f"{base}.marisa"
    output_meta_path = output_meta_path or f"{base}.meta.json"

    logger.info(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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

    orig_size = os.path.getsize(json_path) / (1024 * 1024)
    trie_size = os.path.getsize(output_trie_path) / (1024 * 1024)
    meta_size = os.path.getsize(output_meta_path) / 1024

    logger.info(f"Done! Original JSON: {orig_size:.2f} MB -> Trie: {trie_size:.2f} MB + Meta: {meta_size:.1f} KB (saved {(1 - trie_size/orig_size)*100:.1f}%)")
    return True


def convert_all(data_dir):
    """Convert en, he, and so quadgram models in data_dir."""
    models = ['en_quadgrams.json', 'he_quadgrams.json', 'so_quadgrams.json']
    success = True
    for m in models:
        json_path = os.path.join(data_dir, m)
        if os.path.exists(json_path):
            if not convert_json_to_trie(json_path):
                success = False
        else:
            logger.warning(f"Model {json_path} not found, skipping.")
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert SwitchLang JSON quadgram models to binary tries")
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                        help="Path to data directory containing quadgram JSON models")
    parser.add_argument('--file', help="Convert a single JSON file instead of all models in data-dir")
    args = parser.parse_args()

    if args.file:
        convert_json_to_trie(args.file)
    else:
        convert_all(args.data_dir)
