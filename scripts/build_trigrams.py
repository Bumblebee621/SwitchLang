"""
build_trigrams.py — Offline script to generate trigram frequency tables.

This script generates pre-computed trigram and bigram frequency tables
for English and Hebrew, suitable for the TrigramModel scorer.

Usage:
    python scripts/build_trigrams.py

The script uses built-in representative text samples to create the
frequency tables. For production use, replace these with larger
corpora (Project Gutenberg, Hebrew Wikipedia, etc.).
"""

import json
import os
import sys
from collections import Counter
import download_corpora


def build_trigrams_from_file(file_path):
    """Build trigram and bigram frequency counts from a large text file.
    
    Streams the file line by line to prevent memory exhaustion.

    Args:
        file_path: Path to the plain text corpus.

    Returns:
        Dict with 'trigram_counts', 'bigram_counts', 'vocab_size'.
    """
    quadgram_counts = Counter()
    trigram_counts = Counter()
    bigram_counts = Counter()
    chars = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip()
            # Remove Right-to-Left and Left-to-Right Marks, as users don't type them
            line = line.replace('\u200f', '').replace('\u200e', '')
            
            if not line:
                continue

            for ch in line:
                if not ch.isspace():
                    chars.add(ch)

            words = line.split()
            for word in words:
                # Skip words that are suspiciously long (likely missing spaces)
                if len(word) > 12:
                    continue

                word = ' ' + word + ' '
                # Extract quadgrams
                for i in range(len(word) - 3):
                    quadgram = word[i:i + 4]
                    quadgram_counts[quadgram] += 1
                
                # Extract trigrams and bigrams for fallbacks
                for i in range(len(word) - 2):
                    trigram = word[i:i + 3]
                    bigram = word[i:i + 2]
                    trigram_counts[trigram] += 1
                    bigram_counts[bigram] += 1

    return {
        'quadgram_counts': dict(quadgram_counts),
        'trigram_counts': dict(trigram_counts),
        'bigram_counts': dict(bigram_counts),
        'vocab_size': len(chars) + 1  # +1 for space
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    en_txt_path = os.path.join(data_dir, 'en_corpus.txt')
    he_txt_path = os.path.join(data_dir, 'he_corpus.txt')
    
    if not os.path.exists(en_txt_path) or not os.path.exists(he_txt_path):
        print("ERROR: Corpora text files not found!")
        print("Downloading corpora...")
        download_corpora.main()

    print('Building English trigram model (this may take a minute depending on corpus size)...')
    en_data = build_trigrams_from_file(en_txt_path)
    en_path = os.path.join(data_dir, 'en_trigrams.json')
    with open(en_path, 'w', encoding='utf-8') as f:
        json.dump(en_data, f, ensure_ascii=False, indent=2)
    print(f'  Quadgrams: {len(en_data["quadgram_counts"])}')
    print(f'  Trigrams:  {len(en_data["trigram_counts"])}')
    print(f'  Bigrams:   {len(en_data["bigram_counts"])}')
    print(f'  Vocab:     {en_data["vocab_size"]}')
    print(f'  Saved to: {en_path}')

    print()
    print('Building Hebrew trigram model (this may take a minute depending on corpus size)...')
    he_data = build_trigrams_from_file(he_txt_path)
    he_path = os.path.join(data_dir, 'he_trigrams.json')
    with open(he_path, 'w', encoding='utf-8') as f:
        json.dump(he_data, f, ensure_ascii=False, indent=2)
    print(f'  Quadgrams: {len(he_data["quadgram_counts"])}')
    print(f'  Trigrams:  {len(he_data["trigram_counts"])}')
    print(f'  Bigrams:   {len(he_data["bigram_counts"])}')
    print(f'  Vocab:     {he_data["vocab_size"]}')
    print(f'  Saved to: {he_path}')

    print()
    print('Building collision set...')
    collision_path = os.path.join(data_dir, 'collisions.json')
    collisions = []
    with open(collision_path, 'w', encoding='utf-8') as f:
        json.dump(collisions, f, ensure_ascii=False, indent=2)
    print(f'  Collisions: {len(collisions)}')
    print(f'  Saved to: {collision_path}')

    print()
    print('Done! Trigram data files have been generated.')


if __name__ == '__main__':
    main()
