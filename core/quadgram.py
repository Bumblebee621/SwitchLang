"""
quadgram.py — Character-level quadgram language model using binary RecordTries.

Loads pre-computed quadgram/trigram/bigram counts from memory-mapped MARISA tries
and scores strings by computing log-probability with Laplace smoothing.
"""

import json
import math
import os
import logging
import marisa_trie

logger = logging.getLogger(__name__)


class QuadgramModel:
    """Character-level quadgram scorer backed by a memory-mapped RecordTrie."""

    def __init__(self, model_path):
        """Load quadgram data from a .marisa binary trie file."""
        if not model_path.endswith('.marisa'):
            base, _ = os.path.splitext(model_path)
            model_path = f"{base}.marisa"

        self.path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        meta_path = model_path[:-7] + '.meta.json'
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._trie = marisa_trie.RecordTrie("<I")
        self._trie.mmap(model_path)

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.vocab_size = meta.get('vocab_size', 30)
        self.total_bigrams = meta.get('total_bigrams', 0)
        self._bigram_first_totals = meta.get('bigram_first_totals', {})

    def count(self, ngram):
        """Retrieve frequency count of an n-gram from the underlying trie."""
        res = self._trie.get(ngram)
        return res[0][0] if res else 0

    def score(self, text):
        """Compute the log-probability score of a string.

        Uses the quadgram model with Laplace smoothing:
        P(c4 | c1, c2, c3) = (Count(c1,c2,c3,c4) + 1) / (Count(c1,c2,c3) + V)

        For strings shorter than 4 characters, uses a simplified
        trigram/bigram/unigram fallback.

        Args:
            text: The string to score.

        Returns:
            float log-probability (higher = more likely in this language).
        """
        if len(text) < 2:
            return 0.0

        text = text.lower()
        v = self.vocab_size

        if len(text) == 2:
            count = self.count(text)
            total = self._bigram_first_totals.get(text[0], 0)
            return math.log((count + 1) / (total + v))

        if len(text) == 3:
            tri_count = self.count(text)
            bi_count = self.count(text[:2])
            return math.log((tri_count + 1) / (bi_count + v))

        # Base the score heavily on the absolute probability of the first bigram
        first_bigram = text[:2]
        bi_comp_count = self.count(first_bigram)
        log_prob = math.log((bi_comp_count + 1) / (self.total_bigrams + (v ** 2)))

        for i in range(len(text) - 3):
            quadgram = text[i:i + 4]
            trigram = text[i:i + 3]

            quad_count = self.count(quadgram)
            tri_count = self.count(trigram)

            prob = (quad_count + 1) / (tri_count + v)
            log_prob += math.log(prob)

        return log_prob


def load_models(data_dir, load_so=False):
    """Load English, Hebrew, and optionally Stack Overflow quadgram models.

    Args:
        data_dir: Path to the data/ directory.
        load_so: Whether to also load the Stack Overflow model.

    Returns:
        Dict of {name: QuadgramModel} instances.
    """
    models = {
        'en': QuadgramModel(os.path.join(data_dir, 'en_quadgrams.marisa')),
        'he': QuadgramModel(os.path.join(data_dir, 'he_quadgrams.marisa'))
    }

    if load_so:
        so_path = os.path.join(data_dir, 'so_quadgrams.marisa')
        if os.path.exists(so_path):
            models['so'] = QuadgramModel(so_path)
        else:
            logger.warning('so_quadgrams.marisa not found — technical mode will fall back to standard')

    return models
