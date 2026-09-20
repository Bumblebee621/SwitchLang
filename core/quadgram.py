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
        self._trie = marisa_trie.RecordTrie("<I")
        self._trie.mmap(model_path)

        meta_path = model_path[:-7] + '.meta.json'
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.vocab_size = meta.get('vocab_size', 30)
        self.total_bigrams = meta.get('total_bigrams', 0)
        self._bigram_first_totals = meta.get('bigram_first_totals', {})

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
        trie = self._trie
        v = self.vocab_size

        if len(text) == 2:
            res = trie.get(text)
            count = res[0][0] if res else 0
            total = self._bigram_first_totals.get(text[0], 0)
            return math.log((count + 1) / (total + v))

        if len(text) == 3:
            tri_res = trie.get(text)
            tri_count = tri_res[0][0] if tri_res else 0
            bi_res = trie.get(text[:2])
            bi_count = bi_res[0][0] if bi_res else 0
            return math.log((tri_count + 1) / (bi_count + v))

        # Base the score heavily on the absolute probability of the first bigram
        first_bigram = text[:2]
        bi_comp_res = trie.get(first_bigram)
        bi_comp_count = bi_comp_res[0][0] if bi_comp_res else 0
        log_prob = math.log((bi_comp_count + 1) / (self.total_bigrams + (v ** 2)))

        for i in range(len(text) - 3):
            quadgram = text[i:i + 4]
            trigram = text[i:i + 3]

            q_res = trie.get(quadgram)
            quad_count = q_res[0][0] if q_res else 0
            t_res = trie.get(trigram)
            tri_count = t_res[0][0] if t_res else 0

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
