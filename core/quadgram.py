"""
quadgram.py — Character-level quadgram language model with Laplace smoothing.

Loads pre-computed quadgram/trigram/bigram counts from JSON and scores strings
by computing log-probability under the model.
"""

import json
import math
import os


class QuadgramModel:
    """Character-level quadgram scorer with Laplace (add-1) smoothing."""

    def __init__(self, json_path):
        """Load quadgram data from a JSON file.

        Expected JSON structure:
        {
            "quadgram_counts": {"abcd": 100, ...},
            "trigram_counts": {"abc": 500, ...},
            "bigram_counts": {"ab": 500, ...},
            "vocab_size": 30
        }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.quadgram_counts = data.get('quadgram_counts', {})
        self.trigram_counts = data.get('trigram_counts', {})
        self.bigram_counts = data.get('bigram_counts', {})
        self.vocab_size = data.get('vocab_size', 30)

        # Pre-compute total bigram count across the entire corpus for absolute probabilities
        self.total_bigrams = sum(self.bigram_counts.values())

        # Pre-compute per-first-character bigram totals for O(1) lookup
        # Used by the 2-char fallback in score() instead of scanning the whole dict.
        self._bigram_first_totals = {}
        for k, c in self.bigram_counts.items():
            self._bigram_first_totals[k[0]] = self._bigram_first_totals.get(k[0], 0) + c

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
        log_prob = 0.0
        v = self.vocab_size

        if len(text) == 2:
            bigram = text
            count = self.bigram_counts.get(bigram, 0)
            total = self._bigram_first_totals.get(text[0], 0)
            log_prob = math.log((count + 1) / (total + v))
            return log_prob

        if len(text) == 3:
            trigram = text
            bigram = text[:2]
            tri_count = self.trigram_counts.get(trigram, 0)
            bi_count = self.bigram_counts.get(bigram, 0)
            log_prob = math.log((tri_count + 1) / (bi_count + v))
            return log_prob

        # Base the score heavily on the absolute probability of the first bigram
        first_bigram = text[:2]
        bi_comp_count = self.bigram_counts.get(first_bigram, 0)
        log_prob = math.log((bi_comp_count + 1) / (self.total_bigrams + (v ** 2)))

        for i in range(len(text) - 3):
            quadgram = text[i:i + 4]
            trigram = text[i:i + 3]

            quad_count = self.quadgram_counts.get(quadgram, 0)
            tri_count = self.trigram_counts.get(trigram, 0)

            prob = (quad_count + 1) / (tri_count + v)
            log_prob += math.log(prob)

        return log_prob

    def score_incremental(self, prev2, new_char):
        """Score a single new character given the previous two.

        Useful for real-time per-keystroke evaluation without
        rescoring the entire buffer.

        Args:
            prev2: The two preceding characters (string of length 2).
            new_char: The new character to score.

        Returns:
            float log-probability increment for this quadgram.
        """
        if len(prev2) < 2:
            return 0.0

        prev2 = prev2.lower()
        new_char = new_char.lower()

        quadgram = prev2 + new_char
        trigram = prev2

        quad_count = self.quadgram_counts.get(quadgram, 0)
        tri_count = self.trigram_counts.get(trigram, 0)
        v = self.vocab_size

        return math.log((quad_count + 1) / (tri_count + v))


def load_models(data_dir):
    """Load both English and Hebrew quadgram models.

    Args:
        data_dir: Path to the data/ directory containing
                  en_quadgrams.json and he_quadgrams.json.

    Returns:
        Tuple (en_model, he_model) of QuadgramModel instances.
    """
    en_path = os.path.join(data_dir, 'en_quadgrams.json')
    he_path = os.path.join(data_dir, 'he_quadgrams.json')
    return QuadgramModel(en_path), QuadgramModel(he_path)
