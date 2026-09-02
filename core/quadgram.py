"""
quadgram.py — Character-level quadgram language model with Witten-Bell
interpolated smoothing.

Loads pre-computed quadgram/trigram/bigram counts from JSON and scores strings
by computing log-probability under the model.
"""

import json
import math
import os
import logging

logger = logging.getLogger(__name__)


def _distinct_continuations(counts, ctx_len):
    """Map each context of length ctx_len to N1+(context): the number of
    distinct characters observed following it — the Witten-Bell weight input.
    """
    seen = {}
    for key in counts:
        seen.setdefault(key[:ctx_len], set()).add(key[ctx_len])
    return {ctx: len(conts) for ctx, conts in seen.items()}


class QuadgramModel:
    """Character-level quadgram scorer with Witten-Bell interpolated smoothing.

    P(d|abc) backs off through P(d|bc) and P(d|c) to a Laplace-floored base
    rate, weighted at each level by how well-attested and diverse that
    context is (see evaluation/EXPERIMENTS.md, Arm B). This replaces add-1
    Laplace smoothing, which penalised an unseen continuation in proportion
    to how common its context was — turning a single unfamiliar character
    into a false-positive generator.
    """

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
        # Also doubles as the order-1 context total c(c) for interpolation.
        self._bigram_first_totals = {}
        # Per-second-character bigram totals, used as the order-0 base rate P(d).
        self._char_totals = {}
        for k, c in self.bigram_counts.items():
            self._bigram_first_totals[k[0]] = self._bigram_first_totals.get(k[0], 0) + c
            self._char_totals[k[1]] = self._char_totals.get(k[1], 0) + c

        # N1+(context) at each order, for Witten-Bell interpolation weights.
        self._n1_tri = _distinct_continuations(self.quadgram_counts, 3)
        self._n1_bi = _distinct_continuations(self.trigram_counts, 2)
        self._n1_uni = _distinct_continuations(self.bigram_counts, 1)

    def score(self, text):
        """Compute the log-probability score of a string.

        Uses the quadgram model with Witten-Bell interpolated smoothing:
        P(c4|c1c2c3) backs off through P(c4|c2c3) and P(c4|c3) to a
        Laplace-floored base rate (see _witten_bell_prob).

        For strings shorter than 4 characters, uses a simplified
        Laplace trigram/bigram/unigram fallback — unreachable in production,
        since callers always pass space-padded strings of length >= 4.

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
            log_prob += math.log(self._witten_bell_prob(quadgram))

        return log_prob

    def _witten_bell_prob(self, quadgram):
        """P(d|abc) for a 4-char string 'abcd', via Witten-Bell interpolation.

        Backs off abc -> bc -> c -> a Laplace-floored base rate, trusting
        each context's own maximum-likelihood estimate in proportion to
        how attested and diverse (N1+) that context is — see
        evaluation/EXPERIMENTS.md, Arm B.
        """
        abc, bc, c = quadgram[:3], quadgram[1:3], quadgram[2]
        bcd, cd, d = quadgram[1:4], quadgram[2:4], quadgram[3]

        p = (self._char_totals.get(d, 0) + 1) / (self.total_bigrams + self.vocab_size)
        p = self._interpolate(p, self._bigram_first_totals.get(c, 0),
                               self._n1_uni.get(c, 0), self.bigram_counts.get(cd, 0))
        p = self._interpolate(p, self.bigram_counts.get(bc, 0),
                               self._n1_bi.get(bc, 0), self.trigram_counts.get(bcd, 0))
        p = self._interpolate(p, self.trigram_counts.get(abc, 0),
                               self._n1_tri.get(abc, 0), self.quadgram_counts.get(quadgram, 0))
        return p

    @staticmethod
    def _interpolate(p_backoff, ctx_count, n1_plus, num):
        """One Witten-Bell level: blend this context's ML estimate with the
        lower-order estimate, weighted by ctx_count / (ctx_count + n1_plus).

        Skips the blend (keeps p_backoff) when this context never led to
        the specific continuation being scored — guards against a context
        that is attested but was only ever seen at a word boundary, which
        would otherwise force a spurious probability of exactly zero.
        """
        if not ctx_count or not num:
            return p_backoff
        lam = ctx_count / (ctx_count + n1_plus)
        return lam * (num / ctx_count) + (1 - lam) * p_backoff

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


def load_models(data_dir, load_so=False):
    """Load English, Hebrew, and optionally Stack Overflow quadgram models.

    Args:
        data_dir: Path to the data/ directory.
        load_so: Whether to also load the Stack Overflow model.

    Returns:
        Dict of {name: QuadgramModel} instances.
    """
    en_path = os.path.join(data_dir, 'en_quadgrams.json')
    he_path = os.path.join(data_dir, 'he_quadgrams.json')
    
    models = {
        'en': QuadgramModel(en_path),
        'he': QuadgramModel(he_path)
    }
    
    if load_so:
        so_path = os.path.join(data_dir, 'so_quadgrams.json')
        if os.path.exists(so_path):
            models['so'] = QuadgramModel(so_path)
        else:
            logger.warning('so_quadgrams.json not found — technical mode will fall back to standard')

    return models


def _self_check():
    """Sanity check for _witten_bell_prob: valid probabilities, a familiar
    quadgram scores higher than an unfamiliar one, and a bigram that's only
    ever word-final (zero trigram continuations) doesn't force a spurious
    zero when queried as an interior context.
    """
    import tempfile

    data = {
        'quadgram_counts': {' cat': 50, 'cat ': 50, 'cats': 5},
        'trigram_counts': {' ca': 60, 'cat': 55, 'ats': 5, 'at ': 5},
        'bigram_counts': {' c': 70, 'ca': 60, 'at': 60, 't ': 55, 'ts': 5},
        'vocab_size': 27,
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        model = QuadgramModel(path)

        p_seen = model._witten_bell_prob(' cat')
        p_unseen = model._witten_bell_prob(' cax')
        assert 0 < p_seen <= 1, p_seen
        assert 0 < p_unseen <= 1, p_unseen
        assert p_seen > p_unseen, (p_seen, p_unseen)

        # 't ' is a bigram that only ever occurs word-final (no trigram
        # 't?' continuation exists in this data). Querying it as an
        # interior context (e.g. within 'atzz') must not yield p == 0.
        p_edge = model._witten_bell_prob('atzz')
        assert p_edge > 0, p_edge

        for text in (' cat ', ' cats', 'zzzz'):
            score = model.score(text)
            assert score < 0 and score > float('-inf'), (text, score)
    finally:
        os.remove(path)

    print('quadgram._self_check: OK')


if __name__ == '__main__':
    _self_check()
