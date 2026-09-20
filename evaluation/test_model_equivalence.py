"""
test_model_equivalence.py — Rigorously test numerical equivalence between
the legacy JSON models and the new binary RecordTrie models.
"""

import os
import sys
import math

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quadgram import QuadgramModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def test_model_pair(lang):
    print(f"Testing equivalence for '{lang}' model...")
    json_path = os.path.join(DATA_DIR, f"{lang}_quadgrams.json")
    marisa_path = os.path.join(DATA_DIR, f"{lang}_quadgrams.marisa")

    assert os.path.exists(json_path), f"Missing {json_path}"
    assert os.path.exists(marisa_path), f"Missing {marisa_path}"

    # Force JSON model loading by disabling trie candidate
    # Note: QuadgramModel.__init__ prefers .marisa if given .json, so pass explicit load
    m_trie = QuadgramModel(marisa_path)

    # For JSON model, reference implementation
    class JsonQuadgramModel:
        def __init__(self, path):
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.quadgram_counts = data.get('quadgram_counts', {})
            self.trigram_counts = data.get('trigram_counts', {})
            self.bigram_counts = data.get('bigram_counts', {})
            self.vocab_size = data.get('vocab_size', 30)
            self.total_bigrams = sum(self.bigram_counts.values())
            self._bigram_first_totals = {}
            for k, c in self.bigram_counts.items():
                if k:
                    self._bigram_first_totals[k[0]] = self._bigram_first_totals.get(k[0], 0) + c

        def score(self, text):
            if len(text) < 2:
                return 0.0
            text = text.lower()
            v = self.vocab_size
            if len(text) == 2:
                count = self.bigram_counts.get(text, 0)
                total = self._bigram_first_totals.get(text[0], 0)
                return math.log((count + 1) / (total + v))
            if len(text) == 3:
                tri_count = self.trigram_counts.get(text, 0)
                bi_count = self.bigram_counts.get(text[:2], 0)
                return math.log((tri_count + 1) / (bi_count + v))
            first_bigram = text[:2]
            bi_comp_count = self.bigram_counts.get(first_bigram, 0)
            log_prob = math.log((bi_comp_count + 1) / (self.total_bigrams + (v ** 2)))
            for i in range(len(text) - 3):
                qc = self.quadgram_counts.get(text[i:i + 4], 0)
                tc = self.trigram_counts.get(text[i:i + 3], 0)
                log_prob += math.log((qc + 1) / (tc + v))
            return log_prob

        def score_incremental(self, prev3, new_char):
            if len(prev3) < 3:
                return 0.0
            quadgram = (prev3[-3:] + new_char).lower()
            trigram = prev3[-3:].lower()
            qc = self.quadgram_counts.get(quadgram, 0)
            tc = self.trigram_counts.get(trigram, 0)
            return math.log((qc + 1) / (tc + self.vocab_size))

    m_json = JsonQuadgramModel(json_path)

    # 1. Verify metadata
    assert m_trie.vocab_size == m_json.vocab_size, f"Vocab mismatch: {m_trie.vocab_size} != {m_json.vocab_size}"
    assert m_trie.total_bigrams == m_json.total_bigrams, f"Total bigrams mismatch: {m_trie.total_bigrams} != {m_json.total_bigrams}"
    assert m_trie._bigram_first_totals == m_json._bigram_first_totals, "Bigram first totals mismatch"

    # 2. Test specific keys (existing and missing)
    test_keys = ["the", " the", "tion", "ing ", "כמה", " כמה", "שלום", "quux", "xyz123", "", "a", "ab"]
    for k in test_keys:
        res = m_trie._trie.get(k)
        c_trie = res[0][0] if res else 0
        if len(k) == 4:
            c_json = m_json.quadgram_counts.get(k, 0)
            assert c_json == c_trie, f"Quad mismatch for '{k}': {c_json} != {c_trie}"
        elif len(k) == 3:
            c_json = m_json.trigram_counts.get(k, 0)
            assert c_json == c_trie, f"Tri mismatch for '{k}': {c_json} != {c_trie}"
        elif len(k) == 2:
            c_json = m_json.bigram_counts.get(k, 0)
            assert c_json == c_trie, f"Bi mismatch for '{k}': {c_json} != {c_trie}"

    # 3. Test string scoring across a variety of lengths
    test_strings = [
        "", "a", "ab", "abc", "abcd", "hello", "world", "the quick brown fox jumps over the lazy dog",
        "שלום", "שלום עולם", "בדיקה של מודל השפה החדש",
        "def foo(bar): return bar * 2", "import os, sys, math",
        "completely_unseen_random_quadgram_sequence_xyz_12345"
    ]

    for s in test_strings:
        s_json = m_json.score(s)
        s_trie = m_trie.score(s)
        diff = abs(s_json - s_trie)
        assert diff < 1e-9, f"Score mismatch for '{s}': JSON={s_json}, Trie={s_trie}, diff={diff}"

    # 4. Test incremental scoring
    for prev3 in ["the", "ing", "של ", "def", "xyz"]:
        for new_char in [" ", "e", "ם", "a", "z", "1"]:
            inc_json = m_json.score_incremental(prev3, new_char)
            inc_trie = m_trie.score_incremental(prev3, new_char)
            diff = abs(inc_json - inc_trie)
            assert diff < 1e-9, f"Incremental mismatch for '{prev3}'+'{new_char}': JSON={inc_json}, Trie={inc_trie}"

    print(f"  -> All checks passed for '{lang}' (zero numerical drift, diff < 1e-9).")

def main():
    test_model_pair('en')
    test_model_pair('he')
    if os.path.exists(os.path.join(DATA_DIR, 'so_quadgrams.json')):
        test_model_pair('so')
    print("\nSUCCESS: All models exhibit exact 100% numerical equivalence!")

if __name__ == '__main__':
    main()
