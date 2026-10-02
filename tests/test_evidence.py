"""
test_evidence.py — switch evidence carried across words (CUSUM).
"""

import unittest

from core.engine import EvaluationEngine
from evaluation.benchmark import EvaluationHarness, WordResult


class _Fixed:
    """Stub model: every string scores the same."""
    def __init__(self, value):
        self.value = value

    def score(self, text):
        return self.value


class TestEvidence(unittest.TestCase):

    def test_accumulate_clamps_and_skips_collisions(self):
        self.assertEqual(EvaluationEngine.accumulate(1.0, 2.0, False), 3.0)
        self.assertEqual(EvaluationEngine.accumulate(1.0, -5.0, False), 0.0)
        self.assertEqual(EvaluationEngine.accumulate(1.0, -5.0, True), 1.0)

    def test_two_leaning_words_switch_on_the_second(self):
        engine = EvaluationEngine(_Fixed(-2.0), _Fixed(0.0), enable_logging=False)  # diff = +2
        switch, diff, _, ambiguous = engine.evaluate('ab', 'cd', 3.5, on_delimiter=True)
        self.assertFalse(switch)
        self.assertTrue(ambiguous)
        carried = EvaluationEngine.accumulate(0.0, diff, False)
        switch, _, _, _ = engine.evaluate('ab', 'cd', 3.5, on_delimiter=True, evidence=carried)
        self.assertTrue(switch)

    def test_mixed_counts_over_and_under_correction(self):
        # prefix p1 p2 p3 | suffix s1 s2 s3: p3 and s1 lean, s2 switches.
        # Block = [p3, s1] -> one prefix word wrongly flipped, no suffix word missed.
        script = iter([(False, False), (False, False), (False, True), (False, True), (True, False)])
        harness = EvaluationHarness.__new__(EvaluationHarness)  # no models needed

        def fake(word, *_):
            switched, amb = next(script)
            return WordResult(word, word, word, switched, is_ambiguous=amb)
        harness._simulate_word = fake

        report = harness.test_mixed([('s1 s2 s3', 'p1 p2 p3')], 'en', n_prefix=3, progress=False)

        self.assertEqual((report.over_corrected, report.uncorrected), (1, 0))
        self.assertEqual(report.latency_values, [6])   # 's1 ' + 's2 '


if __name__ == '__main__':
    unittest.main()
