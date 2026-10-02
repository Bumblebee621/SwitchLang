"""
test_benchmark_fn.py — FN word counting when a line switches correctly more than once.
"""

import unittest

from evaluation.benchmark import EvaluationHarness, WordResult


class TestFalseNegativeCounting(unittest.TestCase):

    def test_second_switch_counts_only_its_own_wrong_stretch(self):
        # word1: correct switch | word2: engine flips back (FP) | word3: missed on wrong
        # layout | word4: correct switch again | word5: fine. Only word3 is uncorrected.
        script = iter([True, True, False, True, False])
        harness = EvaluationHarness.__new__(EvaluationHarness)  # no models needed
        harness._simulate_word = lambda word, *_: WordResult(word, word, word, next(script))

        report = harness.test_false_negatives(['w1 w2 w3 w4 w5'], 'en', progress=False)

        self.assertEqual(report.words_not_switched, 1)


if __name__ == '__main__':
    unittest.main()
