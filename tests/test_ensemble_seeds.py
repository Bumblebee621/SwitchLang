"""
test_ensemble_seeds.py — pure helpers of evaluation/ensemble_seeds.py (no training).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ensemble_seeds import ensemble_pair, halve, rank_seeds


class TestEnsembleSeeds(unittest.TestCase):

    def test_halve_alternates_and_is_disjoint(self):
        lines = list('abcdefg')
        even, odd = halve(lines)
        self.assertEqual(even, ['a', 'c', 'e', 'g'])
        self.assertEqual(odd, ['b', 'd', 'f'])
        self.assertEqual(sorted(even + odd), lines)

    def test_ensemble_pair_builds_tuples_in_order(self):
        pairs = [{'en_model_path': f'en{i}', 'he_model_path': f'he{i}'} for i in (2, 0, 1)]
        self.assertEqual(ensemble_pair(pairs), {'en_model_path': ('en2', 'en0', 'en1'),
                                                'he_model_path': ('he2', 'he0', 'he1')})

    def test_rank_seeds_orders_by_cost(self):
        rows = [{'seed': s, 'result': {'score': c}} for s, c in [(0, 3.0), (1, 1.5), (2, 2.0)]]
        self.assertEqual(rank_seeds(rows), [1, 2, 0])


if __name__ == '__main__':
    unittest.main()
