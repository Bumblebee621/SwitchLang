"""
test_tune_char_gru.py — objective A (FN at an FP budget) and the ablation pick.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tune_char_gru import BASELINE_PARAMS, ablations, best, coarse_candidates, fine_candidates, fn_at_budget

BUDGET = {'en': 0.10, 'he': 0.02}


def _fn(delta):
    return {'en': {'fn': delta / 10, 'med': 5.0}, 'he': {'fn': delta / 20, 'med': 4.0}}


class TestObjective(unittest.TestCase):

    def test_first_delta_within_budget_in_both_languages(self):
        fps = {4.5: {'en': 0.3, 'he': 0.01}, 6.0: {'en': 0.08, 'he': 0.03}, 7.5: {'en': 0.05, 'he': 0.02}}
        fn_calls = []

        def fn_at(d):
            fn_calls.append(d)
            return _fn(d)

        res = fn_at_budget(fps.__getitem__, fn_at, BUDGET, deltas=(7.5, 4.5, 6.0))
        self.assertEqual(res['delta'], 7.5)          # 6.0 fails HE, 4.5 fails EN
        self.assertEqual(fn_calls, [7.5])            # FN only measured at the chosen delta
        self.assertAlmostEqual(res['score'], 0.75 + 0.375)
        self.assertEqual(res['med'], {'en': 5.0, 'he': 4.0})

    def test_none_when_budget_never_met(self):
        self.assertIsNone(fn_at_budget(lambda d: {'en': 1.0, 'he': 1.0}, _fn, BUDGET, deltas=(6.0, 9.0)))


class TestAblations(unittest.TestCase):

    def _row(self, floor, margin, score):
        return {'floor': floor, 'margin': margin, 'result': None if score is None else {'score': score}}

    def test_each_restriction_gets_its_own_best(self):
        rows = [self._row(0.05, 3.5, 2.0), self._row(0.0, 3.5, 2.5), self._row(0.0, 5.0, 2.2),
                self._row(0.05, math.inf, 2.4), self._row(0.0, math.inf, None), self._row(0.1, 2.5, None)]
        picked = ablations(rows)
        self.assertEqual(picked['tuned'], rows[0])
        self.assertEqual(picked['no floor, fast switch'], rows[2])
        self.assertEqual(picked['floor, no fast switch'], rows[3])
        self.assertEqual(picked['neither'], rows[4])   # infeasible, still reported

    def test_best_ranks_infeasible_last(self):
        rows = [self._row(0.0, 2.5, None), self._row(0.0, 3.5, 9.0)]
        self.assertEqual(best(rows), rows[1])


class TestCoarseToFine(unittest.TestCase):

    def test_stage1_starts_at_baseline_and_holds_27_unique(self):
        cands = coarse_candidates()
        self.assertEqual(cands[0], BASELINE_PARAMS)
        self.assertEqual(len({tuple(sorted(c.items())) for c in cands}), 27)
        self.assertEqual(len(cands), 27)

    def test_stage2_varies_only_regularization_and_batch(self):
        cands = fine_candidates(BASELINE_PARAMS)
        self.assertEqual(len({tuple(sorted(c.items())) for c in cands}), 18)
        for c in cands:
            self.assertEqual((c['emb_dim'], c['hidden_dim'], c['lr']),
                             (BASELINE_PARAMS['emb_dim'], BASELINE_PARAMS['hidden_dim'], BASELINE_PARAMS['lr']))


if __name__ == '__main__':
    unittest.main()
