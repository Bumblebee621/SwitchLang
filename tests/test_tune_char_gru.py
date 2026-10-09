"""
test_tune_char_gru.py — weighted-cost objective, phase-1 training overlap, ablation pick, coarse-to-fine grids.
"""

import math
import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.tune_char_gru as tune
from scripts.tune_char_gru import BASELINE_PARAMS, ablations, best, coarse_candidates, fine_candidates, weighted_cost

FPS = {6.0: {'en': 0.30, 'he': 0.02}, 7.5: {'en': 0.10, 'he': 0.01}, 9.0: {'en': 0.05, 'he': 0.0}}


def _fn(delta):
    return {'en': {'fn': delta / 10, 'med': 5.0}, 'he': {'fn': delta / 20, 'med': 4.0}}


class TestWeightedCost(unittest.TestCase):

    def test_picks_delta_with_lowest_weighted_cost(self):
        # w=10 costs: Δ6 4.1, Δ7.5 2.225, Δ9 1.85
        res = weighted_cost(FPS.__getitem__, _fn, deltas=(9.0, 6.0, 7.5), w=10)
        self.assertEqual(res['delta'], 9.0)
        self.assertAlmostEqual(res['score'], 1.85)
        self.assertEqual(res['fp'], FPS[9.0])
        self.assertEqual(res['fn'], {'en': 0.9, 'he': 0.45})
        self.assertEqual(res['med'], {'en': 5.0, 'he': 4.0})

    def test_weight_moves_the_choice(self):
        # w=1 costs: Δ6 1.22, Δ7.5 1.235, Δ9 1.40
        self.assertEqual(weighted_cost(FPS.__getitem__, _fn, deltas=(6.0, 7.5, 9.0), w=1)['delta'], 6.0)


class TestPhase1Overlap(unittest.TestCase):

    def test_trains_off_main_thread_and_keeps_rows_aligned(self):
        on_main = []

        def fake_train_pair(hp, searched, split, seed, work_dir, memo):
            on_main.append(threading.current_thread() is threading.main_thread())
            return {'en_model_path': f"m{hp['emb_dim']}_{seed}"}, {'en': float(hp['emb_dim'])}

        def fake_score(pairs, split, floor, margin, data_dir, jobs):
            s = -float(pairs[0]['en_model_path'][1:].split('_')[0])   # bigger emb_dim = better
            zero = {'en': 0.0, 'he': 0.0}
            return {'delta': 6.0, 'fp': zero, 'fn': zero, 'med': zero, 'score': s}

        cands = [{**BASELINE_PARAMS, 'emb_dim': e} for e in (8, 16, 24)]
        with mock.patch.object(tune, 'train_pair', fake_train_pair), mock.patch.object(tune, 'score', fake_score):
            rows = tune.phase1(cands, 'test', ('en',), {}, [0, 1], 'data', 1, 'work', {})

        self.assertEqual(len(on_main), 6)
        self.assertFalse(any(on_main))
        self.assertEqual([r['params']['emb_dim'] for r in rows], [24, 16, 8])
        self.assertEqual(rows[2]['pairs'], [{'en_model_path': 'm8_0'}, {'en_model_path': 'm8_1'}])
        self.assertEqual(rows[2]['val_loss'], {'en': 8.0})


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
        self.assertEqual(picked['neither'], rows[4])   # empty ablation, still reported

    def test_best_ranks_missing_result_last(self):
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
