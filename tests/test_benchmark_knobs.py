"""
test_benchmark_knobs.py — the unigram floor reaches the neural models, and changing the
fast-switch margin reuses a worker's harness (and so its GRU score cache).
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.benchmark as bm
from scripts.train_char_gru import export_numpy_weights, train_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class TestHarnessKnobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        for lang, words in (('en', ['hello', 'world'] * 20), ('he', ['שלום', 'עולם'] * 20)):
            model, info = train_model(words, lang=lang, emb_dim=8, hidden_dim=8, epochs=1, device='cpu')
            base = os.path.join(cls.tmp.name, lang)
            export_numpy_weights(model, info, base)
            cls.paths[f'{lang}_model_path'] = f'{base}.npz'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_floor_reaches_models(self):
        off = bm.EvaluationHarness(DATA_DIR, model_type='neural', floor=0.0, **self.paths)
        on = bm.EvaluationHarness(DATA_DIR, model_type='neural', floor=0.05, **self.paths)
        self.assertIsNone(off.engine.en_model._floor)
        self.assertIsNotNone(on.engine.en_model._floor)
        self.assertNotEqual(off.engine.en_model.score(' hello'), on.engine.en_model.score(' hello'))

    def test_margin_change_reuses_harness(self):
        key = (DATA_DIR, self.paths['en_model_path'], self.paths['he_model_path'],
               'standard', 2, 'neural', True, 0.1, 0.05)
        bm._worker_harnesses.clear()
        bm._run_chunk(('fp', 0, ['hello world'], 'en', 6.0, key, 2.5, {}))
        first = bm._worker_harnesses[key]
        bm._run_chunk(('fp', 0, ['hello world'], 'en', 6.0, key, math.inf, {}))
        self.assertIs(bm._worker_harnesses[key], first)
        self.assertEqual(first.fast_margin, math.inf)


if __name__ == '__main__':
    unittest.main()
