"""
test_neural_ensemble.py — EnsembleNeuralModel averages member probabilities per step and
drops in wherever a CharNeuralModel is used (engine, evaluation harness).
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.benchmark as bm
from core.neural_model import CharNeuralModel, EnsembleNeuralModel
from scripts.train_char_gru import export_numpy_weights, train_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EN_WORDS = ['hello', 'world', 'help', 'word'] * 10
HE_WORDS = ['שלום', 'עולם'] * 20


def _export(tmp, name, words, lang, seed):
    model, info = train_model(words, lang=lang, emb_dim=8, hidden_dim=8, epochs=1,
                              device='cpu', seed=seed)
    base = os.path.join(tmp, name)
    export_numpy_weights(model, info, base)
    return f'{base}.npz'


class TestEnsemble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.en1 = _export(cls.tmp.name, 'en1', EN_WORDS, 'en', 1)
        cls.en2 = _export(cls.tmp.name, 'en2', EN_WORDS, 'en', 2)
        cls.he = _export(cls.tmp.name, 'he', HE_WORDS, 'he', 1)
        cls.a, cls.b = CharNeuralModel(cls.en1), CharNeuralModel(cls.en2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_single_member_matches_model(self):
        ens = EnsembleNeuralModel([self.a])
        self.assertTrue(np.isclose(ens.score(' hello '), self.a.score(' hello ')))
        lp_e, _ = ens.step('h', ens.init_state())
        lp_a, _ = self.a.step('h', self.a.init_state())
        self.assertTrue(np.allclose(lp_e, lp_a))

    def test_two_members_average_probabilities(self):
        ens = EnsembleNeuralModel([self.a, self.b])
        lp_e, h = ens.step('h', ens.init_state())
        lp_a, _ = self.a.step('h', self.a.init_state())
        lp_b, _ = self.b.step('h', self.b.init_state())
        self.assertTrue(np.allclose(lp_e, np.log((np.exp(lp_a) + np.exp(lp_b)) / 2)))
        self.assertEqual(len(h), 2)

    def test_score_is_sum_of_step_log_probs(self):
        ens = EnsembleNeuralModel([self.a, self.b])
        text = ' héllo wörld '  # includes characters outside the vocab
        ids = [ens.char_to_idx.get(c, ens.unk_idx) for c in text.lower()]
        h, total = ens.init_state(), 0.0
        for i in range(len(text) - 1):
            lp, h = ens.step(text[i].lower(), h)
            total += float(lp[ids[i + 1]])
        self.assertTrue(np.isclose(ens.score(text), total))
        self.assertEqual(ens.score('x'), 0.0)

    def test_attributes_come_from_first_member(self):
        ens = EnsembleNeuralModel([self.a, self.b])
        self.assertEqual((ens.lang, ens.unk_idx, ens.pad_idx), ('en', self.a.unk_idx, self.a.pad_idx))
        self.assertIs(ens.char_to_idx, self.a.char_to_idx)
        self.assertEqual(ens.models, [self.a, self.b])

    def test_rejects_empty_and_mismatched_vocab(self):
        with self.assertRaises(ValueError):
            EnsembleNeuralModel([])
        with self.assertRaises(ValueError):
            EnsembleNeuralModel([self.a, CharNeuralModel(self.he)])

    def test_harness_accepts_path_tuple(self):
        h = bm.EvaluationHarness(DATA_DIR, model_type='neural',
                                 en_model_path=(self.en1, self.en2), he_model_path=self.he)
        self.assertIsInstance(h.engine.en_model, EnsembleNeuralModel)
        self.assertEqual(len(h.engine.en_model.models), 2)
        self.assertIsInstance(h.engine.he_model, CharNeuralModel)
        # the harness's incremental scoring path runs on an ensemble
        self.assertIsNotNone(h._neural_start(h.engine.en_model, 'he'))


if __name__ == '__main__':
    unittest.main()
