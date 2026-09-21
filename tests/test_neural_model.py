"""
test_neural_model.py — Unit and performance tests for CharNeuralModel.
"""

import os
import time
import unittest
import numpy as np

from core.neural_model import CharNeuralModel, load_neural_models
from core.keymap import shadow
from core.engine import EvaluationEngine


class TestCharNeuralModel(unittest.TestCase):
    """Test suite for the character-level GRU neural sequence model."""

    @classmethod
    def setUpClass(cls):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        cls.models = load_neural_models(data_dir)
        cls.en_model = cls.models['en']
        cls.he_model = cls.models['he']
        cls.engine = EvaluationEngine(
            cls.en_model, cls.he_model,
            collisions_path=os.path.join(data_dir, 'collisions.json'),
            model_type='neural'
        )

    def test_model_initialization(self):
        """Verify model weights and metadata are loaded correctly."""
        self.assertGreater(self.en_model.vocab_size, 30)
        self.assertGreater(self.he_model.vocab_size, 30)
        self.assertEqual(self.en_model.emb.shape[0], self.en_model.vocab_size)
        self.assertEqual(self.en_model.emb.shape[1], self.en_model.emb_dim)

    def test_scoring_relative_discrimination(self):
        """Valid words in English must score higher on EN model than on HE model, and vice versa."""
        # 'hello' vs Hebrew keys for hello 'יקךךם'
        en_score_hello = self.en_model.score(' hello ')
        he_score_hello = self.he_model.score(' hello ')
        self.assertGreater(en_score_hello, he_score_hello)

        # 'שלום' vs English keys for שלום 'akui'
        he_score_shalom = self.he_model.score(' שלום ')
        en_score_shalom = self.en_model.score(' שלום ')
        self.assertGreater(he_score_shalom, en_score_shalom)

    def test_engine_evaluation_switch_accuracy(self):
        """Engine with neural models must correctly trigger switch on inverted typing."""
        # 1. English typed on Hebrew layout: 'יקךךם' -> 'hello'
        active_he = shadow('hello', 'en_to_he')
        sw, diff, _, _ = self.engine.evaluate(active_he, 'hello', delta=3.5, current_layout='he', on_delimiter=True)
        self.assertTrue(sw)
        self.assertGreater(diff, 5.0)

        # 2. Hebrew typed on English layout: 'akui' -> 'שלום'
        active_en = shadow('שלום', 'he_to_en')
        sw, diff, _, _ = self.engine.evaluate(active_en, 'שלום', delta=3.5, current_layout='en', on_delimiter=True)
        self.assertTrue(sw)
        self.assertGreater(diff, 5.0)

        # 3. Legitimate English: 'hello' on English layout
        sw, diff, _, _ = self.engine.evaluate('hello', active_he, delta=3.5, current_layout='en', on_delimiter=True)
        self.assertFalse(sw)
        self.assertLess(diff, 0.0)

        # 4. Legitimate Hebrew: 'שלום' on Hebrew layout
        sw, diff, _, _ = self.engine.evaluate('שלום', active_en, delta=3.5, current_layout='he', on_delimiter=True)
        self.assertFalse(sw)
        self.assertLess(diff, 0.0)

    def test_step_method(self):
        """Verify incremental step() runs and produces valid log-prob distributions."""
        h = self.en_model.init_state()
        log_probs, h_next = self.en_model.step('a', h)
        self.assertEqual(len(log_probs), self.en_model.vocab_size)
        self.assertEqual(len(h_next), self.en_model.hidden_dim)
        # Sum of softmax probabilities must equal approximately 1.0 (log-sum-exp <= 0)
        prob_sum = np.sum(np.exp(log_probs))
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_inference_latency(self):
        """Verify inference latency is sub-millisecond per word."""
        word = ' testing '
        start = time.perf_counter()
        n_iters = 500
        for _ in range(n_iters):
            self.en_model.score(word)
        elapsed_us = (time.perf_counter() - start) / n_iters * 1e6

        # Average scoring time for an 8-character word must be well below 1,000 microseconds (1ms)
        self.assertLess(elapsed_us, 500.0, f"Scoring too slow: {elapsed_us:.1f} us")


if __name__ == '__main__':
    unittest.main()
