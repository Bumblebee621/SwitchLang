"""
test_tune_char_gru.py — Unit test for reusable CharGRU hyperparameter optimizer.
"""

import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tune_char_gru import tune_hyperparameters, BASELINE_PARAMS


class TestTuneCharGRU(unittest.TestCase):
    """Test suite for Char-GRU hyperparameter tuning routine."""

    def test_tune_hyperparameters_runs_and_returns_valid_ranking(self):
        """Verify tune_hyperparameters executes trials, ranks by validation loss, and identifies optimal params."""
        sample_words = [
            'hello', 'world', 'computer', 'science', 'switch', 'keyboard',
            'neural', 'network', 'learning', 'python', 'fast', 'accuracy'
        ] * 20

        best_params, results = tune_hyperparameters(
            sample_words,
            lang='en',
            trials=2,
            epochs=1,
            val_ratio=0.2,
            seed=42,
            verbose=False,
        )

        # 1. Best params contains all expected hyperparameter keys
        for key in ['emb_dim', 'hidden_dim', 'lr', 'dropout', 'weight_decay', 'batch_size']:
            self.assertIn(key, best_params)

        # 2. Results has correct trial count
        self.assertEqual(len(results), 2)

        # 3. Results are sorted in ascending order of validation loss
        self.assertLessEqual(results[0]['val_loss'], results[1]['val_loss'])

        # 4. Losses and perplexities are valid positive numbers
        for r in results:
            self.assertGreater(r['val_loss'], 0.0)
            self.assertGreater(r['perplexity'], 1.0)
            self.assertGreater(r['params_count'], 0)
            self.assertGreater(r['size_kb'], 0.0)

        # 5. Exactly one result is marked as baseline
        baseline_entries = [r for r in results if r['is_baseline']]
        self.assertEqual(len(baseline_entries), 1)
        self.assertEqual(baseline_entries[0]['params'], BASELINE_PARAMS)


if __name__ == '__main__':
    unittest.main()
