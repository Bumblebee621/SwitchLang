"""
test_train_char_gru.py — val_loss in the model metadata must come from held-out words only.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_char_gru import GRU_HPARAMS, train_model

WORDS = ['hello', 'world', 'keyboard', 'switch', 'neural', 'python'] * 10


class TestTrainValLoss(unittest.TestCase):

    def test_no_val_words_records_none(self):
        _, info = train_model(WORDS, emb_dim=8, hidden_dim=8, epochs=1, device='cpu')
        self.assertIsNone(info['val_loss'])
        self.assertGreater(info['train_loss'], 0.0)

    def test_val_words_records_held_out_loss(self):
        _, info = train_model(WORDS, emb_dim=8, hidden_dim=8, epochs=1, device='cpu',
                              val_words=['letter', 'number'])
        self.assertGreater(info['val_loss'], 0.0)
        self.assertNotEqual(info['val_loss'], info['train_loss'])

    def test_defaults_come_from_gru_hparams(self):
        _, info = train_model(WORDS, epochs=1, device='cpu')
        for key, value in GRU_HPARAMS.items():
            self.assertEqual(info[key], 1 if key == 'epochs' else value, key)

    def test_unknown_hyperparameter_rejected(self):
        with self.assertRaises(TypeError):
            train_model(WORDS, hiden_dim=8, epochs=1, device='cpu')


if __name__ == '__main__':
    unittest.main()
