"""
test_train_char_gru.py — val_loss in the model metadata must come from held-out words only.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_char_gru import train_model

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


if __name__ == '__main__':
    unittest.main()
