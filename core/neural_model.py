"""
neural_model.py — Lightweight character-level GRU neural sequence model for SwitchLang.

Loads pre-trained GRU weights (.npz) and computes log-probability scores using
pure NumPy for ultra-fast, zero-dependency inference (~15 microseconds per step).
"""

import json
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x):
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _log_softmax(x):
    """Numerically stable log_softmax for a 1D vector."""
    m = np.max(x)
    lse = m + np.log(np.sum(np.exp(x - m)))
    return x - lse


class CharNeuralModel:
    """Character-level neural language model backed by a 1-layer GRU in pure NumPy."""

    def __init__(self, model_path):
        """Load CharGRU weights and metadata.
        
        Args:
            model_path: Path to the .npz file (e.g. data/en_char_gru.npz) or base name.
        """
        if not model_path.endswith('.npz'):
            base, _ = os.path.splitext(model_path)
            model_path = f"{base}.npz"

        self.path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Neural model file not found: {model_path}")

        meta_path = model_path[:-4] + '.meta.json'
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Neural model metadata not found: {meta_path}")

        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        self.lang = self.meta.get('lang', 'en')
        self.vocab_size = self.meta['vocab_size']
        self.emb_dim = self.meta['emb_dim']
        self.hidden_dim = self.meta['hidden_dim']
        self.char_to_idx = self.meta['char_to_idx']
        self.unk_idx = self.meta.get('unk_idx', 1)
        self.pad_idx = self.meta.get('pad_idx', 0)

        # Load NumPy weights
        weights = np.load(model_path)
        self.emb = weights['emb']                      # (V, E)
        weight_ih = weights['weight_ih']              # (3*H, E)
        weight_hh = weights['weight_hh']              # (3*H, H)
        bias_ih = weights['bias_ih']                  # (3*H,)
        bias_hh = weights['bias_hh']                  # (3*H,)
        self.weight_fc = weights['weight_fc']          # (V, H)
        self.bias_fc = weights['bias_fc']              # (V,)

        H = self.hidden_dim
        # Split GRU weights into gates (r = reset, z = update, n = candidate)
        self.w_ir = weight_ih[:H]
        self.w_iz = weight_ih[H:2 * H]
        self.w_in = weight_ih[2 * H:3 * H]

        self.w_hr = weight_hh[:H]
        self.w_hz = weight_hh[H:2 * H]
        self.w_hn = weight_hh[2 * H:3 * H]

        self.b_ir = bias_ih[:H]
        self.b_iz = bias_ih[H:2 * H]
        self.b_in = bias_ih[2 * H:3 * H]

        self.b_hr = bias_hh[:H]
        self.b_hz = bias_hh[H:2 * H]
        self.b_hn = bias_hh[2 * H:3 * H]

        # Initial hidden state (all zeros)
        self._initial_h = np.zeros(H, dtype=np.float32)

    def init_state(self):
        """Return a fresh initial hidden state vector."""
        return self._initial_h.copy()

    def step(self, char, h_prev):
        """Run a single GRU step on a character.
        
        Args:
            char: Character string of length 1.
            h_prev: Previous hidden state (1D array of shape (hidden_dim,)).
            
        Returns:
            log_probs: 1D array of shape (vocab_size,) with log P(next_char | history).
            h_next: Next hidden state (1D array of shape (hidden_dim,)).
        """
        token_id = self.char_to_idx.get(char, self.unk_idx)
        x = self.emb[token_id]

        # Gate equations
        r = _sigmoid(self.w_ir @ x + self.b_ir + self.w_hr @ h_prev + self.b_hr)
        z = _sigmoid(self.w_iz @ x + self.b_iz + self.w_hz @ h_prev + self.b_hz)
        n = np.tanh(self.w_in @ x + self.b_in + r * (self.w_hn @ h_prev + self.b_hn))
        h_next = (1.0 - z) * n + z * h_prev

        # Linear projection to vocabulary
        logits = self.weight_fc @ h_next + self.bias_fc
        log_probs = _log_softmax(logits)
        return log_probs, h_next

    def score(self, text):
        """Compute the log-probability score of a string.
        
        Matches QuadgramModel.score(text) interface for seamless drop-in compatibility.
        
        Args:
            text: The text string to score (typically padded with spaces like ' word ').
            
        Returns:
            float log-probability (higher = more likely in this language).
        """
        if len(text) < 2:
            return 0.0

        text = text.lower()
        token_ids = [self.char_to_idx.get(ch, self.unk_idx) for ch in text]

        h = self.init_state()
        total_log_prob = 0.0

        for i in range(len(token_ids) - 1):
            curr_id = token_ids[i]
            next_id = token_ids[i + 1]

            x = self.emb[curr_id]
            r = _sigmoid(self.w_ir @ x + self.b_ir + self.w_hr @ h + self.b_hr)
            z = _sigmoid(self.w_iz @ x + self.b_iz + self.w_hz @ h + self.b_hz)
            n = np.tanh(self.w_in @ x + self.b_in + r * (self.w_hn @ h + self.b_hn))
            h = (1.0 - z) * n + z * h

            logits = self.weight_fc @ h + self.bias_fc
            log_probs = _log_softmax(logits)
            total_log_prob += float(log_probs[next_id])

        return total_log_prob


def load_neural_models(data_dir):
    """Load English and Hebrew CharNeuralModel instances from data directory.
    
    Args:
        data_dir: Path to the data/ directory.
        
    Returns:
        Dict of {'en': CharNeuralModel, 'he': CharNeuralModel}
    """
    en_path = os.path.join(data_dir, 'en_char_gru.npz')
    he_path = os.path.join(data_dir, 'he_char_gru.npz')

    if not os.path.exists(en_path):
        raise FileNotFoundError(f"English neural model not found: {en_path}")
    if not os.path.exists(he_path):
        raise FileNotFoundError(f"Hebrew neural model not found: {he_path}")

    return {
        'en': CharNeuralModel(en_path),
        'he': CharNeuralModel(he_path)
    }
