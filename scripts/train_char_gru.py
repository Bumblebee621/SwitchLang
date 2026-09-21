"""
train_char_gru.py — Train lightweight character-level GRU sequence models for SwitchLang.

Trains a 1-layer Char-GRU language model for English and Hebrew, then exports:
1. Compact NumPy weights (.npz) for zero-dependency runtime (~50 KB).
2. Model metadata (.meta.json) with character vocabulary.
3. ONNX model (.onnx) for optional ONNX Runtime acceleration.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Character vocabularies
EN_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789-.,'\"!?")
HE_CHARS = list(" אבגדהוזחטיכלמנסעפצקרשתךםןףץ0123456789-.,'\"!?")


class CharGRU(nn.Module):
    """Compact 1-layer character GRU language model."""

    def __init__(self, vocab_size, emb_dim=32, hidden_dim=48):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch, seq_len)
            h: Optional initial hidden state (1, batch, hidden_dim)
            
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
            h_n: Final hidden state
        """
        embeds = self.emb(x)
        out, h_n = self.gru(embeds, h)
        logits = self.fc(out)
        return logits, h_n


class WordDataset(Dataset):
    """Dataset of padded character sequences for language modeling."""

    def __init__(self, words, char_to_idx, max_len=16):
        self.samples = []
        unk_idx = char_to_idx.get('<unk>', 1)
        
        for w in words:
            # Padded with spaces: ' word '
            seq = ' ' + w.strip() + ' '
            if len(seq) > max_len:
                seq = seq[:max_len]
            token_ids = [char_to_idx.get(ch, unk_idx) for ch in seq]
            self.samples.append(token_ids)

        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate(batch):
    """Collate function that pads sequences to uniform length."""
    max_len = max(len(s) for s in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s

    x = torch.tensor(padded[:, :-1], dtype=torch.long)
    y = torch.tensor(padded[:, 1:], dtype=torch.long)
    return x, y


def load_words_from_corpus(corpus_path, max_words=500_000, allowed_chars=None):
    """Extract filtered words from corpus file."""
    logger.info("Reading corpus from %s ...", corpus_path)
    words = []
    allowed_set = set(allowed_chars) if allowed_chars else None

    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            for w in line.split():
                if len(w) < 2 or len(w) > 14:
                    continue
                if any('\u0591' <= ch <= '\u05C7' for ch in w):
                    continue
                if allowed_set and any(ch not in allowed_set for ch in w):
                    continue
                words.append(w)
                if len(words) >= max_words:
                    break
            if len(words) >= max_words:
                break

    logger.info("Loaded %d words from %s", len(words), corpus_path)
    return words


def train_model(words, lang='en', emb_dim=32, hidden_dim=48, epochs=4, batch_size=512, lr=0.003):
    """Train the CharGRU model and return (model, vocab_info)."""
    raw_chars = EN_CHARS if lang == 'en' else HE_CHARS
    
    # Vocabulary: 0 = <pad>, 1 = <unk>, followed by characters
    char_list = ['<pad>', '<unk>'] + raw_chars
    char_to_idx = {ch: i for i, ch in enumerate(char_list)}
    idx_to_char = {i: ch for i, ch in enumerate(char_list)}
    vocab_size = len(char_list)

    logger.info("Training %s CharGRU (vocab_size=%d, emb=%d, hidden=%d) on %d words ...",
                lang, vocab_size, emb_dim, hidden_dim, len(words))

    dataset = WordDataset(words, char_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharGRU(vocab_size, emb_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        total_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        elapsed = time.time() - start_time
        logger.info("[%s Epoch %d/%d] Loss: %.4f (Elapsed: %.1fs)", lang, epoch + 1, epochs, avg_loss, elapsed)

    model.eval()
    vocab_info = {
        'lang': lang,
        'vocab_size': vocab_size,
        'emb_dim': emb_dim,
        'hidden_dim': hidden_dim,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'pad_idx': 0,
        'unk_idx': 1,
    }
    return model.cpu(), vocab_info


def export_numpy_weights(model, vocab_info, output_base):
    """Export model weights to .npz and metadata to .meta.json."""
    state = model.state_dict()

    npz_path = f"{output_base}.npz"
    meta_path = f"{output_base}.meta.json"

    weights = {
        'emb': state['emb.weight'].numpy(),
        'weight_ih': state['gru.weight_ih_l0'].numpy(),
        'weight_hh': state['gru.weight_hh_l0'].numpy(),
        'bias_ih': state['gru.bias_ih_l0'].numpy(),
        'bias_hh': state['gru.bias_hh_l0'].numpy(),
        'weight_fc': state['fc.weight'].numpy(),
        'bias_fc': state['fc.bias'].numpy(),
    }

    np.savez_compressed(npz_path, **weights)
    file_size_kb = os.path.getsize(npz_path) / 1024
    logger.info("Saved NumPy weights: %s (%.1f KB)", npz_path, file_size_kb)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata: %s", meta_path)


def export_onnx(model, vocab_info, output_base):
    """Export PyTorch model to ONNX."""
    onnx_path = f"{output_base}.onnx"
    dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long)
    dummy_h = torch.zeros(1, 1, vocab_info['hidden_dim'])

    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_h),
            onnx_path,
            input_names=['input_chars', 'hidden_in'],
            output_names=['logits', 'hidden_out'],
            dynamic_axes={
                'input_chars': {0: 'batch', 1: 'seq_len'},
                'hidden_in': {1: 'batch'},
                'logits': {0: 'batch', 1: 'seq_len'},
                'hidden_out': {1: 'batch'}
            },
            opset_version=14
        )
        file_size_kb = os.path.getsize(onnx_path) / 1024
        logger.info("Saved ONNX model: %s (%.1f KB)", onnx_path, file_size_kb)
    except Exception as e:
        logger.warning("Could not export ONNX model: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight Char-GRU models for SwitchLang")
    parser.add_argument('--data-dir', default='data', help="Data directory containing corpora")
    parser.add_argument('--max-words', type=int, default=500_000, help="Word sample size per language")
    parser.add_argument('--epochs', type=int, default=4, help="Training epochs")
    parser.add_argument('--emb-dim', type=int, default=32, help="Character embedding dimension")
    parser.add_argument('--hidden-dim', type=int, default=48, help="GRU hidden dimension")
    parser.add_argument('--lang', choices=['en', 'he', 'both'], default='both', help="Language to train")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    languages = ['en', 'he'] if args.lang == 'both' else [args.lang]

    for lang in languages:
        corpus_name = f"{lang}_corpus.txt"
        corpus_path = os.path.join(data_dir, corpus_name)
        if not os.path.exists(corpus_path):
            logger.error("Corpus not found: %s", corpus_path)
            continue

        raw_chars = EN_CHARS if lang == 'en' else HE_CHARS
        words = load_words_from_corpus(corpus_path, max_words=args.max_words, allowed_chars=raw_chars)
        model, vocab_info = train_model(
            words,
            lang=lang,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs
        )

        output_base = os.path.join(data_dir, f"{lang}_char_gru")
        export_numpy_weights(model, vocab_info, output_base)
        export_onnx(model, vocab_info, output_base)


if __name__ == '__main__':
    main()
