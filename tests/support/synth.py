"""Synthetic token batches for the encoder maths tests: full control over how many tokens each side has, including none."""
import numpy as np
import torch

from rl.model.arch import SetTransformer
from rl.model.features import TOKEN_FEATURE_DIM

VOCAB = 40


def make_encoder(seed=0, d_model=32, n_heads=2, n_layers=2, dim_ff=64):
    torch.manual_seed(seed)
    enc = SetTransformer(VOCAB, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dim_feedforward=dim_ff)
    enc.eval()
    return enc


def make_tokens(rng, n_mine, n_theirs):
    """(vocab_idx [n], features [n, D]) with the side flag (last column) set per token; mine tokens first."""
    n = n_mine + n_theirs
    vocab_idx = rng.integers(1, VOCAB, size=n)
    feats = (rng.random((n, TOKEN_FEATURE_DIM)) < 0.15).astype(np.float32) * rng.random((n, TOKEN_FEATURE_DIM)).astype(np.float32)
    feats[:, -1] = [1.0] * n_mine + [0.0] * n_theirs
    return vocab_idx, feats


def pad(token_sets, width=None, garbage=None):
    """Stack [(vocab_idx, feats), ...] into padded tensors. `garbage`: an rng that fills the PADDED slots with random
    junk (proving the mask, not the zeros, is what hides them). Returns vocab_idx, features, key_padding_mask."""
    width = max([len(v) for v, _ in token_sets] + [1]) if width is None else width
    b = len(token_sets)
    vi = np.zeros((b, width), dtype=np.int64)
    ft = np.zeros((b, width, TOKEN_FEATURE_DIM), dtype=np.float32)
    mask = np.ones((b, width), dtype=bool)
    for i, (v, f) in enumerate(token_sets):
        n = len(v)
        vi[i, :n], ft[i, :n], mask[i, :n] = v, f, False
        if garbage is not None and n < width:
            vi[i, n:] = garbage.integers(1, VOCAB, size=width - n)
            ft[i, n:] = garbage.random((width - n, TOKEN_FEATURE_DIM)).astype(np.float32)
    return torch.from_numpy(vi), torch.from_numpy(ft), torch.from_numpy(mask)


def run(enc, vi, ft, mask):
    with torch.no_grad():
        return enc(vi, ft, mask, ft[:, :, -1])
