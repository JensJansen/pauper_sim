"""Encoder half of the hand-built-position helpers: the board builders (seat / build / mirror / BUSY) live in the engine
(`game.scenario`) and are re-exported here; this adds `encode`, which runs the real tokenizer + a fixed encoder."""
import torch

from rl.model.arch import SetTransformer, pad_token_batch
from game.scenario import BUSY, NAMES, build, mirror, seat  # noqa: F401  (board builders live in the engine)
from rl.model.features import CardVocab, build_token_set

def make_vocab():
    return CardVocab([[(n, 1) for n in NAMES]])


def make_encoder(vocab, seed=0):
    torch.manual_seed(seed)
    enc = SetTransformer(vocab.size, d_model=32, n_heads=2, n_layers=2, dim_feedforward=64)
    enc.eval()
    return enc


def encode(state, seat_idx, enc, vocab):
    """(mine_summary, theirs_summary, sorted per-token reps) for one seat's view, via the real tokenizer."""
    tokens = build_token_set(state, seat_idx, vocab)
    vi, ft, mask, _ids = pad_token_batch([tokens])
    with torch.no_grad():
        mine, theirs, reps = enc(vi, ft, mask, ft[:, :, -1])
    return mine[0], theirs[0], reps[0, :len(tokens)]


def differs(a, b, tol=1e-5):
    return max((a[0] - b[0]).abs().max().item(), (a[1] - b[1]).abs().max().item()) > tol


def same(a, b, tol=1e-5):
    return not differs(a, b, tol)
