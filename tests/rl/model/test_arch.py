"""Tests for rl.model.arch: token padding, the SetTransformer encoder, and FiLM conditioning."""
import torch
import pytest

import game
from game.state import GameState, Permanent, PlayerState
from rl.model.arch import FiLM, SetTransformer, pad_token_batch
from rl.model.features import CardVocab, build_token_set


def _fixture():
    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    vocab = CardVocab([decklist_a, decklist_b])

    seat0 = PlayerState(on_the_play=True)
    seat0.battlefield = [Permanent(game.CARD_DEFS["Guttersnipe"])]
    seat1 = PlayerState(on_the_play=False)
    seat1.battlefield = [Permanent(game.CARD_DEFS["Kitchen Imp"]), Permanent(game.CARD_DEFS["Sneaky Snacker"])]
    state_a = GameState(on_the_play=True, players=[seat0, seat1])

    seat0b = PlayerState(on_the_play=True)  # a SECOND, differently-sized state -- exercises real padding, not just batch-of-1
    seat1b = PlayerState(on_the_play=False)
    state_b = GameState(on_the_play=True, players=[seat0b, seat1b])

    tokens_a = build_token_set(state_a, 0, vocab)
    tokens_b = build_token_set(state_b, 0, vocab)
    assert len(tokens_a) == 3 and len(tokens_b) == 0, "fixture sanity: expected 3 tokens and 0 tokens respectively"

    return vocab, seat0, seat1, tokens_a, tokens_b


@pytest.mark.slow
def test_pad_token_batch():
    vocab, seat0, _seat1, tokens_a, tokens_b = _fixture()
    vocab_idx, features, mask, identities = pad_token_batch([tokens_a, tokens_b])
    assert identities[0][0] is seat0.battlefield[0], "battlefield token identity must survive padding"
    assert identities[1][0] is None, "the empty-board batch element's padded identity slots must be None"
    assert vocab_idx.shape == (2, 3)
    assert mask[1].all(), "the empty-board batch element must be fully padded"
    assert not mask[0].any(), "the 3-token batch element must have no padding"


@pytest.mark.slow
def test_set_transformer_forward_shapes_and_no_nan():
    vocab, _seat0, _seat1, tokens_a, tokens_b = _fixture()
    vocab_idx, features, mask, _identities = pad_token_batch([tokens_a, tokens_b])
    side_flag = features[:, :, -1]

    net = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    mine_summary, theirs_summary, token_reps = net(vocab_idx, features, mask, side_flag)
    assert mine_summary.shape == (2, 16)
    assert theirs_summary.shape == (2, 16)
    assert token_reps.shape == (2, 3, 16)
    assert torch.isfinite(mine_summary).all() and torch.isfinite(theirs_summary).all(), (
        "an all-padded batch element (state_b) must not produce NaN through the empty-board guard"
    )


@pytest.mark.slow
def test_set_transformer_gradient_flows_to_embedding():
    vocab, _seat0, _seat1, tokens_a, tokens_b = _fixture()
    vocab_idx, features, mask, _identities = pad_token_batch([tokens_a, tokens_b])
    side_flag = features[:, :, -1]

    net = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    mine_summary, theirs_summary, _token_reps = net(vocab_idx, features, mask, side_flag)

    # Gradient sanity: backward pass must reach the embedding table (confirms
    # the whole pipeline is actually differentiable end to end, not just
    # forward-correct).
    loss = mine_summary.sum() + theirs_summary.sum()
    loss.backward()
    assert net.embedding.weight.grad is not None
    assert torch.isfinite(net.embedding.weight.grad).all()


@pytest.mark.slow
def test_set_transformer_permutation_invariance():
    vocab, _seat0, _seat1, tokens_a, tokens_b = _fixture()
    vocab_idx, features, mask, _identities = pad_token_batch([tokens_a, tokens_b])
    side_flag = features[:, :, -1]

    net = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)

    # Permutation invariance: pooled output must not depend on token ORDER
    # within a batch element (the whole point of a set encoder).
    net.eval()
    with torch.no_grad():
        tokens_a_shuffled = [tokens_a[2], tokens_a[0], tokens_a[1]]
        vocab_idx2, features2, mask2, _identities2 = pad_token_batch([tokens_a_shuffled, tokens_b])
        side_flag2 = features2[:, :, -1]
        mine2, theirs2, _ = net(vocab_idx2, features2, mask2, side_flag2)
        mine1, theirs1, _ = net(vocab_idx, features, mask, side_flag)
        assert torch.allclose(mine1, mine2, atol=1e-5), "pooled summary must be permutation-invariant to token order"
        assert torch.allclose(theirs1, theirs2, atol=1e-5)


@pytest.mark.slow
def test_film_shapes_and_identity_init():
    vocab, _seat0, _seat1, tokens_a, tokens_b = _fixture()
    vocab_idx, features, mask, _identities = pad_token_batch([tokens_a, tokens_b])
    side_flag = features[:, :, -1]

    net = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=2, dim_feedforward=32)
    _mine_summary, theirs_summary, _token_reps = net(vocab_idx, features, mask, side_flag)

    film = FiLM(condition_dim=16, layer_dims=[32, 32])
    params = film(theirs_summary)
    assert len(params) == 2
    for gamma, beta in params:
        assert gamma.shape == (2, 32) and beta.shape == (2, 32)
        assert torch.isfinite(gamma).all() and torch.isfinite(beta).all()
    # At a freshly-initialized net (raw output near 0), gamma should start near 1.0 (identity modulation).
    assert torch.allclose(params[0][0].mean(), torch.tensor(1.0), atol=0.5)
