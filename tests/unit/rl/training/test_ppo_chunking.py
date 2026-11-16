"""Chunked PPO update (`max_token_rows`): the same update as one pass per minibatch, up to floating-point rounding.

The properties that matter: the chunker covers every episode once and respects its budget, the accumulated gradient equals the
single-pass gradient (compared as gradients, never as parameters after an Adam step -- Adam turns rounding noise in a
near-zero gradient into a full-size step, which is not a difference in the update), and the knob reaches ppo_update from the
run config. Episodes are ragged and carry real token lists, so padding, the widest-board term and the pointer head are all in
play."""
import copy

import numpy as np
import pytest
import torch

from rl.league import league_runner
from rl.model.deck import SCALAR_FEATURE_DIM
from rl.model.features import TOKEN_FEATURE_DIM
from rl.training.ppo import _pack_episodes, ppo_update
from rl.training.train import RolloutBuffer
from support.ppo_reference import N_FIXED, sgd, tiny_net


# ---- the chunker ---------------------------------------------------------------------------------------------------------

def _cost(chunk, widths, spans):
    longest = max(e - s for s, e in chunk)
    return len(chunk) * longest * max(widths[spans.index(sp)] for sp in chunk)


def test_every_episode_lands_in_exactly_one_chunk_and_chunks_respect_the_budget():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 17))
        lengths = rng.integers(1, 400, size=n)
        ends = np.cumsum(lengths)
        spans = [(int(end - n_steps), int(end)) for end, n_steps in zip(ends, lengths)]
        widths = [int(w) for w in rng.integers(1, 90, size=n)]
        budget = int(rng.integers(500, 60000))
        chunks = _pack_episodes(spans, widths, budget)
        flat = [sp for c in chunks for sp in c]
        assert sorted(flat) == sorted(spans)  # covered once, none dropped or duplicated
        for c in chunks:
            if len(c) > 1:  # only a lone episode may exceed the budget (it cannot be split)
                assert _cost(c, widths, spans) <= budget


def test_a_budget_larger_than_the_minibatch_gives_one_chunk_and_a_tiny_one_gives_one_episode_each():
    spans = [(0, 10), (10, 40), (40, 45)]
    widths = [3, 8, 1]
    assert len(_pack_episodes(spans, widths, 10**9)) == 1
    chunks = _pack_episodes(spans, widths, 1)
    assert [len(c) for c in chunks] == [1, 1, 1]
    assert [c[0] for c in chunks] == [(10, 40), (0, 10), (40, 45)]  # longest first


def test_neighbours_of_similar_length_share_a_chunk_so_padding_stays_small():
    spans = [(0, 100), (100, 200), (200, 210), (210, 221)]
    chunks = _pack_episodes(spans, [1, 1, 1, 1], 100 * 2 * 1)  # room for two 100-step episodes, not four rows of 100
    assert sorted(len(c) for c in chunks) == [2, 2]
    assert {(0, 100), (100, 200)} in [set(c) for c in chunks]  # the two long ones together, the two short ones together


def test_an_empty_minibatch_has_no_chunks():
    assert _pack_episodes([], [], 1000) == []


# ---- the update ------------------------------------------------------------------------------------------------------------

def make_token_buffer(lengths, seed=0, n_fixed=N_FIXED):
    """Ragged episodes with real token lists (1..9 tokens per state), pointer-capable masks and a mix of fixed and pointer
    actions, so the encoder, the widest-board slice and the pointer head are all exercised."""
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer()
    for n in lengths:
        for step in range(n):
            k = int(rng.integers(0, 10))
            tokens = []
            for t in range(k):
                row = rng.random(TOKEN_FEATURE_DIM).astype(np.float32)
                row[-1] = float(t % 2)  # side flag
                tokens.append((int(rng.integers(1, 5)), row, None))
            mask = np.concatenate([rng.random(n_fixed) < 0.6, rng.random(k) < 0.5])
            if not mask.any():
                mask[0] = True
            action = int(rng.choice(np.flatnonzero(mask)))
            buf.add(tokens, rng.normal(size=SCALAR_FEATURE_DIM).astype(np.float32), mask, action,
                    float(rng.normal(-1.0, 0.3)), float(rng.normal(0.0, 0.5)), float(rng.normal(0.0, 1.0)), step == n - 1)
    return buf


def _grads(net, buf, **kw):
    """The gradient of the last minibatch, left on the parameters after the update (lr 0: nothing moves)."""
    net = copy.deepcopy(net)
    stats = {}
    ppo_update(net, sgd(net, lr=0.0), buf, "cpu", n_epochs=1, target_kl=None, stats_out=stats, **kw)
    return torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]), stats


@pytest.mark.parametrize("budget", [1, 40, 300, 2000, 10**9])
def test_chunked_gradient_equals_the_single_pass_gradient(budget):
    net = tiny_net(seed=5)
    buf = make_token_buffer([7, 30, 3, 18, 11], seed=2)
    ref, ref_stats = _grads(net, buf, batch_size=16)  # all five episodes in one minibatch
    got, stats = _grads(net, buf, batch_size=16, max_token_rows=budget)
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-6), float((got - ref).abs().max())
    assert torch.nn.functional.cosine_similarity(got, ref, dim=0) == pytest.approx(1.0, abs=1e-5)
    for k in ref_stats:  # the reported means are sums of per-chunk terms over the same denominator
        assert stats[k] == pytest.approx(ref_stats[k], rel=1e-4, abs=1e-6), k


def test_chunking_does_not_depend_on_how_the_minibatches_are_cut():
    """Several minibatches and epochs, lr 0 so the weights stay put: the update-wide means are a property of the data."""
    net = tiny_net(seed=6)
    buf = make_token_buffer([9, 25, 4, 14, 6, 21, 2], seed=3)
    _, ref = _grads(net, buf, batch_size=3)
    _, got = _grads(net, buf, batch_size=3, max_token_rows=60)
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-3, abs=1e-6), k


def test_a_whole_update_lands_on_the_same_weights():
    """With plain SGD (linear in the gradient, so rounding noise stays rounding noise) two real optimizer steps agree."""
    net = tiny_net(seed=8)
    buf = make_token_buffer([12, 5, 28, 9, 16, 3], seed=4)
    a, b = copy.deepcopy(net), copy.deepcopy(net)
    for model, kw in ((a, {}), (b, {"max_token_rows": 50})):
        np.random.seed(11)  # same minibatch shuffle
        ppo_update(model, sgd(model, lr=0.05), buf, "cpu", n_epochs=2, batch_size=3, target_kl=None, **kw)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, rtol=1e-3, atol=1e-5)


def test_a_lone_episode_over_the_budget_still_trains():
    net = tiny_net(seed=9)
    buf = make_token_buffer([40], seed=5)
    ref, _ = _grads(net, buf, batch_size=16)
    got, _ = _grads(net, buf, batch_size=16, max_token_rows=1)  # cannot be split: one chunk, over budget
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("bad", [0, -5])
def test_a_non_positive_budget_is_rejected(bad):
    net = tiny_net()
    with pytest.raises(AssertionError, match="max_token_rows"):
        ppo_update(net, sgd(net), make_token_buffer([4]), "cpu", max_token_rows=bad)


# ---- the knob reaches ppo_update from the run config ---------------------------------------------------------------------

def test_max_token_rows_defaults_to_off_and_is_passed_through_for_every_update(monkeypatch):
    assert league_runner.PPO_DEFAULTS["max_token_rows"] is None
    seen = []

    def spy(net, optimizer, buf, device, **kw):
        seen.append(kw["max_token_rows"])
        return (0.0,) * 5 + (1, 0.0, 0.0)

    monkeypatch.setattr(league_runner, "ppo_update", spy)
    buf = make_token_buffer([3])
    for cap in (None, 40000):
        ctx = {"hp": {**league_runner.PPO_DEFAULTS, "max_token_rows": cap}, "live_nets": {"a": 1, "b": 2},
               "optimizers": {"a": 1, "b": 2}, "device": "cpu", "train_deck": True, "train_set": {"a", "b"}}
        seen.clear()
        league_runner._update_round(ctx, "a", {"a": buf, "b": buf}, 4, 0.01)  # own bucket + a salvaged opponent bucket
        assert seen == [cap, cap]
