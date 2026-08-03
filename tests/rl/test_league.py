"""Tests for rl.league.LeaguePool: opponent sampling, snapshot eviction, and
load caching.

No real game simulation needed here -- this module's own logic (sampling
distribution, snapshot eviction, cache invalidation) is what's under test,
not the token architecture (already covered by rl.deck/rl.arch's own
self-checks).
"""
import os
import random

import pytest

from rl.arch import SetTransformer
from rl.agent import AlwaysKeep, SeatAgent
from rl.deck import DeckNetwork
from rl.league import LeaguePool
from rl.mulligan import MulliganNet

DECK_NAMES = ["deck_a", "deck_b"]


def _fresh_pool(tmp_path):
    return LeaguePool(str(tmp_path), DECK_NAMES, max_snapshots_per_deck=3)


def _registered_pool(tmp_path):
    """A fresh pool with 5 snapshots registered on deck_a (more than the cap
    of 3), plus the shared/fake net+mulligan-net used to register them --
    the shared setup most of this module's checks build on."""
    pool = _fresh_pool(tmp_path)
    shared = SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)
    fake_net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    fake_mull = MulliganNet(shared, hidden=8)  # snapshots are whole agents now (deck + mulligan)
    for _ in range(5):
        pool.register_snapshot("deck_a", fake_net, fake_mull)
    return pool, shared, fake_net, fake_mull


@pytest.mark.slow
def test_league_pool_starts_with_no_snapshots(tmp_path):
    pool = _fresh_pool(tmp_path)
    assert pool.snapshots == {"deck_a": [], "deck_b": []}


@pytest.mark.slow
def test_sample_opponent_with_no_snapshots_is_always_live(tmp_path):
    pool = _fresh_pool(tmp_path)
    rng = random.Random(0)
    # With no snapshots yet, every sample must be (some deck, None) -- only "live" is available.
    for _ in range(20):
        name, path = pool.sample_opponent("deck_a", rng)
        assert name in DECK_NAMES
        assert path is None


@pytest.mark.slow
def test_register_snapshot_evicts_oldest_past_cap(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    # Register more snapshots than the cap -- oldest must be evicted, cache invalidated with it.
    assert len(pool.snapshots["deck_a"]) == 3, "eviction window must cap at max_snapshots_per_deck"
    remaining_ids = [i for i, _p in pool.snapshots["deck_a"]]
    assert remaining_ids == [2, 3, 4], f"oldest snapshots (0, 1) must be evicted first, got ids {remaining_ids}"
    for i in (0, 1):
        assert not os.path.exists(os.path.join(str(tmp_path), "deck_a", f"snapshot_{i}.pt")), \
            "evicted snapshot file must be deleted from disk"


@pytest.mark.slow
def test_sample_opponent_default_checkpoint_rate_stays_live(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    rng = random.Random(0)
    # DEFAULT checkpoint_rate (0.0): must stay live-only EVEN THOUGH deck_a
    # now has 3 real snapshots on disk -- a full snapshot window must never
    # silently imply mostly-checkpoint sampling (see this module's own
    # docstring).
    for _ in range(200):
        name, path = pool.sample_opponent("deck_a", rng)
        if name == "deck_a":
            assert path is None, "checkpoint_rate defaults to 0.0 -- must never draw a snapshot"


@pytest.mark.slow
def test_sample_opponent_checkpoint_rate_one_always_draws_snapshot(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    rng = random.Random(0)
    # checkpoint_rate=1.0: must always return a real snapshot path when
    # the sampled deck actually has any.
    saw_snapshot = False
    for _ in range(200):
        name, path = pool.sample_opponent("deck_a", rng, checkpoint_rate=1.0)
        if name == "deck_a":
            saw_snapshot = True
            assert path is not None and os.path.exists(path), "checkpoint_rate=1.0 must always draw a snapshot when one exists"
        else:
            assert path is None, "deck_b has zero snapshots -- must fall back to live regardless of checkpoint_rate"
    assert saw_snapshot, "expected at least one deck_a draw across 200 samples"


@pytest.mark.slow
def test_sample_opponent_checkpoint_rate_half_tracks_requested_fraction(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    rng = random.Random(0)
    # checkpoint_rate=0.5: the live/checkpoint split must roughly track the
    # requested rate, independent of how many snapshots exist (3 here) -- not
    # drift with snapshot count the way a 1/(1+N) split would.
    # Filtered to deck_a draws only: deck_b has zero snapshots and would
    # always read as "live" regardless of rate, diluting the measured
    # fraction if mixed in (opponent-deck selection is a SEPARATE uniform
    # draw from the live/checkpoint decision -- see this function's own
    # docstring on the two-level structure).
    deck_a_draws = [p for n, p in (pool.sample_opponent("deck_a", rng, checkpoint_rate=0.5) for _ in range(4000)) if n == "deck_a"]
    checkpoint_frac = sum(1 for p in deck_a_draws if p is not None) / len(deck_a_draws)
    assert abs(checkpoint_frac - 0.5) < 0.05, f"checkpoint_rate=0.5 should draw a snapshot ~50% of the time deck_a is picked, got {checkpoint_frac:.3f}"


@pytest.mark.slow
def test_load_snapshot_agent_loads_and_caches(tmp_path):
    pool, shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    # load_snapshot_agent must load a frozen SeatAgent (deck + mulligan) and be cached.
    deck_ctx = (None, [("Pass", None, None)] * 4)
    _sid, path = pool.snapshots["deck_a"][0]
    loaded = pool.load_snapshot_agent(path, shared, deck_ctx)
    loaded_again = pool.load_snapshot_agent(path, shared, deck_ctx)
    assert loaded is loaded_again, "repeat loads of the same snapshot must hit the cache, not reconstruct"
    assert isinstance(loaded, SeatAgent), "a loaded snapshot is a frozen SeatAgent"
    assert isinstance(loaded.mulligan, MulliganNet), "a snapshot saved WITH a mulligan net must round-trip it"
    assert all(not p.requires_grad for p in loaded.main.parameters()), "a snapshot's deck net must be fully frozen"
    assert all(not p.requires_grad for p in loaded.mulligan.parameters()), "a snapshot's mulligan net must be fully frozen"


@pytest.mark.slow
def test_load_snapshot_agent_legacy_deck_only_falls_back_to_always_keep(tmp_path):
    import torch

    pool, shared, fake_net, _fake_mull = _registered_pool(tmp_path)
    deck_ctx = (None, [("Pass", None, None)] * 4)
    # A deck-only snapshot (no mulligan state) must still load, falling back to AlwaysKeep.
    legacy_path = os.path.join(str(tmp_path), "deck_b", "snapshot_0.pt")
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    torch.save({"state_dict": fake_net.state_dict(),
                "trunk_hidden": tuple(l.out_features for l in fake_net.trunk_layers)}, legacy_path)
    legacy = pool.load_snapshot_agent(legacy_path, shared, deck_ctx)
    assert isinstance(legacy, SeatAgent) and isinstance(legacy.mulligan, AlwaysKeep), \
        "a deck-only snapshot must load with an AlwaysKeep pregame decider"


@pytest.mark.slow
def test_register_snapshot_eviction_invalidates_load_cache(tmp_path):
    pool, shared, fake_net, fake_mull = _registered_pool(tmp_path)
    deck_ctx = (None, [("Pass", None, None)] * 4)
    _sid, path = pool.snapshots["deck_a"][0]
    pool.load_snapshot_agent(path, shared, deck_ctx)  # populate the cache
    assert path in pool._net_cache

    # A freshly-registered snapshot that evicts `path` must invalidate its cache entry.
    pool.register_snapshot("deck_a", fake_net, fake_mull)
    assert path not in pool._net_cache, "evicting a snapshot must drop it from the load cache too"


@pytest.mark.slow
def test_fresh_pool_rediscovers_snapshots_from_disk(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)
    # A fresh LeaguePool pointed at the same root_dir must rediscover existing snapshots from disk.
    pool2 = LeaguePool(str(tmp_path), DECK_NAMES, max_snapshots_per_deck=3)
    assert pool2.snapshots["deck_a"] == pool.snapshots["deck_a"], "snapshot discovery from disk must match what's actually there"
