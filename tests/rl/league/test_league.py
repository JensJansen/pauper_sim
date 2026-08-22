"""Tests for rl.league.league.LeaguePool: opponent sampling, snapshot eviction, and
load caching.

No real game simulation needed here -- this module's own logic (sampling
distribution, snapshot eviction, cache invalidation) is what's under test,
not the token architecture (already covered by rl.model.deck/rl.model.arch's own
self-checks).
"""
import os
import random

import pytest
from types import SimpleNamespace

from rl.model.arch import SetTransformer
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.model.deck import DeckNetwork
from rl.league.league import LeaguePool
from rl.model.mulligan import MulliganNet

DECK_NAMES = ["deck_a", "deck_b"]


def _fresh_pool(tmp_path):
    return LeaguePool(str(tmp_path), DECK_NAMES, max_snapshots_per_deck=3)


def _registered_pool(tmp_path):
    """A fresh pool with 5 snapshots registered on deck_a (more than the cap
    of 3), plus the shared/fake net+mulligan-net used to register them --
    the shared setup most of this module's checks build on."""
    pool = _fresh_pool(tmp_path)
    # Production encoder architecture, not a shrunken one: load_snapshot_agent
    # rebuilds the encoder at SetTransformer's own defaults (see its assert),
    # so a snapshot written at a custom width could never be loaded back.
    shared = SetTransformer(vocab_size=5)
    fake_net = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4)
    fake_mull = MulliganNet(fake_net.encoder, hidden=8)  # snapshots are whole agents now (deck + mulligan)
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
    # Register more snapshots than the cap -- oldest must be evicted (out of
    # the ACTIVE sampling pool), cache invalidated with it.
    assert len(pool.snapshots["deck_a"]) == 3, "eviction window must cap at max_snapshots_per_deck"
    remaining_ids = [i for i, _p in pool.snapshots["deck_a"]]
    assert remaining_ids == [2, 3, 4], f"oldest snapshots (0, 1) must be evicted first, got ids {remaining_ids}"
    for i in (0, 1):
        assert not os.path.exists(os.path.join(str(tmp_path), "deck_a", f"snapshot_{i}.pt")), \
            "an evicted snapshot must no longer be in the active pool directory"
        archived_path = os.path.join(str(tmp_path), "deck_a", "archive", f"snapshot_{i}.pt")
        assert os.path.exists(archived_path), \
            "an evicted snapshot must be ARCHIVED, not deleted -- see register_snapshot's own comment on why"


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
    deck_ctx = (SimpleNamespace(size=5), [("Pass", None, None)] * 4)  # .size sizes the encoder the snapshot loads into
    _sid, path = pool.snapshots["deck_a"][0]
    loaded = pool.load_snapshot_agent(path, deck_ctx)
    loaded_again = pool.load_snapshot_agent(path, deck_ctx)
    assert loaded is loaded_again, "repeat loads of the same snapshot must hit the cache, not reconstruct"
    assert isinstance(loaded, SeatAgent), "a loaded snapshot is a frozen SeatAgent"
    assert isinstance(loaded.mulligan, MulliganNet), "a snapshot saved WITH a mulligan net must round-trip it"
    assert all(not p.requires_grad for p in loaded.main.parameters()), "a snapshot's deck net must be fully frozen"
    assert all(not p.requires_grad for p in loaded.mulligan.parameters()), "a snapshot's mulligan net must be fully frozen"


@pytest.mark.slow
def test_load_snapshot_agent_legacy_deck_only_falls_back_to_always_keep(tmp_path):
    import torch

    pool, shared, fake_net, _fake_mull = _registered_pool(tmp_path)
    deck_ctx = (SimpleNamespace(size=5), [("Pass", None, None)] * 4)  # .size sizes the encoder the snapshot loads into
    # A deck-only snapshot (no mulligan state) must still load, falling back to AlwaysKeep.
    legacy_path = os.path.join(str(tmp_path), "deck_b", "snapshot_0.pt")
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    torch.save({"state_dict": fake_net.state_dict(),
                "trunk_hidden": tuple(l.out_features for l in fake_net.trunk_layers)}, legacy_path)
    legacy = pool.load_snapshot_agent(legacy_path, deck_ctx)
    assert isinstance(legacy, SeatAgent) and isinstance(legacy.mulligan, AlwaysKeep), \
        "a deck-only snapshot must load with an AlwaysKeep pregame decider"


@pytest.mark.slow
def test_register_snapshot_eviction_invalidates_load_cache(tmp_path):
    pool, shared, fake_net, fake_mull = _registered_pool(tmp_path)
    deck_ctx = (SimpleNamespace(size=5), [("Pass", None, None)] * 4)  # .size sizes the encoder the snapshot loads into
    _sid, path = pool.snapshots["deck_a"][0]
    pool.load_snapshot_agent(path, deck_ctx)  # populate the cache
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


@pytest.mark.slow
def test_pfsp_cold_start_is_uniform_like_the_old_behavior(tmp_path):
    # No recorded games anywhere -- every candidate gets the SAME weight
    # (neutral 0.5 win-rate prior), so this must reproduce the old uniform
    # distribution's statistical shape even though the sampling ALGORITHM
    # changed (rng.choices, not rng.choice) -- same guarantee every pre-PFSP
    # test in this file already relies on without being rewritten for it.
    pool = _fresh_pool(tmp_path)
    rng = random.Random(0)
    counts = {"deck_a": 0, "deck_b": 0}
    for _ in range(4000):
        name, _path = pool.sample_opponent("deck_a", rng)
        counts[name] += 1
    frac_a = counts["deck_a"] / 4000
    assert abs(frac_a - 0.5) < 0.05, f"cold start (no data) must sample both decks ~uniformly, got deck_a={frac_a:.3f}"


@pytest.mark.slow
def test_pfsp_weights_toward_the_deck_currently_beating_it(tmp_path):
    pool = _fresh_pool(tmp_path)
    rng = random.Random(0)
    # deck_a training seat has recorded losing ALL 20 games it's played
    # against deck_b, and winning all 20 against itself -- PFSP should now
    # strongly favor deck_b (currently beating it) over deck_a (mirror,
    # currently a lock).
    for _ in range(20):
        pool.record_outcome("deck_a", ("deck", "deck_b"), won=False)
        pool.record_outcome("deck_a", ("deck", "deck_a"), won=True)
    counts = {"deck_a": 0, "deck_b": 0}
    for _ in range(4000):
        name, _path = pool.sample_opponent("deck_a", rng)
        counts[name] += 1
    frac_b = counts["deck_b"] / 4000
    # weight(deck_b) = 0.1 + (1-0)**1 = 1.1; weight(deck_a) = 0.1 + (1-1)**1 = 0.1
    # -> expected P(deck_b) = 1.1 / 1.2 ≈ 0.917
    assert frac_b > 0.8, f"a deck currently beating training_deck_name must be heavily favored, got P(deck_b)={frac_b:.3f}"


@pytest.mark.slow
def test_pfsp_never_starves_an_opponent_entirely(tmp_path):
    # Even a deck the training deck has NEVER lost to keeps a nonzero floor
    # weight (PFSP_FLOOR) -- driving it to exactly 0 would silently reintroduce
    # catastrophic forgetting, the exact failure this pool's history exists to
    # prevent (see this module's own docstring).
    pool = _fresh_pool(tmp_path)
    rng = random.Random(1)
    for _ in range(50):
        pool.record_outcome("deck_a", ("deck", "deck_b"), won=True)  # deck_a beats deck_b every single time
    saw_deck_b = any(pool.sample_opponent("deck_a", rng)[0] == "deck_b" for _ in range(2000))
    assert saw_deck_b, "an opponent the training deck always beats must still be sampled occasionally, never starved to zero"


@pytest.mark.slow
def test_pfsp_weights_snapshot_choice_within_a_deck_too(tmp_path):
    pool, _shared, _fake_net, _fake_mull = _registered_pool(tmp_path)  # deck_a has 3 snapshots: ids 2, 3, 4
    rng = random.Random(0)
    ids = [sid for sid, _path in pool.snapshots["deck_a"]]
    # deck_b training seat loses to snapshot ids[0] every time, beats the other two every time.
    for _ in range(20):
        pool.record_outcome("deck_b", ("snapshot", "deck_a", ids[0]), won=False)
        pool.record_outcome("deck_b", ("snapshot", "deck_a", ids[1]), won=True)
        pool.record_outcome("deck_b", ("snapshot", "deck_a", ids[2]), won=True)
    chosen_ids = []
    for _ in range(3000):
        name, path = pool.sample_opponent("deck_b", rng, checkpoint_rate=1.0)
        if name == "deck_a":
            chosen_ids.append(next(sid for sid, p in pool.snapshots["deck_a"] if p == path))
    frac_hardest = chosen_ids.count(ids[0]) / len(chosen_ids)
    assert frac_hardest > 0.7, (
        f"within one deck's own snapshot window, the specific snapshot currently winning "
        f"most often must be favored, got P(hardest snapshot)={frac_hardest:.3f}"
    )


@pytest.mark.slow
def test_opponent_stats_persist_and_reload_and_eviction_drops_stale_entries(tmp_path):
    pool = _fresh_pool(tmp_path)
    pool.record_outcome("deck_a", ("deck", "deck_b"), won=True)
    pool.record_outcome("deck_a", ("deck", "deck_b"), won=False)
    pool.save_opponent_stats()

    pool2 = LeaguePool(str(tmp_path), DECK_NAMES, max_snapshots_per_deck=3)
    assert pool2.opponent_stats["deck_a"][("deck", "deck_b")] == (1, 2), \
        "opponent_stats must round-trip through save/load exactly (1 win, 2 games)"

    # Fill deck_b's snapshot window to exactly its cap (nothing evicted yet),
    # inject a stat against the one about to age out, then push it out with one
    # more registration -- the stale snapshot-level stat must be dropped, not
    # because it was measured wrong, but because it's no longer sample-able by
    # ANYONE and would otherwise be permanent dead weight.
    shared = SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)
    fake_net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    for _ in range(3):  # cap=3 -- fills the window exactly, no eviction yet
        pool2.register_snapshot("deck_b", fake_net)
    oldest_id = pool2.snapshots["deck_b"][0][0]
    pool2.opponent_stats["deck_a"][("snapshot", "deck_b", oldest_id)] = (5, 5)
    assert ("snapshot", "deck_b", oldest_id) in pool2.opponent_stats["deck_a"]

    pool2.register_snapshot("deck_b", fake_net)  # 4th registration -> evicts oldest_id now
    assert ("snapshot", "deck_b", oldest_id) not in pool2.opponent_stats["deck_a"], \
        "an evicted snapshot's stats must be dropped from every training deck's opponent_stats"


# --- PFSP_POWER direction (2026-08-13) ---


def _stats_for_a_hopeless_matchup(pool, training_deck):
    """training_deck loses ~1% to 'boss', ~50% in the mirror, wins the rest --
    the real 4-deck shape (elves won 1.1% vs mono_red_rally, 50.1% in mirror)."""
    for opp, (wins, games) in {"boss": (10, 1000), training_deck: (500, 1000),
                               "easy_a": (900, 1000), "easy_b": (940, 1000)}.items():
        pool.opponent_stats[training_deck][("deck", opp)] = (wins, games)


def _share_of_hardest(power, seed=0):
    import random as _random
    from rl.league.league import LeaguePool
    import tempfile
    pool = LeaguePool(tempfile.mkdtemp(), ["me", "boss", "easy_a", "easy_b"], pfsp_power=power)
    _stats_for_a_hopeless_matchup(pool, "me")
    rng = _random.Random(seed)
    picks = [pool.sample_opponent("me", rng)[0] for _ in range(4000)]
    return picks.count("boss") / len(picks), picks.count("me") / len(picks)


@pytest.mark.slow
def test_share_of_the_hardest_matchup_decreases_as_pfsp_power_decreases():
    """THE invariant the 2026-08-06 change violated, with no test to catch it.

    weight = FLOOR + (1 - win_rate) ** POWER, and (1 - win_rate) < 1, so a
    LARGER exponent SHARPENS concentration onto the opponent you lose to most.
    The change was made to reduce that concentration and did the opposite,
    sending three of four decks to 58-77% of their training games in matchups
    they win <25% of."""
    shares = {p: _share_of_hardest(p) for p in (0.5, 1.0, 2.0)}
    hardest = {p: s[0] for p, s in shares.items()}
    mirror = {p: s[1] for p, s in shares.items()}
    assert hardest[0.5] < hardest[1.0] < hardest[2.0], (
        f"share of the unwinnable matchup must be monotone INCREASING in POWER, got {hardest}")
    assert mirror[0.5] > mirror[2.0], (
        f"and the mirror -- the only matchup with real advantage variance -- must gain, got {mirror}")


@pytest.mark.slow
def test_default_pfsp_power_is_the_flattening_one():
    from rl.league.league import PFSP_POWER
    assert PFSP_POWER == 0.5, "0.5 flattens; 2.0 was backwards (see this module's own comment)"


@pytest.mark.slow
def test_pfsp_power_is_per_pool_not_a_global():
    """It has to be settable per league so re-tuning is a config edit. A module
    constant read at call time would make the two pools below agree."""
    assert _share_of_hardest(0.5)[0] != _share_of_hardest(2.0)[0]
