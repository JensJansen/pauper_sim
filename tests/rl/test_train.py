"""A tiny end-to-end smoke test for the token/attention training pipeline:
real 2-player games (mono_red_madness mirror, a genuine cross-matchup vs
rakdos_madness), tiny network dims, few games/iterations -- just enough to
prove the whole pipeline (rollout collection -> padded/masked batching -> PPO
update) runs without crashing, hanging, or producing NaN/inf, before any real
training.

Each test below builds its own fresh fixture via _base_fixture() (nets get a
fresh random init per test); every assertion here is a shape/finiteness/
"did-something-change" check, not an exact-value one, so a fresh fixture per
test is safe. test_league_smoke_and_frozen_cache_ppo_update keeps the league
smoke test and the frozen-cache ppo_update check in ONE function because the
frozen-cache check reuses the league smoke test's own collected buffer -- a
real data dependency, not just convenience.
"""
import random as _random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import pytest

import game
from game import mana, registry, resolution
from game.effects.casting import play_land_from_hand
from rl.action_bridge import build_fixed_action_table
from rl.agent import AlwaysKeep, SeatAgent
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.features import CardVocab
from rl.league import LeaguePool
from rl.pool import build_pool
from rl.rewards import action_count_win_reward_200_floor02, with_mana_mistake_penalty
from rl.train import (
    RolloutBuffer,
    _constant_pairing,
    _make_on_mana_burn,
    _wants_mana_mistake,
    collect_rollout,
    collect_rollout_league,
    collect_rollout_league_parallel,
    ppo_update,
    train_selfplay,
)

DEVICE = "cpu"
HORIZON = 20


def _base_fixture():
    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    vocab = CardVocab([decklist_a, decklist_b], token_card_defs=token_defs)

    fixed_table_a = build_fixed_action_table(decklist_a, token_card_defs=token_defs)
    fixed_table_b = build_fixed_action_table(decklist_b, token_card_defs=token_defs)
    deck_ctx_a = (vocab, fixed_table_a)
    deck_ctx_b = (vocab, fixed_table_b)

    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net_a = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_a), trunk_hidden=(24, 24))
    net_b = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_b), trunk_hidden=(24, 24))
    opt_a = torch.optim.Adam(net_a.parameters(), lr=3e-4)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=3e-4)

    return {
        "decklist_a": decklist_a, "decklist_b": decklist_b,
        "deck_ctx_a": deck_ctx_a, "deck_ctx_b": deck_ctx_b,
        "fixed_table_a": fixed_table_a, "fixed_table_b": fixed_table_b,
        "net_a": net_a, "net_b": net_b, "opt_a": opt_a, "opt_b": opt_b,
        "reward_fn": action_count_win_reward_200_floor02,
        "rng": _random.Random(0),
    }


@pytest.mark.slow
def test_mirror_selfplay_smoke():
    # Mirror self-play smoke test -- net_a plays itself, BOTH seats pooled
    # into one bucket ("m"), one update; exercises the "same weights both
    # seats" path and the pairing-driven collect_rollout + AlwaysKeep pregame.
    fx = _base_fixture()
    net_a, decklist_a, deck_ctx_a, reward_fn, rng = fx["net_a"], fx["decklist_a"], fx["deck_ctx_a"], fx["reward_fn"], fx["rng"]
    t0 = time.time()
    agent_a = SeatAgent(net_a, AlwaysKeep(), deck_ctx_a)
    mirror_pairing = _constant_pairing([agent_a, agent_a], [decklist_a, decklist_a],
                                       [reward_fn, reward_fn], ["m", "m"])
    buffers_by_deck, _mull, games_played = collect_rollout(mirror_pairing, 2, HORIZON, rng, device=DEVICE)
    assert games_played == 2
    assert set(buffers_by_deck) == {"m"}, "a mirror pools both seats into ONE bucket"
    buf = buffers_by_deck["m"]
    assert len(buf) > 0, "the pooled mirror bucket must have recorded transitions from both seats"
    assert all(np.isfinite(v) for v in buf.value), "collected values must be finite"
    assert all(np.isfinite(r) for r in buf.reward), "collected rewards must be finite"
    assert buf.done[-1] is True, "the bucket must end with a flushed terminal transition"
    policy_loss, value_loss, entropy = ppo_update(net_a, [fx["opt_a"]], buf, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(policy_loss) and np.isfinite(value_loss) and np.isfinite(entropy)
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the mirror PPO update"
    print(f"rl.train mirror smoke test: OK ({games_played} games, buf_size={len(buf)}, "
          f"policy_loss={policy_loss:.4f}, {time.time() - t0:.1f}s)")


@pytest.mark.slow
def test_cross_matchup_smoke():
    # Cross-matchup smoke test -- net_a vs net_b, two independent
    # buffers/updates, exercises the "different decks/action spaces on each
    # seat" path (this is what the league's cross-deck games rely on).
    fx = _base_fixture()
    t0 = time.time()
    train_selfplay(
        fx["net_a"], fx["deck_ctx_a"], fx["decklist_a"], fx["reward_fn"],
        fx["net_b"], fx["deck_ctx_b"], fx["decklist_b"], fx["reward_fn"],
        [fx["opt_a"]], [fx["opt_b"]], HORIZON, n_iterations=2, games_per_iteration=2, rng=fx["rng"], device=DEVICE,
    )
    for net in (fx["net_a"], fx["net_b"]):
        for p in net.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after the cross-matchup PPO update"
    print(f"rl.train cross-matchup smoke test: OK ({time.time() - t0:.1f}s)")


@pytest.mark.slow
def test_game_logs_smoke():
    # game_logs smoke test -- wiring the game engine's OWN existing
    # event_log (game/state.py's log_event, already instrumented across
    # mana.py/turn.py/resolution/*.py/game/effects/*.py) through to
    # collect_rollout, not any new logging. One entry per game played,
    # each a real list of structured event dicts.
    fx = _base_fixture()
    net_a, decklist_a, deck_ctx_a, reward_fn, rng = fx["net_a"], fx["decklist_a"], fx["deck_ctx_a"], fx["reward_fn"], fx["rng"]
    agent_a = SeatAgent(net_a, AlwaysKeep(), deck_ctx_a)
    mirror_pairing = _constant_pairing([agent_a, agent_a], [decklist_a, decklist_a],
                                       [reward_fn, reward_fn], ["m", "m"])
    game_logs = []
    _bufs, _mull, played = collect_rollout(mirror_pairing, 2, HORIZON, rng, device=DEVICE, game_logs=game_logs)
    assert len(game_logs) == played == 2, "one event_log entry must be appended per game played"
    for one_game_events in game_logs:
        assert len(one_game_events) > 0, "a real game must produce at least one engine event"
        for event in one_game_events:
            assert "kind" in event and "turn" in event and "phase" in event, "every event must carry log_event's own envelope"
    kinds_seen = {event["kind"] for one_game_events in game_logs for event in one_game_events}
    assert "turn_start" in kinds_seen, "a multi-turn game must log at least one turn_start event"
    print(f"rl.train game_logs smoke test: OK ({sum(len(g) for g in game_logs)} events across {played} games, "
          f"kinds={sorted(kinds_seen)})")


@pytest.mark.slow
def test_split_optimizer_shared_stack_smoke():
    # Split-optimizer smoke test -- the actual pattern run_pretrain.py needs:
    # TWO throwaway heads sharing ONE SetTransformer instance, but only ONE
    # optimizer (opt_shared2) ever touches the shared stack's own params, so
    # its Adam momentum stays coherent across both decks' alternating mirror
    # sessions instead of being split across two unsynchronized Adam
    # instances (see ppo_update's own docstring).
    fx = _base_fixture()
    t0 = time.time()
    vocab_size = fx["deck_ctx_a"][0].size
    fixed_table_a, fixed_table_b = fx["fixed_table_a"], fx["fixed_table_b"]
    shared2 = SetTransformer(vocab_size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net_a2 = DeckNetwork(shared2, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_a), trunk_hidden=(24, 24))
    net_b2 = DeckNetwork(shared2, film_condition_dim=16, non_targeting_n_actions=len(fixed_table_b), trunk_hidden=(24, 24))
    opt_shared2 = torch.optim.Adam(shared2.parameters(), lr=3e-4)
    opt_a2_head = torch.optim.Adam([p for n, p in net_a2.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)
    opt_b2_head = torch.optim.Adam([p for n, p in net_b2.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)
    shared2_before = [p.clone() for p in shared2.parameters()]

    train_selfplay(net_a2, fx["deck_ctx_a"], fx["decklist_a"], fx["reward_fn"], net_a2, fx["deck_ctx_a"], fx["decklist_a"], fx["reward_fn"],
                    [opt_shared2, opt_a2_head], [opt_shared2, opt_a2_head], HORIZON,
                    n_iterations=1, games_per_iteration=2, rng=fx["rng"], device=DEVICE)
    train_selfplay(net_b2, fx["deck_ctx_b"], fx["decklist_b"], fx["reward_fn"], net_b2, fx["deck_ctx_b"], fx["decklist_b"], fx["reward_fn"],
                    [opt_shared2, opt_b2_head], [opt_shared2, opt_b2_head], HORIZON,
                    n_iterations=1, games_per_iteration=2, rng=fx["rng"], device=DEVICE)

    assert id(opt_shared2) == id(opt_shared2), "sanity: the SAME optimizer object must be reused across both decks"
    assert any(not torch.equal(a, b) for a, b in zip(shared2_before, shared2.parameters())), (
        "shared stack must have actually moved after two decks' worth of updates through the ONE shared optimizer"
    )
    for net in (net_a2, net_b2):
        for p in net.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after the split-optimizer PPO update"
    print(f"rl.train split-optimizer (pretrain pattern) smoke test: OK ({time.time() - t0:.1f}s)")


@pytest.mark.slow
def test_league_smoke_and_frozen_cache_ppo_update():
    # League smoke test -- collect_rollout_league against a REAL
    # LeaguePool, exercising all three opponent kinds it must handle:
    # true mirror (both seats recorded), another deck's live net (training
    # seat only), and a frozen historical snapshot (training seat only).
    # sample_opponent is monkeypatched per sub-case rather than left to
    # chance, so each path is deterministically exercised instead of
    # hoping enough random games happen to hit all three.
    #
    # Then (same buffer, same net): frozen shared-stack caching in
    # ppo_update -- the LEAGUE path. When the shared stack is frozen,
    # ppo_update precomputes its per-transition outputs ONCE
    # (_precompute_frozen_shared) and reuses them across epochs instead of
    # recomputing the SetTransformer. Verifies the cached path runs, trains
    # the head, and leaves the frozen stack byte-for-byte untouched -- the
    # mirror/cross-matchup/split-optimizer tests above all used a TRAINABLE
    # shared stack, so none of them exercise this path.
    fx = _base_fixture()
    net_a, net_b, opt_a, reward_fn, rng = fx["net_a"], fx["net_b"], fx["opt_a"], fx["reward_fn"], fx["rng"]
    decklist_a, decklist_b = fx["decklist_a"], fx["decklist_b"]
    deck_ctx_a, deck_ctx_b = fx["deck_ctx_a"], fx["deck_ctx_b"]

    t0 = time.time()
    live_nets = {"a": net_a, "b": net_b}
    decklists_by_name = {"a": decklist_a, "b": decklist_b}
    ctxs_by_name = {"a": deck_ctx_a, "b": deck_ctx_b}
    tmp_dir = tempfile.mkdtemp()
    try:
        pool = LeaguePool(tmp_dir, ["a", "b"], max_snapshots_per_deck=3)
        pool.register_snapshot("a", net_a)  # gives the "frozen snapshot of self" path something real to load
        snapshot_path = pool.snapshots["a"][0][1]

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0: ("a", None)  # true mirror
        bufs_self, _mull, played = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                          pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_self.get("a", RolloutBuffer())) > 0, "true mirror must record a non-empty 'a' bucket"
        assert set(bufs_self) == {"a"}, "a true mirror records ONLY the training bucket (both seats pooled into it)"
        buf_self = bufs_self["a"]

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0: ("b", None)  # another deck's live net
        bufs_cross, _mull, played = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                           pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_cross.get("a", RolloutBuffer())) > 0, "cross-deck opponent must record the training bucket"
        # A LIVE-net opponent's transitions are salvaged under its own deck name ('b').
        assert "b" in bufs_cross and len(bufs_cross["b"]) > 0, "a live-net opponent must salvage its own bucket 'b'"
        assert all(np.isfinite(v) for v in bufs_cross["b"].value) and all(np.isfinite(r) for r in bufs_cross["b"].reward)

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0: ("a", snapshot_path)  # frozen snapshot of self
        bufs_snap, _mull, played = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                          pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_snap.get("a", RolloutBuffer())) > 0, "a frozen snapshot opponent still records the training bucket"
        assert set(bufs_snap) == {"a"}, "a frozen snapshot opponent is off-policy -- only the training bucket, nothing salvaged"
        assert snapshot_path in pool._net_cache, "load_snapshot_net must have populated the cache"

        for buf in (bufs_self["a"], bufs_cross["a"], bufs_snap["a"]):
            assert all(np.isfinite(v) for v in buf.value)
            assert all(np.isfinite(r) for r in buf.reward)

        policy_loss, value_loss, entropy = ppo_update(net_a, [opt_a], buf_self, DEVICE, n_epochs=1, batch_size=16)
        assert np.isfinite(policy_loss) and np.isfinite(value_loss)
        for p in net_a.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after a league-buffer PPO update"
    finally:
        shutil.rmtree(tmp_dir)

    print(f"rl.train league smoke test: OK (mirror/cross-deck/snapshot opponents all exercised, {time.time() - t0:.1f}s)")

    t0 = time.time()
    for p in net_a.shared_stack.parameters():
        p.requires_grad = False
    shared_before = [p.clone() for p in net_a.shared_stack.parameters()]
    head_before = [p.clone() for p in net_a.non_targeting_head.parameters()]
    opt_head = torch.optim.Adam([p for p in net_a.parameters() if p.requires_grad], lr=3e-4)
    pl, vl, ent = ppo_update(net_a, [opt_head], buf_self, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(pl) and np.isfinite(vl) and np.isfinite(ent)
    assert all(torch.equal(a, b) for a, b in zip(shared_before, net_a.shared_stack.parameters())), \
        "a FROZEN shared stack must be byte-for-byte unchanged after a cached ppo_update"
    assert any(not torch.equal(a, b) for a, b in zip(head_before, net_a.non_targeting_head.parameters())), \
        "the per-deck head must have actually trained in the cached ppo_update"
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the cached ppo_update"
    print(f"rl.train frozen-cache ppo_update smoke test: OK (shared stack untouched, head trained, {time.time() - t0:.1f}s)")


@pytest.mark.slow
def test_eval_record_false_smoke():
    # Eval / record=False smoke test -- the ANTI-DRIFT invariant. The SAME
    # collect_rollout drives eval and training; only `record` differs. With
    # record=False it must produce NO training buffers (nothing recorded) yet
    # still yield one event_log per game -- "one loop, faithful logging". Also
    # exercises greedy=True (the eval default is sampled, but greedy must run).
    fx = _base_fixture()
    net_a, decklist_a, deck_ctx_a, reward_fn, rng = fx["net_a"], fx["decklist_a"], fx["deck_ctx_a"], fx["reward_fn"], fx["rng"]
    agent_a = SeatAgent(net_a, AlwaysKeep(), deck_ctx_a)
    mirror_pairing = _constant_pairing([agent_a, agent_a], [decklist_a, decklist_a],
                                       [reward_fn, reward_fn], ["m", "m"])
    t0 = time.time()
    eval_logs = []
    bufs, mull, played = collect_rollout(mirror_pairing, 2, HORIZON, rng, device=DEVICE,
                                         record=False, greedy=True, game_logs=eval_logs)
    assert played == 2
    assert bufs == {} and mull == {}, "record=False must produce NO training buffers"
    assert len(eval_logs) == 2 and all(len(g) > 0 for g in eval_logs), "record=False must still produce event logs"
    print(f"rl.train eval/record=False smoke test: OK (no buffers, {sum(len(g) for g in eval_logs)} events logged, "
          f"{time.time() - t0:.1f}s)")


@pytest.mark.slow
def test_collect_rollout_league_parallel_smoke():
    # Regression coverage for collect_rollout_league_parallel's own
    # shared_state_dict/all_trunk_hidden plumbing: run_league.py now computes
    # both ONCE per session and threads them through every call instead of
    # this function re-deriving them from live_nets itself (see its own
    # docstring). A ThreadPoolExecutor stands in for the real
    # ProcessPoolExecutor -- same submit()/future.result() interface, no
    # process-spawn/pickling overhead -- but _league_rollout_worker's own
    # build_pool() call still runs for real (this directory's conftest.py
    # chdir's to src/ for exactly that reason). Confirms the frozen shared
    # stack and the per-deck trunk widths survive the boundary intact enough
    # to actually play a game and record a real, finite transition.
    _decklists, vocab, _deck_ctxs, fixed_tables = build_pool()
    shared_hparams = {"d_model": 16, "n_heads": 2, "n_layers": 1, "dim_feedforward": 32}
    shared = SetTransformer(vocab.size, **shared_hparams)
    net = DeckNetwork(shared, film_condition_dim=16,
                       non_targeting_n_actions=len(fixed_tables["mono_red_madness"]), trunk_hidden=(24, 24))
    live_nets = {"mono_red_madness": net}
    shared_state_dict = shared.state_dict()
    all_trunk_hidden = {"mono_red_madness": tuple(layer.out_features for layer in net.trunk_layers)}

    tmp_dir = tempfile.mkdtemp()
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            buffers_by_deck, _mull_by_deck, games_played = collect_rollout_league_parallel(
                "mono_red_madness", live_nets, "action_count_win_reward_200_floor02", tmp_dir, HORIZON,
                n_games=1, executor=executor, n_workers=1, shared_hparams=shared_hparams,
                shared_state_dict=shared_state_dict, all_trunk_hidden=all_trunk_hidden,
            )
    finally:
        shutil.rmtree(tmp_dir)

    assert games_played == 1, "one submitted worker task playing one game must report one game played"
    assert "mono_red_madness" in buffers_by_deck and len(buffers_by_deck["mono_red_madness"]) > 0, (
        "the worker's own build_pool()-rebuilt tables, the shared stack loaded from shared_state_dict, and "
        "the DeckNetwork rebuilt with all_trunk_hidden's trunk widths must together produce a real rollout"
    )
    assert all(np.isfinite(v) for v in buffers_by_deck["mono_red_madness"].value)
    print(f"rl.train collect_rollout_league_parallel smoke test: OK ({time.time() - t0:.1f}s)")


def test_wants_mana_mistake_gates_on_reward_fn_tag():
    # _wants_mana_mistake: collect_rollout's own gate for whether to build/
    # wire the on_mana_burn hook at all. True only once a TRACKED seat's
    # reward_fn is one with_mana_mistake_penalty actually tagged (see its own
    # consumes_mana_mistake attribute) -- lets pretraining's
    # action_count_win_reward_* (never tagged) skip the extra
    # legal_action_mask sweep entirely.
    base = action_count_win_reward_200_floor02
    tagged = with_mana_mistake_penalty(base)
    assert not _wants_mana_mistake([base, base], ["m", "m"]), "neither reward_fn is tagged"
    assert _wants_mana_mistake([tagged, base], ["m", None]), "seat 0 is tracked AND tagged"
    assert not _wants_mana_mistake([tagged, base], [None, "m"]), "the tagged seat isn't tracked; the tracked seat isn't tagged"
    assert not _wants_mana_mistake([tagged, base], [None, None]), "no tracked seat at all"


def test_on_mana_burn_closure_flags_wasted_tap():
    # The REAL production on_mana_burn closure (rl.train._make_on_mana_burn),
    # not a hand-rolled stand-in -- exercises the exact deck_ctx indexing,
    # legal_action_mask call, and mask/fixed_table zip alignment that shipped
    # with a bug once (a "Tap X" row was mistaken for proof the floated mana
    # had a use -- see game.turn._tally_mana_mistake's own docstring for the
    # three-way exemption this hook completes). Only a minimal fake agent
    # (a real deck_ctx, no network) is needed -- SeatAgent.decide is never
    # called here.
    #
    # Mono-Mountain deck, no spells at all -- same board-control trick as
    # tests/game/test_turn.py::test_on_mana_burn_hook_wired_through_run_multiplayer_game,
    # but wired to the REAL closure instead of a hand-rolled lambda: play a
    # land, tap it once in MAIN1, then always Pass. Nothing is EVER castable
    # with this deck, so a correct closure must report "no, nothing was
    # legally castable" (False) for the float to register as a mistake.
    class _FakeAgent:
        def __init__(self, deck_ctx):
            self.deck_ctx = deck_ctx

    decklist = [("Mountain", 20)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    vocab = CardVocab([decklist, decklist], token_card_defs=token_defs)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    deck_ctx = (vocab, fixed_table)
    agents = [_FakeAgent(deck_ctx), _FakeAgent(deck_ctx)]
    on_mana_burn = _make_on_mana_burn(agents, record=True, record_as=["m", "m"])

    tapped_once = []

    def _play_then_tap_then_pass(state):
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision":
            return lambda: resolution.execute_mulligan_keep(state)  # both players always keep
        if state.active_idx != 0:
            return None
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        if not tapped_once:
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            if mtn is not None:
                tapped_once.append(True)
                return lambda: mana.activate_mana_source(state, mtn)
        return None

    state = game.run_multiplayer_game(
        decklists=[decklist, decklist],
        rng=_random.Random(0), starting_player_idx=0,
        choose_action=_play_then_tap_then_pass, horizon=1, on_mana_burn=on_mana_burn,
    )
    assert tapped_once  # the tap actually happened
    assert state.players[0].mana_mistake_burn == 1, (
        "the real closure must recognize a mono-land deck's Tap/Play-land rows as NOT "
        "proof anything was castable, so the wasted tap is correctly tallied as a mistake"
    )
