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

import numpy as np
import torch
import pytest

import game
from game import mana, registry, resolution
from game.effects.casting import play_land_from_hand
from rl.action_bridge import build_fixed_action_table
from rl.agent import AlwaysKeep, DecisionResult, SeatAgent
from rl.arch import SetTransformer
from rl.deck import DeckNetwork, SCALAR_FEATURE_DIM
from rl.features import CardVocab
from rl.league import LeaguePool
from rl.ppo import ppo_update
from rl.rewards import action_count_win_reward_200_floor02, with_dense_mana_burn_penalty, with_mana_mistake_penalty
from rl.train import (
    RolloutBuffer,
    _constant_pairing,
    _make_on_mana_burn,
    _wants_mana_mistake,
    collect_rollout,
    collect_rollout_league,
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
    (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
     epochs_run, explained_variance, adv_std) = ppo_update(
        net_a, [fx["opt_a"]], buf, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(policy_loss) and np.isfinite(value_loss) and np.isfinite(entropy)
    assert np.isfinite(approx_kl) and np.isfinite(clip_fraction)
    assert 1 <= epochs_run <= 2
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
    assert "game_over" in kinds_seen, "collect_rollout must log the outcome itself, not leave it to be inferred"
    for one_game_events in game_logs:
        game_over_events = [e for e in one_game_events if e["kind"] == "game_over"]
        assert len(game_over_events) == 1, "exactly one game_over event per game"
        assert game_over_events[0]["winner"] in (0, 1, None), "winner must be a seat index or None (timeout)"
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

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: ("a", None)  # true mirror
        bufs_self, _mull, played, outcomes_self = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                          pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_self.get("a", RolloutBuffer())) > 0, "true mirror must record a non-empty 'a' bucket"
        assert set(bufs_self) == {"a"}, "a true mirror records ONLY the training bucket (both seats pooled into it)"
        # 0 entries iff that single game hit a horizon timeout (no winner) -- excluded
        # entirely rather than recorded as a loss, see collect_rollout_league's own docstring.
        assert len(outcomes_self) <= 1 and all(o == ("a", None, o[2]) for o in outcomes_self), \
            "a mirror's outcome, when present, is keyed by ('a', None, <won>)"
        buf_self = bufs_self["a"]

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: ("b", None)  # another deck's live net
        bufs_cross, _mull, played, outcomes_cross = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                           pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_cross.get("a", RolloutBuffer())) > 0, "cross-deck opponent must record the training bucket"
        # A LIVE-net opponent's transitions are salvaged under its own deck name ('b').
        assert "b" in bufs_cross and len(bufs_cross["b"]) > 0, "a live-net opponent must salvage its own bucket 'b'"
        assert all(np.isfinite(v) for v in bufs_cross["b"].value) and all(np.isfinite(r) for r in bufs_cross["b"].reward)
        assert len(outcomes_cross) <= 1 and all(o == ("b", None, o[2]) for o in outcomes_cross), \
            "a live cross-deck outcome, when present, is keyed by ('b', None, <won>)"

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: ("a", snapshot_path)  # frozen snapshot of self
        bufs_snap, _mull, played, outcomes_snap = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                          pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_snap.get("a", RolloutBuffer())) > 0, "a frozen snapshot opponent still records the training bucket"
        assert set(bufs_snap) == {"a"}, "a frozen snapshot opponent is off-policy -- only the training bucket, nothing salvaged"
        assert snapshot_path in pool._net_cache, "load_snapshot_net must have populated the cache"
        snap_id = pool.snapshots["a"][0][0]
        assert len(outcomes_snap) <= 1 and all(o == ("a", snap_id, o[2]) for o in outcomes_snap), \
            "a frozen-snapshot outcome, when present, must resolve the snapshot PATH back to its id, keyed by ('a', <id>, <won>)"

        for buf in (bufs_self["a"], bufs_cross["a"], bufs_snap["a"]):
            assert all(np.isfinite(v) for v in buf.value)
            assert all(np.isfinite(r) for r in buf.reward)

        (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
         epochs_run, explained_variance, adv_std) = ppo_update(
            net_a, [opt_a], buf_self, DEVICE, n_epochs=1, batch_size=16)
        assert np.isfinite(policy_loss) and np.isfinite(value_loss)
        assert np.isfinite(approx_kl) and np.isfinite(clip_fraction) and epochs_run == 1
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
    pl, vl, ent, akl, cf, ep, ev, astd = ppo_update(net_a, [opt_head], buf_self, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(pl) and np.isfinite(vl) and np.isfinite(ent)
    assert np.isfinite(akl) and np.isfinite(cf) and 1 <= ep <= 2
    # explained_variance: finite and <= 1 by construction. NOT asserted
    # positive -- an untrained critic on a smoke-test buffer legitimately
    # scores below zero (worse than predicting the mean), which is the honest
    # reading and exactly why this is recorded next to the raw value_loss.
    assert np.isfinite(ev) and ev <= 1.0
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


def test_dense_mana_burn_credit_lands_on_the_tap_not_the_pass():
    # rl.train's on_single_pip_burn hook + open_taps bookkeeping: the whole
    # phase's dense mana-burn charge must land on the Tap action that
    # produced the float, not on the Pass that happened to be pending when
    # the phase boundary crossed -- the mis-attribution a 2026-08-10 one-off
    # probe (since retired) found against the real production reward
    # (see with_dense_mana_burn_penalty's own docstring: only 1.8% of real
    # transitions ever carried a nonzero charge, and Tap wasn't preferred
    # over Pass at all). Scripted mono-Mountain deck (nothing ever castable,
    # so the tap's own pip is guaranteed to survive to the clear): play a
    # land, tap it once in MAIN1, then always Pass.
    #
    # A real SeatAgent/DeckNetwork isn't needed -- collect_rollout's own
    # choose_action closure only ever reads ppo_entry's action_idx (against
    # a REAL fixed_table, for real "Tap"/"Pass" label lookups) and deck_ctx;
    # everything else a buffer entry stores (tokens/scalar/mask/logp/value)
    # is inert placeholder data this test never reads back.
    decklist = [("Mountain", 20)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    tap_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Tap Mountain")
    pass_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
    land_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name.startswith("Play land"))

    class _ScriptedAgent:
        deck_ctx = (None, fixed_table)

        def __init__(self):
            self.tapped = False

        def decide(self, state, seat, horizon, device, greedy=False):
            pend = state.pending_resolution
            if pend is not None and pend["kind"] == "mulligan_decision":
                return DecisionResult(lambda: resolution.execute_mulligan_keep(state), None, None, False)
            if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
                executor = lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
                return DecisionResult(executor, (None, None, None, land_idx, 0.0, 0.0), None, False)
            if not self.tapped:
                mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
                if mtn is not None:
                    self.tapped = True
                    executor = lambda: mana.activate_mana_source(state, mtn)
                    return DecisionResult(executor, (None, None, None, tap_idx, 0.0, 0.0), None, False)
            return DecisionResult(None, (None, None, None, pass_idx, 0.0, 0.0), None, True)

    base = lambda state, done, horizon: 0.0
    reward_fn = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=2.0)
    agents = [_ScriptedAgent(), _ScriptedAgent()]
    pairing = _constant_pairing(agents, [decklist, decklist], [reward_fn, reward_fn], ["m", None])

    bufs, _mull, played = collect_rollout(pairing, 1, horizon=3, rng=_random.Random(0), device=DEVICE, record=True)
    assert played == 1
    buf = bufs["m"]
    tap_rewards = [buf.reward[i] for i in range(len(buf)) if buf.action[i] == tap_idx]
    pass_rewards = [buf.reward[i] for i in range(len(buf)) if buf.action[i] == pass_idx]
    assert tap_rewards, "the scripted tap must have actually been recorded"
    assert any(r < -1e-9 for r in tap_rewards), "the Tap action must carry the real negative dense charge"
    assert all(r == 0.0 for r in pass_rewards), "Pass must NOT absorb the charge anymore -- that was the bug"


@pytest.mark.slow
def test_winner_only_mana_burn_charges_the_winner_and_drops_the_loser():
    # rl.train's deferred_charges + _winner_only_burn_for: with a WINNER-ONLY
    # reward (rl.rewards.with_dense_mana_burn_penalty(refund_on_loss=True),
    # i.e. deploy_reward_v5/v6's wrap), a seat's dense burn charges are held for
    # the whole game and applied at the terminal flush ONLY if it won. A
    # losing seat's trajectory must come out bit-for-bit identical to one
    # that never burnt anything at all.
    #
    # That neutrality is the entire point of DEFERRING rather than charging-
    # then-refunding: PPO trains on GAE advantages, so a charge written at
    # step t lands in delta_t immediately while a terminal refund only reaches
    # it discounted by (gamma*gae_lambda)^k -- see with_dense_mana_burn_
    # penalty's own docstring. A refund-based implementation would leave a
    # residue here; this asserts there is none.
    #
    # Same scripted mono-Mountain setup as test_dense_mana_burn_credit_lands_
    # on_the_tap_not_the_pass above (nothing castable, so every tapped pip
    # survives to the phase clear), but with a deck small enough to DECK OUT
    # within the horizon -- that produces a real winner and a real loser in
    # ONE game, exercising both branches against the same rollout.
    decklist = [("Mountain", 8)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    tap_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Tap Mountain")
    pass_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
    land_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name.startswith("Play land"))

    class _ScriptedAgent:
        deck_ctx = (None, fixed_table)

        def __init__(self):
            self.tapped = False

        def decide(self, state, seat, horizon, device, greedy=False):
            pend = state.pending_resolution
            if pend is not None and pend["kind"] == "mulligan_decision":
                return DecisionResult(lambda: resolution.execute_mulligan_keep(state), None, None, False)
            if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
                executor = lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
                return DecisionResult(executor, (None, None, None, land_idx, 0.0, 0.0), None, False)
            if not self.tapped:
                mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
                if mtn is not None:
                    self.tapped = True
                    executor = lambda: mana.activate_mana_source(state, mtn)
                    return DecisionResult(executor, (None, None, None, tap_idx, 0.0, 0.0), None, False)
            return DecisionResult(None, (None, None, None, pass_idx, 0.0, 0.0), None, True)

    base = lambda state, done, horizon: 0.0
    reward_fn = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0,
                                              game_penalty_cap=2.0, refund_on_loss=True)
    agents = [_ScriptedAgent(), _ScriptedAgent()]
    # Both seats recorded, into SEPARATE buckets, so winner and loser can be
    # told apart afterwards.
    pairing = _constant_pairing(agents, [decklist, decklist], [reward_fn, reward_fn], ["seat0", "seat1"])

    winner = {}
    bufs, _mull, played = collect_rollout(pairing, 1, horizon=6, rng=_random.Random(0), device=DEVICE,
                                           record=True, on_game_end=lambda s: winner.setdefault("idx", s.winner))
    assert played == 1
    assert winner["idx"] is not None, "this fixture must deck someone out -- otherwise neither branch is exercised"

    won_buf = bufs[f"seat{winner['idx']}"]
    lost_buf = bufs[f"seat{1 - winner['idx']}"]

    # WINNER: the charge applies, and still lands on the Tap (not the Pass) --
    # deferring must not disturb the attribution the non-deferred path already
    # guarantees (test_dense_mana_burn_credit_lands_on_the_tap_not_the_pass).
    won_taps = [won_buf.reward[i] for i in range(len(won_buf)) if won_buf.action[i] == tap_idx]
    won_passes = [won_buf.reward[i] for i in range(len(won_buf)) if won_buf.action[i] == pass_idx]
    assert won_taps, "the winner's scripted tap must have been recorded"
    assert any(r < -1e-9 for r in won_taps), "the winner's Tap must carry the deferred charge once applied"
    assert all(r == 0.0 for r in won_passes), "Pass must never absorb the charge"

    # LOSER: nothing was ever written. Not "smaller", not "refunded to about
    # zero" -- EXACTLY zero on every single transition, the guarantee a
    # terminal refund could not have provided.
    assert len(lost_buf), "the loser's trajectory must have been recorded at all"
    assert all(r == 0.0 for r in lost_buf.reward), (
        "a losing seat must pay nothing for mana burnt -- every transition exactly 0.0"
    )
    lost_taps = [i for i in range(len(lost_buf)) if lost_buf.action[i] == tap_idx]
    assert lost_taps, "the loser also tapped -- otherwise this proves nothing about dropping its charge"


@pytest.mark.slow
def test_explained_variance_is_zero_not_one_when_returns_are_constant():
    """The whole reason explained_variance was added (2026-08-13): value_loss
    is a raw MSE with no scale attached, so a critic predicting a CONSTANT that
    happens to be right reports a tiny loss and looks excellent. That is the
    live competing explanation for this project's own flat 0.01 value_loss --
    gamma/gae_lambda discounting early-game returns to ~0 makes them trivially
    predictable. A degenerate target must therefore score 0, not 1."""
    shared = SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)
    for p in shared.parameters():
        p.requires_grad = False
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    buf = RolloutBuffer()
    # Every transition identical, reward 0, done immediately -> zero-variance
    # returns. A "perfect" critic here has explained nothing.
    for _ in range(8):
        buf.add([], np.zeros(SCALAR_FEATURE_DIM, dtype=np.float32),
                np.ones(4, dtype=bool), 0, 0.0, 0.0, 0.0, True)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    *_rest, ev, _astd = ppo_update(net, [opt], buf, DEVICE, n_epochs=1, batch_size=4)
    assert ev == 0.0, f"constant returns must explain nothing, got {ev}"


@pytest.mark.slow
def test_adv_norm_floor_defaults_to_a_true_no_op_and_damps_degenerate_batches():
    """The guard on rl.ppo's advantage normalization.

    Unguarded, (adv - mean) / (std + 1e-8) rescales EVERY batch to unit
    variance, including one where nearly every trajectory returned the same
    outcome -- so critic noise is promoted to a full-scale gradient. Three of
    four decks spent 58-77% of training in matchups they win <25% of, and
    flattening PFSP only reaches ~56%, so this is structural rather than
    incidental.

    Two properties, both load-bearing:
      - floor=0.0 (the default) must reproduce the historical behavior EXACTLY,
        so shipping the knob changes nothing until it is deliberately set;
      - a floor above the batch's own spread must SHRINK the advantages rather
        than normalize them to unit scale.
    """
    import numpy as np

    def normalized(adv_raw, floor):
        adv = np.array(adv_raw, dtype=np.float32)
        return (adv - adv.mean()) / (max(float(adv.std()), floor) + 1e-8)

    degenerate = [-1.0, -1.0, -1.0, -0.98, -1.02]  # a lost matchup: nothing differentiates
    healthy = [-1.0, 1.0, -1.0, 1.0, 0.2]

    unguarded = normalized(degenerate, 0.0)
    assert abs(float(unguarded.std()) - 1.0) < 1e-3, "floor=0 must still normalize to unit variance"

    guarded = normalized(degenerate, 0.5)
    assert float(guarded.std()) < 0.05, (
        f"a batch with no real signal must stay small, got std={guarded.std():.3f}")

    # ...and the same floor must not disturb a batch that DOES carry signal.
    assert np.allclose(normalized(healthy, 0.5), normalized(healthy, 0.0), atol=1e-5), (
        "a floor below the batch's own spread must be inert")


@pytest.mark.slow
def test_ppo_update_reports_raw_adv_std():
    """adv_norm_floor cannot be set responsibly without knowing the actual
    advantage scale, which nothing ever recorded. It is returned and logged so
    the value comes from the measured distribution rather than a guess -- the
    mistake that made PFSP_POWER=2.0 the leading cause of a 60,001-game
    regression."""
    fx = _base_fixture()
    agent_a = SeatAgent(fx["net_a"], AlwaysKeep(), fx["deck_ctx_a"])
    pairing = _constant_pairing([agent_a, agent_a], [fx["decklist_a"]] * 2,
                                [fx["reward_fn"]] * 2, ["m", "m"])
    bufs, _mull, _played = collect_rollout(pairing, 2, HORIZON, fx["rng"], device=DEVICE)
    *_rest, _ev, adv_std = ppo_update(fx["net_a"], [fx["opt_a"]], bufs["m"], DEVICE,
                                      n_epochs=1, batch_size=16)
    assert np.isfinite(adv_std) and adv_std >= 0.0
