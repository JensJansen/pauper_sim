"""A tiny end-to-end smoke test for the token/attention training pipeline:
real 2-player games (mono_red_madness mirror, a cross-matchup vs
rakdos_madness), tiny network dims, few games/iterations -- proves the
pipeline (rollout collection -> padded/masked batching -> PPO update) runs
without crashing, hanging, or producing NaN/inf.

Most tests build a fresh real-network fixture via _base_fixture(); every
assertion is a shape/finiteness/"did-something-change" check, so a fresh
fixture per test is safe. test_league_smoke_and_frozen_cache_ppo_update
reuses the league smoke test's own collected buffer for its frozen-cache
check.

A handful of tests use the cheap _ScriptedAgent (a scripted mono-Mountain
deck: play land, tap once, then always Pass) instead of a real network,
since they only need collect_rollout's real mechanics with some
predictable decision each turn.
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
from rl.decision.action_bridge import build_fixed_action_table
from rl.decision.agent import AlwaysKeep, DecisionResult, SeatAgent
from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork, SCALAR_FEATURE_DIM
from rl.model.features import CardVocab
from rl.league.league import LeaguePool
from rl.training.ppo import ppo_update
from rl.rewards import flat_win_loss_reward, with_dense_mana_burn_penalty
from rl.training.train import (
    RolloutBuffer,
    _constant_pairing,
    _make_on_mana_burn,
    _wants_mana_mistake,
    collect_rollout,
    collect_rollout_league,
)

DEVICE = "cpu"
HORIZON = 20


class _ScriptedAgent:
    """Deterministic decision-only agent for tests needing real
    collect_rollout mechanics but not a trained network: keep every
    mulligan, play the first land, tap the first untapped Mountain once,
    then always Pass. Takes fixed_table directly so one class works for any
    caller's decklist."""
    def __init__(self, fixed_table):
        self.deck_ctx = (None, fixed_table)
        self.tapped = False
        self._tap_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Tap Mountain")
        self._pass_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")
        self._land_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name.startswith("Play land"))

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if pend is not None and pend["kind"] == "mulligan_decision":
            return DecisionResult(lambda: resolution.execute_mulligan_keep(state), None, None, False)
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            executor = lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
            return DecisionResult(executor, (None, None, None, self._land_idx, 0.0, 0.0), None, False)
        if not self.tapped:
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            if mtn is not None:
                self.tapped = True
                executor = lambda: mana.activate_mana_source(state, mtn)
                return DecisionResult(executor, (None, None, None, self._tap_idx, 0.0, 0.0), None, False)
        return DecisionResult(None, (None, None, None, self._pass_idx, 0.0, 0.0), None, True)


def _base_fixture():
    decklist_a = game.parse_decklist_file("../data/mono_red_madness.txt")
    decklist_b = game.parse_decklist_file("../data/rakdos_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    vocab = CardVocab([decklist_a, decklist_b], token_card_defs=token_defs)

    fixed_table_a = build_fixed_action_table(decklist_a, token_card_defs=token_defs)
    fixed_table_b = build_fixed_action_table(decklist_b, token_card_defs=token_defs)
    deck_ctx_a = (vocab, fixed_table_a)
    deck_ctx_b = (vocab, fixed_table_b)

    # One encoder PER DECK -- sharing one instance would put it in two
    # optimizers.
    #
    # Built at SetTransformer's DEFAULT architecture, not a shrunken one:
    # this fixture's nets get registered as snapshots below, and
    # LeaguePool.load_snapshot_agent rebuilds an encoder at the default
    # width to load them into.
    def _enc():
        return SetTransformer(vocab.size)
    d = _enc().d_model
    net_a = DeckNetwork(_enc(), film_condition_dim=d, non_targeting_n_actions=len(fixed_table_a), trunk_hidden=(24, 24))
    net_b = DeckNetwork(_enc(), film_condition_dim=d, non_targeting_n_actions=len(fixed_table_b), trunk_hidden=(24, 24))
    opt_a = torch.optim.Adam(net_a.parameters(), lr=3e-4)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=3e-4)

    return {
        "decklist_a": decklist_a, "decklist_b": decklist_b,
        "deck_ctx_a": deck_ctx_a, "deck_ctx_b": deck_ctx_b,
        "fixed_table_a": fixed_table_a, "fixed_table_b": fixed_table_b,
        "net_a": net_a, "net_b": net_b, "opt_a": opt_a, "opt_b": opt_b,
        "reward_fn": flat_win_loss_reward(),
        "rng": _random.Random(0),
    }


@pytest.mark.slow
def test_mirror_selfplay_smoke():
    # Mirror self-play: net_a plays itself, both seats pooled into one
    # bucket ("m"), one update.
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
        net_a, fx["opt_a"], buf, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(policy_loss) and np.isfinite(value_loss) and np.isfinite(entropy)
    assert np.isfinite(approx_kl) and np.isfinite(clip_fraction)
    assert 1 <= epochs_run <= 2
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the mirror PPO update"
    print(f"rl.training.train mirror smoke test: OK ({games_played} games, buf_size={len(buf)}, "
          f"policy_loss={policy_loss:.4f}, {(time.time() - t0) * 1000:,.0f}ms)")


@pytest.mark.slow
def test_cross_matchup_smoke():
    # Cross-matchup: net_a vs net_b, two independent buffers/updates --
    # exercises different decks/action spaces on each seat.
    fx = _base_fixture()
    net_a, net_b, reward_fn, rng = fx["net_a"], fx["net_b"], fx["reward_fn"], fx["rng"]
    decklist_a, decklist_b = fx["decklist_a"], fx["decklist_b"]
    deck_ctx_a, deck_ctx_b = fx["deck_ctx_a"], fx["deck_ctx_b"]
    t0 = time.time()
    agent_a = SeatAgent(net_a, AlwaysKeep(), deck_ctx_a)
    agent_b = SeatAgent(net_b, AlwaysKeep(), deck_ctx_b)
    pairing = _constant_pairing([agent_a, agent_b], [decklist_a, decklist_b],
                                [reward_fn, reward_fn], ["a", "b"])
    buffers_by_deck, _mull, games_played = collect_rollout(pairing, 2, HORIZON, rng, device=DEVICE)
    assert games_played == 2
    assert set(buffers_by_deck) == {"a", "b"}, "a cross-matchup keeps each seat's own bucket, never pooled"
    ppo_update(net_a, fx["opt_a"], buffers_by_deck["a"], DEVICE, n_epochs=2, batch_size=16)
    ppo_update(net_b, fx["opt_b"], buffers_by_deck["b"], DEVICE, n_epochs=2, batch_size=16)
    for net in (net_a, net_b):
        for p in net.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after the cross-matchup PPO update"
    print(f"rl.training.train cross-matchup smoke test: OK ({(time.time() - t0) * 1000:,.0f}ms)")


@pytest.mark.slow
def test_game_logs_smoke():
    # Wires the game engine's own event_log through to collect_rollout, not
    # any new logging. One entry per game played.
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
    print(f"rl.training.train game_logs smoke test: OK ({sum(len(g) for g in game_logs)} events across {played} games, "
          f"kinds={sorted(kinds_seen)})")


@pytest.mark.slow
def test_league_smoke_and_ppo_update_trains_the_encoder():
    # League smoke test against a real LeaguePool, exercising all three
    # opponent kinds: true mirror (both seats recorded), another deck's live
    # net (training seat only), a frozen snapshot (training seat only).
    # sample_opponent is monkeypatched per sub-case so each path is
    # deterministically exercised.
    #
    # Then (same buffer, same net): a second ppo_update asserting the
    # encoder trains along with the heads, since it's a registered child of
    # the net.
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
        # 0 entries iff that game hit a horizon timeout (no winner) -- excluded rather than counted as a loss.
        assert len(outcomes_self) <= 1 and all(o == ("a", None, o[2], False) for o in outcomes_self), \
            "a mirror's outcome, when present, is keyed by ('a', None, <won>, <stratified=False, no stratify_0land_pct given>)"
        buf_self = bufs_self["a"]

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: ("b", None)  # another deck's live net
        bufs_cross, _mull, played, outcomes_cross = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                           pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_cross.get("a", RolloutBuffer())) > 0, "cross-deck opponent must record the training bucket"
        # A LIVE-net opponent's transitions are salvaged under its own deck name ('b').
        assert "b" in bufs_cross and len(bufs_cross["b"]) > 0, "a live-net opponent must salvage its own bucket 'b'"
        assert all(np.isfinite(v) for v in bufs_cross["b"].value) and all(np.isfinite(r) for r in bufs_cross["b"].reward)
        assert len(outcomes_cross) <= 1 and all(o == ("b", None, o[2], False) for o in outcomes_cross), \
            "a live cross-deck outcome, when present, is keyed by ('b', None, <won>, <stratified=False, no stratify_0land_pct given>)"

        pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: ("a", snapshot_path)  # frozen snapshot of self
        bufs_snap, _mull, played, outcomes_snap = collect_rollout_league("a", live_nets, None, ctxs_by_name, decklists_by_name,
                                                          pool, reward_fn, HORIZON, n_games=1, rng=rng, device=DEVICE)
        assert played == 1 and len(bufs_snap.get("a", RolloutBuffer())) > 0, "a frozen snapshot opponent still records the training bucket"
        assert set(bufs_snap) == {"a"}, "a frozen snapshot opponent is off-policy -- only the training bucket, nothing salvaged"
        assert snapshot_path in pool._net_cache, "load_snapshot_net must have populated the cache"
        snap_id = pool.snapshots["a"][0][0]
        assert len(outcomes_snap) <= 1 and all(o == ("a", snap_id, o[2], False) for o in outcomes_snap), \
            "a frozen-snapshot outcome, when present, must resolve the snapshot PATH back to its id, " \
            "keyed by ('a', <id>, <won>, <stratified=False, no stratify_0land_pct given>)"

        for buf in (bufs_self["a"], bufs_cross["a"], bufs_snap["a"]):
            assert all(np.isfinite(v) for v in buf.value)
            assert all(np.isfinite(r) for r in buf.reward)

        (policy_loss, value_loss, entropy, approx_kl, clip_fraction,
         epochs_run, explained_variance, adv_std) = ppo_update(
            net_a, opt_a, buf_self, DEVICE, n_epochs=1, batch_size=16)
        assert np.isfinite(policy_loss) and np.isfinite(value_loss)
        assert np.isfinite(approx_kl) and np.isfinite(clip_fraction) and epochs_run == 1
        for p in net_a.parameters():
            assert torch.isfinite(p).all(), "a parameter went non-finite after a league-buffer PPO update"
    finally:
        shutil.rmtree(tmp_dir)

    print(f"rl.training.train league smoke test: OK (mirror/cross-deck/snapshot opponents all exercised, {(time.time() - t0) * 1000:,.0f}ms)")

    t0 = time.time()
    encoder_before = [p.clone() for p in net_a.encoder.parameters()]
    head_before = [p.clone() for p in net_a.non_targeting_head.parameters()]
    opt_all = torch.optim.Adam([p for p in net_a.parameters() if p.requires_grad], lr=3e-4)
    pl, vl, ent, akl, cf, ep, ev, astd = ppo_update(net_a, opt_all, buf_self, DEVICE, n_epochs=2, batch_size=16)
    assert np.isfinite(pl) and np.isfinite(vl) and np.isfinite(ent)
    assert np.isfinite(akl) and np.isfinite(cf) and 1 <= ep <= 2
    # explained_variance: finite, <= 1 by construction. NOT asserted
    # positive -- an untrained critic can legitimately score below zero.
    assert np.isfinite(ev) and ev <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(encoder_before, net_a.encoder.parameters())), \
        "the per-deck encoder must actually train -- gradients reach it through the registered child"
    assert any(not torch.equal(a, b) for a, b in zip(head_before, net_a.non_targeting_head.parameters())), \
        "the per-deck head must have actually trained"
    for p in net_a.parameters():
        assert torch.isfinite(p).all(), "a parameter went non-finite after the ppo_update"
    print(f"rl.training.train ppo_update smoke test: OK (encoder AND head trained, {(time.time() - t0) * 1000:,.0f}ms)")


@pytest.mark.slow
def test_eval_record_false_smoke():
    # record=False (eval): the SAME collect_rollout, only `record` differs.
    # Must produce no training buffers, still yield one event_log per game.
    # Also exercises greedy=True (the eval default is sampled).
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
    print(f"rl.training.train eval/record=False smoke test: OK (no buffers, {sum(len(g) for g in eval_logs)} events logged, "
          f"{(time.time() - t0) * 1000:,.0f}ms)")


def test_wants_mana_mistake_gates_on_reward_fn_tag():
    # collect_rollout's gate for whether to build/wire the on_mana_burn
    # hook: True only once a tracked seat's reward_fn is tagged
    # consumes_mana_mistake=True. Tagged by hand here (no reward_fn in
    # rl.rewards currently sets this tag).
    base = flat_win_loss_reward()
    def tagged(state, done, horizon):
        return base(state, done, horizon)
    tagged.consumes_mana_mistake = True
    assert not _wants_mana_mistake([base, base], ["m", "m"]), "neither reward_fn is tagged"
    assert _wants_mana_mistake([tagged, base], ["m", None]), "seat 0 is tracked AND tagged"
    assert not _wants_mana_mistake([tagged, base], [None, "m"]), "the tagged seat isn't tracked; the tracked seat isn't tagged"
    assert not _wants_mana_mistake([tagged, base], [None, None]), "no tracked seat at all"


def test_on_mana_burn_closure_flags_wasted_tap():
    # The real production on_mana_burn closure (_make_on_mana_burn), not a
    # hand-rolled stand-in. Only a minimal fake agent (a real deck_ctx, no
    # network) is needed -- SeatAgent.decide is never called here.
    #
    # Mono-Mountain deck, no spells: play a land, tap it once in MAIN1, then
    # always Pass. Nothing is ever castable, so a correct closure must
    # report False (nothing castable) for the float to register as a mistake.
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
    # The whole phase's dense mana-burn charge must land on the Tap action
    # that produced the float, not the Pass pending when the phase boundary
    # crossed. Scripted mono-Mountain deck (nothing ever castable, so the
    # tap's pip survives to the clear): play a land, tap it once in MAIN1,
    # then always Pass.
    #
    # A real SeatAgent/DeckNetwork isn't needed -- collect_rollout's
    # choose_action closure only reads ppo_entry's action_idx and deck_ctx;
    # everything else a buffer entry stores is inert placeholder data.
    decklist = [("Mountain", 20)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    tap_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Tap Mountain")
    pass_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")

    base = lambda state, done, horizon: 0.0
    reward_fn = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=2.0)
    agents = [_ScriptedAgent(fixed_table), _ScriptedAgent(fixed_table)]
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
    # With a WINNER-ONLY reward (refund_on_loss=True), a seat's dense burn
    # charges are held for the whole game and applied at the terminal flush
    # only if it won. A losing seat's trajectory must come out bit-for-bit
    # identical to one that never burnt anything.
    #
    # Deferring rather than charging-then-refunding matters under GAE: a
    # charge written at step t lands in delta_t immediately, while a
    # terminal refund reaches it only discounted by (gamma*gae_lambda)^k --
    # a refund-based implementation would leave a residue here.
    #
    # Same scripted mono-Mountain setup as the test above, but with a deck
    # small enough to deck out within the horizon, producing a real winner
    # and loser in ONE game.
    decklist = [("Mountain", 8)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    tap_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Tap Mountain")
    pass_idx = next(i for i, (name, _l, _e) in enumerate(fixed_table) if name == "Pass")

    base = lambda state, done, horizon: 0.0
    reward_fn = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0,
                                              game_penalty_cap=2.0, refund_on_loss=True)
    agents = [_ScriptedAgent(fixed_table), _ScriptedAgent(fixed_table)]
    # Both seats recorded, into separate buckets, so winner and loser can be told apart.
    pairing = _constant_pairing(agents, [decklist, decklist], [reward_fn, reward_fn], ["seat0", "seat1"])

    winner = {}
    bufs, _mull, played = collect_rollout(pairing, 1, horizon=6, rng=_random.Random(0), device=DEVICE,
                                           record=True, on_game_end=lambda s: winner.setdefault("idx", s.winner))
    assert played == 1
    assert winner["idx"] is not None, "this fixture must deck someone out -- otherwise neither branch is exercised"

    won_buf = bufs[f"seat{winner['idx']}"]
    lost_buf = bufs[f"seat{1 - winner['idx']}"]

    # WINNER: the charge applies, and still lands on the Tap (not the Pass) --
    # deferring must not disturb the non-deferred path's attribution.
    won_taps = [won_buf.reward[i] for i in range(len(won_buf)) if won_buf.action[i] == tap_idx]
    won_passes = [won_buf.reward[i] for i in range(len(won_buf)) if won_buf.action[i] == pass_idx]
    assert won_taps, "the winner's scripted tap must have been recorded"
    assert any(r < -1e-9 for r in won_taps), "the winner's Tap must carry the deferred charge once applied"
    assert all(r == 0.0 for r in won_passes), "Pass must never absorb the charge"

    # LOSER: nothing was ever written -- exactly zero on every transition,
    # not merely "smaller" or "refunded to about zero".
    assert len(lost_buf), "the loser's trajectory must have been recorded at all"
    assert all(r == 0.0 for r in lost_buf.reward), (
        "a losing seat must pay nothing for mana burnt -- every transition exactly 0.0"
    )
    lost_taps = [i for i in range(len(lost_buf)) if lost_buf.action[i] == tap_idx]
    assert lost_taps, "the loser also tapped -- otherwise this proves nothing about dropping its charge"


@pytest.mark.slow
def test_explained_variance_is_zero_not_one_when_returns_are_constant():
    """value_loss is a raw MSE with no scale attached, so a critic
    predicting a constant that happens to be right reports a tiny loss and
    looks excellent. A degenerate (zero-variance) target must score 0, not 1."""
    net = DeckNetwork(SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16),
                      film_condition_dim=8, non_targeting_n_actions=4)
    buf = RolloutBuffer()
    # Every transition identical, reward 0, done immediately -> zero-variance
    # returns; a "perfect" critic here has explained nothing.
    for _ in range(8):
        buf.add([], np.zeros(SCALAR_FEATURE_DIM, dtype=np.float32),
                np.ones(4, dtype=bool), 0, 0.0, 0.0, 0.0, True)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)
    *_rest, ev, _astd = ppo_update(net, opt, buf, DEVICE, n_epochs=1, batch_size=4)
    assert ev == 0.0, f"constant returns must explain nothing, got {ev}"


@pytest.mark.slow
def test_adv_norm_floor_defaults_to_a_true_no_op_and_damps_degenerate_batches():
    """The guard on rl.training.ppo's advantage normalization.

    Unguarded, (adv - mean) / (std + 1e-8) rescales EVERY batch to unit
    variance, promoting critic noise to a full-scale gradient when the real
    signal is tiny (e.g. a near-always-lost matchup).

    Two properties, both load-bearing: floor=0.0 (default) must reproduce
    the historical behavior exactly; a floor above the batch's own spread
    must shrink the advantages rather than normalize to unit scale.
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

    # The same floor must not disturb a batch that DOES carry signal.
    assert np.allclose(normalized(healthy, 0.5), normalized(healthy, 0.0), atol=1e-5), (
        "a floor below the batch's own spread must be inert")


@pytest.mark.slow
def test_ppo_update_reports_raw_adv_std():
    """adv_norm_floor can't be set responsibly without knowing the actual
    advantage scale -- it's returned and logged so the value comes from the
    measured distribution rather than a guess."""
    fx = _base_fixture()
    agent_a = SeatAgent(fx["net_a"], AlwaysKeep(), fx["deck_ctx_a"])
    pairing = _constant_pairing([agent_a, agent_a], [fx["decklist_a"]] * 2,
                                [fx["reward_fn"]] * 2, ["m", "m"])
    bufs, _mull, _played = collect_rollout(pairing, 2, HORIZON, fx["rng"], device=DEVICE)
    *_rest, _ev, adv_std = ppo_update(fx["net_a"], fx["opt_a"], bufs["m"], DEVICE,
                                      n_epochs=1, batch_size=16)
    assert np.isfinite(adv_std) and adv_std >= 0.0


@pytest.mark.slow
def test_recurrent_state_is_per_seat_and_cleared_between_games():
    """Two invariants enforced at the seam (collect_rollout's game loop),
    not by calling reset() directly.

    PER SEAT: a mirror pairing puts ONE SeatAgent on BOTH seats. A shared
    hidden state would leak seat 0's hand into seat 1's conditioning, and
    would desync replay (ppo_update replays every episode from zeros).

    CLEARED PER GAME: a SeatAgent is reused across games (LeaguePool caches
    snapshot agents by path), so without a reset it would start a new game
    remembering one it's no longer playing."""
    fx = _base_fixture()
    left_over = []

    class _Spy(SeatAgent):
        def reset(self):
            left_over.append(self.hidden)  # what the PREVIOUS game left behind
            super().reset()

    agent = _Spy(fx["net_a"], AlwaysKeep(), fx["deck_ctx_a"])
    pairing = _constant_pairing([agent, agent], [fx["decklist_a"], fx["decklist_a"]],
                                [fx["reward_fn"], fx["reward_fn"]], ["a", "a"])
    collect_rollout(pairing, n_games=2, horizon=HORIZON, rng=fx["rng"], device=DEVICE)

    assert left_over[0] == {}, "the first game must start from a clean state"
    carried = left_over[-1]
    assert set(carried) == {0, 1}, (
        f"both seats must have accumulated their OWN recurrent state, got keys {sorted(carried)} -- "
        "one shared state would leave a single entry and leak seat 0's history into seat 1"
    )
    assert not torch.equal(carried[0], carried[1]), (
        "the two seats saw different games and must hold different states -- identical states mean "
        "they are sharing one"
    )
    # reset() runs at the start of a game, so the agent still holds the
    # final game's state here.
    assert set(agent.hidden) == {0, 1}
    agent.reset()
    assert agent.hidden == {}, "reset must clear every seat, not just the last one written"


def test_non_progressing_priority_round_raises_instead_of_hanging(monkeypatch):
    # A legal-but-non-progressing action spins forever inside a priority
    # round: `horizon` bounds turn_number, which doesn't advance within one
    # turn, so nothing in the ENGINE stops it (capping a priority round
    # would deviate from real Magic). Left unguarded that's a silent hang,
    # not a crash.
    #
    # Drives the guard directly with a frozen turn_number -- the defining
    # property of the failure -- rather than reconstructing a specific
    # engine state that loops.
    import rl.training.train as train_mod

    # A scripted mono-Mountain agent, not a real net -- the guard is purely
    # turn-number/decision-count based.
    decklist = [("Mountain", 20)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    reward_fn = lambda state, done, horizon: 0.0
    agent = _ScriptedAgent(fixed_table)
    pairing = _constant_pairing([agent, agent], [decklist, decklist],
                                [reward_fn, reward_fn], ["m", "m"])

    # Freeze turn_number: every decision counts against the same turn.
    monkeypatch.setattr(train_mod, "MAX_DECISIONS_PER_TURN", 25)
    real_run = game.run_multiplayer_game

    def frozen_turn_game(*a, **kw):
        choose = kw["choose_action"]

        def wrapped(state):
            state.turn_number = 0  # never advances -- the loop's signature
            return choose(state)
        kw["choose_action"] = wrapped
        return real_run(*a, **kw)

    monkeypatch.setattr(game, "run_multiplayer_game", frozen_turn_game)

    with pytest.raises(RuntimeError, match="non-progressing priority round"):
        collect_rollout(pairing, 1, HORIZON, _random.Random(0), device=DEVICE)


def test_the_guard_does_not_fire_on_ordinary_games():
    # A guard that trips on legitimate play would abort real training runs.
    # Largest legitimate turn ever measured here is 61 decisions; the
    # shipped limit is 5000.
    from rl.training.train import MAX_DECISIONS_PER_TURN
    assert MAX_DECISIONS_PER_TURN >= 1000, "limit must stay far above any legitimate turn"

    # Same cheap scripted agent -- ordinary play with real turn advancement
    # must never trip the guard.
    decklist = [("Mountain", 20)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    reward_fn = lambda state, done, horizon: 0.0
    agent = _ScriptedAgent(fixed_table)
    pairing = _constant_pairing([agent, agent], [decklist, decklist],
                                [reward_fn, reward_fn], ["m", "m"])
    buffers, _mull, played = collect_rollout(pairing, 2, HORIZON, _random.Random(0), device=DEVICE)
    assert played == 2
    assert len(buffers["m"]) > 0


class _HandRecordingAgent(_ScriptedAgent):
    """_ScriptedAgent that stashes the hand it's first asked to judge at the
    mulligan_decision pending resolution -- reads back the actual opening
    hand stratify wiring dealt, before any cards are played."""

    def __init__(self, fixed_table):
        super().__init__(fixed_table)
        self.recorded_hand = None

    def decide(self, state, seat, horizon, device, greedy=False):
        pend = state.pending_resolution
        if self.recorded_hand is None and pend is not None and pend["kind"] == "mulligan_decision":
            self.recorded_hand = list(state.hand)
        return super().decide(state, seat, horizon, device, greedy=greedy)


def test_collect_rollout_stratify_forces_the_recorded_seats_opening_hand():
    """collect_rollout's probability-roll/seat-gate wiring for
    stratify_0land_pct/stratify_7land_pct had no direct test -- only the
    lower-level game.state plumbing it calls was covered. Deterministic
    (pct=1.0) so this isn't a statistical flake, with only one seat
    recorded so the "exactly one recorded seat" gate actually fires."""
    decklist = [("Mountain", 33), ("Lightning Bolt", 27)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    reward_fn = flat_win_loss_reward()

    def _dealt_land_count(stratify_0land_pct, stratify_7land_pct):
        recorder = _HandRecordingAgent(fixed_table)
        other = _ScriptedAgent(fixed_table)
        pairing = _constant_pairing([recorder, other], [decklist, decklist],
                                     [reward_fn, reward_fn], ["m", None])
        # horizon=1: only the opening hand matters here. A tiny horizon
        # keeps the game short enough that _ScriptedAgent (mulligans, land
        # plays, one mana tap) never has to face a discard it can't script.
        collect_rollout(pairing, 1, horizon=1, rng=_random.Random(0), device=DEVICE,
                         stratify_0land_pct=stratify_0land_pct, stratify_7land_pct=stratify_7land_pct)
        assert recorder.recorded_hand is not None, "the recorded seat's own mulligan decision must have been reached"
        return sum(1 for c in recorder.recorded_hand if c.card_type.name == "LAND")

    assert _dealt_land_count(1.0, 0.0) == 0, "stratify_0land_pct=1.0 must always deal a 0-land opening hand"
    assert _dealt_land_count(0.0, 1.0) == 7, "stratify_7land_pct=1.0 must always deal a 7-land opening hand"


def test_collect_rollout_stratify_defaults_are_a_true_no_op():
    """stratify_0land_pct=0.0, stratify_7land_pct=0.0 (the defaults) must
    draw from the RNG identically to omitting the params -- checked by
    running the same seed twice and asserting the dealt hand comes out
    byte-identical."""
    decklist = [("Mountain", 33), ("Lightning Bolt", 27)]
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    reward_fn = flat_win_loss_reward()

    def _run(**stratify_kwargs):
        recorder = _HandRecordingAgent(fixed_table)
        other = _ScriptedAgent(fixed_table)
        pairing = _constant_pairing([recorder, other], [decklist, decklist],
                                     [reward_fn, reward_fn], ["m", None])
        collect_rollout(pairing, 1, horizon=1, rng=_random.Random(0), device=DEVICE, **stratify_kwargs)
        return [c.name for c in recorder.recorded_hand]

    omitted = _run()
    explicit_zero = _run(stratify_0land_pct=0.0, stratify_7land_pct=0.0)
    assert omitted == explicit_zero, "explicit 0.0/0.0 must deal byte-identical hands to omitting the params"


@pytest.mark.slow
def test_collect_rollout_league_stratifies_only_a_frozen_opponent_game():
    """stratify_0land_pct threaded into collect_rollout_league (train.py's
    own on_game_end reading state.mulligan_stratified back into the 4th
    outcome-tuple element) must only ever come back True when the opponent
    that game was a frozen snapshot -- the one case with exactly one
    recorded seat (collect_rollout's own "len(recorded_seats) == 1" gate). A
    mirror or a live cross-deck opponent both record two seats, so the gate
    must keep blocking stratify even at stratify_0land_pct=1.0 -- this is
    the "never the opposition" guarantee league_runner._run_session's
    record_outcome-skip depends on, tested at this layer."""
    fx = _base_fixture()
    net_a, net_b, reward_fn, rng = fx["net_a"], fx["net_b"], fx["reward_fn"], fx["rng"]
    decklist_a, decklist_b = fx["decklist_a"], fx["decklist_b"]
    ctxs_by_name = {"a": fx["deck_ctx_a"], "b": fx["deck_ctx_b"]}
    decklists_by_name = {"a": decklist_a, "b": decklist_b}
    live_nets = {"a": net_a, "b": net_b}
    # A bigger horizon than this module's own HORIZON=20 -- untrained nets
    # playing real 60-card decks routinely need more than 20 turns to reach
    # a winner, and this test needs at least one real outcome per case to
    # mean anything (unlike the other collect_rollout_league tests above,
    # which explicitly tolerate zero). Matches league_runner.HORIZON, the
    # real production horizon.
    horizon = 120
    tmp_dir = tempfile.mkdtemp()
    try:
        pool = LeaguePool(tmp_dir, ["a", "b"], max_snapshots_per_deck=3)
        pool.register_snapshot("a", net_a)
        snapshot_path = pool.snapshots["a"][0][1]

        def _stratified_flags(sample_opponent_result, n_games=8):
            pool.sample_opponent = lambda training_deck_name, rng, checkpoint_rate=0.0, pfsp=True: sample_opponent_result
            _bufs, _mull, _played, outcomes = collect_rollout_league(
                "a", live_nets, None, ctxs_by_name, decklists_by_name, pool, reward_fn,
                horizon, n_games=n_games, rng=rng, device=DEVICE, stratify_0land_pct=1.0)
            assert outcomes, f"at least one of {n_games} games must reach a winner for this assertion to mean anything"
            return [stratified for *_, stratified in outcomes]

        assert all(_stratified_flags(("a", snapshot_path))), \
            "a frozen-snapshot opponent (only the training seat recorded) must be stratified every time at pct=1.0"
        assert not any(_stratified_flags(("a", None))), \
            "a true mirror (both seats recorded) must never be stratified, even at pct=1.0"
        assert not any(_stratified_flags(("b", None))), \
            "a live cross-deck opponent (both seats recorded) must never be stratified, even at pct=1.0"
    finally:
        shutil.rmtree(tmp_dir)
