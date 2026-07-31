"""Migrated from src/rl/agent.py's __main__ self-check."""
import random
import types

import pytest

import drl_env
import game
from game.state import GameState, PlayerState
from rl.action_bridge import build_fixed_action_table
from rl.agent import DECK_SIZE_CAP, OPPONENT_HAND_SIZE_CAP, AlwaysKeep, SeatAgent, _scalar_features
from rl.arch import SetTransformer
from rl.features import CardVocab
from rl.mulligan import MulliganNet


def _mulligan_decision_state():
    """A REAL state parked at the first mulligan decision (advance the
    coroutine one step) -- no guessing at GameState internals."""
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    vocab = CardVocab([decklist])
    state = game.turn.new_multiplayer_game_state([decklist, decklist], starting_player_idx=0, rng=random.Random(0))
    gen = game.turn.game_coroutine(state)
    next(gen)
    assert state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision", (
        f"expected first decision to be a mulligan, got {state.pending_resolution}"
    )
    return decklist, vocab, state


@pytest.mark.slow
def test_always_keep_routes_mulligan_decision():
    # AlwaysKeep routes mulligan_decision to a keep executor, records nothing.
    _decklist, vocab, state = _mulligan_decision_state()
    seat = state.active_idx
    ak_agent = SeatAgent(main=None, mulligan=AlwaysKeep(), deck_ctx=(vocab, [], ()))
    dr = ak_agent.decide(state, seat, horizon=120, device="cpu")
    assert callable(dr.executor) and dr.ppo_entry is None and dr.mull_entry is None and not dr.is_pass


@pytest.mark.slow
def test_always_keep_refuses_mulligan_bottom():
    # AlwaysKeep must refuse mulligan_bottom (unreachable under keep-always).
    raised = False
    try:
        AlwaysKeep().decide(types.SimpleNamespace(pending_resolution={"kind": "mulligan_bottom"}))
    except AssertionError as e:
        raised = "mulligan_bottom" in str(e)
    assert raised, "AlwaysKeep must assert on mulligan_bottom"


@pytest.mark.slow
def test_mulligan_net_branch_routes_and_records():
    # MulliganNet branch: routes to the net and records a 'decision' transition.
    _decklist, vocab, state = _mulligan_decision_state()
    seat = state.active_idx
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    mnet = MulliganNet(shared, hidden=16)
    net_agent = SeatAgent(main=None, mulligan=mnet, deck_ctx=(vocab, [], ()))
    dr2 = net_agent.decide(state, seat, 120, "cpu")
    assert callable(dr2.executor) and dr2.ppo_entry is None
    assert dr2.mull_entry is not None and dr2.mull_entry[0] == "decision", (
        f"MulliganNet decision must record a 'decision' transition, got {dr2.mull_entry}"
    )


@pytest.mark.slow
def test_main_fixed_table_has_no_pregame_action():
    # Phase-4 invariant: the main net's fixed table has ZERO pregame actions, so in
    # a pregame state its legal mask is all-False and the SeatAgent MUST intercept
    # (never reach _policy_decision). Guards against a future re-add of the actions.
    decklist, _vocab, state = _mulligan_decision_state()
    ftable = build_fixed_action_table(decklist, pending_kinds=game.derive_pending_kinds(decklist))
    assert not any(drl_env.legal_action_mask(state, ftable)), (
        "regression: the main net's fixed table has a legal action in a pregame state -- "
        "pregame actions must stay removed (the mulligan model owns them)"
    )


@pytest.mark.slow
def test_scalar_features_library_hand_size_and_stack_targets():
    # _scalar_features: library size (mine/opponent), opponent hand size, and
    # stack-targets-me/opponent (the "player" target kind -- the one case
    # with no token to carry a bit on; see rl.features._stack_target_map and
    # its own build_token_set self-check for the three OBJECT target kinds).
    sf_seat0 = PlayerState(on_the_play=True)
    sf_seat0.library = [game.CARD_DEFS["Mountain"]] * 40
    sf_seat0.hand = [game.CARD_DEFS["Lightning Bolt"]] * 3
    sf_seat1 = PlayerState(on_the_play=False)
    sf_seat1.library = [game.CARD_DEFS["Mountain"]] * 10
    sf_seat1.hand = [game.CARD_DEFS["Lightning Bolt"]] * 5
    sf_state = GameState(on_the_play=True, players=[sf_seat0, sf_seat1])
    # seat 1's Lava Dart targets seat 0 as a PLAYER (controller=1).
    sf_state.stack = [{"card_def": game.CARD_DEFS["Lava Dart"], "resolve": None, "controller": 1,
                        "targets": (("player", 0),)}]

    sf0 = _scalar_features(sf_state, 0, horizon=40)
    sf1 = _scalar_features(sf_state, 1, horizon=40)
    # Tail layout (see _scalar_features's own append order): [..., my_library,
    # opp_library, opp_hand, stack_targets_me, stack_targets_opponent].
    my_lib_i, opp_lib_i, opp_hand_i, targets_me_i, targets_opp_i = -5, -4, -3, -2, -1
    assert sf0[my_lib_i] == 40 / DECK_SIZE_CAP and sf0[opp_lib_i] == 10 / DECK_SIZE_CAP
    assert sf1[my_lib_i] == 10 / DECK_SIZE_CAP and sf1[opp_lib_i] == 40 / DECK_SIZE_CAP, (
        "library-size scalars must flip with perspective, not stay pinned to seat 0/1"
    )
    assert sf0[opp_hand_i] == 5 / OPPONENT_HAND_SIZE_CAP, "seat 0's opponent-hand-size view must be seat 1's hand"
    assert sf1[opp_hand_i] == 3 / OPPONENT_HAND_SIZE_CAP, "seat 1's opponent-hand-size view must be seat 0's hand"
    # Lava Dart (controller=1) targets seat 0 as a player -- seat 0 sees
    # "stack targets me", seat 1 sees "stack targets opponent", never both.
    assert sf0[targets_me_i] == 1.0 and sf0[targets_opp_i] == 0.0
    assert sf1[targets_me_i] == 0.0 and sf1[targets_opp_i] == 1.0
