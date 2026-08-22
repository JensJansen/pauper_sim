"""Tests for game.resolution's targeting handlers (handlers_targeting.py):
choose-permanent (own/opponent's), choose-any-target (creature or player),
up-to-N multi-target selection, and refizzle_if_now_targetless (re-validating
a predicate-driven target against live battlefield state after a
state-based action). Exercises these primitives directly against hand-built
states, bypassing drl_env entirely (no card wires into every one of these
yet)."""

import pytest

from game.cards import CardDef, CardType, EffectId
from game.resolution import (
    begin_choose_any_target,
    begin_choose_opponent_permanent,
    begin_choose_permanent,
    begin_choose_up_to_any_target,
    begin_search_fetch,
    choose_any_target_creature_options,
    choose_any_target_options,
    choose_opponent_permanent_options,
    execute_choose_any_target_creature,
    execute_choose_any_target_player,
    execute_choose_opponent_permanent_option,
    refizzle_if_now_targetless,
)
from game.state import GameState, Permanent, PlayerState


def _permanent(name, card_type):
    return Permanent(CardDef(name, card_type, None, None))


def test_choose_opponent_permanent_targets_specific_slot():
    # Targets state.opponent's battlefield, addressed by (name, slot).
    # Simulates blocking's own defender-decision channel by setting
    # active_idx directly to "the defender."
    attacker_bogle_1 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2.slot = 2
    attacker_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2, attacker_land]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective

    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert choose_opponent_permanent_options(state) == [("Slippery Bogle", 1), ("Slippery Bogle", 2)]  # the Forest never qualifies
    execute_choose_opponent_permanent_option(state, "Slippery Bogle", 2)
    assert completed == [("Slippery Bogle", 2)]  # the SPECIFIC slot chosen, not an arbitrary same-named match


def test_choose_opponent_permanent_empty_options_fizzles():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == [None]


def _open_choose_opponent_permanent(state, predicate, on_complete):
    state.active_idx = 1  # simulating the defender's own already-flipped perspective
    begin_choose_opponent_permanent(state, predicate, on_complete)


def _open_choose_permanent(state, predicate, on_complete):
    begin_choose_permanent(state, predicate, on_complete)


def _open_choose_any_target_creature_only(state, predicate, on_complete):
    # allow_players=False, optional=False -- the one configuration that can
    # ever go all-False.
    begin_choose_any_target(state, predicate, on_complete, allow_players=False, optional=False)


@pytest.mark.parametrize(
    "open_resolution",
    [_open_choose_opponent_permanent, _open_choose_permanent, _open_choose_any_target_creature_only],
    ids=["choose_opponent_permanent", "choose_permanent", "choose_any_target_creature_only"],
)
def test_refizzle_if_now_targetless_fizzles(open_resolution):
    # begin_choose_* only validates non-empty options ONCE, at open time.
    # Simulates the gap directly: open with one legal target, remove it
    # (standing in for an SBA), then confirm the re-check fizzles cleanly
    # with None instead of leaving a dead resolution.
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    completed = []
    open_resolution(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == []  # still open -- one legal target existed at open time
    state.players[0].battlefield = []  # the SBA's own effect: the only target just died
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_leaves_still_legal_resolution_alone():
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert refizzle_if_now_targetless(state) is False
    assert completed == []  # untouched -- the target is still there
    assert state.pending_resolution is not None


def test_refizzle_if_now_targetless_ignores_unrelated_kinds():
    # search_fetch reads the library, which state_based_actions never
    # mutates, so it's not in refizzle_if_now_targetless's covered set.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.library = []
    completed = []
    begin_search_fetch(state, lambda c: True, lambda s, name: completed.append(name))
    assert completed == [None]  # already fizzled by its OWN open-time check
    assert state.pending_resolution is None
    assert refizzle_if_now_targetless(state) is False  # nothing pending -- no-op


def test_choose_any_target_creature_and_player():
    # A single target spanning BOTH battlefields' creatures plus either
    # player. Creatures addressed by (side, name, slot); players by index.
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    assert choose_any_target_creature_options(state) == [(0, "Grizzly Bears", 1), (1, "Grizzly Bears", 1)]  # both sides, Forest excluded
    assert ("player", 0) in choose_any_target_options(state) and ("player", 1) in choose_any_target_options(state)
    execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)  # the OPPONENT's copy specifically
    assert completed == [("creature", 1, "Grizzly Bears", 1)]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    execute_choose_any_target_player(state, 0)  # legal to target yourself (real Magic)
    assert completed == [("player", 0)]


def test_choose_any_target_no_players_no_creatures_fizzles():
    empty = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    empty.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    completed = []
    begin_choose_any_target(empty, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t), allow_players=False)
    assert completed == [None]


def test_choose_any_target_no_players_with_creature_offers_creatures_only():
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: None, allow_players=False)
    assert all(o[0] == "creature" for o in choose_any_target_options(state))


def test_choose_up_to_any_target_board_identity_exclusion():
    # Two same-named "Bear" (distinct slots) -- both reachable, picking
    # slot 1 excludes only it (slot 2 stays choosable).
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.active_idx = 0
    bear1 = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    bear1.slot = 1
    bear2 = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    bear2.slot = 2
    st.players[0].battlefield = [bear1, bear2]
    descs = []
    begin_choose_up_to_any_target(st, lambda p: p.card_type == CardType.CREATURE, 2, lambda s, d: descs.extend(d))
    opts = choose_any_target_creature_options(st)
    assert (0, "Bear", 1) in opts and (0, "Bear", 2) in opts
    execute_choose_any_target_creature(st, 0, "Bear", 1)
    opts = choose_any_target_creature_options(st)
    assert (0, "Bear", 1) not in opts and (0, "Bear", 2) in opts  # identity (slot) exclusion
    execute_choose_any_target_creature(st, 0, "Bear", 2)
    assert st.pending_resolution is None and descs == [("creature", 0, "Bear", 1), ("creature", 0, "Bear", 2)]
