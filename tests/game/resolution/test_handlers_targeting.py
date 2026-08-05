"""Tests for game.resolution's targeting handlers (handlers_targeting.py):
choose-permanent (own/opponent's), choose-any-target (creature or player),
up-to-N multi-target selection, and refizzle_if_now_targetless (re-validating
a predicate-driven target against live battlefield state after a
state-based action). Exercises these primitives directly against hand-built
states, bypassing drl_env entirely (no card wires into every one of these
yet)."""

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
    # Cross-player targeting: begin_choose_opponent_permanent
    # targets state.opponent's battlefield, addressed by (name, slot) --
    # not name alone, since two same-named OPPOSING permanents aren't
    # necessarily interchangeable. Only correct once the referencing player
    # is already the active one (blocking's own defender-decision channel
    # flips active_idx before ever calling this) -- simulated here by
    # setting active_idx directly to "the defender," same as that channel
    # would.
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
    # Empty-options safety net: no eligible opposing permanent -> fizzles
    # immediately with None, same convention as begin_choose_permanent.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == [None]


def test_refizzle_if_now_targetless_fizzles_choose_opponent_permanent():
    # begin_choose_opponent_permanent only validates non-empty options ONCE,
    # at open time. If a state_based_actions pass removes the only legal
    # target before the next decision point, the pending resolution would
    # otherwise sit there with an all-False mask and no recovery --
    # game.turn._run_priority_round_gen's own refizzle_if_now_targetless call
    # is what catches that. Simulates the gap directly: open with one legal
    # target, remove it (standing in for an SBA), then confirm the re-check
    # fizzles cleanly with None instead of leaving a dead resolution.
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == []  # still open -- one legal target existed at open time
    state.players[0].battlefield = []  # the SBA's own effect: the only target just died
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_fizzles_choose_permanent():
    # Same gap, own-battlefield half (begin_choose_permanent).
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    completed = []
    begin_choose_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == []
    state.players[0].battlefield = []
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_fizzles_choose_any_target_creature_only():
    # Same gap, choose_any_target's creature-only mode (allow_players=False,
    # optional=False -- the one configuration that can ever go all-False; with
    # allow_players=True a player is always legal, and optional=True always
    # offers a decline, so neither ever needs this re-check).
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    completed = []
    begin_choose_any_target(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
        allow_players=False, optional=False,
    )
    assert completed == []
    state.players[0].battlefield = []
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_leaves_still_legal_resolution_alone():
    # No-op when the pending resolution's options are still non-empty -- the
    # common case, every priority-loop iteration.
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
    # search_fetch reads the library, which state_based_actions never mutates --
    # not in refizzle_if_now_targetless's covered set, so even an empty-options
    # search_fetch (its own open-time safety net already fizzled it) is simply
    # not its concern. No pending resolution at all is the other no-op case.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.library = []
    completed = []
    begin_search_fetch(state, lambda c: True, lambda s, name: completed.append(name))
    assert completed == [None]  # already fizzled by its OWN open-time check
    assert state.pending_resolution is None
    assert refizzle_if_now_targetless(state) is False  # nothing pending -- no-op


def test_choose_any_target_creature_and_player():
    # "Any target" (begin_choose_any_target): a single target spanning BOTH
    # battlefields' creatures plus either player -- real Magic's "any
    # target" (Lightning Bolt). Creatures addressed by (side, name, slot)
    # so a same-named creature on each side stays distinguishable; players
    # by index.
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
    # allow_players=False + no creature anywhere -> immediate None (fizzle/
    # can't-target), same empty-options net as the other primitives.
    empty = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    empty.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    completed = []
    begin_choose_any_target(empty, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t), allow_players=False)
    assert completed == [None]


def test_choose_any_target_no_players_with_creature_offers_creatures_only():
    # allow_players=False WITH a creature -> creatures only, no player option offered
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: None, allow_players=False)
    assert all(o[0] == "creature" for o in choose_any_target_options(state))


def test_choose_up_to_any_target_board_identity_exclusion():
    # board up-to-2: two same-named "Bear" (distinct slots) -- both
    # reachable, picking slot 1 excludes only it (slot 2 stays choosable).
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
