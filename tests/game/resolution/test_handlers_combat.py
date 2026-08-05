"""Tests for game.resolution's combat handlers (handlers_combat.py):
declaring blockers, including gang-blocking. Exercises these primitives
directly against hand-built states, bypassing drl_env entirely (no card
wires into every one of these yet)."""

from game.cards import CardDef, CardType
from game.resolution import (
    begin_declare_blockers,
    choose_opponent_permanent_options,
    complete_resolution,
    declare_blocker_assignment,
    execute_choose_opponent_permanent_option,
)
from game.state import GameState, Permanent, PlayerState


def _permanent(name, card_type):
    return Permanent(CardDef(name, card_type, None, None))


def test_declare_blockers_gang_blocking_and_done():
    # Blocking: begin_declare_blockers/
    # declare_blocker_assignment, driven directly against a hand-built
    # state (bypassing game.turn._declare_blockers_gen's active_idx-flip --
    # simulated here the same way the cross-player check above does, by
    # setting active_idx to "the defender" up front). Also bypasses
    # drl_env's own _assign_blocker_legal eligibility gate -- this
    # exercises the resolution primitives directly, so a "re-open
    # begin_declare_blockers after each assignment" step is done by hand
    # here rather than relying on drl_env._assign_blocker_execute's own
    # nested on_complete to do it.
    bear = _permanent("Bear", CardType.CREATURE)
    wolf = _permanent("Wolf", CardType.CREATURE)
    grizzly = _permanent("Grizzly Bears", CardType.CREATURE)
    panther = _permanent("Panther", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [bear, wolf]
    state.players[0].attackers = [bear, wolf]
    state.players[1].battlefield = [grizzly, panther]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []  # real attackers declared -- does not auto-complete
    assert state.pending_resolution["kind"] == "declare_blockers"

    # Assign Grizzly Bears to block Bear specifically (not Wolf) -- the
    # nested choose_opponent_permanent offers both.
    step1_done = []
    declare_blocker_assignment(state, grizzly, on_complete=lambda s: step1_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step1_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly]}  # attacker -> LIST of blockers (gang-blocking)

    # GANG-BLOCKING: re-open the consult and assign Panther to the SAME
    # attacker (Bear). An already-blocked attacker is STILL offered --
    # multiple blockers may pile onto one attacker.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    step2_done = []
    declare_blocker_assignment(state, panther, on_complete=lambda s: step2_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]  # Bear STILL offered -- gang-block
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step2_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly, panther]}  # two blockers on one attacker

    # "Done blocking" (drl_env's action): closes a still-open
    # declare_blockers resolution outright, no assignment required.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    complete_resolution(state)
    assert completed == [True]


def test_declare_blockers_no_attackers_auto_completes():
    # No attackers at all: auto-completes immediately, same empty-options
    # precedent as begin_choose_permanent/begin_search_fetch.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 1
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == [True]


def test_declare_blocker_assignment_extra_predicate():
    # declare_blocker_assignment's extra_predicate: this module stays
    # effect-agnostic (see its own module docstring) and doesn't import
    # game.effects.stats itself, so the actual restriction is supplied by
    # the CALLER (drl_env._assign_blocker_execute, using game.has_keyword)
    # -- this proves the parameter itself is correctly applied on top of
    # the usual "unblocked attacker" filter, using a plain stand-in
    # predicate rather than a real keyword lookup.
    flyer = _permanent("Flyer", CardType.CREATURE)
    grounded = _permanent("Grounded", CardType.CREATURE)
    non_flying_blocker = _permanent("Non-Flying Blocker", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [flyer, grounded]
    state.players[0].attackers = [flyer, grounded]
    state.players[1].battlefield = [non_flying_blocker]
    state.active_idx = 1

    completed = []
    declare_blocker_assignment(
        state, non_flying_blocker, on_complete=lambda s: completed.append(True),
        extra_predicate=lambda p: p is not flyer,  # stand-in: "flyer needs a flying blocker, and this one isn't"
    )
    assert choose_opponent_permanent_options(state) == [("Grounded", 1)]  # Flyer excluded by extra_predicate
    execute_choose_opponent_permanent_option(state, "Grounded", 1)
    assert completed == [True]
    assert state.players[0].blocked_by == {grounded: [non_flying_blocker]}
