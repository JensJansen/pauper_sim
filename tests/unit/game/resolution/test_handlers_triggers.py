"""Tests for game.resolution's trigger-ordering handler (handlers_triggers.py):
when 2+ of one player's own triggers are ready to move onto the stack at
once, the player chooses the placement order. Exercises this primitive
directly against a hand-built state, bypassing drl_env entirely."""

from game.cards import CardDef, CardType
from game.resolution import begin_order_triggers, execute_order_triggers_option, order_triggers_options
from game.state import GameState


def test_order_triggers_placement_order_is_lifo_at_resolution():
    # PLACEMENT order, not resolution order (the stack is LIFO). Driven
    # directly against a hand-built state with plain no-op resolve
    # functions, since only the ordering mechanism itself is under test.
    resolved_order = []
    entry_a = {"card_def": CardDef("Trigger A", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    entry_b = {"card_def": CardDef("Trigger B", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    state = GameState(on_the_play=True)
    completed = []
    begin_order_triggers(state, [entry_a, entry_b], on_complete=lambda s: completed.append(True))
    assert order_triggers_options(state) == ["Trigger A", "Trigger B"]

    execute_order_triggers_option(state, "Trigger A")  # placed FIRST -- resolves LAST
    assert completed == []  # one more still to place
    assert state.stack == [entry_a]
    assert order_triggers_options(state) == ["Trigger B"]  # already-placed one no longer offered

    execute_order_triggers_option(state, "Trigger B")  # placed LAST -- resolves FIRST
    assert completed == [True]
    assert state.stack == [entry_a, entry_b]  # placement order: A then B
    assert state.pending_resolution is None

    while state.stack:  # LIFO: B (placed last) actually resolves first
        entry = state.stack.pop()
        entry["resolve"](state, entry["card_def"])
    assert resolved_order == ["Trigger B", "Trigger A"]
