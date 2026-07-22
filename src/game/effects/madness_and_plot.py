"""The Madness "cast for its madness cost" path and the Plot "pay cost,
exile instead of resolving" path -- these need game.mana.begin_pay_cost,
which game.resolution can't import (mana.py imports resolution.py at its
own top level; the reverse would cycle). This module is free to depend on
both resolution.py and mana.py, so it's where that orchestration lives
instead. See docs/MADNESS_DECKS_PLAN.md items 1/3/4/7. This is the one
piece of the old effects_common.py that module's own docstring was
actually about -- everything else that used to share the file with it
(combat, casting, the stack, ...) has moved to its own module."""

from .. import mana, registry, resolution
from .stack import push_to_stack


def execute_madness_cast(state):
    """Model chose "cast" for a pending madness_decision (itself now only
    ever offered from inside the madness trigger's own stack resolve, see
    triggers._trigger_resolve -- docs/PRIORITY_PLAN.md item 1): pay the
    card's madness cost, then push its effect onto the stack (see
    stack.push_to_stack) instead of resolving it immediately -- a real,
    independent stack entry that gets its own priority round before it
    resolves, same as any other cast. Then calls the enclosing
    madness_decision's own on_complete, a no-op today (see
    triggers._trigger_resolve's own docstring for why the old recursive
    "keep draining" continuation isn't needed anymore).

    Captures card_def/on_complete from the CURRENT pending_resolution
    before begin_pay_cost overwrites it with its own "pay_cost" one --
    same nested-callback shape flashback_dread_return
    (game.catalog.black_cards) already uses for its own multi-step
    chain."""
    pending = state.pending_resolution
    card_def = pending["card_def"]
    outer_on_complete = pending["on_complete"]
    madness_spec = registry.EFFECT_REGISTRY[card_def.effect_id]["madness"]
    resolution._remove_one_from_exile(state, card_def)

    def _after_pay(s):
        push_to_stack(s, card_def, madness_spec["resolve"])
        outer_on_complete(s)

    mana.begin_pay_cost(state, madness_spec["cost"], on_complete=_after_pay)


def plot_to_exile(state, card_def):
    """Plot's own resolve shape (MADNESS_DECKS_PLAN.md item 4): pay the
    plot cost, move hand -> exile with this turn's stamp instead of
    running the card's real effect. Generic Plot-mechanic plumbing (any
    future "plot" card reuses this unchanged, same precedent as
    execute_madness_cast above) -- currently only Highway Robbery
    (game.catalog.red_cards) has a "plot" registry spec."""
    state.hand.remove(card_def)
    state.exile.append((card_def, state.turn_number))


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.madness_and_plot`
    # from src/. plot_to_exile directly (execute_madness_cast is exercised
    # together with triggers.py/stack.py's own machinery -- the full
    # discard -> exile + queue -> drain -> decision -> pay -> resolve chain
    # -- in effects/integration_check.py, since it's genuinely a
    # multi-module scenario, not just this one function).
    from ..cards import CardDef, CardType, EffectId
    from ..state import GameState

    state = GameState(on_the_play=True)
    plot_card = CardDef("Fake Plot Spell", CardType.SORCERY, {"generic": 1}, EffectId.FILLER)
    state.hand = [plot_card]
    plot_to_exile(state, plot_card)
    assert state.hand == []
    assert state.exile == [(plot_card, state.turn_number)]

    print("madness_and_plot.py plot_to_exile self-check: OK")
