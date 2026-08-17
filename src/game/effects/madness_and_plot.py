"""The Madness "cast for its madness cost" and Plot "pay cost, exile instead
of resolving" paths -- both need mana.begin_pay_cost, which resolution can't
import (mana imports resolution; the reverse would cycle). This module depends
on both, so the orchestration lives here."""

from .. import mana, registry, resolution
from .stack import push_to_stack


def execute_madness_cast(state):
    """Model chose "cast" for a pending madness_decision (offered from inside
    the madness trigger's own stack resolve): pay the madness cost, then push
    the effect onto the stack (its own priority round, like any cast), then call
    the decision's on_complete (a no-op today). Captures card_def/on_complete
    before begin_pay_cost overwrites pending with its own "pay_cost".

    Exile removal happens in _after_pay, NOT before begin_pay_cost -- the same
    "never touch a zone until on_complete fires" contract every begin_pay_cost
    caller keeps: begin_pay_cost is only opened once plan_payment has confirmed
    the cost is payable (and it asserts that itself), so there is no undo path
    to worry about."""
    pending = state.pending_resolution
    card_def = pending["card_def"]
    outer_on_complete = pending["on_complete"]
    madness_spec = registry.EFFECT_REGISTRY[card_def.effect_id]["madness"]

    def _after_pay(s):
        resolution._remove_one_from_exile(s, card_def)
        if madness_spec.get("precast_choice"):
            # A madness spell that targets (Fiery Temper) locks its target
            # AS it's put on the stack, real Magic -- its resolve chooses the
            # target and does its own push_to_stack, same precast_choice
            # contract the normal cast path uses (drl_env._precast_choice_execute).
            madness_spec["resolve"](s, card_def)
        else:
            push_to_stack(s, card_def, madness_spec["resolve"], reserves_hand_card=False)
        outer_on_complete(s)

    mana.begin_pay_cost(state, madness_spec["cost"], on_complete=_after_pay)


def plot_to_exile(state, card_def):
    """Plot's own resolve shape: pay the
    plot cost, move hand -> exile with this turn's stamp instead of
    running the card's real effect. Generic Plot-mechanic plumbing (any
    future "plot" card reuses this unchanged, same precedent as
    execute_madness_cast above) -- currently only Highway Robbery
    (game.catalog.red_cards) has a "plot" registry spec."""
    state.hand.remove(card_def)
    state.exile.append((card_def, state.turn_number))
