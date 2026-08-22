"""Madness ("cast for its madness cost") and Plot ("pay cost, exile instead
of resolving") -- both need mana.begin_pay_cost, which resolution can't
import (mana imports resolution), so this orchestration lives here."""

from .. import mana, registry, resolution
from .stack import push_to_stack


def execute_madness_cast(state):
    """Model chose "cast" for a pending madness_decision: pay the madness
    cost, then push the effect onto the stack, then call the decision's
    on_complete. Captures card_def/on_complete before begin_pay_cost
    overwrites pending with its own "pay_cost"."""
    pending = state.pending_resolution
    card_def = pending["card_def"]
    outer_on_complete = pending["on_complete"]
    madness_spec = registry.EFFECT_REGISTRY[card_def.effect_id]["madness"]

    def _after_pay(s):
        resolution._remove_one_from_exile(s, card_def)
        if madness_spec.get("precast_choice"):
            # A targeting madness spell (Fiery Temper) locks its target as it's
            # put on the stack; its own resolve does the push_to_stack.
            madness_spec["resolve"](s, card_def)
        else:
            push_to_stack(s, card_def, madness_spec["resolve"], reserves_hand_card=False)
        outer_on_complete(s)

    mana.begin_pay_cost(state, madness_spec["cost"], on_complete=_after_pay)


def plot_to_exile(state, card_def):
    """Plot's resolve: pay the plot cost, move hand -> exile with this
    turn's stamp, instead of running the card's real effect."""
    state.hand.remove(card_def)
    state.exile.append((card_def, state.turn_number))
