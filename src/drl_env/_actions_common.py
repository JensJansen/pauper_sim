"""Shared infrastructure for the drl_env action-table modules: the one sentinel
a gate-free-vs-gated legal() closure stamps itself with when it uses the plain
no-pending-resolution gate (used by _actions_cast/_actions_cast_altzone/
_actions_combat/_actions_resolution -- _actions_mana uses a different gate
scheme instead, see its own docstring, and _actions_table only assembles rows
the other modules already gated, so neither imports it), plus
_hand_count_available -- a plain hand tally needed by both _actions_cast's
own hand-cast factories and _actions_cast_altzone's _alt_cast_legal. It lives
here, not in either of them, specifically to avoid a cast <-> cast_altzone
import cycle: _actions_cast.py separately imports _with_chosen_copy FROM
_actions_cast_altzone (for its own _graveyard_ability_execute), so
_actions_cast_altzone importing _hand_count_available back FROM
_actions_cast would close a loop -- routing the shared helper through this
common module instead keeps both edges one-directional."""

_GATE_NO_PENDING = object()  # sentinel: this closure's own first check is "state.pending_resolution is None" -- see legal_action_mask's own docstring


def _hand_count_available(state, name):
    """How many copies of `name` in state.hand are castable right now -- just
    the count in hand. A spell LEAVES hand the instant it is put on the stack
    (game.effects.stack.push_to_stack removes it at cast), so a copy already
    paid-for and awaiting resolution is simply not in state.hand and can't be
    re-cast -- the card physically leaving hand IS the re-cast guard, so a
    plain hand tally is all this needs."""
    return sum(1 for c in state.hand if c.name == name)


__all__ = [
    '_GATE_NO_PENDING',
    '_hand_count_available',
]
