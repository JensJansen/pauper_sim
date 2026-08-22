"""Shared infrastructure for the drl_env action-table modules: the
no-pending-resolution gate sentinel (used by _actions_cast,
_actions_cast_altzone, _actions_combat, _actions_resolution) and
_hand_count_available, shared by _actions_cast and _actions_cast_altzone.
Lives here rather than in either of them to avoid a cast <-> cast_altzone
import cycle."""

_GATE_NO_PENDING = object()  # marks a legal() closure gated on state.pending_resolution is None


def _hand_count_available(state, name):
    """Count of `name` currently in state.hand. A cast spell leaves hand
    immediately (push_to_stack removes it), so this doubles as the
    re-cast guard."""
    return sum(1 for c in state.hand if c.name == name)


__all__ = [
    '_GATE_NO_PENDING',
    '_hand_count_available',
]
