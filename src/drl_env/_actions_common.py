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


def _flag_signature(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(v) for v in value)))
    return ("obj", id(value))


def copy_signature(state, permanent):
    """Everything that can make two same-named permanents NOT interchangeable for an action: their own runtime state
    (tapped, summoning sickness, counters, damage, type override, flags) and the Auras attached to them -- on EITHER
    battlefield, since an Aura may enchant the other player's permanent. Two copies with equal signatures are the same
    choice (rules 602.2/605.3: the player chooses which permanent to tap or activate, so when the choice cannot matter
    it is not asked)."""
    auras = tuple(sorted(
        (a.card_def.name, tuple(sorted((k, _flag_signature(v)) for k, v in a.flags.items() if k not in ("enchanting", "_logged_pt"))))
        for pl in state.players for a in pl.battlefield if a.flags.get("enchanting") is permanent))
    return (
        permanent.card_def.name, permanent.tapped, permanent.summoning_sick, permanent.damage_marked,
        tuple(sorted(permanent.counters.items())), permanent.type_override,
        tuple(sorted((k, _flag_signature(v)) for k, v in permanent.flags.items() if k != "_logged_pt")),
        permanent.temp_power, permanent.temp_toughness, tuple(sorted(permanent.temp_keywords)), auras,
    )


def distinct_copies(state, permanents):
    """One representative per class of interchangeable permanents (see copy_signature), in the given order."""
    seen, out = set(), []
    for p in permanents:
        sig = copy_signature(state, p)
        if sig not in seen:
            seen.add(sig)
            out.append(p)
    return out


__all__ = [
    '_GATE_NO_PENDING',
    '_hand_count_available',
    'copy_signature',
    'distinct_copies',
]
