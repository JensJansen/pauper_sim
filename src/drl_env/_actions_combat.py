"""Combat actions: declaring attackers, assigning blockers (incl. Done and
Menace), and the permanently-masked trample-to-player row (kept for
fixed-table shape stability). legal(state)/execute(state) factory pairs
build_action_table calls once per matching creature/slot."""

import game

from ._actions_common import _GATE_NO_PENDING


def _find_on_battlefield(state, name, slot):
    """The (name, slot) permanent on state.battlefield, or None. Plain
    O(battlefield_size) scan: the "Attack: "/"Assign Blocker: " rows this
    backs are filtered out of the production table before
    drl_env.legal_action_mask ever sees them (rl.decision.action_bridge's
    build_fixed_action_table routes attacking/blocking through the pointer
    head instead, via game.creature_attack_eligible/creature_block_eligible
    directly) -- so this only ever runs from a direct unit test invoking
    _attack_legal/_assign_blocker_legal against the raw, unfiltered table,
    never from a hot per-decision sweep. A former sweep-scoped cache here
    (removed 2026-08-25, having been measured dead on every production call
    path) existed to serve exactly that sweep, which no longer reaches this
    code."""
    return next((p for p in state.battlefield if p.card_def.name == name and p.slot == slot), None)


def _attack_legal(name, slot):
    """Legal only during Phase.DECLARE_ATTACKERS, only for the turn owner
    (active_idx == turn_player_idx, since declaring an attacker is a
    turn-based special action, not a priority action), and only if the
    permanent at this (name, slot) is attack-eligible
    (game.creature_attack_eligible: untapped, not summoning sick unless
    haste). Attacking is optional; Pass with zero attackers is legal.

    _GATE_NO_PENDING: a DECLARE_ATTACKERS window never coexists with an
    open pending_resolution."""
    def legal(state):
        if state.phase is not game.turn.Phase.DECLARE_ATTACKERS:
            return False
        if state.active_idx != state.turn_player_idx:
            return False
        p = _find_on_battlefield(state, name, slot)
        return p is not None and game.creature_attack_eligible(state, p)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _attack_execute(name, slot):
    """Declares the permanent at this (name, slot) as an attacker --
    exact-slot addressing lets a model distinguish an Aura-enchanted copy
    from a plain one of the same name."""
    def execute(state):
        permanent = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_attack_eligible(state, p)
        )
        game.declare_attacker(state, permanent)
    return execute


def _assign_blocker_legal(name, slot):
    """"Assign Blocker: <name> (slot j)" -- legal only while a
    "declare_blockers" resolution is pending, and the permanent at this
    (name, slot) is block-eligible (game.creature_block_eligible):
    untapped, not already assigned this combat. Unlike attacking, neither
    summoning sickness nor Defender excludes a blocker."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "declare_blockers":
            return False
        p = _find_on_battlefield(state, name, slot)
        return p is not None and game.creature_block_eligible(state, p)
    legal._pending_gate = frozenset({"declare_blockers"})
    return legal


def _assign_blocker_execute(name, slot):
    """Parks the permanent at this (name, slot) as a blocker, then hands
    off to game.declare_blocker_assignment, which nests a
    choose_opponent_permanent sub-resolution restricted by extra_predicate
    to attackers this blocker may legally block (game.can_block). Once
    that completes, re-opens begin_declare_blockers so the defender can
    assign another blocker or choose Done.

    extra_predicate here and the eligibility check in
    creature_block_eligible must stay the same function (both must consult
    can_block, honoring reach/flying restrictions identically) -- a
    mismatch can make the legality check pass while the nested choice
    matches zero attackers, leaving declare-blockers unable to terminate."""
    def execute(state):
        blocker = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_block_eligible(state, p)
        )
        outer_on_complete = state.pending_resolution["on_complete"]
        game.declare_blocker_assignment(
            state, blocker, on_complete=lambda s: game.begin_declare_blockers(s, outer_on_complete),
            extra_predicate=lambda attacker: game.can_block(state, blocker, attacker),
        )
    return execute


def _done_blocking_legal(state):
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "declare_blockers":
        return False
    # Menace (509.1c): can't finish a declaration leaving a menace attacker
    # blocked by exactly one creature. No undo: if no second blocker is
    # available, this stays illegal, the declaration is abandoned via
    # zero-legal-actions, and combat.enforce_menace drops the illegal block.
    return not game.menace_block_incomplete(state)


_done_blocking_legal._pending_gate = frozenset({"declare_blockers"})


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _assign_damage_to_opponent_legal(state):
    """Always False. Trample-through is now a forced, automatic outcome of
    assign_combat_damage once every blocker in the split is at its lethal
    cap (702.19b/510.1c) -- never an agent choice. This row stays
    registered, permanently illegal, only so the fixed action table's
    length doesn't change and break existing checkpoints (rl.model.deck)."""
    return False


_assign_damage_to_opponent_legal._pending_gate = frozenset({"assign_combat_damage"})


def _assign_damage_to_opponent_execute(state):
    raise AssertionError("unreachable: _assign_damage_to_opponent_legal is always False")


__all__ = [
    '_attack_legal',
    '_attack_execute',
    '_assign_blocker_legal',
    '_assign_blocker_execute',
    '_done_blocking_legal',
    '_done_blocking_execute',
    '_assign_damage_to_opponent_legal',
    '_assign_damage_to_opponent_execute',
    '_find_on_battlefield',
]
