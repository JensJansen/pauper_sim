"""Combat actions: declaring attackers, assigning blockers (incl. Done and
Menace), and the trample combat-damage-to-player row. legal(state)/
execute(state) factory pairs build_action_table (drl_env._actions_table)
calls once per matching creature/slot."""

import game

from ._actions_common import _GATE_NO_PENDING

_battlefield_lookup_cache = None  # (state, {(name, slot): Permanent}) -- valid only for the duration of one legal_action_mask sweep, same lifecycle as the mana-side caches in _actions_mana


def _cached_battlefield_lookup(state):
    """Sweep-scoped {(name, slot): Permanent} lookup for state.battlefield --
    same "profiled, not guessed" caching pattern as _cached_mana_ability_options
    just below: _attack_legal/
    _assign_blocker_legal each independently scanned the WHOLE battlefield
    with any(...) to find one specific (name, slot), once per action-table
    entry -- for a deck with many creature copies (boggles' Auras/tokens)
    that's O(action_table_size x battlefield_size) repeated work every
    sweep (profiled: 2 closures alone accounted for ~3.4M calls across a
    single 8192-step training burst). Building this dict once per sweep
    turns each of those checks into an O(1) lookup. Safe for the same
    reason _cached_mana_ability_options is: a legal_action_mask sweep only
    ever calls legal_fns, never an execute_* function, so state can't change
    mid-sweep. (name, slot) is a safe dict key here because it is unique
    per side -- state.battlefield is always ONE side's own,
    active-relative zone (see this module's other active-relative
    docstrings), never two players' permanents mixed in one sweep."""
    global _battlefield_lookup_cache
    if _battlefield_lookup_cache is None or _battlefield_lookup_cache[0] is not state:
        _battlefield_lookup_cache = (state, {(p.card_def.name, p.slot): p for p in state.battlefield})
    return _battlefield_lookup_cache[1]


def _attack_legal(name, slot):
    """Legal only during Phase.DECLARE_ATTACKERS, and only for the true
    turn owner (state.active_idx == state.turn_player_idx,
   ) -- declaring an attacker is a turn-based
    special action, not a priority action, so the non-turn player must
    never be allowed to declare one just because state.phase (a single
    shared field describing the TURN's phase) happens to match during
    their own priority window. And only if the specific physical
    permanent occupying this (name, slot)
    permanent-identity design -- is currently attack-eligible
    (game.creature_attack_eligible): untapped, and not summoning sick
    unless it has haste. Attacking stays fully optional: a model can leave
    any subset of eligible creatures back, Pass with zero attackers
    declared is still legal (same as always -- state.attackers simply
    starts, and can stay, empty for this turn).

    Gated _GATE_NO_PENDING (owner-confirmed): a genuine DECLARE_ATTACKERS
    window never coexists with an open pending_resolution -- anything an
    attack declaration itself triggers (e.g. "on attacks" abilities) gets
    its own later priority window, before declare-blockers, not during
    declare-attackers. So `pending is not None` already rules this out;
    the phase/turn-owner checks below still gate the (much more common)
    case where pending IS None but it isn't a legal attack window."""
    def legal(state):
        if state.phase is not game.turn.Phase.DECLARE_ATTACKERS:
            return False
        if state.active_idx != state.turn_player_idx:
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_attack_eligible(state, p)
    legal._pending_gate = _GATE_NO_PENDING
    return legal


def _attack_execute(name, slot):
    """Declares the specific physical permanent occupying this (name,
    slot) as an attacker -- exact-slot addressing lets a model distinguish
    an Aura-enchanted copy (different effective power) from a plain one of
    the same name."""
    def execute(state):
        permanent = next(
            p for p in state.battlefield
            if p.card_def.name == name and p.slot == slot and game.creature_attack_eligible(state, p)
        )
        game.declare_attacker(state, permanent)
    return execute


def _assign_blocker_legal(name, slot):
    """One "Assign Blocker: <name> (slot j)" action -- legal only while a "declare_blockers" resolution
    is pending (game.turn._declare_blockers_gen has already flipped
    state.active_idx to the defender by the time this is ever checked)
    and the specific physical permanent at this (name, slot) is currently
    block-eligible (game.creature_block_eligible): untapped, not already
    assigned to block something else this combat. Unlike attacking,
    neither summoning sickness nor Defender excludes a blocker -- see
    creature_block_eligible's own docstring for why."""
    def legal(state):
        pending = state.pending_resolution
        if pending is None or pending["kind"] != "declare_blockers":
            return False
        p = _cached_battlefield_lookup(state).get((name, slot))
        return p is not None and game.creature_block_eligible(state, p)
    legal._pending_gate = frozenset({"declare_blockers"})
    return legal


def _assign_blocker_execute(name, slot):
    """Parks the specific physical permanent at this (name, slot) as a
    blocker, then hands off to game.declare_blocker_assignment, which
    nests a cross-player choose_opponent_permanent sub-resolution to pick
    which of the attacker's declared, not-yet-blocked attackers it
    blocks -- restricted by extra_predicate to attackers this specific
    blocker is actually allowed to block. That predicate is game.can_block
    itself (resolution can't compute it, see declare_blocker_assignment's own
    docstring for why the predicate has to come from here instead), which is
    the SAME function game.creature_block_eligible uses to decide this action
    is legal at all. Once that completes, re-opens begin_declare_blockers
    (via the captured outer on_complete) so the defender can assign
    another blocker or choose Done -- same nested-callback shape
    execute_madness_cast already uses for its own multi-step chain.

    These two MUST be the same function. This used to inline a flying-only
    subset -- `not attacker.flying or blocker.flying` -- while
    creature_block_eligible consulted can_block, which also honors REACH and
    "can't be blocked except by creatures with flying". A reach blocker facing
    a flying attacker therefore passed the legality check and then matched no
    attacker under this predicate: the nested choice found zero candidates,
    recorded nothing, and re-opened declare-blockers on a byte-identical
    state, so the same action stayed legal forever. Observed 2026-08-19 as
    elves' Generous Ent (reach) against dmir_terror's Sneaky Snacker (flying):
    the declare-blockers round never ended, state.turn_number never advanced
    (so `horizon`, which bounds TURNS, never fired), and the rollout buffer
    grew ~1 GB/min until the collect worker died pickling its result."""
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
    # Menace (509.1c): can't FINISH a block declaration that leaves a menace
    # attacker blocked by exactly one creature -- the defender must add a
    # second blocker. No undo available (by design -- see
    # game.menace_block_incomplete's own docstring): if no second blocker is
    # available, this stays illegal and game.turn._declare_blockers_gen's own
    # zero-legal-actions check abandons the declaration, after which
    # combat.enforce_menace drops the illegal lone block at combat damage.
    # (This used to cite a "phase action cap" forcing completion. No such cap
    # exists -- corrected 2026-08-19.)
    return not game.menace_block_incomplete(state)


_done_blocking_legal._pending_gate = frozenset({"declare_blockers"})


def _done_blocking_execute(state):
    game.complete_resolution(state)


def _assign_damage_to_opponent_legal(state):
    """The trample "assign this combat-damage point to the defending player"
    action -- legal only during an assign_combat_damage resolution whose
    attacker HAS trample and still has points to assign (the blockers are
    the pointer half of this decision; this fixed action is the player
    half). NOT a targeting-prefixed name, so build_fixed_action_table keeps
    it in the fixed table rather than stripping it to the pointer scheme."""
    pending = state.pending_resolution
    return (
        pending is not None and pending["kind"] == "assign_combat_damage"
        and pending["has_trample"] and pending["remaining"] > 0
    )


_assign_damage_to_opponent_legal._pending_gate = frozenset({"assign_combat_damage"})


def _assign_damage_to_opponent_execute(state):
    game.execute_assign_combat_damage_to_player(state)


__all__ = [
    '_attack_legal',
    '_attack_execute',
    '_assign_blocker_legal',
    '_assign_blocker_execute',
    '_done_blocking_legal',
    '_done_blocking_execute',
    '_assign_damage_to_opponent_legal',
    '_assign_damage_to_opponent_execute',
    '_battlefield_lookup_cache',
    '_cached_battlefield_lookup',
]
