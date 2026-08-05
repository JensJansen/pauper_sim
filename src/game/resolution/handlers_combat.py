"""Combat-specific resolutions: declaring blockers (including gang-blocking)
and, for a multi-blocked attacker, the controller's own combat-damage-split
assignment. Re-exported via game.resolution so `from ..resolution import X`
in the catalogs keeps resolving."""

from ._core import begin_resolution, complete_resolution
from .handlers_targeting import begin_choose_opponent_permanent


def begin_declare_blockers(state, on_complete):
    """The defending player assigns 0+ of their own untapped creatures to
    block the active player's declared attackers, one assignment at a
    time -- each pairing an "Assign Blocker: <name> (slot j)" action
    (drl_env, picks one of THIS player's own untapped, not-yet-used
    creatures) with a nested begin_choose_opponent_permanent picking
    which of the attacker's declared attackers it blocks -- until the
    defender chooses Done. Gang-blocking IS allowed:
    many blockers may pile onto one attacker (each a separate "Assign
    Blocker" action, blocked_by[attacker] is a LIST). Still at most one
    attacker per blocker (a committed blocker isn't reassignable --
    creature_block_eligible excludes it), and no menace (nothing forces an
    attacker to be blocked by 2+).

    Only ever entered with state.active_idx already flipped to the
    defender (game.turn._declare_blockers_gen) -- state.battlefield/
    state.opponent below only mean the right thing once that's true; the
    hidden-information fix this whole mechanism depends on.

    Auto-completes immediately if the active player (the attacker, from
    the defender's own point of view) declared no attackers at all --
    nothing to block, same empty-options precedent as
    begin_choose_permanent/begin_search_fetch."""
    begin_resolution(state, "declare_blockers", on_complete)
    if not state.opponent.attackers:
        complete_resolution(state)


def declare_blocker_assignment(state, blocker, on_complete, extra_predicate=lambda p: True):
    """One "Assign Blocker: <name> (slot j)" action's actual effect
    (drl_env already picked the specific eligible `blocker` permanent):
    nests a begin_choose_opponent_permanent choosing which of the
    attacker's declared, not-yet-blocked attackers this blocker is
    assigned to (or None, if none remain -- shouldn't happen given the
    action's own legality check, but never crashes either way), appends the
    blocker to state.opponent.blocked_by[attacker] (a LIST -- gang-blocking),
    then calls
    on_complete -- which drl_env uses to re-open begin_declare_blockers
    so the defender can assign another blocker or finish.

    extra_predicate(attacker) -> bool: an additional restriction beyond
    "is a currently-unblocked attacker" -- e.g. flying's own blocking
    restriction. Supplied by the CALLER (drl_env)
    rather than computed here: this module stays effect-agnostic (see its
    own module docstring) and doesn't import game.effects.stats itself, so
    it has no way to ask "does this creature have flying" on its own.
    Defaults to "no extra restriction," unchanged
    from before this parameter existed -- a wasted "Assign Blocker" action
    (parking a blocker with nothing legal left for it to block, once this
    predicate is applied) just re-opens the consult with nothing recorded,
    same graceful no-op as the "no attackers left at all" case."""
    def _on_attacker_chosen(s, choice):
        if choice is not None:
            name, slot = choice
            attacker = next(p for p in s.opponent.attackers if p.card_def.name == name and p.slot == slot)
            s.opponent.blocked_by.setdefault(attacker, []).append(blocker)  # gang-blocking: one attacker, many blockers
            s.log_event(
                "block_assigned", blocker=(blocker.card_def.name, blocker.slot), attacker=(name, slot),
            )
        on_complete(s)

    # GANG-BLOCKING: an already-blocked attacker is STILL a legal choice --
    # multiple creatures may block the same attacker (the `p not in
    # blocked_by` exclusion that enforced 1-blocker-per-attacker is gone).
    # A blocker still blocks exactly one attacker (enforced by
    # creature_block_eligible, which drops a creature already committed).
    begin_choose_opponent_permanent(
        state,
        lambda p: p in state.opponent.attackers and extra_predicate(p),
        _on_attacker_chosen,
    )


def begin_assign_combat_damage(state, attacker, blockers, power, has_trample, on_complete):
    """A MULTI-blocked attacker's controller freely assigns `power` points
    of the attacker's combat damage across `blockers` -- any portion to any
    blocker, non-lethal allowed (user spec) -- plus to the defending player
    if the attacker has trample. One point at a time: assign_combat_damage_
    options (pick a blocker, (name, slot)-addressed for the pointer head) /
    execute_assign_combat_damage_option, or execute_assign_combat_damage_to_
    player (trample only, a fixed action), until every point is spent. The
    finished split is stashed on attacker.flags['combat_damage_split'] =
    ({blocker: amount}, opponent_amount) for combat_damage_step to apply --
    NOT passed through complete_resolution's own *args (which would try to
    log a Permanent-keyed dict, not serialisable). Only ever opened for 2+
    blockers -- a lone blocker has no choice (combat_damage_step auto-
    assigns). Auto-finishes immediately if power is 0."""
    begin_resolution(state, "assign_combat_damage", on_complete,
                     attacker=attacker, blockers=list(blockers), remaining=power, amounts={}, opponent=0,
                     has_trample=has_trample)
    if power <= 0:
        _finish_assign_combat_damage(state)


def _finish_assign_combat_damage(state):
    pending = state.pending_resolution
    pending["attacker"].flags["combat_damage_split"] = (dict(pending["amounts"]), pending["opponent"])
    complete_resolution(state)


def assign_combat_damage_options(state):
    """Every blocker is a choosable target for the next damage point (a
    blocker may take any number of points, so all stay offered until
    remaining hits 0), (name, slot)-addressed like choose_opponent_permanent
    for the pointer head. The trample 'assign to the player' option is a
    separate fixed action -- same choose-vs-fixed split blocking uses."""
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    return sorted((b.card_def.name, b.slot) for b in pending["blockers"])


def execute_assign_combat_damage_option(state, name, slot):
    pending = state.pending_resolution
    blocker = next(b for b in pending["blockers"] if b.card_def.name == name and b.slot == slot)
    pending["amounts"][blocker] = pending["amounts"].get(blocker, 0) + 1
    pending["remaining"] -= 1
    if pending["remaining"] <= 0:
        _finish_assign_combat_damage(state)


def execute_assign_combat_damage_to_player(state):
    """Trample: assign the next damage point to the defending player instead
    of a blocker. Only legal while has_trample (drl_env gates the action)."""
    pending = state.pending_resolution
    pending["opponent"] += 1
    pending["remaining"] -= 1
    if pending["remaining"] <= 0:
        _finish_assign_combat_damage(state)
