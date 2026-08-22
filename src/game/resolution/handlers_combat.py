"""Combat-specific resolutions: declaring blockers (including gang-blocking)
and, for a multi-blocked attacker, the controller's combat-damage-split
assignment."""

from ._core import begin_resolution, complete_resolution
from .handlers_targeting import begin_choose_opponent_permanent


def begin_declare_blockers(state, on_complete):
    """The defending player assigns 0+ of their own untapped creatures to
    block the active player's declared attackers, one assignment at a time
    -- each "Assign Blocker: <name> (slot j)" action pairs with a nested
    begin_choose_opponent_permanent picking which declared attacker it
    blocks -- until the defender chooses Done. Gang-blocking is allowed:
    many blockers may pile onto one attacker (blocked_by[attacker] is a
    LIST). Still at most one attacker per blocker, and no menace.

    Only ever entered with state.active_idx already flipped to the
    defender (game.turn._declare_blockers_gen) -- state.battlefield/
    state.opponent below only mean the right thing once that's true.

    Auto-completes immediately if the attacker declared no attackers at
    all -- nothing to block."""
    begin_resolution(state, "declare_blockers", on_complete)
    if not state.opponent.attackers:
        complete_resolution(state)


def declare_blocker_assignment(state, blocker, on_complete, extra_predicate=lambda p: True):
    """One "Assign Blocker: <name> (slot j)" action's actual effect: nests a
    begin_choose_opponent_permanent choosing which declared, not-yet-blocked
    attacker this blocker is assigned to (or None if none remain), appends
    the blocker to state.opponent.blocked_by[attacker] (a LIST --
    gang-blocking), then calls on_complete, which drl_env uses to re-open
    begin_declare_blockers.

    extra_predicate(attacker) -> bool: an additional restriction beyond "is
    a currently-unblocked attacker" (e.g. flying), supplied by the CALLER
    since this module stays effect-agnostic."""
    def _on_attacker_chosen(s, choice):
        if choice is not None:
            name, slot = choice
            attacker = next(p for p in s.opponent.attackers if p.card_def.name == name and p.slot == slot)
            s.opponent.blocked_by.setdefault(attacker, []).append(blocker)  # gang-blocking: one attacker, many blockers
            s.log_event(
                "block_assigned", blocker=(blocker.card_def.name, blocker.slot), attacker=(name, slot),
            )
        on_complete(s)

    # Gang-blocking: an already-blocked attacker is still a legal choice.
    begin_choose_opponent_permanent(
        state,
        lambda p: p in state.opponent.attackers and extra_predicate(p),
        _on_attacker_chosen,
    )


def begin_assign_combat_damage(state, attacker, blockers, power, has_trample, lethal_by_blocker, on_complete):
    """A MULTI-blocked attacker's controller assigns `power` points of
    combat damage across `blockers`, one point at a time
    (assign_combat_damage_options -> execute_assign_combat_damage_option,
    (name, slot)-addressed): any portion to any blocker up to its own
    lethal_by_blocker[blocker] cap, never more (no overkill). Once every
    blocker is at its cap, the remainder is forced, never an agent choice:
    it spills to the defending player if the attacker has trample
    (702.19b/510.1c), or piles onto the last blocker otherwise (see
    _autoresolve_if_no_choices_left). The finished split is stashed on
    attacker.flags['combat_damage_split'] = ({blocker: amount},
    opponent_amount) for combat_damage_step to apply -- NOT passed through
    complete_resolution's *args (a Permanent-keyed dict isn't loggable).
    Only ever opened for 2+ blockers; auto-finishes immediately if power is
    0 or every blocker's cap is already 0."""
    begin_resolution(state, "assign_combat_damage", on_complete,
                     attacker=attacker, blockers=list(blockers), remaining=power, amounts={}, opponent=0,
                     has_trample=has_trample, lethal_by_blocker=dict(lethal_by_blocker))
    _autoresolve_if_no_choices_left(state)


def _finish_assign_combat_damage(state):
    pending = state.pending_resolution
    pending["attacker"].flags["combat_damage_split"] = (dict(pending["amounts"]), pending["opponent"])
    complete_resolution(state)


def _autoresolve_if_no_choices_left(state):
    """Once every blocker still in the split has been assigned its own
    lethal_by_blocker cap, nothing about where the rest of this attacker's
    power goes is a real decision anymore -- so it's applied directly
    instead of asking the agent to spend the remaining points one at a
    time. Trample: the remainder spills to the defending player (702.19b).

    RULES EXCEPTION (owner-approved 2026-08-20), non-trample case: real
    Magic still requires that leftover be assigned to a blocking creature
    even though every blocker here is already at its own lethal cap and it
    changes nothing (that blocker is already dead). Rather than model which
    specific blocker "in reality" absorbs that dead-letter overkill, this
    engine piles all of it onto the last blocker in the split -- mirroring
    _default_damage_assignment's own identical exception on the auto path.
    Inert in practice (nothing in this card pool reads how much *excess*
    damage an already-dead blocker received), but flagged per this repo's
    rules-faithfulness mandate rather than silently assumed."""
    pending = state.pending_resolution
    if pending is None or pending["kind"] != "assign_combat_damage":
        return
    if pending["remaining"] <= 0:
        _finish_assign_combat_damage(state)
        return
    lethal_by_blocker = pending["lethal_by_blocker"]
    amounts = pending["amounts"]
    if any(amounts.get(b, 0) < lethal_by_blocker[b] for b in pending["blockers"]):
        return  # a real choice still remains -- wait for the next action
    if pending["has_trample"]:
        pending["opponent"] += pending["remaining"]
    else:
        last = pending["blockers"][-1]
        amounts[last] = amounts.get(last, 0) + pending["remaining"]
    pending["remaining"] = 0
    _finish_assign_combat_damage(state)


def assign_combat_damage_options(state):
    """Every blocker still under its own lethal_by_blocker cap, (name,
    slot)-addressed. No separate "assign to the player" option:
    trample-through is never a choice, see _autoresolve_if_no_choices_left."""
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    lethal_by_blocker = pending["lethal_by_blocker"]
    amounts = pending["amounts"]
    return sorted(
        (b.card_def.name, b.slot) for b in pending["blockers"] if amounts.get(b, 0) < lethal_by_blocker[b]
    )


def execute_assign_combat_damage_option(state, name, slot):
    pending = state.pending_resolution
    blocker = next(b for b in pending["blockers"] if b.card_def.name == name and b.slot == slot)
    pending["amounts"][blocker] = pending["amounts"].get(blocker, 0) + 1
    pending["remaining"] -= 1
    _autoresolve_if_no_choices_left(state)
