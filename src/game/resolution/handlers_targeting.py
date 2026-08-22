"""Choosing a target: one's own/opponent's permanent, any-target (creature or
player), a target player, or a spell on the stack -- the begin_/options/
execute_ trio per kind, plus refizzle_if_now_targetless (re-validates a
predicate-driven target against live battlefield state after a state-based
action)."""

from ._core import begin_resolution, complete_resolution


def begin_choose_permanent(state, predicate, on_complete):
    """The model picks ONE of its own battlefield permanents, addressed by
    the exact (name, slot) it occupies, not by name alone (two same-named
    permanents stop being interchangeable once an Aura attaches to only
    one). on_complete(state, (name, slot)_or_None); fizzles with None if
    nothing matches."""
    begin_resolution(state, "choose_permanent", on_complete, predicate=predicate)
    if not choose_permanent_options(state):
        complete_resolution(state, None)


def choose_permanent_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted((p.card_def.name, p.slot) for p in state.battlefield if predicate(p))


def execute_choose_permanent_option(state, name, slot):
    complete_resolution(state, (name, slot))


def begin_choose_up_to_any_target(state, predicate, max_targets, on_complete, allow_players=False):
    """Choose UP TO max_targets DISTINCT any-targets (a creature on EITHER
    battlefield, optionally a player), one at a time -- the board analog of
    begin_choose_up_to_graveyard. Each pick excludes those already chosen by
    (side, name, slot), and is optional (declining ends selection early).

    Runs on_complete(state, descriptors) with the 0..max_targets chosen
    target descriptors -- ("creature", side, name, slot) or ("player", idx).
    The caller captures each as the ability is put on the stack, then at
    resolution acts on the still-legal captured targets, fully fizzling
    only if ALL are illegal (608.2c)."""
    chosen = []

    def _already(p):
        side = next(i for i, pl in enumerate(state.players) if p in pl.battlefield)
        key = (side, p.card_def.name, p.slot)
        return any(d is not None and d[0] == "creature" and (d[1], d[2], d[3]) == key for d in chosen)

    def _step():
        if len(chosen) >= max_targets:
            on_complete(state, chosen)
            return

        def _picked(state, descriptor):
            if descriptor is None:  # declined -> stop early (the "up to" slack)
                on_complete(state, chosen)
                return
            chosen.append(descriptor)
            _step()

        begin_choose_any_target(
            state, lambda p: predicate(p) and not _already(p), _picked,
            allow_players=allow_players, optional=True,
        )

    _step()


def begin_choose_target_player(state, on_complete):
    """"Target player" -- addressed by index into state.players. The active
    player is always a legal target (real Magic never excludes its own
    caster), so this never auto-completes. Backed by drl_env's fixed
    "Target: yourself"/"Target: opponent" actions. on_complete(state, idx)."""
    begin_resolution(state, "choose_target_player", on_complete)


def execute_choose_target_player_option(state, idx):
    complete_resolution(state, idx)


def begin_choose_any_target(state, predicate, on_complete, allow_players=True, optional=False):
    """A single target chosen from BOTH players' battlefields at once, plus
    (allow_players) either player -- real Magic's "any target". Burn
    (Lightning Bolt) uses allow_players=True; "target creature" effects
    spanning both sides (Pinnacle Kill-Ship, Quirion Ranger) pass
    allow_players=False.

    Faithful targeting contract: the caller resolves the returned
    descriptor to the exact object at cast/activation time and captures it,
    then rechecks its legality at resolution, fizzling if no longer legal.

    on_complete(state, target) where target is one of:
      ("player", idx)                -- a player, addressed by index
      ("creature", side, name, slot) -- a creature, addressed by the
                                         controlling player's index plus
                                         its own (name, slot)
      None                           -- only when allow_players=False and no
                                         creature matches

    predicate(permanent) filters the creature half only.

    optional=True is "up to one target" (Pinnacle Kill-Ship): may decline
    (execute_choose_any_target_decline -> None) even when legal targets
    exist. optional=False auto-completes with None only when there's no
    legal target at all."""
    begin_resolution(
        state, "choose_any_target", on_complete, predicate=predicate, allow_players=allow_players, optional=optional,
    )
    if not optional and not choose_any_target_options(state):
        complete_resolution(state, None)  # allow_players=False and no legal creature -- nothing to target


def choose_any_target_creature_options(state):
    """The (side, name, slot) creature half of a choose_any_target -- every
    matching creature on either battlefield. Split out from the player half
    so the action layer can route creatures through the identity pointer
    scheme (rl.decision.action_bridge) and players through fixed actions."""
    predicate = state.pending_resolution["predicate"]
    return sorted(
        (side, p.card_def.name, p.slot)
        for side, player in enumerate(state.players)
        for p in player.battlefield
        if predicate(p)
    )


def choose_any_target_options(state):
    """Both halves together -- creatures (side, name, slot) plus, when
    allow_players, each player ("player", idx). Used for the empty-options
    safety-net check in begin_choose_any_target."""
    options = [("creature", side, name, slot) for side, name, slot in choose_any_target_creature_options(state)]
    if state.pending_resolution["allow_players"]:
        options += [("player", idx) for idx in range(len(state.players))]
    return options


def execute_choose_any_target_creature(state, side, name, slot):
    complete_resolution(state, ("creature", side, name, slot))


def execute_choose_any_target_player(state, idx):
    complete_resolution(state, ("player", idx))


def execute_choose_any_target_decline(state):
    """Decline an "up to one target" (optional) choose_any_target -- resolve
    with no target. Only offered when the pending was begun optional=True."""
    complete_resolution(state, None)


def begin_choose_opponent_permanent(state, predicate, on_complete):
    """Like begin_choose_permanent, but targets the OPPONENT's battlefield
    (state.opponent) -- the general cross-player targeting primitive that
    blocking uses. Addressed by (name, slot). on_complete(state, (name,
    slot)_or_None); fizzles with None if nothing matches.

    Only correct with the referencing player's own perspective already
    active (state.active_idx) -- blocking flips active_idx to the defender
    before this runs, so state.opponent correctly means "the attacker"."""
    begin_resolution(state, "choose_opponent_permanent", on_complete, predicate=predicate)
    if not choose_opponent_permanent_options(state):
        complete_resolution(state, None)


def choose_opponent_permanent_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted((p.card_def.name, p.slot) for p in state.opponent.battlefield if predicate(p))


def execute_choose_opponent_permanent_option(state, name, slot):
    complete_resolution(state, (name, slot))


def refizzle_if_now_targetless(state):
    """Re-validates a pending resolution whose legal options depend on LIVE
    battlefield state via a predicate -- choose_permanent, choose_opponent_
    permanent, and choose_any_target's creature half (see begin_choose_any_
    target) -- the three kinds state_based_actions (a creature dying) can
    invalidate after their one-time open-time empty-options check already
    passed. Mirrors that same fizzle-with-None completion (608.2c). Call
    alongside check_state_based_actions -- see game.turn.
    _run_priority_round_gen. Returns True if it fizzled something."""
    pending = state.pending_resolution
    if pending is None:
        return False
    kind = pending["kind"]
    if kind == "choose_permanent" and not choose_permanent_options(state):
        complete_resolution(state, None)
        return True
    if kind == "choose_opponent_permanent" and not choose_opponent_permanent_options(state):
        complete_resolution(state, None)
        return True
    if (kind == "choose_any_target" and not pending["optional"] and not pending["allow_players"]
            and not choose_any_target_options(state)):
        complete_resolution(state, None)
        return True
    return False


def begin_choose_stack_target(state, predicate, on_complete):
    """Choose a SPELL on the stack (Counterspell: any; Dispel: instant; Spell
    Pierce: noncreature) to counter. `predicate(entry)` further narrows the
    spell entries. Pointer-addressed, not by name, since the spell being
    countered is often the OPPONENT's. Options are the matching stack
    entries themselves, by object identity. Fizzles (on_complete(None)) if
    nothing matches."""
    begin_resolution(state, "choose_stack_target", on_complete, predicate=predicate)
    if not choose_stack_target_options(state):
        complete_resolution(state, None)


def choose_stack_target_options(state):
    """The matching stack entries themselves (objects), not names. No
    dedup, no sort: rl.decision.action_bridge masks/executes by object
    identity (id()-keyed, since a stack entry is an unhashable dict)."""
    predicate = state.pending_resolution["predicate"]
    return [e for e in state.stack if e.get("is_spell") and predicate(e)]


def execute_choose_stack_target_option(state, entry):
    """`entry` is the exact chosen stack-entry object; the on_complete
    consumer (_cast_counter) acts on that exact entry."""
    complete_resolution(state, entry)
