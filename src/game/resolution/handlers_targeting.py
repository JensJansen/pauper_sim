"""Choosing a target: one's own/opponent's permanent, any-target (creature or
player), a target player, or a spell on the stack -- the begin_/options/
execute_ trio per kind, plus refizzle_if_now_targetless (re-validates a
predicate-driven target against live battlefield state after a state-based
action). Re-exported via game.resolution so `from ..resolution import X` in
the catalogs keeps resolving."""

from ._core import begin_resolution, complete_resolution


def begin_choose_permanent(state, predicate, on_complete):
    """The model picks ONE of its own battlefield permanents, addressed by
    the exact (name, slot) it occupies, not by name alone: two same-named
    permanents stop being interchangeable the moment an Aura attaches to
    only one of them, or a caller needs the EXACT physical permanent it
    chose to still be there later (see cast_aura's own targeting contract).
    on_complete(state, (name, slot)_or_None) runs once decided. Same
    empty-options safety net as begin_search_fetch -- fizzles immediately
    with None if nothing matches."""
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
    battlefield, optionally a player), one at a time -- the BOARD analog of
    begin_choose_up_to_graveyard, for "up to N target creatures/permanents".
    Each pick excludes those already chosen by its (side, name, slot) -- slot
    distinguishes same-named copies, so two distinct same-named creatures are
    both reachable -- and is optional (declining ends the selection early).

    Runs on_complete(state, descriptors) with the 0..max_targets chosen target
    descriptors -- ("creature", side, name, slot) or ("player", idx). The caller
    captures each (casting.capture_any_target) as the ability is put on the
    stack, then at resolution acts on the still-legal captured targets, fully
    fizzling only if ALL are illegal (608.2c). Player targets aren't de-duped
    (no pool card needs up-to-N with repeatable player targets); add it if one
    arrives. See project_targeted_triggered_abilities."""
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
    """"Target player" -- addressed by index into state.players, not by
    name (unlike every other choose_* primitive here: a player isn't
    fungible-by-name the way two same-named cards are, and there's no
    other identifier to use). The active player themselves is ALWAYS a
    legal target -- a real Magic legality fact, "target player" never
    excludes its own caster -- so, unlike begin_choose_permanent/
    begin_search_fetch's own empty-battlefield/empty-library safety nets,
    this never auto-completes: at least one legal target (yourself)
    always exists, even alone in a 1-player game. Real, explicit choice
    every time, drl_env's own fixed "Target: yourself"/"Target: opponent"
    actions (the latter only legal once a second PlayerState actually
    exists) -- never a silently-assumed default. on_complete(state, idx)
    runs once chosen."""
    begin_resolution(state, "choose_target_player", on_complete)


def execute_choose_target_player_option(state, idx):
    complete_resolution(state, idx)


def begin_choose_any_target(state, predicate, on_complete, allow_players=True, optional=False):
    """A single target chosen from BOTH players' battlefields at once, plus
    (allow_players) either player -- real Magic's "any target" (a creature,
    a player, or a planeswalker/battle; this pool has no planeswalkers or
    battles). Burn (Lightning Bolt "3 damage to any target") uses
    allow_players=True; "target creature" effects that still span both
    sides (Pinnacle Kill-Ship's "up to one target creature", Quirion
    Ranger's "untap target creature") pass allow_players=False.

    Faithful targeting contract (same as casting.cast_aura, generalized to
    span sides and players): the caller resolves the returned descriptor to
    the EXACT object right now (cast/activation time) and captures it, then
    rechecks that exact object's legality when the spell/ability resolves
    off the stack -- fizzling if the chosen target is no longer legal.

    on_complete(state, target) where target is one of:
      ("player", idx)            -- a player, addressed by index
      ("creature", side, name, slot) -- a creature, addressed by the index
                                    of the player who controls it plus its
                                    own (name, slot); side disambiguates two
                                    same-named creatures on opposite battle-
                                    fields
      None                       -- only when allow_players=False AND no
                                    creature matches (an empty "up to one"
                                    or a can't-be-activated target choice);
                                    with allow_players=True a player is
                                    always legal, so None never happens.

    predicate(permanent) filters the creature half only; players are never
    filtered (a player is always a legal "any target").

    optional=True is "up to one target" (Pinnacle Kill-Ship): the chooser may
    decline (execute_choose_any_target_decline -> None) even when legal
    targets exist -- so this never auto-completes (the decline action is
    always available). optional=False auto-completes with None only when
    there's no legal target at all (allow_players=False and no creature)."""
    begin_resolution(
        state, "choose_any_target", on_complete, predicate=predicate, allow_players=allow_players, optional=optional,
    )
    if not optional and not choose_any_target_options(state):
        complete_resolution(state, None)  # allow_players=False and no legal creature -- nothing to target


def choose_any_target_creature_options(state):
    """The (side, name, slot) creature half of a choose_any_target -- every
    matching creature on EITHER battlefield. Split out from the player half
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
    safety-net check in begin_choose_any_target; the action layer consults
    the two halves separately."""
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
    with no target (the ability still resolves, doing nothing to a creature).
    Only offered when the pending was begun optional=True."""
    complete_resolution(state, None)


def begin_choose_opponent_permanent(state, predicate, on_complete):
    """Like begin_choose_permanent, but targets the OPPONENT's battlefield
    (state.opponent -- only meaningful in a 2-player game) instead of the
    active player's own -- the general cross-player targeting primitive
    that blocking uses. Addressed by (name, slot), same as
    begin_choose_permanent: two same-named OPPOSING permanents are not an
    arbitrary pick either -- an Aura-enchanted attacker and a plain one of
    the same name are not interchangeable for a blocker choosing between
    them. on_complete(state, (name, slot)_or_None) runs once decided. Same
    empty-options safety net as begin_choose_permanent/begin_search_fetch
    -- fizzles immediately with None if nothing matches.

    Only correct when called with the referencing player's own
    perspective actually active (state.active_idx) -- e.g. blocking's own
    defender-decision channel temporarily flips active_idx to the
    defender before this ever runs, exactly so state.opponent correctly
    means "the attacker" from the defender's point of view instead of
    leaking the defender's own hand as if it belonged to whoever was
    active a moment ago."""
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
    permanent, and choose_any_target's creature half (only when not optional
    and not allow_players, its own only all-False-capable configuration; see
    begin_choose_any_target's own docstring) -- the three kinds
    state_based_actions (a creature dying) can invalidate AFTER their own
    one-time open-time empty-options check already passed. Every other
    choose_*/search_fetch kind here reads the library/graveyard/hand/stack
    instead, none of which check_state_based_actions ever mutates, so they
    can't develop this same gap once past their own open-time check.

    Mirrors exactly the empty-options -> fizzle-with-None completion each of
    these three already performs at open time (608.2c: an effect with no
    legal target does nothing) -- same outcome, just re-checked on every
    subsequent decision point too, not only the first. Call this right alongside
    check_state_based_actions (the only thing that can invalidate these
    predicates mid-resolution) -- see game.turn._run_priority_round_gen.
    Returns True if it fizzled something (the pending resolution is now
    gone), else False."""
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
    spell entries (entries are the stack's own {"card_def","is_spell",...}
    dicts). POINTER-addressed (rl.decision.action_bridge), not by name: the spell
    being countered is very often the OPPONENT's, so no per-deck "Choose: X"
    row could ever represent it. Options are the matching stack ENTRIES
    themselves, by object identity, so two simultaneous same-named spells
    stay independently addressable too, not just the topmost of them.
    Fizzles immediately (on_complete
    (None)) if nothing matches -- though a counter spell's own extra_legal
    already requires a legal target to be cast at all."""
    begin_resolution(state, "choose_stack_target", on_complete, predicate=predicate)
    if not choose_stack_target_options(state):
        complete_resolution(state, None)


def choose_stack_target_options(state):
    """The matching stack entries themselves (objects), NOT names -- see
    begin_choose_stack_target's own docstring for why. No dedup, no sort:
    rl.decision.action_bridge masks/executes by object identity (id()-keyed, since a
    stack entry is an unhashable dict), not by name."""
    predicate = state.pending_resolution["predicate"]
    return [e for e in state.stack if e.get("is_spell") and predicate(e)]


def execute_choose_stack_target_option(state, entry):
    """`entry` is the exact chosen stack-entry object -- the on_complete
    consumer (_cast_counter) acts on that exact entry, so a specific spell
    among simultaneous same-named copies is reachable."""
    complete_resolution(state, entry)
