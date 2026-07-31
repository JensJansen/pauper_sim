"""Every deck-agnostic pending-resolution handler (search_fetch,
choose_permanent, scry/surveil, discard, sacrifice, combat-damage
assignment, ...): each kind's begin_/options/execute_ trio, reused across
card implementations rather than duplicated per card. Re-exported via
game.resolution so `from ..resolution import X` in the catalogs keeps
resolving."""

from .. import registry
from ..cards import CardDef, CardType
from ._core import begin_resolution, complete_resolution


def begin_search_fetch(state, predicate, on_complete, optional=False):
    """The model picks ONE library card by name, among distinct names
    currently matching `predicate`, to fetch -- one action per matching
    name (search_fetch_options), not a full reveal (search effects look at
    the whole library, already-known information by elimination, not a
    scry-style reveal of previously-hidden cards). on_complete(state,
    chosen_name) runs once decided. If nothing in the library matches
    right now (legality only guarantees the *cost* was payable, not that a
    target still exists -- e.g. every land could already be drawn),
    fizzles immediately with chosen_name=None instead of leaving a
    resolution with zero legal options.

    optional=True (Gatecreeper Vine's "may search"; Expedition Map/Crop
    Rotation's mandatory fetches leave this False, unchanged) offers a
    dedicated decline via the environment's own action, not folded into
    search_fetch_options' name list -- same treatment Ancient Stirrings'
    decline already gets."""
    begin_resolution(state, "search_fetch", on_complete, predicate=predicate, optional=optional)
    if not search_fetch_options(state):
        complete_resolution(state, None)


def search_fetch_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted({c.name for c in state.library if predicate(c)})


def execute_search_fetch_option(state, name):
    complete_resolution(state, name)


def execute_search_fetch_decline(state):
    complete_resolution(state, None)


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


def begin_choose_graveyard_card(state, predicate, on_complete, graveyard=None, optional=False):
    """Pick ONE card from a graveyard, among those matching predicate --
    used by Dread Return's reanimation target (game.catalog.black_cards) and
    by Relic of Progenitus' own repeatable exile ability alike; living here
    rather than in a deck-specific catalog file is what lets both share the
    identical primitive.

    The pick is BY OBJECT IDENTITY: on_complete receives the exact chosen
    object (a CardInstance for a real graveyard -- so two same-named copies are
    distinct and individually reachable; a CardDef for the DEFERRED
    Mesmeric-Fiend hand-reveal path, where interned same-named copies are still
    indistinguishable until hand instances land). Same empty-options safety net
    as begin_search_fetch/begin_choose_permanent.

    graveyard=None defaults to state.graveyard (the active player's own,
    via the active-idx proxy) -- Dread Return's reanimation target is
    always its own controller's graveyard, never anyone else's. Pass an
    explicit graveyard list to target a DIFFERENT player's graveyard
    instead -- Relic of Progenitus' own ability can target either player.
    The card is picked by whoever is the current active player: when the
    RULES say the TARGET player chooses (Relic targeting the opponent), the
    caller flips active_idx to that player first and restores it after (see
    activate_relic_of_progenitus_exile), so the correct player -- and, in
    training, their own net -- makes the choice. (Where the CASTER is the one
    who chooses, e.g. Mesmeric Fiend picking from a revealed hand, no flip is
    needed.)

    optional=True is a "you may exile a card from your graveyard" choice
    (Masked Vandal's ETB): the model may decline outright even when legal
    cards exist (execute_choose_graveyard_card_decline -> None), the same
    dedicated-decline treatment begin_search_fetch's own optional already
    gets. optional=False (Dread Return, Relic -- unchanged) is mandatory
    when any card matches; either way the empty-options safety net still
    completes with None when nothing matches at all."""
    if graveyard is None:
        graveyard = state.graveyard
    begin_resolution(state, "choose_graveyard_card", on_complete, predicate=predicate, graveyard=graveyard, optional=optional)
    if not choose_graveyard_card_options(state):
        complete_resolution(state, None)


def choose_graveyard_card_options(state):
    """The matching graveyard cards themselves (objects), NOT names -- so two
    same-named copies are DISTINCT choices (a real graveyard holds CardInstances;
    the deferred Mesmeric-Fiend hand-reveal path holds CardDefs). No dedup, no
    sort: rl.action_bridge masks/executes by object identity (the token carries
    the same object), not by name."""
    pending = state.pending_resolution
    return [c for c in pending["graveyard"] if pending["predicate"](c)]


def execute_choose_graveyard_card_option(state, card):
    """`card` is the exact chosen object (a CardInstance from a graveyard, or a
    CardDef from the deferred hand-reveal path) -- the on_complete consumer acts
    on that exact object, so a specific copy among same-named duplicates is
    reachable."""
    complete_resolution(state, card)


def execute_choose_graveyard_card_decline(state):
    """Decline an OPTIONAL choose_graveyard_card (Masked Vandal's "you may
    exile a creature card from your graveyard") -- resolve with None, exiling
    nothing. Only ever offered by the environment while the pending was begun
    optional=True, same convention as execute_search_fetch_decline."""
    complete_resolution(state, None)


def begin_choose_cast_copy(state, name, on_complete, reserved_cost=None):
    """WHICH physical copy of `name` in your own graveyard is being cast (or
    having its graveyard ability activated) -- Flashback/Escape/graveyard-ability.

    Real Magic 601.2a: the object being cast is chosen when the spell is
    ANNOUNCED, before costs are paid, and with two same-named cards in a
    graveyard that is a real choice the PLAYER makes. It is only observable when
    something else references one specific copy -- but then it matters a great
    deal: a Rooftop Percher trigger on the stack targets copy A by object
    identity, and casting copy A in response removes it, making Percher's target
    illegal so that target fizzles (608.2b/c), while casting copy B leaves A to
    be exiled. Both are legal lines; the agent has to be able to aim.

    MANDATORY -- no decline (the caster already committed to casting by taking
    the flashback/activate action, whose own legality already required a
    same-named card here). That is why this is a distinct kind rather than a
    reuse of choose_graveyard_card, whose optional-decline machinery would be
    permanently-illegal noise; it also keeps this a POINTER-only pending that
    adds ZERO fixed action rows (see rl.action_bridge), so no deck's action-space
    width changes.

    reserved_cost: the mana cost (a plain {color: n} dict, or None) that
    on_complete is ABOUT to pay via begin_pay_cost once a copy is chosen --
    known in full up front (drl_env._actions._graveyard_ability_execute/
    _flashback_execute already read it off the registry/card_def before this
    ever opens). Stashed on the pending purely so drl_env._actions._filter_
    would_strand_payment can protect it: mana abilities/filters stay legal
    "in ANY priority window" (605.1a/605.3b) DURING this choice, same as
    during an already-open pay_cost, so the agent could otherwise filter away
    the exact colored pip this ability is guaranteed to need one step later,
    leaving pay_cost to open already unpayable with an all-False mask. Only
    choose_cast_copy needs this: every other pointer
    pending in this pool (choose_permanent, choose_any_target, ...) either
    can't precede a cost payment at all, or (choose_graveyard_card feeding
    Dread Return's own creature-sacrifice cost) pays with a non-mana resource
    a filter can't touch.

    Always the caster's OWN graveyard (state.graveyard, active-idx proxied) --
    no card in this pool casts from an opponent's graveyard. on_complete receives
    the exact chosen CardInstance. Callers only open this when 2+ copies exist;
    with one copy there is no choice to make (drl_env._actions._graveyard_instance
    resolves it directly)."""
    begin_resolution(state, "choose_cast_copy", on_complete, name=name, reserved_cost=reserved_cost)


def choose_cast_copy_options(state):
    """The matching graveyard INSTANCES themselves (objects), not names -- the
    whole point is telling same-named copies apart. No dedup, no sort:
    rl.action_bridge masks/executes by object identity, and the observation
    token for each instance carries that same object (plus a
    targeted_by_mine/targeted_by_theirs bit -- which is what makes "cast the
    copy they're pointing at" a LEARNABLE choice rather than an invisible
    coin flip)."""
    pending = state.pending_resolution
    return [c for c in state.graveyard if c.name == pending["name"]]


def execute_choose_cast_copy_option(state, card):
    """`card` is the exact chosen graveyard CardInstance -- the on_complete
    consumer (a flashback/escape/graveyard-ability execute closure) proceeds to
    pay costs and resolve using that exact object."""
    complete_resolution(state, card)


def begin_choose_up_to_graveyard(state, predicate, max_targets, on_complete, graveyard=None):
    """Choose UP TO max_targets DISTINCT cards from a graveyard (default the
    active player's; pass a combined list for "from graveyardS"), one at a time.
    Each pick excludes those already chosen BY OBJECT IDENTITY -- so two
    same-named copies are both reachable -- and is optional: declining ends the
    selection early (the "up to" slack), as does running out of eligible cards.
    Runs on_complete(state, chosen_instances) with the 0..max_targets chosen
    instances (a LIST). The N-ary multi-target primitive behind "exile up to N
    target cards from graveyard(s)" (Rooftop Percher); pairs with a resolution
    that acts on still-legal captured instances and fully fizzles only if all
    are gone (608.2c). See project_targeted_triggered_abilities."""
    gy = state.graveyard if graveyard is None else graveyard
    chosen = []

    def _step():
        if len(chosen) >= max_targets:
            on_complete(state, chosen)
            return

        def _picked(state, card):
            if card is None:  # declined, or no eligible card left -> stop early
                on_complete(state, chosen)
                return
            chosen.append(card)
            _step()

        begin_choose_graveyard_card(
            state, lambda c: predicate(c) and c not in chosen, _picked, graveyard=gy, optional=True,
        )

    _step()


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
    scheme (rl.action_bridge) and players through fixed actions."""
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


def explore(state, creature):
    """A creature explores (Map token, Fanatical Offering): reveal the top
    card of the exploring player's library -- a land goes to hand; a nonland
    puts a +1/+1 counter on the creature, then "you may put that card into
    your graveyard" (keep on top or bin), which is exactly surveil 1 on that
    same still-on-top card. Empty library: nothing to reveal, no-op."""
    if not state.library:
        return
    top = state.library[0]
    if top.card_type == CardType.LAND:
        state.library.pop(0)
        state.hand.append(top)
        state.log_event("explore", card=top.name, result="land_to_hand", creature=(creature.card_def.name, creature.slot))
    else:
        creature.counters["+1/+1"] = creature.counters.get("+1/+1", 0) + 1
        state.log_event("explore", card=top.name, result="plus_counter", creature=(creature.card_def.name, creature.slot))
        surveil(state, 1)  # "you may put it into your graveyard" == surveil 1 (keep on top or graveyard)


def begin_exile_n_from_graveyard(state, n, on_complete, predicate=None):
    """Exile n cards from the active player's own graveyard as a COST, the
    model choosing which one at a time (chained begin_choose_graveyard_card).
    predicate (default any) narrows eligible cards. Used by Delve (exile N to
    pay {N} of a spell's generic cost, Gurmag Angler) and reusable by any
    other "exile N from your graveyard" cost. Runs on_complete(state) once n
    are exiled (or eligible cards run out)."""
    pred = predicate or (lambda c: True)

    def _step(remaining):
        if remaining <= 0 or not any(pred(c) for c in state.graveyard):
            on_complete(state)
            return

        def _chosen(state, chosen):
            state.graveyard.remove(chosen)  # the exact chosen instance; exiled, untracked
            state.log_event("zone_move", card=chosen.name, from_zone="graveyard", to_zone="exile_untracked", reason="exile_cost")
            _step(remaining - 1)

        begin_choose_graveyard_card(state, pred, _chosen)

    _step(n)


def begin_tuck_to_library(state, card_def, owner_idx, on_complete=None):
    """Deem Inferior: a permanent's OWNER (active_idx flipped to them) puts
    its card into their library second-from-the-top or on the bottom -- their
    choice, backed by two drl_env actions ("Tuck: 2nd from top" / "Tuck:
    bottom"). The permanent must already be removed from the battlefield by
    the caller."""
    original = state.active_idx
    state.active_idx = owner_idx
    begin_resolution(
        state, "tuck_position", on_complete or (lambda s: None),
        tuck_card=card_def, owner_idx=owner_idx, original_idx=original,
    )


def execute_tuck_position(state, position):
    pending = state.pending_resolution
    card_def = pending["tuck_card"]
    owner = state.players[pending["owner_idx"]]
    original = pending["original_idx"]
    on_complete = pending["on_complete"]
    state.pending_resolution = None
    state.active_idx = original
    if position == "top2":
        owner.library.insert(min(1, len(owner.library)), card_def)  # second from the top
    else:
        owner.library.append(card_def)  # bottom
    state.log_event("tuck", card=card_def.name, position=position, owner_idx=pending["owner_idx"])
    on_complete(state)


def begin_may_transform(state, permanent, revealed_card=None):
    """A "you may transform this creature" choice (Delver of Secrets, once an
    instant/sorcery is revealed). Two drl_env actions -- "Transform" /
    "Don't transform" -- back it; execute_may_transform applies or skips.
    revealed_card: the name of the top-of-library card the trigger looked at
    (the instant/sorcery that enables the transform) -- logged as the reveal
    iff the player actually reveals it, i.e. transforms (Delver's "may reveal"
    and "if revealed, transform" collapse to one choice in this engine)."""
    begin_resolution(state, "may_transform", lambda s: None, permanent=permanent, revealed_card=revealed_card)


def execute_may_transform(state, do_transform):
    pending = state.pending_resolution
    permanent = pending["permanent"]
    if do_transform:
        # Revealing the top card is what enables (and, for an I/S, forces) the
        # transform -- so a transform IS a reveal. Log it before the flip.
        revealed = pending.get("revealed_card")
        if revealed is not None:
            state.log_event("reveal", card=revealed, source=(permanent.card_def.name, permanent.slot),
                            from_zone="library", reason="delver")
        # Flip to the back face: set the marker flag (effects.stats reads it for the
        # 3/2-flying stats) AND swap the Permanent's own card_def to the back face, so
        # the game state's identity -- name, every log event, RL perception -- becomes
        # Insectile Aberration, not Delver. front_card_def is kept so the permanent
        # reverts to its front face if it later leaves the battlefield (a DFC is only
        # its back face while on the battlefield).
        spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("transform", {})
        from_name = permanent.card_def.name
        permanent.flags["transformed"] = True
        back = spec.get("card_def")
        if back is not None:
            permanent.flags["front_card_def"] = permanent.card_def
            permanent.card_def = back
        state.log_event("transform", permanent=(from_name, permanent.slot), to_card=permanent.card_def.name,
                        power=spec.get("power"), toughness=spec.get("toughness"))
    complete_resolution(state)


def begin_may_copy(state, on_complete):
    """A "you may copy this spell" choice (Chain Lightning's rider, after the
    {R}{R} has already been paid -- the second, independent "may" in "they may
    pay {R}{R}. If the player does, they may copy this spell"). on_complete(
    state, do_copy: bool)."""
    begin_resolution(state, "may_copy", on_complete)


def execute_may_copy(state, do_copy):
    complete_resolution(state, do_copy)


def begin_may_cast(state, on_complete):
    """A "you may cast this card [without paying its mana cost]" choice --
    Cascade's own may-cast of the hit card (Maelstrom Colossus). Two drl_env
    actions -- "Cast (may)" / "Decline (may)" -- back it. It is the caster's
    own decision (no active_idx flip). on_complete(state, do_cast: bool)."""
    begin_resolution(state, "may_cast", on_complete)


def execute_may_cast(state, do_cast):
    complete_resolution(state, do_cast)


def begin_choose_room(state, options, on_complete):
    """Undercity venture: the venturing player picks which of `options` (the
    1 or 2 rooms the current room leads to) to enter next. on_complete(state,
    room_name) fires with the chosen room. Effect-agnostic here -- the actual
    room entry + its effect live in game.effects.undercity."""
    begin_resolution(state, "choose_room", on_complete, options=tuple(options))


def choose_room_options(state):
    return list(state.pending_resolution["options"])


def execute_choose_room_option(state, name):
    complete_resolution(state, name)


_ANY_COLORS = ("W", "U", "B", "R", "G")  # "any color" -- the five colors, never colorless {C}


def begin_choose_mana_color(state, on_complete):
    """"Add one mana of any color" (Chromatic Star): the activating player picks
    one of the five colors. A mana ability, so it resolves immediately (no
    stack); on_complete(state, color) is what actually adds the mana. Always
    has five legal options, so it never softlocks."""
    begin_resolution(state, "choose_mana_color", on_complete)


def choose_mana_color_options(state):
    return list(_ANY_COLORS)


def execute_choose_mana_color(state, color):
    complete_resolution(state, color)


def begin_choose_cast_mode(state, card_def, modes, on_complete):
    """A modal spell's own mode choice (601.2b: chosen as the spell is cast,
    before its total cost is even calculated) -- shared "Mode 1".."Mode 5"
    drl_env buttons, capped per-card by len(modes), instead of a per-card,
    per-mode fixed-table row. Covers both cast_modes (Winding Way: no X) and
    x_cast_modes (Nyxborn Hydra: mode chosen first, X chosen next via
    begin_choose_cast_x). modes: tuple of (extra_legal_or_None,
    afford_check) pairs, one per mode in this card's own registry order --
    afford_check(state) -> bool reports whether THIS mode is currently
    payable at all (a fixed cost for cast_modes, "some X in range is
    affordable" for x_cast_modes), letting the shared buttons mask per-mode
    without a per-card row. on_complete(state, mode_index) fires with the
    chosen zero-based index into that card's own registry mode order."""
    begin_resolution(state, "choose_cast_mode", on_complete, card_def=card_def, modes=modes)


def choose_cast_mode_options(state):
    return list(range(len(state.pending_resolution["modes"])))


def execute_choose_cast_mode_option(state, mode_index):
    complete_resolution(state, mode_index)


def begin_choose_cast_x(state, base_cost, max_x, on_complete):
    """An X-cost spell's own X value (601.2f: determined before the total
    cost is paid) -- shared "X=0".."X=10" drl_env buttons, capped per-card/
    mode by max_x, instead of a per-(mode,X) fixed-table row. base_cost is
    this mode's own cost before X's generic is added; each shared button
    computes base_cost+n itself to check affordability -- the exact
    per-value masking a flat per-row enumeration already did, just
    re-encoded as a resolution. on_complete(state, x) fires with the
    chosen X."""
    begin_resolution(state, "choose_cast_x", on_complete, base_cost=base_cost, max_x=max_x)


def choose_cast_x_options(state):
    return list(range(state.pending_resolution["max_x"] + 1))


def execute_choose_cast_x_option(state, x):
    complete_resolution(state, x)


def begin_choose_delve_amount(state, card_def, max_n, on_complete):
    """Delve's own exile-amount choice (702.66) -- shared "Delve 0".."Delve
    6" drl_env buttons, capped per-card by max_n, instead of a per-N
    fixed-table row. Opens BEFORE the existing exile sub-cost
    (begin_exile_n_from_graveyard) and the reduced-cost payment, matching
    real sequencing: choose how much to delve, then pay accordingly.
    card_def lets each shared button check both graveyard size and the
    reduced cost's affordability. on_complete(state, n) fires with the
    chosen amount."""
    begin_resolution(state, "choose_delve_amount", on_complete, card_def=card_def, max_n=max_n)


def choose_delve_amount_options(state):
    return list(range(state.pending_resolution["max_n"] + 1))


def execute_choose_delve_amount_option(state, n):
    complete_resolution(state, n)


def begin_mana_subdecision(state, source, target_predicate):
    """Opens a gate-free mana ability's own multi-step choice (currently:
    Saruli Caretaker's "tap another creature, then choose a color") --
    deliberately NOT begin_resolution/state.pending_resolution (a single
    slot that would be clobbered; see state.mana_subdecision's own
    docstring for the full reasoning). No on_complete callback -- unlike
    pending_resolution, this mechanism has exactly one, fixed two-stage
    shape and always ends the same way (execute_mana_subdecision_color taps
    the source and produces mana), so there's nothing for a caller to
    customize."""
    state.mana_subdecision = {
        "stage": "choose_target", "source": source, "target_predicate": target_predicate, "target": None,
    }
    state.log_event("mana_subdecision_begin", source=(source.card_def.name, source.slot))


def execute_mana_subdecision_target(state, target):
    """Records the chosen tap target and advances to the color stage --
    mutates the SAME dict in place (not complete_resolution-style clear +
    fire), since source/target are still needed for the final step."""
    sub = state.mana_subdecision
    sub["target"] = target
    sub["stage"] = "choose_color"
    state.log_event("mana_subdecision_target", target=(target.card_def.name, target.slot))


def execute_mana_subdecision_color(state, color):
    """Closes the sub-decision, THEN taps the target and produces mana --
    clearing state.mana_subdecision before firing effects mirrors
    complete_resolution's own "clear pending before on_complete" order.
    Tap-target-then-activate-source order matches exactly what the old
    single-atomic-row implementation did, preserving identical observable
    behavior."""
    from ..mana import activate_mana_source  # call-time import -- mana imports resolution, see pay_unless_pay's own comment

    sub = state.mana_subdecision
    source, target = sub["source"], sub["target"]
    state.mana_subdecision = None
    target.tapped = True
    state.log_event("mana_subdecision_complete", color=color)
    activate_mana_source(state, source, color)


def begin_throne_reveal(state, n, on_complete):
    """Undercity's Throne of the Dead Three: reveal the top `n` library cards;
    the venturer picks one CREATURE card from among them
    (throne_reveal_options). Every exit path (a pick, or the empty-reveal
    auto-complete when no creature is revealed) returns the unchosen revealed
    cards to the library and shuffles it -- done here so the library is always
    left consistent; on_complete(state, chosen_card_def | None) then only has
    to place the chosen creature (battlefield + counters + hexproof, in
    game.effects.undercity)."""
    revealed = state.library[:n]
    del state.library[:n]
    begin_resolution(state, "throne_reveal", on_complete, revealed=revealed)
    if not throne_reveal_options(state):
        _finish_throne(state, None)  # no creature among the revealed cards


def throne_reveal_options(state):
    return sorted({c.name for c in state.pending_resolution["revealed"] if c.card_type == CardType.CREATURE})


def execute_throne_reveal_option(state, name):
    chosen = next(c for c in state.pending_resolution["revealed"] if c.name == name and c.card_type == CardType.CREATURE)
    _finish_throne(state, chosen)


def _finish_throne(state, chosen):
    revealed = state.pending_resolution["revealed"]
    rest = [c for c in revealed if c is not chosen] if chosen is not None else list(revealed)
    state.library.extend(rest)
    state.rng.shuffle(state.library)
    complete_resolution(state, chosen)


def begin_choose_stack_target(state, predicate, on_complete):
    """Choose a SPELL on the stack (Counterspell: any; Dispel: instant; Spell
    Pierce: noncreature) to counter. `predicate(entry)` further narrows the
    spell entries (entries are the stack's own {"card_def","is_spell",...}
    dicts). POINTER-addressed (rl.action_bridge), not by name: the spell
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
    rl.action_bridge masks/executes by object identity (id()-keyed, since a
    stack entry is an unhashable dict), not by name."""
    predicate = state.pending_resolution["predicate"]
    return [e for e in state.stack if e.get("is_spell") and predicate(e)]


def execute_choose_stack_target_option(state, entry):
    """`entry` is the exact chosen stack-entry object -- the on_complete
    consumer (_cast_counter) acts on that exact entry, so a specific spell
    among simultaneous same-named copies is reachable."""
    complete_resolution(state, entry)


def begin_pay_unless(state, payer_idx, cost, on_result):
    """A "player may pay `cost` to decide an outcome" prompt. `cost` is a mana
    cost dict ({"generic": 2} for Spell Pierce / Ward; {"B": 1} for Nihil
    Spellbomb's "you may pay {B}"). `payer_idx` (the countered spell's
    controller / the warded permanent's opponent / the Nihil controller)
    decides whether to pay: active_idx is flipped to them for the decision and
    restored once it's made. Backed by two drl_env actions -- "Pay (unless)"
    (only when affordable) and "Don't pay (unless)"; the mana itself routes
    through begin_pay_cost. on_result(state, paid: bool) fires with the
    outcome -- the caller then counters/draws/etc. accordingly."""
    original = state.active_idx
    state.active_idx = payer_idx
    begin_resolution(state, "pay_unless", None, cost=cost, on_result=on_result, original_idx=original)


def pay_unless_pay(state):
    """The payer chose to pay: hand off to begin_pay_cost (active_idx stays on
    the payer so they tap their OWN sources); once the mana is paid, restore
    active_idx and report paid=True."""
    from ..mana import begin_pay_cost  # call-time import -- mana imports resolution, so avoid a load cycle

    pending = state.pending_resolution
    cost, on_result, original = pending["cost"], pending["on_result"], pending["original_idx"]
    state.pending_resolution = None

    def _paid(state):
        state.active_idx = original
        on_result(state, True)

    begin_pay_cost(state, cost, on_complete=_paid)


def pay_unless_decline(state):
    pending = state.pending_resolution
    on_result, original = pending["on_result"], pending["original_idx"]
    state.pending_resolution = None
    state.active_idx = original
    on_result(state, False)


def begin_scry_surveil(state, kind, n, on_complete):
    """Reveal the top n library cards; the model decides keep-on-top or
    dispose for each one in turn (scry_surveil_options/
    execute_scry_surveil_option below), then -- if 2+ were kept -- the
    order to put them back in. Kept cards return to the library top in
    that model-chosen order; disposed cards go to the library bottom in
    random order (kind="scry") or the graveyard (kind="surveil") -- their
    order is never a model decision, since nothing here ever reads it
    again."""
    revealed = state.library[:n]
    del state.library[:n]
    begin_resolution(state, kind, on_complete, remaining=revealed, kept=[], disposed=[], ordered=None)
    # Empty-reveal safety net: an empty (or shorter-than-n) library reveals
    # nothing, so there is no keep/dispose decision to make and no ordering
    # -- the resolution would otherwise sit forever with remaining=[],
    # kept=[], ordered=None, which scry_surveil_options returns [] for and
    # _keep_dispose_legal refuses (remaining is falsy), i.e. ZERO legal
    # actions -> softlock. Same immediate-complete safety net begin_choose_
    # permanent/begin_search_fetch already apply for their own empty-options
    # case.
    if not revealed:
        _finish_scry_surveil(state)


def scry_surveil_options(state):
    """While deciding (remaining non-empty): keep or dispose the current
    (front of remaining) card. While ordering (remaining empty, 2+ kept,
    not yet all placed): one option per distinct name still waiting to be
    placed on top."""
    pending = state.pending_resolution
    if pending["remaining"]:
        return ["keep", "dispose"]
    if pending["ordered"] is not None:
        return sorted({c.name for c in pending["kept"]})
    return []


def _finish_scry_surveil(state):
    pending = state.pending_resolution
    kept_final = pending["ordered"] if pending["ordered"] is not None else pending["kept"]
    disposed = pending["disposed"]
    state.library[0:0] = kept_final
    disposed_to = "library_bottom" if pending["kind"] == "scry" else "graveyard"
    if pending["kind"] == "scry":
        state.rng.shuffle(disposed)
        state.library.extend(disposed)
    else:  # surveil
        for c in disposed:
            state.move_card(c, state.graveyard)
    state.log_event(
        "zone_move", reason=pending["kind"], kept_to_library_top=[c.name for c in kept_final],
        disposed_to=disposed_to, disposed=[c.name for c in disposed],
    )
    complete_resolution(state)


def execute_scry_surveil_option(state, option):
    pending = state.pending_resolution
    if pending["remaining"]:
        card = pending["remaining"].pop(0)
        (pending["kept"] if option == "keep" else pending["disposed"]).append(card)
        if pending["remaining"]:
            return  # more cards still to decide
        if len(pending["kept"]) <= 1:
            _finish_scry_surveil(state)  # 0 or 1 kept -- no ordering choice to make
        else:
            pending["ordered"] = []  # 2+ kept -- enter the ordering phase
        return

    # Ordering phase: option is the name of the next card to place on top.
    idx = next(i for i, c in enumerate(pending["kept"]) if c.name == option)
    pending["ordered"].append(pending["kept"].pop(idx))
    if not pending["kept"]:
        _finish_scry_surveil(state)


def scry(state, n):
    """Scry n (Candy Trail's ETB): see begin_scry_surveil."""
    begin_scry_surveil(state, "scry", n, on_complete=lambda s: None)


def surveil(state, n):
    """Surveil n (Conduit Pylons' ETB, Tocasia's Dig Site's ability): see
    begin_scry_surveil."""
    begin_scry_surveil(state, "surveil", n, on_complete=lambda s: None)


def begin_put_on_top_from_hand(state, n, on_complete):
    """Brainstorm's "put N cards from your hand on top of your library in any
    order." The model picks N hand cards one at a time by name (the generic
    "Choose: X" dispatch); the FIRST picked ends up on top (drawn first), so
    the model fully controls the order. Fewer than N in hand just places
    whatever's there. on_complete(state) runs once placement is done."""
    begin_resolution(state, "put_on_top", on_complete, remaining=n, placed=[])
    if not put_on_top_options(state):
        _finish_put_on_top(state)


def put_on_top_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    # Excludes a copy already reserved on the stack (a spell cast in response,
    # paid for and awaiting resolution): real Magic -- a card on the stack is
    # NOT in your hand, so Brainstorm's "put two cards from your hand on top"
    # can't pick it. The engine defers the on-stack card's own hand-removal to
    # its resolution (game.effects.stack.push_to_stack), leaving it physically
    # in the hand list, so this must subtract reservations exactly like
    # discard_options does -- without it, Brainstorm could move a still-on-the-
    # stack spell onto the library, and that spell's later resolve would fail
    # to find its card in hand.
    return _available_hand_names(state)


def _finish_put_on_top(state):
    pending = state.pending_resolution
    state.library[0:0] = pending["placed"]  # placed[0] on top
    state.log_event("put_on_top", cards=[c.name for c in pending["placed"]])
    complete_resolution(state)


def execute_put_on_top_option(state, name):
    pending = state.pending_resolution
    idx = next(i for i, c in enumerate(state.hand) if c.name == name)
    pending["placed"].append(state.hand.pop(idx))
    pending["remaining"] -= 1
    if pending["remaining"] <= 0 or not state.hand:
        _finish_put_on_top(state)


def begin_ponder(state, on_complete):
    """Ponder's "look at the top three cards, then put them back in any order.
    You may shuffle." The model either orders the revealed cards back on top
    one at a time (execute_ponder_option, via the generic "Choose: X"
    dispatch; FIRST placed ends on top) OR shuffles the whole library
    (execute_ponder_shuffle -- a dedicated "Shuffle (Ponder)" action, legal
    only before any card has been placed, drl_env._ponder_shuffle_legal).
    on_complete (Ponder's own "Draw a card") runs either way."""
    revealed = state.library[:3]
    del state.library[:3]
    begin_resolution(state, "ponder", on_complete, remaining=revealed, ordered=[])
    if not revealed:
        complete_resolution(state)  # empty library: nothing to arrange -- on_complete (the draw) still runs


def ponder_options(state):
    return sorted({c.name for c in state.pending_resolution["remaining"]})


def execute_ponder_option(state, name):
    pending = state.pending_resolution
    idx = next(i for i, c in enumerate(pending["remaining"]) if c.name == name)
    pending["ordered"].append(pending["remaining"].pop(idx))
    if not pending["remaining"]:
        state.library[0:0] = pending["ordered"]  # ordered[0] on top
        state.log_event("ponder", ordered=[c.name for c in pending["ordered"]], shuffled=False)
        complete_resolution(state)


def execute_ponder_shuffle(state):
    """"You may shuffle": put the looked-at cards back and shuffle the whole
    library. Only reachable before any card has been ordered (the action's
    own legality gate)."""
    pending = state.pending_resolution
    state.library.extend(pending["remaining"])  # order irrelevant -- about to shuffle
    state.rng.shuffle(state.library)
    state.log_event("ponder", shuffled=True)
    complete_resolution(state)


def begin_discard(state, n, optional, on_complete):
    """Discard n cards from hand, one at a time -- the model picks which,
    by name, same by-name fungibility every other resolution here uses.
    optional=True additionally allows declining outright, discarding
    nothing at all (Melded Moxite/Highway Robbery's "you may discard a
    card"); optional=False is mandatory, discarding as many as n allows,
    down to whatever's actually in hand if it runs out first (Faithless
    Looting's draw-2-discard-2 -- though by the time its own discard step
    runs, its own draw-2 already guarantees 2 cards are available, so this
    only ever matters as a defensive fallback, never a real case in either
    new deck; Grab the Prize's discard-as-an-additional-cost, paid via
    this same resolution before the spell's own effect runs, gated
    separately by its own extra_legal check requiring 1+ discardable
    card).

    Deliberately does NOT decide what happens to a discarded card beyond
    moving it out of hand:
    Madness reroutes qualifying cards to exile (plus a queued cast-or-
    graveyard decision, drained only once the enclosing action's entire
    effect is done, never mid-discard) instead of the plain graveyard trip
    this module implements alone today. on_complete(state, discarded_cards)
    once n discards are made, the model declines, or hand runs out of
    cards to offer -- discarded_cards is the list[CardDef] actually
    discarded this resolution, in discard order (empty if declined or
    hand was empty from the start). Callers that only care whether
    anything was discarded can just check bool(discarded_cards) (Highway
    Robbery/Melded Moxite's own "if you do, draw two cards"); Grab the
    Prize needs to inspect which specific card it was."""
    begin_resolution(state, "discard", on_complete, remaining=n, optional=optional, discarded_cards=[])
    if not discard_options(state):
        complete_resolution(state, [])


def _available_hand_names(state):
    """Distinct names in hand still available to discard/pay as a cost --
    excluding any copy already reserved on state.stack (paid for, awaiting
    resolution; see game.effects.stack.push_to_stack). That card's own
    resolve function hasn't removed it from hand yet (deferred until it
    actually resolves), so it's still physically present here, but
    offering it as a discard option (from an instant-speed activated
    ability like Blood's sac-for-a-card, which -- unlike a cast -- is
    never blocked by a non-empty stack) would let it be discarded twice
    over: once here, once more when its own stack entry finally tries to
    remove it. Same exclusion drl_env._hand_count_available applies, just
    for hand-count-based discard legality instead of cast legality. Shared by
    discard_options and discard_or_sacrifice_discard_options below."""
    stacked_counts = {}
    for entry in state.stack:
        if not entry["reserves_hand_card"]:
            continue
        name = entry["card_def"].name
        stacked_counts[name] = stacked_counts.get(name, 0) + 1
    hand_counts = {}
    for c in state.hand:
        hand_counts[c.name] = hand_counts.get(c.name, 0) + 1
    return sorted(name for name, count in hand_counts.items() if count > stacked_counts.get(name, 0))


def discard_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    return _available_hand_names(state)


def execute_discard_decline(state):
    """Only ever offered by the environment while
    state.pending_resolution['optional'] is True -- not itself enforced
    here, same convention as execute_search_fetch_decline."""
    complete_resolution(state, state.pending_resolution["discarded_cards"])


def _discard_one(state, card):
    """Move `card` out of hand into the graveyard, EXCEPT a Madness card,
    which goes to exile with a queued cast-or-graveyard decision instead
    -- a real-rules replacement effect that applies to ANY discard,
    regardless of why the card was discarded (a Madness card discarded to
    pay Highway Robbery's own optional cost triggers exactly the same way
    as one discarded by Faithless Looting). Queued rather than offered
    immediately: the model only sees the cast-or-graveyard decision once
    the enclosing action's entire effect is fully resolved (the Madness
    cast-or-graveyard cross-cutting rule). Shared by
    execute_discard_option's own per-card loop and execute_discard_or_
    sacrifice_option's single optional discard."""
    state.hand.remove(card)
    madness_spec = registry.EFFECT_REGISTRY.get(card.effect_id, {}).get("madness")
    if madness_spec is not None:
        state.exile.append((card, None))
        state.trigger_queue.append({"type": "decision", "kind": "madness", "card_def": card})
        state.log_event("zone_move", card=card.name, from_zone="hand", to_zone="exile", reason="discard_madness")
    else:
        state.move_card(card, state.graveyard)
        state.log_event("zone_move", card=card.name, from_zone="hand", to_zone="graveyard", reason="discard")


def execute_discard_option(state, name):
    pending = state.pending_resolution
    card = next(c for c in state.hand if c.name == name)
    _discard_one(state, card)
    pending["discarded_cards"].append(card)
    pending["remaining"] -= 1
    if pending["remaining"] <= 0 or not discard_options(state):
        complete_resolution(state, pending["discarded_cards"])


def begin_discard_or_sacrifice(state, sac_predicate, on_complete):
    """"You may discard a card or sacrifice a [land]. If you do, ..."
    (Highway Robbery) -- ONE optional decision offering two different
    cost shapes at once, unlike begin_discard's own single-cost-type
    optionality. Kept as a single exactly-one-of-these-or-neither choice,
    not two independent optional costs -- real text is "a card OR a
    [land]," never both. on_complete(state, paid) -- paid is True iff
    either a discard or a sacrifice actually happened, False if declined
    or nothing was payable to begin with; callers that only care whether
    anything was paid (Highway Robbery's own "if you do, draw two cards")
    just branch on that bool, same shape begin_discard's own
    bool(discarded_cards) contract already has."""
    begin_resolution(state, "discard_or_sacrifice", on_complete, sac_predicate=sac_predicate)
    if not discard_or_sacrifice_discard_options(state) and not discard_or_sacrifice_sacrifice_options(state):
        complete_resolution(state, False)


def discard_or_sacrifice_discard_options(state):
    return _available_hand_names(state)


def discard_or_sacrifice_sacrifice_options(state):
    pending = state.pending_resolution
    return sorted({p.card_def.name for p in state.battlefield if pending["sac_predicate"](p)})


def execute_discard_or_sacrifice_option(state, mode, name):
    if mode == "discard":
        card = next(c for c in state.hand if c.name == name)
        _discard_one(state, card)
    else:
        permanent = next(p for p in state.battlefield if p.card_def.name == name)
        state.battlefield.remove(permanent)
        state.move_card(permanent.card_def, state.graveyard)
        state.log_event(
            "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
            to_zone="graveyard", reason="sacrifice",
        )
        from ..effects.shared import fire_sacrifice_triggers
        fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator
    complete_resolution(state, True)


def execute_discard_or_sacrifice_decline(state):
    complete_resolution(state, False)


def begin_mulligan(state, on_complete):
    """Pregame: this player already has an opening 7-card hand (dealt by
    state.new_multiplayer_game_state's own eager draw(7)) --
    decide keep or mulligan (London Mulligan). Driven by
    game.turn._run_mulligan_gen, once per player, before turn 1 ever starts.

    No hand-contents event of its own: the opening hand and every redraw are
    already logged as library->hand "draw" zone_moves by GameState.draw (the
    single generic draw hook), the mulligan itself by execute_mulligan_take's
    "mulligan_take" zone_move, and the London bottoming by the per-card
    "mulligan_bottom" zone_moves. Together those reconstruct exactly what
    each player saw and did, so a separate "mulligan_hand" event would only
    duplicate the draw events."""
    begin_resolution(state, "mulligan_decision", on_complete)


def mulligan_decision_options(state):
    return ["keep", "mulligan"]


def execute_mulligan_keep(state):
    """Keep the current hand. London Mulligan: put a number of cards equal
    to mulligans already taken this game onto the library bottom, model-
    chosen -- opens a "mulligan_bottom" resolution for exactly that many
    (capped at hand size, in case someone ever mulligans past 7) before
    completing; on_complete only runs once the whole keep (bottoming
    included) is done."""
    on_complete = state.pending_resolution["on_complete"]
    n = min(state.mulligans_taken, len(state.hand))
    if n <= 0:
        complete_resolution(state)
        return
    state.pending_resolution = None
    begin_bottom(state, n, on_complete)


def execute_mulligan_take(state):
    """Take a mulligan: shuffle the current hand back into the library,
    redraw a fresh 7, increment mulligans_taken, then offer the same
    keep-or-mulligan decision again -- London Mulligan allows this as many
    times as the model likes, bounded only by library size like any other
    draw."""
    mulliganed = [c.name for c in state.hand]
    state.library.extend(state.hand)
    state.hand = []
    state.rng.shuffle(state.library)
    state.mulligans_taken += 1
    state.log_event("zone_move", cards=mulliganed, from_zone="hand", to_zone="library", reason="mulligan_take")
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    state.draw(7)
    begin_mulligan(state, on_complete)


def begin_bottom(state, n, on_complete):
    """Put exactly n cards from hand on the library bottom, model-chosen
    one at a time, in the order chosen -- London Mulligan's own "any order"
    (never read back by anything in this engine, so pick order = final
    order, same fungible-by-name picking as begin_discard). Deliberately
    not begin_discard itself -- its Madness routing is discard-specific and
    wrong here."""
    begin_resolution(state, "mulligan_bottom", on_complete, remaining=n)
    if not bottom_options(state):
        complete_resolution(state)


def bottom_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    return sorted({c.name for c in state.hand})


def execute_bottom_option(state, name):
    pending = state.pending_resolution
    card = next(c for c in state.hand if c.name == name)
    state.hand.remove(card)
    state.library.append(card)
    state.log_event("zone_move", card=name, from_zone="hand", to_zone="library_bottom", reason="mulligan_bottom")
    pending["remaining"] -= 1
    if pending["remaining"] <= 0 or not bottom_options(state):
        complete_resolution(state)


def _remove_one_from_exile(state, card_def):
    """First state.exile entry for this exact card_def object -- CardDefs
    are shared/interned per name (game.registry.CARD_DEFS holds one per
    distinct name, not per physical copy), so identity comparison here
    correctly matches "a copy of this card," same fungible-by-name
    convention every other zone in this engine already relies on."""
    entry = next(e for e in state.exile if e[0] is card_def)
    state.exile.remove(entry)


def begin_madness_decision(state, card_def, on_complete):
    """A qualifying card was just exiled by a discard (see
    execute_discard_option) -- offer "cast it for its madness cost" or
    "let it go to the graveyard." Only ever entered via the trigger-queue
    drain in game/effects/triggers.py, once the discard's enclosing action
    is fully done -- never mid-discard."""
    begin_resolution(state, "madness_decision", on_complete, card_def=card_def)


def madness_decision_options(state):
    return ["cast", "decline"]


def execute_madness_decline(state):
    pending = state.pending_resolution
    card_def = pending["card_def"]
    _remove_one_from_exile(state, card_def)
    state.move_card(card_def, state.graveyard)
    state.log_event("zone_move", card=card_def.name, from_zone="exile", to_zone="graveyard", reason="madness_decline")
    complete_resolution(state)


# "cast" isn't handled here -- paying the madness cost needs
# game.mana.begin_pay_cost, which this module can't import (see the
# module docstring) -- see game.effects.madness_and_plot.execute_madness_cast.


def begin_order_triggers(state, entries, on_complete):
    """2+ of the active player's own
    triggers are ready to move onto the stack at once (e.g. Faithless
    Looting's discard-2 hitting two Madness cards in the same discard, two
    Sneaky Snackers both crossing their own draw-count trigger on the same
    draw, or two targeting ETBs entering simultaneously) -- real Magic lets
    that player choose the PLACEMENT order (603.3b), not a fixed queue
    order. Only ever a single player's own triggers here -- APNAP ordering
    BETWEEN players is handled one level up, by
    game.effects.triggers.promote_triggers_to_stack's own group placement,
    before this is ever opened for either player's group.

    entries: list of {"card_def", "resolve"} dicts (plain triggers, already
    stack-ready) or {"card_def", "permanent", "targeting": True} dicts
    (target-at-promotion ETBs -- placing one runs its OWN etb_trigger hook
    instead, see execute_order_triggers_option's targeting branch) -- built
    by game.effects.triggers.promote_triggers_to_stack, which is
    also what turns each queued trigger's own (type, kind) into the right
    resolve function -- this module only ever deals in the stack's own
    generic shape, never trigger-specific semantics, same reverse-import
    reason execute_madness_cast's own cost-payment lives in
    game/effects/madness_and_plot.py instead of here).

    Picks one at a time; each pick is placed immediately
    (execute_order_triggers_option below), not deferred to the end --
    PLACEMENT order, not resolution order. Since the stack is LIFO, whichever
    PLAIN entry is placed LAST resolves FIRST (a targeting entry's own effect
    goes on the stack at ITS placement moment too, once it locks targets,
    following that same LIFO rule). on_complete(state) once every entry has
    been placed."""
    begin_resolution(state, "order_triggers", on_complete, remaining=list(entries))


def order_triggers_options(state):
    return sorted({e["card_def"].name for e in state.pending_resolution["remaining"]})


def execute_order_triggers_option(state, name):
    pending = state.pending_resolution
    idx = next(i for i, e in enumerate(pending["remaining"]) if e["card_def"].name == name)
    entry = pending["remaining"].pop(idx)
    if entry.get("targeting"):
        # Target-at-promotion (game.effects.triggers.promote_triggers_to_
        # stack's own docstring): placing this entry means running its OWN
        # etb_trigger hook NOW -- it opens target selection and pushes its
        # own stack entry once targets are locked, rather than this function
        # pushing a plain resolve directly. Same two-way branch
        # promote_triggers_to_stack's own single-entry fast path uses.
        registry.EFFECT_REGISTRY[entry["card_def"].effect_id]["etb_trigger"](state, entry["permanent"])
    else:
        # Same controller field push_to_stack itself stamps on every entry
        # -- state.active_idx here is still the
        # trigger owner (nothing else can interleave mid-resolution), so this
        # is the correct moment to record it, same reasoning push_to_stack's
        # own docstring gives.
        entry["controller"] = state.active_idx
        # Every entry reaching here originates from triggers.promote_triggers_
        # to_stack (begin_order_triggers's own docstring: queued triggers only,
        # never a real cast) -- same "never reserves a hand card" reasoning
        # push_to_stack(..., reserves_hand_card=False) applies for the
        # single-trigger branch right above this function's own caller.
        entry["reserves_hand_card"] = False
        entry["is_spell"] = False  # a triggered ability, not a spell -- never a Counterspell target
        state.stack.append(entry)  # already the stack's own native {"card_def", "resolve"} shape
        # A triggered ability going on the stack moves no card (is_spell=False), so
        # emit no card zone_move -- same reasoning as push_to_stack / resolve_top_of_stack.
    if not pending["remaining"]:
        complete_resolution(state)


def begin_sacrifice(state, predicate, n, on_complete):
    """Choose and sacrifice n of your own battlefield permanents matching
    predicate, one at a time -- same by-name fungibility every other
    resolution here uses. A predicate-based primitive covering both
    creature-sacrifice costs (Dread Return's own Flashback:
    `begin_sacrifice(state, lambda p: p.card_type == CardType.CREATURE, 3,
    on_complete)`) and land-sacrifice costs (Fireblast's alt-cost, Lava
    Dart's Flashback, Highway Robbery's discard-or-sac choice).

    Caller's own legality check guarantees n eligible permanents exist
    before this is ever offered (same "guaranteed payable, not a maybe"
    contract every alternate cost path here already follows) -- the
    n<=0/empty-options branch below is pure belt-and-suspenders, matching
    every other pending kind here. on_complete(state, True) once n are
    sacrificed (False only via that defensive n<=0 fallback)."""
    begin_resolution(state, "sacrifice", on_complete, predicate=predicate, remaining=n)
    if not sacrifice_options(state):
        complete_resolution(state, n <= 0)


def sacrifice_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    predicate = pending["predicate"]
    return sorted({p.card_def.name for p in state.battlefield if predicate(p)})


def execute_sacrifice_option(state, name):
    pending = state.pending_resolution
    predicate = pending["predicate"]
    permanent = next(p for p in state.battlefield if p.card_def.name == name and predicate(p))
    state.battlefield.remove(permanent)
    # 400.7 linked-ability tracking: see state_based._destroy_creature's own
    # comment -- stash the fresh graveyard instance so an ltb_trigger needing
    # the EXACT card that left (Lembas) can reference it, not bridge by name.
    permanent.flags["graveyard_instance"] = state.move_card(permanent.card_def, state.graveyard)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    # Leaves-the-battlefield triggered ability (Mesmeric Fiend, sacrificed --
    # e.g. to Dread Return's Flashback). Same three lines as state_based.
    # _queue_leave_triggers, inlined here because resolution can't import
    # state_based without a cycle (state_based imports resolution).
    ltb_spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if ltb_spec.get("ltb_trigger") is not None:
        state.trigger_queue.append({"type": "ltb", "card_def": permanent.card_def, "permanent": permanent})
    from ..effects.shared import fire_sacrifice_triggers
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator / Writhing Chrysalis
    pending["remaining"] -= 1
    if pending["remaining"] <= 0:
        complete_resolution(state, True)

