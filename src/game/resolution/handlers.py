"""Every deck-agnostic pending-resolution handler (search_fetch,
choose_permanent, scry/surveil, discard, sacrifice, combat-damage
assignment, ...): each kind's begin_/options/execute_ trio. Split out of
resolution.py unchanged; re-exported via game.resolution so
`from ..resolution import X` in the catalogs keeps resolving."""

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
    the exact (name, slot) it occupies
    "Permanent identity" gap, closed here: same (name, slot) addressing
    begin_choose_opponent_permanent already uses, not the old
    fungible-by-name shortcut (two same-named permanents stop being
    interchangeable the moment an Aura attaches to only one of them, or a
    caller needs the EXACT physical permanent it chose to still be there
    later -- see cast_aura's own targeting contract). on_complete(state,
    (name, slot)_or_None) runs once decided. Same empty-options safety net
    as begin_search_fetch -- fizzles immediately with None if nothing
    matches."""
    begin_resolution(state, "choose_permanent", on_complete, predicate=predicate)
    if not choose_permanent_options(state):
        complete_resolution(state, None)


def choose_permanent_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted((p.card_def.name, p.slot) for p in state.battlefield if predicate(p))


def execute_choose_permanent_option(state, name, slot):
    complete_resolution(state, (name, slot))


def begin_choose_graveyard_card(state, predicate, on_complete, graveyard=None, optional=False):
    """Pick ONE card from a graveyard by name, among those matching
    predicate -- Dread Return's reanimation target originally
    (game.catalog.black_cards), promoted here once Relic of Progenitus'
    own repeatable exile ability needed the identical primitive too (see
    this module's own docstring: a deck-specific kind moves here the
    moment something ELSE reuses it). Same fungible-by-name simplification,
    same empty-options safety net as begin_search_fetch/begin_choose_
    permanent.

    graveyard=None defaults to state.graveyard (the active player's own,
    via the active-idx proxy) -- Dread Return's reanimation target is
    always its own controller's graveyard, never anyone else's. Pass an
    explicit graveyard list to target a DIFFERENT player's graveyard
    instead -- Relic of Progenitus' own ability can target either player,
    real "choose which card" ability text notwithstanding: the target's
    own choice is simplified to the ACTIVATING player's, same "no
    observable difference in this solitaire sim, nothing depends on WHO
    picks" reasoning already applied elsewhere (Grab the Prize's own
    discard timing).

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
    pending = state.pending_resolution
    return sorted({c.name for c in pending["graveyard"] if pending["predicate"](c)})


def execute_choose_graveyard_card_option(state, name):
    complete_resolution(state, name)


def execute_choose_graveyard_card_decline(state):
    """Decline an OPTIONAL choose_graveyard_card (Masked Vandal's "you may
    exile a creature card from your graveyard") -- resolve with None, exiling
    nothing. Only ever offered by the environment while the pending was begun
    optional=True, same convention as execute_search_fetch_decline."""
    complete_resolution(state, None)


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
   , first used by blocking. Addressed by (name,
    slot), not name alone: unlike begin_choose_permanent's own
    fungible-by-name simplification, two same-named OPPOSING permanents
    are exactly the case "Permanent identity"
    section flags -- an Aura-enchanted attacker and a plain one of the
    same name are not an arbitrary pick for a blocker to choose between.
    on_complete(state, (name, slot)_or_None) runs once decided. Same
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
    (drl_env.py, picks one of THIS player's own untapped, not-yet-used
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
    (drl_env.py already picked the specific eligible `blocker` permanent):
    nests a begin_choose_opponent_permanent choosing which of the
    attacker's declared, not-yet-blocked attackers this blocker is
    assigned to (or None, if none remain -- shouldn't happen given the
    action's own legality check, but never crashes either way), appends the
    blocker to state.opponent.blocked_by[attacker] (a LIST -- gang-blocking),
    then calls
    on_complete -- which drl_env.py uses to re-open begin_declare_blockers
    so the defender can assign another blocker or finish.

    extra_predicate(attacker) -> bool: an additional restriction beyond
    "is a currently-unblocked attacker" -- e.g. flying's own blocking
    restriction. Supplied by the CALLER (drl_env.py)
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

        def _chosen(state, name):
            found = next(c for c in state.graveyard if c.name == name)
            state.graveyard.remove(found)  # exiled, untracked
            state.log_event("zone_move", card=found.name, from_zone="graveyard", to_zone="exile_untracked", reason="exile_cost")
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


def begin_may_transform(state, permanent):
    """A "you may transform this creature" choice (Delver of Secrets, once an
    instant/sorcery is revealed). Two drl_env actions -- "Transform" /
    "Don't transform" -- back it; execute_may_transform applies or skips."""
    begin_resolution(state, "may_transform", lambda s: None, permanent=permanent)


def execute_may_transform(state, do_transform):
    pending = state.pending_resolution
    permanent = pending["permanent"]
    if do_transform:
        permanent.flags["transformed"] = True
        state.log_event("transform", permanent=(permanent.card_def.name, permanent.slot))
    complete_resolution(state)


def begin_may_copy(state, on_complete):
    """A "you may copy this spell" choice (Chain Lightning's rider, after the
    {R}{R} has already been paid -- the second, independent "may" in "they may
    pay {R}{R}. If the player does, they may copy this spell"). on_complete(
    state, do_copy: bool)."""
    begin_resolution(state, "may_copy", on_complete)


def execute_may_copy(state, do_copy):
    complete_resolution(state, do_copy)


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
    dicts). Options are the distinct names of matching spell entries; the
    chosen entry (the TOPMOST of that name -- most recently cast) is passed to
    on_complete. Fizzles immediately (on_complete(None)) if nothing matches --
    though a counter spell's own extra_legal already requires a legal target
    to be cast at all."""
    begin_resolution(state, "choose_stack_target", on_complete, predicate=predicate)
    if not choose_stack_target_options(state):
        complete_resolution(state, None)


def choose_stack_target_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted({e["card_def"].name for e in state.stack if e.get("is_spell") and predicate(e)})


def execute_choose_stack_target_option(state, name):
    predicate = state.pending_resolution["predicate"]
    # Topmost (last-pushed) matching spell of this name -- LIFO, the spell
    # most recently put on the stack.
    entry = next(
        e for e in reversed(state.stack)
        if e.get("is_spell") and predicate(e) and e["card_def"].name == name
    )
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
    # case; scry/surveil was the one begin_* missing it. Surfaced by a
    # monster_tron game (a deck that both scries/surveils AND decks itself
    # out in long games) via the token action mask going all-False.
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
        state.graveyard.extend(disposed)
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
    # to find its card in hand (confirmed via a Gurmag-Angler-mid-delve-cast +
    # Brainstorm + Mental Note line in cross-deck self-play).
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
    remove it. Same fix as drl_env._hand_count_available, just for
    hand-count-based discard legality instead of cast legality. Shared by
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
        state.graveyard.append(card)
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
        state.graveyard.append(permanent.card_def)
        state.log_event(
            "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
            to_zone="graveyard", reason="sacrifice",
        )
    complete_resolution(state, True)


def execute_discard_or_sacrifice_decline(state):
    complete_resolution(state, False)


def begin_mulligan(state, on_complete):
    """Pregame: this player already has an opening 7-card hand (dealt by
    state.new_game_state/new_multiplayer_game_state's own eager draw(7)) --
    decide keep or mulligan (London Mulligan). Driven by
    game.turn.run_mulligan_phase/_run_mulligan_gen, once per player, before
    turn 1 ever starts.

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
    state.graveyard.append(card_def)
    state.log_event("zone_move", card=card_def.name, from_zone="exile", to_zone="graveyard", reason="madness_decline")
    complete_resolution(state)


# "cast" isn't handled here -- paying the madness cost needs
# game.mana.begin_pay_cost, which this module can't import (see the
# module docstring) -- see game.effects.madness_and_plot.execute_madness_cast.


def begin_order_triggers(state, entries, on_complete):
    """2+ of the active player's own
    triggers are ready to move onto the stack at once (e.g. Faithless
    Looting's discard-2 hitting two Madness cards in the same discard, or
    two Sneaky Snackers both crossing their own draw-count trigger on the
    same draw) -- real Magic lets that player choose the PLACEMENT order
    (603.3b: APNAP among different players, but this engine only ever
    queues triggers for the active player -- see game.effects.triggers.
    promote_triggers_to_stack's own docstring for why that's sufficient
    given the current card pool), not a fixed queue order.

    entries: list of {"card_def", "resolve"} dicts, already stack-ready
    (built by game.effects.triggers.promote_triggers_to_stack, which is
    also what turns each queued trigger's own (type, kind) into the right
    resolve function -- this module only ever deals in the stack's own
    generic shape, never trigger-specific semantics, same reverse-import
    reason execute_madness_cast's own cost-payment lives in
    game/effects/madness_and_plot.py instead of here).

    Picks one at a time; each pick is pushed onto state.stack immediately
    (execute_order_triggers_option below), not deferred to the end --
    PLACEMENT order, not resolution order. Since the stack is LIFO,
    whichever entry is placed LAST resolves FIRST. on_complete(state) once
    every entry has been placed."""
    begin_resolution(state, "order_triggers", on_complete, remaining=list(entries))


def order_triggers_options(state):
    return sorted({e["card_def"].name for e in state.pending_resolution["remaining"]})


def execute_order_triggers_option(state, name):
    pending = state.pending_resolution
    idx = next(i for i, e in enumerate(pending["remaining"]) if e["card_def"].name == name)
    entry = pending["remaining"].pop(idx)
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
    resolution here uses. Generalizes what was originally Dread Return's
    own Flashback-cost-only "sacrifice_creatures" resolution
    (game.catalog.black_cards) into a predicate-based primitive reusable by land-
    sacrifice costs too (Fireblast's alt-cost, Lava Dart's Flashback,
    Highway Robbery's discard-or-sac choice
    5); Dread Return's own creature-sacrifice is just
    `begin_sacrifice(state, lambda p: p.card_type ==
    CardType.CREATURE, 3, on_complete)` now, no separate function needed.

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
    state.graveyard.append(permanent.card_def)
    state.log_event(
        "zone_move", permanent=(permanent.card_def.name, permanent.slot), from_zone="battlefield",
        to_zone="graveyard", reason="sacrifice",
    )
    # Leaves-the-battlefield triggered ability (Mesmeric Fiend, sacrificed --
    # e.g. to Dread Return's Flashback). Same three lines as state_based.
    # _queue_leave_triggers, inlined here because resolution.py can't import
    # state_based without a cycle (state_based imports resolution).
    ltb_spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {})
    if ltb_spec.get("ltb_trigger") is not None:
        state.trigger_queue.append({"type": "ltb", "card_def": permanent.card_def, "permanent": permanent})
    from ..effects.shared import fire_sacrifice_triggers
    fire_sacrifice_triggers(state, state.active_idx, permanent.card_def)  # Gixian Infiltrator / Writhing Chrysalis
    pending["remaining"] -= 1
    if pending["remaining"] <= 0:
        complete_resolution(state, True)


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.resolution`
    # from src/. Exercises begin_discard directly against a hand-built
    # state, bypassing drl_env.py entirely (no card wires into this
    # primitive yet -- deck assembly is out of scope for this plan).
    from ..cards import CardDef, CardType
    from ..state import GameState

    def _card(name):
        return CardDef(name, CardType.SORCERY, {"generic": 1}, None)

    # Mandatory discard of fewer cards than n asks for: never crashes,
    # stops once hand is exhausted instead of running remaining negative.
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 2, optional=False, on_complete=lambda s, cards: completed.append(cards))
    assert discard_options(state) == ["A"]
    execute_discard_option(state, "A")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A"]
    assert state.hand == []
    assert [c.name for c in state.graveyard] == ["A"]

    # Mandatory discard of exactly n, from a larger hand.
    state = GameState(on_the_play=True)
    state.hand = [_card("A"), _card("B"), _card("C")]
    completed = []
    begin_discard(state, 2, optional=False, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_option(state, "A")
    assert completed == []  # one more still required
    execute_discard_option(state, "B")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A", "B"]
    assert [c.name for c in state.hand] == ["C"]
    assert sorted(c.name for c in state.graveyard) == ["A", "B"]

    # Optional discard, declined: hand/graveyard untouched, still completes
    # with an empty discarded_cards list (Highway Robbery/Melded Moxite's
    # own "if you do" check reads bool(discarded_cards) for exactly this).
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 1, optional=True, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_decline(state)
    assert completed == [[]]
    assert [c.name for c in state.hand] == ["A"]
    assert state.graveyard == []

    # Optional discard, taken.
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 1, optional=True, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_option(state, "A")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A"]
    assert state.hand == []
    assert [c.name for c in state.graveyard] == ["A"]

    # Mulligan (London style): begin_mulligan/execute_mulligan_take loop
    # twice (redraw to 7 each time, mulligans_taken incrementing), then
    # execute_mulligan_keep bottoms exactly mulligans_taken (2) cards before
    # completing.
    events = []
    state = GameState(on_the_play=True, event_log=events)
    state.library = [_card(f"L{i}") for i in range(20)]
    state.rng.shuffle(state.library)
    state.draw(7)  # new_multiplayer_game_state's own eager opening draw -- begin_mulligan's own precondition
    completed = []
    begin_mulligan(state, on_complete=lambda s: completed.append(True))
    assert mulligan_decision_options(state) == ["keep", "mulligan"]
    assert state.pending_resolution["kind"] == "mulligan_decision"
    assert len(state.hand) == 7

    execute_mulligan_take(state)
    assert state.mulligans_taken == 1
    assert len(state.hand) == 7  # redrawn fresh, not bottomed yet
    assert state.pending_resolution["kind"] == "mulligan_decision"

    execute_mulligan_take(state)
    assert state.mulligans_taken == 2
    assert len(state.hand) == 7
    assert completed == []  # still deciding -- on_complete hasn't fired

    execute_mulligan_keep(state)
    assert completed == []  # not yet -- 2 cards still need to be bottomed
    assert state.pending_resolution["kind"] == "mulligan_bottom"
    bottomed = []
    while state.pending_resolution is not None:
        name = bottom_options(state)[0]
        bottomed.append(name)
        execute_bottom_option(state, name)
    assert completed == [True]
    assert len(state.hand) == 5  # 7 - 2 bottomed
    assert [c.name for c in state.library[-2:]] == bottomed  # bottomed, in the order chosen

    # There is no "mulligan_hand" event any more. Every hand SEEN is a
    # library->hand "draw" zone_move (GameState.draw, the single generic
    # hook): three here -- the opener, then the two redraws -- 7 cards each,
    # in order.
    draws = [e["cards"] for e in events if e.get("reason") == "draw"]
    assert len(draws) == 3 and all(len(d) == 7 for d in draws)
    # Each thrown-back hand (mulligan_take) is exactly the hand drawn just
    # before it, so draws[0] and draws[1] are the two mulliganed hands.
    takes = [e["cards"] for e in events if e.get("reason") == "mulligan_take"]
    assert takes == draws[:2]  # seen == thrown back
    assert [e["card"] for e in events if e.get("reason") == "mulligan_bottom"] == bottomed

    # Keeping with 0 mulligans taken never opens a mulligan_bottom at all.
    state = GameState(on_the_play=True)
    state.library = [_card(f"L{i}") for i in range(20)]
    state.draw(7)
    completed = []
    begin_mulligan(state, on_complete=lambda s: completed.append(True))
    execute_mulligan_keep(state)
    assert completed == [True]
    assert state.pending_resolution is None
    assert len(state.hand) == 7

    print("resolution.py mulligan self-check: OK")

    # Madness routing: a discarded card whose EffectId has a "madness"
    # registry spec goes to exile + the trigger queue, not the graveyard.
    # No real madness card exists yet (deck assembly is out of scope), so
    # this borrows EffectId.FILLER for the duration of the check, saving
    # and restoring its real (empty) registry entry around it.
    from ..cards import EffectId

    _filler_entry_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"R": 1}, "resolve": lambda s, c: None}}
    try:
        madness_card = CardDef("Fake Madness Card", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [madness_card]
        completed = []
        begin_discard(state, 1, optional=False, on_complete=lambda s, cards: completed.append(cards))
        execute_discard_option(state, "Fake Madness Card")
        assert len(completed) == 1 and completed[0] == [madness_card]
        assert state.hand == [] and state.graveyard == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Madness Card"]
        assert state.trigger_queue == [{"type": "decision", "kind": "madness", "card_def": madness_card}]

        # Promoting the queue (game.effects.triggers.promote_triggers_to_
        # stack's job in real play,) and declining:
        # back out of exile, into the graveyard.
        state.trigger_queue.clear()
        drain_completed = []
        begin_madness_decision(state, madness_card, on_complete=lambda s: drain_completed.append(True))
        assert madness_decision_options(state) == ["cast", "decline"]
        execute_madness_decline(state)
        assert drain_completed == [True]
        assert state.exile == []
        assert [c.name for c in state.graveyard] == ["Fake Madness Card"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_entry_backup

    # begin_sacrifice: predicate-based, not hardcoded to creatures --
    # exercise both a creature predicate (Dread Return's own shape, post-
    # migration) and a land predicate (Fireblast/Lava Dart's shape,
    #) against the same primitive.
    from ..state import Permanent

    def _permanent(name, card_type):
        return Permanent(CardDef(name, card_type, None, None))

    state = GameState(on_the_play=True)
    state.battlefield = [
        _permanent("Bear", CardType.CREATURE),
        _permanent("Wolf", CardType.CREATURE),
        _permanent("Mountain", CardType.LAND),
    ]
    completed = []
    begin_sacrifice(state, lambda p: p.card_def.card_type == CardType.CREATURE, 2, lambda s, ok: completed.append(ok))
    assert sacrifice_options(state) == ["Bear", "Wolf"]  # the Mountain never qualifies
    execute_sacrifice_option(state, "Bear")
    assert completed == []
    execute_sacrifice_option(state, "Wolf")
    assert completed == [True]
    assert sorted(p.card_def.name for p in state.battlefield) == ["Mountain"]
    assert sorted(c.name for c in state.graveyard) == ["Bear", "Wolf"]

    state = GameState(on_the_play=True)
    state.battlefield = [_permanent("Mountain", CardType.LAND), _permanent("Bear", CardType.CREATURE)]
    completed = []
    begin_sacrifice(state, lambda p: p.card_def.name == "Mountain", 1, lambda s, ok: completed.append(ok))
    assert sacrifice_options(state) == ["Mountain"]  # the Bear never qualifies, even though it's a permanent
    execute_sacrifice_option(state, "Mountain")
    assert completed == [True]
    assert [p.card_def.name for p in state.battlefield] == ["Bear"]

    print("resolution.py discard self-check: OK")

    # Cross-player targeting: begin_choose_opponent_permanent
    # targets state.opponent's battlefield, addressed by (name, slot) --
    # not name alone, since two same-named OPPOSING permanents aren't
    # necessarily interchangeable. Only correct once the referencing player is already the
    # active one (blocking's own defender-decision channel flips
    # active_idx before ever calling this) -- simulated here by setting
    # active_idx directly to "the defender," same as that channel would.
    from ..state import PlayerState

    attacker_bogle_1 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2.slot = 2
    attacker_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2, attacker_land]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective

    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert choose_opponent_permanent_options(state) == [("Slippery Bogle", 1), ("Slippery Bogle", 2)]  # the Forest never qualifies
    execute_choose_opponent_permanent_option(state, "Slippery Bogle", 2)
    assert completed == [("Slippery Bogle", 2)]  # the SPECIFIC slot chosen, not an arbitrary same-named match

    # Empty-options safety net: no eligible opposing permanent -> fizzles
    # immediately with None, same convention as begin_choose_permanent.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == [None]

    print("resolution.py cross-player targeting self-check: OK")

    # "Any target" (begin_choose_any_target): a single target spanning BOTH
    # battlefields' creatures plus either player -- real Magic's "any
    # target" (Lightning Bolt). Creatures addressed by (side, name, slot)
    # so a same-named creature on each side stays distinguishable; players
    # by index. allow_players=False narrows it to "target creature" spanning
    # sides (Kill-Ship/Quirion), with the empty-options net if none match.
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    assert choose_any_target_creature_options(state) == [(0, "Grizzly Bears", 1), (1, "Grizzly Bears", 1)]  # both sides, Forest excluded
    assert ("player", 0) in choose_any_target_options(state) and ("player", 1) in choose_any_target_options(state)
    execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)  # the OPPONENT's copy specifically
    assert completed == [("creature", 1, "Grizzly Bears", 1)]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    execute_choose_any_target_player(state, 0)  # legal to target yourself (real Magic)
    assert completed == [("player", 0)]

    # allow_players=False + no creature anywhere -> immediate None (fizzle/
    # can't-target), same empty-options net as the other primitives.
    empty = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    empty.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    completed = []
    begin_choose_any_target(empty, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t), allow_players=False)
    assert completed == [None]
    # allow_players=False WITH a creature -> creatures only, no player option offered
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: None, allow_players=False)
    assert all(o[0] == "creature" for o in choose_any_target_options(state))

    print("resolution.py any-target self-check: OK")

    # Blocking: begin_declare_blockers/
    # declare_blocker_assignment, driven directly against a hand-built
    # state (bypassing game.turn._declare_blockers_gen's active_idx-flip --
    # simulated here the same way the cross-player check above does, by
    # setting active_idx to "the defender" up front). Also bypasses
    # drl_env.py's own _assign_blocker_legal eligibility gate -- this
    # exercises the resolution.py primitives directly, so a "re-open
    # begin_declare_blockers after each assignment" step is done by hand
    # here rather than relying on drl_env._assign_blocker_execute's own
    # nested on_complete to do it.
    bear = _permanent("Bear", CardType.CREATURE)
    wolf = _permanent("Wolf", CardType.CREATURE)
    grizzly = _permanent("Grizzly Bears", CardType.CREATURE)
    panther = _permanent("Panther", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [bear, wolf]
    state.players[0].attackers = [bear, wolf]
    state.players[1].battlefield = [grizzly, panther]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []  # real attackers declared -- does not auto-complete
    assert state.pending_resolution["kind"] == "declare_blockers"

    # Assign Grizzly Bears to block Bear specifically (not Wolf) -- the
    # nested choose_opponent_permanent offers both.
    step1_done = []
    declare_blocker_assignment(state, grizzly, on_complete=lambda s: step1_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step1_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly]}  # attacker -> LIST of blockers (gang-blocking)

    # GANG-BLOCKING: re-open the consult and assign Panther to the SAME
    # attacker (Bear). An already-blocked attacker is STILL offered -- the
    # old 1:1 "already spoken for" exclusion is gone -- and multiple
    # blockers may pile onto one attacker.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    step2_done = []
    declare_blocker_assignment(state, panther, on_complete=lambda s: step2_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]  # Bear STILL offered -- gang-block
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step2_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly, panther]}  # two blockers on one attacker

    # "Done blocking" (drl_env.py's action): closes a still-open
    # declare_blockers resolution outright, no assignment required.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    complete_resolution(state)
    assert completed == [True]

    # No attackers at all: auto-completes immediately, same empty-options
    # precedent as begin_choose_permanent/begin_search_fetch.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 1
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == [True]

    print("resolution.py blocking self-check: OK")

    # declare_blocker_assignment's extra_predicate: this module stays effect-agnostic (see
    # its own module docstring) and doesn't import game.effects.stats
    # itself, so the actual restriction is supplied by the CALLER
    # (drl_env._assign_blocker_execute, using game.has_keyword) -- this
    # proves the parameter itself
    # is correctly applied on top of the usual "unblocked attacker" filter,
    # using a plain stand-in predicate rather than a real keyword lookup.
    flyer = _permanent("Flyer", CardType.CREATURE)
    grounded = _permanent("Grounded", CardType.CREATURE)
    non_flying_blocker = _permanent("Non-Flying Blocker", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [flyer, grounded]
    state.players[0].attackers = [flyer, grounded]
    state.players[1].battlefield = [non_flying_blocker]
    state.active_idx = 1

    completed = []
    declare_blocker_assignment(
        state, non_flying_blocker, on_complete=lambda s: completed.append(True),
        extra_predicate=lambda p: p is not flyer,  # stand-in: "flyer needs a flying blocker, and this one isn't"
    )
    assert choose_opponent_permanent_options(state) == [("Grounded", 1)]  # Flyer excluded by extra_predicate
    execute_choose_opponent_permanent_option(state, "Grounded", 1)
    assert completed == [True]
    assert state.players[0].blocked_by == {grounded: [non_flying_blocker]}

    print("resolution.py extra_predicate (flying-restriction wiring) self-check: OK")

    # begin_order_triggers: 2+ simultaneous
    # triggers get a real placement-order choice -- PLACEMENT order, not
    # resolution order (the stack is LIFO). Driven directly against a
    # hand-built state, bypassing game.effects.triggers.promote_triggers_
    # to_stack entirely (this module doesn't import game.effects.triggers
    # -- see its own docstring), using plain no-op resolve functions since
    # only the ordering mechanism itself is under test here.
    resolved_order = []
    entry_a = {"card_def": CardDef("Trigger A", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    entry_b = {"card_def": CardDef("Trigger B", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    state = GameState(on_the_play=True)
    completed = []
    begin_order_triggers(state, [entry_a, entry_b], on_complete=lambda s: completed.append(True))
    assert order_triggers_options(state) == ["Trigger A", "Trigger B"]

    execute_order_triggers_option(state, "Trigger A")  # placed FIRST -- resolves LAST
    assert completed == []  # one more still to place
    assert state.stack == [entry_a]
    assert order_triggers_options(state) == ["Trigger B"]  # already-placed one no longer offered

    execute_order_triggers_option(state, "Trigger B")  # placed LAST -- resolves FIRST
    assert completed == [True]
    assert state.stack == [entry_a, entry_b]  # placement order: A then B
    assert state.pending_resolution is None

    while state.stack:  # LIFO: B (placed last) actually resolves first
        entry = state.stack.pop()
        entry["resolve"](state, entry["card_def"])
    assert resolved_order == ["Trigger B", "Trigger A"]

    print("resolution.py begin_order_triggers self-check: OK")

    # explore (Map token / Fanatical Offering): a land on top goes to hand; a
    # nonland puts a +1/+1 counter on the exploring creature, then surveil 1
    # (keep on top or bin) on that same card.
    from ..state import Permanent as _Perm

    state = GameState(on_the_play=True)
    creature = _Perm(CardDef("Explorer", CardType.CREATURE, None, None, power=1, toughness=1))
    state.battlefield = [creature]
    state.library = [CardDef("A Land", CardType.LAND, None, None), CardDef("A Spell", CardType.INSTANT, {"U": 1}, None)]
    explore(state, creature)
    assert [c.name for c in state.hand] == ["A Land"]  # land -> hand
    assert creature.counters.get("+1/+1", 0) == 0  # no counter for a land

    state = GameState(on_the_play=True)
    creature = _Perm(CardDef("Explorer", CardType.CREATURE, None, None, power=1, toughness=1))
    state.battlefield = [creature]
    state.library = [CardDef("A Spell", CardType.INSTANT, {"U": 1}, None), CardDef("Next", CardType.LAND, None, None)]
    explore(state, creature)
    assert creature.counters["+1/+1"] == 1  # nonland -> +1/+1
    assert state.pending_resolution["kind"] == "surveil"  # then "may put it in graveyard"
    execute_scry_surveil_option(state, "dispose")
    assert [c.name for c in state.graveyard] == ["A Spell"]
    print("resolution.py explore self-check: OK")
