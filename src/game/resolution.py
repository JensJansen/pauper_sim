"""Pending resolution: a decision point that takes more than one action to
fully resolve -- paying a cost one tap at a time, walking a scry/surveil,
choosing a search target -- because the model, not an automatic solver,
makes every one of these choices.

This module holds the generic core (begin_resolution/complete_resolution)
plus every deck-agnostic resolution kind: search_fetch, choose_permanent,
choose_graveyard_card (Dread Return's own reanimation-target pick
originally, promoted here once Relic of Progenitus' repeatable exile
ability needed the identical primitive too), scry/surveil, discard,
discard_or_sacrifice (Highway Robbery's own "discard a card or sacrifice
a land" -- two different cost shapes under one optional decision),
madness_decision, sacrifice (predicate-based -- Dread Return's creature
sacrifice and Fireblast/Lava Dart/Highway Robbery's land sacrifice are
the same primitive, different predicates; see MADNESS_DECKS_PLAN.md item
5). Deck-specific kinds (Ancient Stirrings' take-one-or-decline, Lead the
Stampede's select_to_hand) still live with their owning deck instead,
since nothing else currently reuses them -- the exact bar
choose_graveyard_card/discard_or_sacrifice both just crossed.

References game.registry.EFFECT_REGISTRY only from inside function
bodies, via `registry.EFFECT_REGISTRY` -- same lazy-lookup convention
mana.py/several of game/effects/*.py already use, for the same reason
(registry.py's own import chain reaches this module before EFFECT_REGISTRY
exists). Deliberately does NOT import game.mana (mana.py imports THIS
module at its own top level; the reverse import would cycle) -- so the
madness "cast for its madness cost" path's cost payment lives in
game/effects/madness_and_plot.py instead, which is free to depend on both
this module and mana.py. See docs/MADNESS_DECKS_PLAN.md items 1/3.
"""

from . import registry


def begin_resolution(state, kind, on_complete, **fields):
    """Start a pending resolution. on_complete(state) runs once it's fully
    resolved (via repeated calls into that kind's own option/execute
    functions) -- it may itself begin a further resolution, so multi-step
    effects chain naturally through nested callbacks rather than needing a
    single monolithic resolution type."""
    state.pending_resolution = {"kind": kind, "on_complete": on_complete, **fields}


def complete_resolution(state, *args):
    """*args is an optional payload for kinds whose completion carries a
    result the caller needs (e.g. search_fetch's chosen card name) --
    on_complete(state) for kinds that don't (e.g. pay_cost)."""
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    on_complete(state, *args)


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
    the exact (name, slot) it occupies -- docs/MULTIPLAYER_GAPS.md's
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


def begin_choose_graveyard_card(state, predicate, on_complete, graveyard=None):
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
    discard timing)."""
    if graveyard is None:
        graveyard = state.graveyard
    begin_resolution(state, "choose_graveyard_card", on_complete, predicate=predicate, graveyard=graveyard)
    if not choose_graveyard_card_options(state):
        complete_resolution(state, None)


def choose_graveyard_card_options(state):
    pending = state.pending_resolution
    return sorted({c.name for c in pending["graveyard"] if pending["predicate"](c)})


def execute_choose_graveyard_card_option(state, name):
    complete_resolution(state, name)


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


def begin_choose_opponent_permanent(state, predicate, on_complete):
    """Like begin_choose_permanent, but targets the OPPONENT's battlefield
    (state.opponent -- only meaningful in a 2-player game) instead of the
    active player's own -- the general cross-player targeting primitive
    (docs/COMBAT_PLAN.md), first used by blocking. Addressed by (name,
    slot), not name alone: unlike begin_choose_permanent's own
    fungible-by-name simplification, two same-named OPPOSING permanents
    are exactly the case docs/MULTIPLAYER_GAPS.md's "Permanent identity"
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
    which of the attacker's declared, not-yet-blocked attackers it
    blocks -- until the defender chooses Done (docs/COMBAT_PLAN.md). No
    gang-blocking/menace: at most one blocker per attacker, at most one
    attacker per blocker.

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
    action's own legality check, but never crashes either way), records
    state.opponent.blocked_by[attacker] = blocker, then calls
    on_complete -- which drl_env.py uses to re-open begin_declare_blockers
    so the defender can assign another blocker or finish.

    extra_predicate(attacker) -> bool: an additional restriction beyond
    "is a currently-unblocked attacker" -- e.g. flying's own blocking
    restriction (docs/COMBAT_PLAN.md step 7: a flying attacker can only be
    blocked by a flying blocker). Supplied by the CALLER (drl_env.py)
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
            s.opponent.blocked_by[attacker] = blocker
        on_complete(s)

    begin_choose_opponent_permanent(
        state,
        lambda p: p in state.opponent.attackers and p not in state.opponent.blocked_by and extra_predicate(p),
        _on_attacker_chosen,
    )


def execute_choose_opponent_permanent_option(state, name, slot):
    complete_resolution(state, (name, slot))


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
    if pending["kind"] == "scry":
        state.rng.shuffle(disposed)
        state.library.extend(disposed)
    else:  # surveil
        state.graveyard.extend(disposed)
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
    moving it out of hand -- see docs/MADNESS_DECKS_PLAN.md items 1/3:
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
    the enclosing action's entire effect is fully resolved (docs/
    MADNESS_DECKS_PLAN.md items 1/3's cross-cutting rule). Shared by
    execute_discard_option's own per-card loop and execute_discard_or_
    sacrifice_option's single optional discard."""
    state.hand.remove(card)
    madness_spec = registry.EFFECT_REGISTRY.get(card.effect_id, {}).get("madness")
    if madness_spec is not None:
        state.exile.append((card, None))
        state.trigger_queue.append({"type": "decision", "kind": "madness", "card_def": card})
    else:
        state.graveyard.append(card)


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
    complete_resolution(state, True)


def execute_discard_or_sacrifice_decline(state):
    complete_resolution(state, False)


def begin_mulligan(state, on_complete):
    """Pregame: this player already has an opening 7-card hand (dealt by
    state.new_game_state/new_multiplayer_game_state's own eager draw(7)) --
    decide keep or mulligan (London Mulligan). Driven by
    game.turn.run_mulligan_phase/_run_mulligan_gen, once per player, before
    turn 1 ever starts."""
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
    state.library.extend(state.hand)
    state.hand = []
    state.rng.shuffle(state.library)
    state.mulligans_taken += 1
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
    complete_resolution(state)


# "cast" isn't handled here -- paying the madness cost needs
# game.mana.begin_pay_cost, which this module can't import (see the
# module docstring) -- see game.effects.madness_and_plot.execute_madness_cast.


def begin_order_triggers(state, entries, on_complete):
    """docs/PRIORITY_PLAN.md item 1: 2+ of the active player's own
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
    # (docs/PRIORITY_PLAN.md) -- state.active_idx here is still the
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
    state.stack.append(entry)  # already the stack's own native {"card_def", "resolve"} shape
    if not pending["remaining"]:
        complete_resolution(state)


def begin_sacrifice(state, predicate, n, on_complete):
    """Choose and sacrifice n of your own battlefield permanents matching
    predicate, one at a time -- same by-name fungibility every other
    resolution here uses. Generalizes what was originally Dread Return's
    own Flashback-cost-only "sacrifice_creatures" resolution
    (game.catalog.black_cards) into a predicate-based primitive reusable by land-
    sacrifice costs too (Fireblast's alt-cost, Lava Dart's Flashback,
    Highway Robbery's discard-or-sac choice -- MADNESS_DECKS_PLAN.md item
    5); Dread Return's own creature-sacrifice is just
    `begin_sacrifice(state, lambda p: p.card_def.card_type ==
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
    pending["remaining"] -= 1
    if pending["remaining"] <= 0:
        complete_resolution(state, True)


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.resolution`
    # from src/. Exercises begin_discard directly against a hand-built
    # state, bypassing drl_env.py entirely (no card wires into this
    # primitive yet -- deck assembly is out of scope for this plan).
    from .cards import CardDef, CardType
    from .state import GameState

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
    state = GameState(on_the_play=True)
    state.library = [_card(f"L{i}") for i in range(20)]
    state.rng.shuffle(state.library)
    state.draw(7)  # new_game_state's own eager opening draw -- begin_mulligan's own precondition
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
    from .cards import EffectId

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
        # stack's job in real play, docs/PRIORITY_PLAN.md item 1) and declining:
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
    # MADNESS_DECKS_PLAN.md item 5) against the same primitive.
    from .state import Permanent

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

    # Cross-player targeting (docs/COMBAT_PLAN.md): begin_choose_opponent_permanent
    # targets state.opponent's battlefield, addressed by (name, slot) --
    # not name alone, since two same-named OPPOSING permanents aren't
    # necessarily interchangeable (docs/MULTIPLAYER_GAPS.md's "Permanent
    # identity"). Only correct once the referencing player is already the
    # active one (blocking's own defender-decision channel flips
    # active_idx before ever calling this) -- simulated here by setting
    # active_idx directly to "the defender," same as that channel would.
    from .state import PlayerState

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

    # Blocking (docs/COMBAT_PLAN.md): begin_declare_blockers/
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
    assert state.players[0].blocked_by == {bear: grizzly}  # keyed by the ATTACKER, not the blocker

    # Re-open the consult (as drl_env._assign_blocker_execute's own nested
    # on_complete would) and assign Panther to the one remaining attacker --
    # Bear is no longer offered, already spoken for.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    step2_done = []
    declare_blocker_assignment(state, panther, on_complete=lambda s: step2_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Wolf", 1)]  # Bear no longer offered -- already blocked
    execute_choose_opponent_permanent_option(state, "Wolf", 1)
    assert step2_done == [True]
    assert state.players[0].blocked_by == {bear: grizzly, wolf: panther}

    # Both attackers now blocked: a further assignment attempt (only
    # reachable here because this test bypasses drl_env's own eligibility
    # gate, which wouldn't offer this action in real play once every
    # attacker's spoken for) finds no remaining unblocked attacker and
    # fizzles immediately with None, same empty-options safety net as
    # begin_choose_opponent_permanent's own -- never crashes, blocked_by
    # stays exactly as it was.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    step3_done = []
    declare_blocker_assignment(state, grizzly, on_complete=lambda s: step3_done.append(True))
    assert step3_done == [True]  # fizzled synchronously -- no eligible attacker left to choose
    assert state.players[0].blocked_by == {bear: grizzly, wolf: panther}  # unchanged
    assert state.pending_resolution is None  # the fizzle already completed it -- nothing left open

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

    # declare_blocker_assignment's extra_predicate (docs/COMBAT_PLAN.md
    # step 7's flying restriction): this module stays effect-agnostic (see
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
    assert state.players[0].blocked_by == {grounded: non_flying_blocker}

    print("resolution.py extra_predicate (flying-restriction wiring) self-check: OK")

    # begin_order_triggers (docs/PRIORITY_PLAN.md item 1): 2+ simultaneous
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
