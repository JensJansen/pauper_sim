"""Library/graveyard/hand manipulation: search/fetch, graveyard-card choices
(single and up-to-N), explore, exile-N-from-graveyard (Delve), tuck-to-
library, scry/surveil, put-N-on-top, ponder, discard (including Madness
routing), the combined discard-or-sacrifice cost, and sacrifice. Re-exported
via game.resolution so `from ..resolution import X` in the catalogs keeps
resolving."""

from .. import registry
from ..cards import CardType
from ._core import begin_resolution, complete_resolution
from ..effects.shared import shuffle_library
from ..state import known_top_prefix
from .handlers_targeting import begin_choose_permanent


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
    # The controller chose these and obviously remembers them -- record it so
    # the agent's observation matches what a real player knows. Prepended, not
    # assigned: an earlier Brainstorm's cards may still be sitting underneath
    # these, and they are still known. game.state.known_top_prefix validates the
    # whole list against the real library at read time, so a stale tail can
    # never produce a false claim.
    placer = state.players[state.active_idx]
    placer.known_top = list(pending["placed"]) + known_top_prefix(placer)
    placer.known_top_library_len = len(placer.library)
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
    shuffle_library(state)
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
    bool(discarded_cards) contract already has.

    The SACRIFICE half is a real per-instance player choice, same fix as
    begin_sacrifice's own: picking "sacrifice" (execute_discard_or_
    sacrifice_trigger_sacrifice) opens a nested choose_permanent
    sub-decision for WHICH exact permanent pays it, instead of an
    arbitrary first-same-name match -- a Utopia-Sprawl/Abundant-Growth-
    enchanted land is not interchangeable with a bare one of the same
    name."""
    begin_resolution(state, "discard_or_sacrifice", on_complete, sac_predicate=sac_predicate)
    if not discard_or_sacrifice_discard_options(state) and not discard_or_sacrifice_can_sacrifice(state):
        complete_resolution(state, False)


def discard_or_sacrifice_discard_options(state):
    return _available_hand_names(state)


def discard_or_sacrifice_can_sacrifice(state):
    """Whether the SACRIFICE half has any eligible permanent right now --
    gates both the trigger action's own legal() (drl_env._discard_or_
    sacrifice_trigger_sacrifice_legal) and this function's own begin-time
    check above."""
    pending = state.pending_resolution
    return any(pending["sac_predicate"](p) for p in state.battlefield)


def execute_discard_or_sacrifice_discard(state, name):
    card = next(c for c in state.hand if c.name == name)
    _discard_one(state, card)
    complete_resolution(state, True)


def execute_discard_or_sacrifice_trigger_sacrifice(state):
    """Opens the exact (name, slot) choose_permanent sub-decision for WHICH
    permanent pays this optional cost -- same real per-instance choice
    begin_sacrifice's own predicate-driven picks get (see its own
    docstring), never an arbitrary first-match. Captures the outer
    on_complete/sac_predicate before opening the nested resolution --
    begin_choose_permanent's own begin_resolution call replaces
    state.pending_resolution wholesale, same "capture first" contract
    declare_blocker_assignment's own nested chain already documents.
    Actually sacrificing the chosen permanent goes through state_based.
    sacrifice_to_graveyard, the one canonical "leave the battlefield by
    sacrifice" path (token-ceases / DFC-front-face / dies-trigger handling
    shared with Lembas, Chromatic Star, ...) instead of a second, inlined
    copy of that logic."""
    from ..effects.state_based import sacrifice_to_graveyard
    outer_on_complete = state.pending_resolution["on_complete"]
    sac_predicate = state.pending_resolution["sac_predicate"]

    def _after_choice(state, choice):
        name, slot = choice
        permanent = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        sacrifice_to_graveyard(state, permanent)
        outer_on_complete(state, True)

    begin_choose_permanent(state, sac_predicate, _after_choice)


def execute_discard_or_sacrifice_decline(state):
    complete_resolution(state, False)


def begin_sacrifice(state, predicate, n, on_complete):
    """Choose and sacrifice n of your own battlefield permanents matching
    predicate, one at a time. Real Magic (602.5a/602.5g): naming WHICH
    permanent pays a "sacrifice a [type]" cost is the PLAYER'S OWN choice
    among every eligible one -- two same-named permanents stop being
    interchangeable the instant one differs from the other (an attached
    Aura, a counter, ...), so this delegates to begin_choose_permanent (the
    SAME exact-(name, slot) primitive Aura targets/Crop Rotation's own
    sacrifice cost already use) once per permanent, chained via nested
    on_complete -- same repeated-resolution shape declare_blocker_
    assignment/begin_choose_delve_amount's own callers already use
    elsewhere. Actually sacrificing each chosen permanent goes through
    state_based.sacrifice_to_graveyard, the one canonical "leave the
    battlefield by sacrifice" path (token-ceases / DFC-front-face /
    dies-trigger handling all shared with Lembas, Chromatic Star, Nihil
    Spellbomb, ...) instead of a second, inlined copy of that logic.

    A predicate-based primitive covering both creature-sacrifice costs
    (Dread Return's own Flashback: `begin_sacrifice(state, lambda p:
    p.card_type == CardType.CREATURE, 3, on_complete)`) and land-sacrifice
    costs (Fireblast's alt-cost, Lava Dart's Flashback, Crop Rotation's own
    sacrifice).

    Caller's own legality check guarantees n eligible permanents exist
    before this is ever offered (same "guaranteed payable, not a maybe"
    contract every alternate cost path here already follows) --
    on_complete(state, True) once n are sacrificed (False only via the
    defensive n<=0 fallback, matching every other pending kind here)."""
    if n <= 0:
        on_complete(state, False)
        return
    _sacrifice_n_via_choice(state, predicate, n, on_complete)


def _sacrifice_n_via_choice(state, predicate, remaining, on_complete):
    from ..effects.state_based import sacrifice_to_graveyard

    def _after_choice(state, choice):
        if choice is None:  # belt-and-suspenders -- begin_choose_permanent's own empty-options fallback
            on_complete(state, False)
            return
        name, slot = choice
        permanent = next(p for p in state.battlefield if p.card_def.name == name and p.slot == slot)
        sacrifice_to_graveyard(state, permanent)
        if remaining - 1 <= 0:
            on_complete(state, True)
        else:
            _sacrifice_n_via_choice(state, predicate, remaining - 1, on_complete)

    begin_choose_permanent(state, predicate, _after_choice)
