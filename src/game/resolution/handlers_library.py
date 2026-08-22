"""Library/graveyard/hand manipulation: search/fetch, graveyard-card choices
(single and up-to-N), explore, exile-N-from-graveyard (Delve), tuck-to-
library, scry/surveil, put-N-on-top, ponder, discard (including Madness
routing), the combined discard-or-sacrifice cost, and sacrifice."""

from .. import registry
from ..cards import CardType
from ._core import begin_resolution, complete_resolution
from ..effects.shared import shuffle_library
from .handlers_targeting import begin_choose_permanent


def begin_search_fetch(state, predicate, on_complete, optional=False):
    """The model picks ONE library card by name, among distinct names
    matching `predicate`, to fetch -- one action per matching name, not a
    full reveal. on_complete(state, chosen_name); fizzles immediately with
    None if nothing in the library matches right now.

    optional=True (Gatecreeper Vine's "may search") offers a dedicated
    decline action rather than folding it into search_fetch_options."""
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
    used by Dread Return's reanimation target and Relic of Progenitus'
    repeatable exile ability alike. The pick is BY OBJECT IDENTITY:
    on_complete receives the exact chosen object (a CardInstance for a real
    graveyard; a CardDef for the deferred Mesmeric-Fiend hand-reveal path).

    graveyard=None defaults to state.graveyard (active-idx proxied). Pass an
    explicit graveyard list to target a different player's (Relic of
    Progenitus): when the rules say the TARGET player chooses, the caller
    flips active_idx to that player first and restores it after.

    optional=True (Masked Vandal's ETB) lets the model decline outright
    even when legal cards exist; optional=False (Dread Return, Relic) is
    mandatory when any card matches."""
    if graveyard is None:
        graveyard = state.graveyard
    begin_resolution(state, "choose_graveyard_card", on_complete, predicate=predicate, graveyard=graveyard, optional=optional)
    if not choose_graveyard_card_options(state):
        complete_resolution(state, None)


def choose_graveyard_card_options(state):
    """The matching graveyard cards themselves (objects), not names -- so two
    same-named copies are distinct choices. No dedup, no sort:
    rl.decision.action_bridge masks/executes by object identity."""
    pending = state.pending_resolution
    return [c for c in pending["graveyard"] if pending["predicate"](c)]


def execute_choose_graveyard_card_option(state, card):
    """`card` is the exact chosen object; the on_complete consumer acts on
    that exact object, so a specific copy among duplicates is reachable."""
    complete_resolution(state, card)


def execute_choose_graveyard_card_decline(state):
    """Decline an OPTIONAL choose_graveyard_card (Masked Vandal's "you may
    exile a creature card from your graveyard") -- resolve with None, exiling
    nothing."""
    complete_resolution(state, None)


def begin_choose_up_to_graveyard(state, predicate, max_targets, on_complete, graveyard=None):
    """Choose UP TO max_targets DISTINCT cards from a graveyard (default the
    active player's; pass a combined list for "from graveyardS"), one at a
    time. Each pick excludes those already chosen BY OBJECT IDENTITY, and is
    optional: declining ends selection early, as does running out of
    eligible cards. Runs on_complete(state, chosen_instances), a LIST of
    0..max_targets. The N-ary primitive behind "exile up to N target cards
    from graveyard(s)" (Rooftop Percher); fully fizzles only if all
    captured instances are gone by resolution (608.2c)."""
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
    puts a +1/+1 counter on the creature, then surveil 1 on that same
    still-on-top card. Empty library: nothing to reveal, no-op."""
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


def begin_exile_n_from_graveyard(state, n, on_complete, predicate=None, mid_cast=False):
    """Exile n cards from the active player's own graveyard as a COST, the
    model choosing which one at a time (chained begin_choose_graveyard_card).
    predicate (default any) narrows eligible cards. Used by Delve (Gurmag
    Angler) and reusable by any "exile N from your graveyard" cost. Runs
    on_complete(state) once n are exiled (or eligible cards run out).

    mid_cast marks each step's pending as part of an in-flight cast, making
    mana abilities illegal for its duration (CR 601.2f) -- Delve's exile
    happens between announcing the spell and the payment opening, so a tap
    taken here could narrow a color choice the payment depends on. Per-call,
    not per-KIND: choose_graveyard_card is also used at resolution time
    (Masked Vandal, Relic), where mana abilities are legal."""
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
        if mid_cast:
            state.pending_resolution["mid_cast"] = True

    _step(n)


def begin_tuck_to_library(state, card_def, owner_idx, on_complete=None):
    """Deem Inferior: a permanent's OWNER (active_idx flipped to them) puts
    its card into their library second-from-the-top or on the bottom, their
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
    dispose for each in turn, then -- if 2+ were kept -- the order to put
    them back in. Kept cards return to the library top in that order;
    disposed cards go to the library bottom in random order (kind="scry")
    or the graveyard (kind="surveil")."""
    revealed = state.library[:n]
    del state.library[:n]
    begin_resolution(state, kind, on_complete, remaining=revealed, kept=[], disposed=[], ordered=None)
    # Empty-reveal safety net: nothing to decide, so finish immediately
    # instead of leaving a zero-legal-actions resolution open.
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
    # Excludes a copy already reserved on the stack: a card on the stack is
    # NOT in your hand (real Magic), even though the engine defers its
    # hand-removal until it resolves -- same exclusion discard_options uses.
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
    """Ponder's "look at the top three cards, then put them back in any
    order. You may shuffle." The model either orders the revealed cards
    back on top one at a time (execute_ponder_option; FIRST placed ends on
    top) OR shuffles the whole library (execute_ponder_shuffle, legal only
    before any card has been placed). on_complete (Ponder's "Draw a card")
    runs either way."""
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
    """Discard n cards from hand, one at a time -- the model picks which, by
    name. optional=True additionally allows declining outright, discarding
    nothing (Highway Robbery's "you may discard a card"); optional=False is
    mandatory, discarding as many as n allows, down to whatever's actually
    in hand if it runs out first.

    Does not decide what happens to a discarded card beyond moving it out
    of hand: Madness reroutes qualifying cards to exile instead of the
    plain graveyard trip this module implements alone. on_complete(state,
    discarded_cards) fires once n discards are made, the model declines, or
    hand runs out -- discarded_cards is the list[CardDef] actually
    discarded, in order (empty if declined or hand was empty)."""
    begin_resolution(state, "discard", on_complete, remaining=n, optional=optional, discarded_cards=[])
    if not discard_options(state):
        complete_resolution(state, [])


def _available_hand_names(state):
    """Distinct names in hand still available to discard/pay as a cost --
    excluding any copy already reserved on state.stack (paid for, awaiting
    resolution; its own hand-removal is deferred until it resolves, so it's
    still physically in hand but must not be offered twice). Shared by
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
    state.pending_resolution['optional'] is True."""
    complete_resolution(state, state.pending_resolution["discarded_cards"])


def _discard_one(state, card):
    """Move `card` out of hand into the graveyard, EXCEPT a Madness card,
    which goes to exile with a queued cast-or-graveyard decision instead --
    a real-rules replacement effect that applies to ANY discard, regardless
    of why the card was discarded. Queued rather than offered immediately:
    the model only sees the decision once the enclosing action's entire
    effect is fully resolved. Shared by execute_discard_option's per-card
    loop and execute_discard_or_sacrifice_option's single optional discard."""
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
    (Highway Robbery) -- ONE optional decision offering two different cost
    shapes at once (real text is "a card OR a [land]," never both).
    on_complete(state, paid) -- True iff either a discard or a sacrifice
    happened, False if declined or nothing was payable.

    The SACRIFICE half is a real per-instance player choice: picking
    "sacrifice" opens a nested choose_permanent sub-decision for WHICH
    exact permanent pays it, not an arbitrary first-same-name match."""
    begin_resolution(state, "discard_or_sacrifice", on_complete, sac_predicate=sac_predicate)
    if not discard_or_sacrifice_discard_options(state) and not discard_or_sacrifice_can_sacrifice(state):
        complete_resolution(state, False)


def discard_or_sacrifice_discard_options(state):
    return _available_hand_names(state)


def discard_or_sacrifice_can_sacrifice(state):
    """Whether the SACRIFICE half has any eligible permanent right now --
    gates both drl_env's own legal() check and this function's begin-time
    check above."""
    pending = state.pending_resolution
    return any(pending["sac_predicate"](p) for p in state.battlefield)


def execute_discard_or_sacrifice_discard(state, name):
    card = next(c for c in state.hand if c.name == name)
    _discard_one(state, card)
    complete_resolution(state, True)


def execute_discard_or_sacrifice_trigger_sacrifice(state):
    """Opens the exact (name, slot) choose_permanent sub-decision for WHICH
    permanent pays this optional cost, never an arbitrary first-match.
    Captures the outer on_complete/sac_predicate before opening the nested
    resolution, since begin_choose_permanent replaces state.pending_
    resolution wholesale. Sacrificing the chosen permanent goes through
    state_based.sacrifice_to_graveyard, the one canonical "leave the
    battlefield by sacrifice" path."""
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
    permanent pays a "sacrifice a [type]" cost is the player's own choice
    among every eligible one, so this delegates to begin_choose_permanent
    once per permanent, chained via nested on_complete. Sacrificing each
    chosen permanent goes through state_based.sacrifice_to_graveyard.
    Covers both creature-sacrifice costs (Dread Return's Flashback) and
    land-sacrifice costs (Fireblast's alt-cost, Crop Rotation).

    Caller's legality check guarantees n eligible permanents exist before
    this is offered -- on_complete(state, True) once n are sacrificed
    (False only via the defensive n<=0 fallback)."""
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
