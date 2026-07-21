"""Pending resolution: a decision point that takes more than one action to
fully resolve -- paying a cost one tap at a time, walking a scry/surveil,
choosing a search target -- because the model, not an automatic solver,
makes every one of these choices.

This module holds the generic core (begin_resolution/complete_resolution)
plus every deck-agnostic resolution kind: search_fetch, choose_permanent,
scry/surveil, discard, madness_decision, sacrifice (predicate-based --
Dread Return's creature sacrifice and Fireblast/Lava Dart/Highway
Robbery's land sacrifice are the same primitive, different predicates;
see MADNESS_DECKS_PLAN.md item 5). Deck-specific kinds (Ancient Stirrings'
take-one-or-decline, Lead the Stampede's select_to_hand, Dread Return's
choose_graveyard_card) still live with their owning deck instead, since
nothing else currently reuses them.

References game.registry.EFFECT_REGISTRY only from inside function
bodies, via `registry.EFFECT_REGISTRY` -- same lazy-lookup convention
mana.py/effects_common.py already use, for the same reason (registry.py's
own import chain reaches this module before EFFECT_REGISTRY exists).
Deliberately does NOT import game.mana (mana.py imports THIS module at
its own top level; the reverse import would cycle) -- so the madness
"cast for its madness cost" path's cost payment lives in
effects_common.py instead, which is free to depend on both this module
and mana.py. See docs/MADNESS_DECKS_PLAN.md items 1/3.
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
    """The model picks ONE of its own battlefield permanents, by name,
    among those matching `predicate` -- e.g. Crop Rotation's sacrifice
    target. Same fungible-by-name simplification as mana's tap_cost_options:
    which physical copy doesn't matter, only which name. on_complete(state,
    chosen_name) runs once decided. Same empty-options safety net as
    begin_search_fetch -- fizzles immediately with chosen_name=None if
    nothing matches."""
    begin_resolution(state, "choose_permanent", on_complete, predicate=predicate)
    if not choose_permanent_options(state):
        complete_resolution(state, None)


def choose_permanent_options(state):
    predicate = state.pending_resolution["predicate"]
    return sorted({p.card_def.name for p in state.battlefield if predicate(p)})


def execute_choose_permanent_option(state, name):
    complete_resolution(state, name)


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


def discard_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    return sorted({c.name for c in state.hand})


def execute_discard_decline(state):
    """Only ever offered by the environment while
    state.pending_resolution['optional'] is True -- not itself enforced
    here, same convention as execute_search_fetch_decline."""
    complete_resolution(state, state.pending_resolution["discarded_cards"])


def execute_discard_option(state, name):
    pending = state.pending_resolution
    card = next(c for c in state.hand if c.name == name)
    state.hand.remove(card)
    madness_spec = registry.EFFECT_REGISTRY.get(card.effect_id, {}).get("madness")
    if madness_spec is not None:
        # Discard into exile instead of the graveyard, per Madness --
        # queue the cast-for-madness-cost-or-graveyard decision rather
        # than offering it immediately: the model only sees it once the
        # enclosing action's entire effect is fully resolved (see
        # docs/MADNESS_DECKS_PLAN.md items 1/3's cross-cutting rule).
        state.exile.append((card, None))
        state.trigger_queue.append({"type": "decision", "kind": "madness", "card_def": card})
    else:
        state.graveyard.append(card)
    pending["discarded_cards"].append(card)
    pending["remaining"] -= 1
    if pending["remaining"] <= 0 or not discard_options(state):
        complete_resolution(state, pending["discarded_cards"])


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
    drain in effects_common.py, once the discard's enclosing action is
    fully done -- never mid-discard."""
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
# module docstring) -- see effects_common.execute_madness_cast.


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

        # Draining the queue (effects_common.drain_trigger_queue's job in
        # real play) and declining: back out of exile, into the graveyard.
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
