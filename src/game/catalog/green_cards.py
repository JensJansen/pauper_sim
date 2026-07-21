"""Green-identity card catalog: every card whose real mana cost is
mono-green (or, for lands with no cost, whose only mana output is green).
Every card's cost/type/oracle-text below is a direct Scryfall pull,
except creature power/toughness, which is a design choice, not Scryfall
data. Bramble Wurm (Tron filler, real cost {6}{G}) files here rather
than colorless_cards.py -- verified via Scryfall, not guessed. Sagu
Wildling is implemented as its Adventure sorcery half only ("Roost
Seek": search a basic land to hand) -- the creature side is dropped per
design discussion, so this entry is CardType.SORCERY, not CREATURE, even
though it keeps the printed card's name (decklist readability + Scryfall
art lookup). "defender" marks the four defender creatures Overgrown
Battlement's own mana ability counts (itself included)."""

from .. import resolution
from ..cards import CardDef, CardType, EffectId
from ..effects_common import cast_permanent_from_hand, enters_battlefield, find_and_remove_by_name
from ..mana import COLORS

GREEN_CARD_CATALOG = {
    "Forest": CardDef("Forest", CardType.LAND, None, EffectId.FOREST),
    "Generous Ent": CardDef(
        "Generous Ent", CardType.CREATURE, {"generic": 5, "G": 1}, EffectId.GENEROUS_ENT,
        forestcycling_cost={"generic": 1},
    ),
    "Masked Vandal": CardDef("Masked Vandal", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.MASKED_VANDAL),
    "Saruli Caretaker": CardDef("Saruli Caretaker", CardType.CREATURE, {"G": 1}, EffectId.SARULI_CARETAKER, defender=True),
    "Overgrown Battlement": CardDef(
        "Overgrown Battlement", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.OVERGROWN_BATTLEMENT, defender=True,
    ),
    "Wall of Roots": CardDef("Wall of Roots", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.WALL_OF_ROOTS, defender=True),
    "Sagu Wildling": CardDef("Sagu Wildling", CardType.SORCERY, {"G": 1}, EffectId.ROOST_SEEK),
    "Gatecreeper Vine": CardDef(
        "Gatecreeper Vine", CardType.CREATURE, {"generic": 1, "G": 1}, EffectId.GATECREEPER_VINE, defender=True,
    ),
    "Nyxborn Hydra": CardDef("Nyxborn Hydra", CardType.CREATURE, {"G": 1}, EffectId.NYXBORN_HYDRA),
    "Quirion Ranger": CardDef("Quirion Ranger", CardType.CREATURE, {"G": 1}, EffectId.QUIRION_RANGER),
    "Winding Way": CardDef("Winding Way", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.WINDING_WAY),
    "Lead the Stampede": CardDef("Lead the Stampede", CardType.SORCERY, {"generic": 2, "G": 1}, EffectId.LEAD_THE_STAMPEDE),
    "Land Grant": CardDef("Land Grant", CardType.SORCERY, {"generic": 1, "G": 1}, EffectId.LAND_GRANT),
    "Crop Rotation": CardDef("Crop Rotation", CardType.INSTANT, {"G": 1}, EffectId.CROP_ROTATION),
    "Ancient Stirrings": CardDef("Ancient Stirrings", CardType.SORCERY, {"G": 1}, EffectId.ANCIENT_STIRRINGS),
    "Bramble Wurm": CardDef("Bramble Wurm", CardType.FILLER, None, EffectId.FILLER),
}


def _is_defender(permanent):
    return permanent.card_def.extra.get("defender", False)


def forestcycle_generous_ent(state, card_def):
    """{1}, discard this card from hand: search library for a Forest, put
    into hand, shuffle. Only one possible target name, so this resolves
    immediately -- no model choice/pending resolution needed."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    found = find_and_remove_by_name(state, "Forest")
    state.rng.shuffle(state.library)
    if found:
        state.hand.append(found)


def _saruli_caretaker_extra_available(state, permanent):
    """Saruli Caretaker's mana ability costs {T}, tap an untapped creature
    you control (not itself) -- not offered as a mana source unless
    another untapped creature exists to pay that extra cost."""
    return any(
        p is not permanent and not p.tapped and p.card_def.card_type == CardType.CREATURE
        for p in state.battlefield
    )


def _saruli_caretaker_on_tap(state, permanent):
    """Which specific other creature gets tapped doesn't matter (same
    fungible-by-name simplification used throughout this engine) --
    auto-picks the first untapped one. Recorded on Saruli's own flags so
    on_tap_undo can reverse exactly this tap if the payment is abandoned."""
    other = next(
        (p for p in state.battlefield
         if p is not permanent and not p.tapped and p.card_def.card_type == CardType.CREATURE),
        None,
    )
    if other is not None:
        other.tapped = True
        permanent.flags["tapped_other"] = other


def _saruli_caretaker_on_tap_undo(state, permanent):
    other = permanent.flags.pop("tapped_other", None)
    if other is not None:
        other.tapped = False


def _wall_of_roots_on_tap(state, permanent):
    """Put a -0/-1 counter on this creature: add {G}, once each turn --
    modeled per design discussion as a plain ("fixed", "G") source (once-
    per-turn already falls out of tapping) plus this activation counter,
    rather than a general counters/toughness/state-based-death system.
    Dies on its 5th use."""
    permanent.flags["roots_activations"] = permanent.flags.get("roots_activations", 0) + 1
    if permanent.flags["roots_activations"] >= 5:
        state.battlefield.remove(permanent)
        state.graveyard.append(permanent.card_def)


def _wall_of_roots_on_tap_undo(state, permanent):
    permanent.flags["roots_activations"] -= 1
    if permanent not in state.battlefield:
        state.battlefield.append(permanent)
        state.graveyard.remove(permanent.card_def)


def _search_to_hand(state, name):
    """Shared on_complete callback for search-and-reshuffle-into-hand
    effects (Roost Seek, Gatecreeper Vine)."""
    found = find_and_remove_by_name(state, name) if name is not None else None
    state.rng.shuffle(state.library)
    if found:
        state.hand.append(found)


def cast_roost_seek(state, card_def):
    """Sagu Wildling's Adventure sorcery half -- the only half this
    simulator implements. {G}: search library for a basic land. Two
    possible names here (Forest or Swamp), a real model choice, unlike
    Land Grant's single fixed target below."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, lambda s, name: _search_to_hand(s, name))


def gatecreeper_vine_etb(state):
    """ETB: may search a basic land to hand -- optional even when a target
    exists, unlike Expedition Map/Crop Rotation's mandatory fetches."""
    resolution.begin_search_fetch(
        state, lambda c: c.card_type == CardType.LAND, lambda s, name: _search_to_hand(s, name), optional=True,
    )


def cast_land_grant(state, card_def):
    """Search library for a Forest specifically -- single target name, so
    this resolves immediately, no pending resolution. Serves both the
    normal {1}{G} cast and the free alt-cost cast below -- they differ
    only in how the cost was paid."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    found = find_and_remove_by_name(state, "Forest")
    state.rng.shuffle(state.library)
    if found:
        state.hand.append(found)


def land_grant_alt_cost_legal(state):
    """Land Grant's free alt-cost ("reveal your hand" instead of paying):
    legal only with no land cards in hand. Revealing the hand has no
    simulator-visible effect (solitaire, no opponent to show it to) --
    this predicate is the only real consequence of that clause."""
    return not any(c.card_type == CardType.LAND for c in state.hand)


def quirion_ranger_untap_legal(state, permanent):
    """Return a Forest you control to hand: untap target creature. Once
    each turn (the used_this_turn flag, reset for every permanent by
    untap_step regardless of which card set it -- same mechanism Barrels
    of Blasting Jelly's filter ability already relies on). No {T} in this
    ability's real cost, so -- unlike every other activated ability here
    -- it doesn't require this permanent itself to be untapped."""
    if permanent.flags.get("used_this_turn", False):
        return False
    return any(p.card_def.name == "Forest" for p in state.battlefield)


def quirion_ranger_untap_resolve(state, permanent):
    permanent.flags["used_this_turn"] = True
    forest = next(p for p in state.battlefield if p.card_def.name == "Forest")
    state.battlefield.remove(forest)
    state.hand.append(forest.card_def)

    def _on_chosen(state, name):
        if name is None:
            return
        target = next(p for p in state.battlefield if p.card_def.name == name)
        target.tapped = False

    resolution.begin_choose_permanent(state, lambda p: p.card_def.card_type == CardType.CREATURE, _on_chosen)


def _cast_winding_way(state, card_def, chosen_type):
    """Choose creature or land at cast time -- two separate action-table
    entries, not a pending resolution. Reveal top 4; matches to hand, the
    rest to the graveyard. Fully deterministic given the chosen type -- no
    further model choice."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    revealed = state.library[:4]
    del state.library[:4]
    for card in revealed:
        if card.card_type == chosen_type:
            state.hand.append(card)
        else:
            state.graveyard.append(card)


def cast_winding_way_creature(state, card_def):
    _cast_winding_way(state, card_def, CardType.CREATURE)


def cast_winding_way_land(state, card_def):
    _cast_winding_way(state, card_def, CardType.LAND)


def begin_select_to_hand(state, n, eligible_predicate, on_complete):
    """Lead the Stampede: reveal top n; the model decides keep-to-hand
    (only if eligible_predicate matches) or bottom for each in turn, then
    -- if 2+ went to the bottom -- the order to put them there. Mirrors
    game.resolution.begin_scry_surveil's remaining/kept/disposed/ordered
    shape exactly, except "kept" lands in hand (not library top) and only
    eligible cards may be kept."""
    revealed = state.library[:n]
    del state.library[:n]
    resolution.begin_resolution(
        state, "select_to_hand", on_complete,
        remaining=revealed, eligible=eligible_predicate, kept=[], disposed=[], ordered=None,
    )
    if not revealed:
        # Library was already empty (this deck's own mill-out combo can get
        # here) -- nothing to decide, so complete immediately instead of
        # leaving a pending resolution with zero legal actions.
        resolution.complete_resolution(state)


def select_to_hand_options(state):
    """While deciding (remaining non-empty): keep (only if the front card
    is eligible) or bottom. While ordering (remaining empty, 2+ disposed,
    not yet all placed): one option per distinct name still waiting to be
    bottomed."""
    pending = state.pending_resolution
    if pending["remaining"]:
        front = pending["remaining"][0]
        return ["keep", "bottom"] if pending["eligible"](front) else ["bottom"]
    if pending["ordered"] is not None:
        return sorted({c.name for c in pending["disposed"]})
    return []


def _finish_select_to_hand(state):
    pending = state.pending_resolution
    state.hand.extend(pending["kept"])
    disposed_final = pending["ordered"] if pending["ordered"] is not None else pending["disposed"]
    state.library.extend(disposed_final)
    resolution.complete_resolution(state)


def execute_select_to_hand_option(state, option):
    pending = state.pending_resolution
    if pending["remaining"]:
        card = pending["remaining"].pop(0)
        (pending["kept"] if option == "keep" else pending["disposed"]).append(card)
        if pending["remaining"]:
            return  # more cards still to decide
        if len(pending["disposed"]) <= 1:
            _finish_select_to_hand(state)  # 0 or 1 bottomed -- no ordering choice to make
        else:
            pending["ordered"] = []  # 2+ bottomed -- enter the ordering phase
        return

    # Ordering phase: option is the name of the next card to bottom.
    idx = next(i for i, c in enumerate(pending["disposed"]) if c.name == option)
    pending["ordered"].append(pending["disposed"].pop(idx))
    if not pending["disposed"]:
        _finish_select_to_hand(state)


def cast_lead_the_stampede(state, card_def):
    """{2}{G}: look at top 5, may reveal any number of creatures to hand,
    rest to the bottom in any order."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    begin_select_to_hand(state, 5, lambda c: c.card_type == CardType.CREATURE, on_complete=lambda s: None)


def is_noncreature_colorless(card_def):
    if card_def.card_type in (CardType.CREATURE, CardType.FILLER):
        return False
    if card_def.cast_cost is None:
        return True  # a land -- no mana cost, therefore colorless
    return not any(k in COLORS for k in card_def.cast_cost)


def cast_crop_rotation(state, card_def):
    """{G}, sacrifice a land: search library for a land, put it directly
    onto the battlefield (its own normal tapped/ETB rules apply), shuffle.
    Both the sacrifice target and the fetch target are the model's choice
    (begins a choose_permanent resolution for the sacrifice, chaining into
    a search_fetch resolution for the fetch). Caller has already paid the
    {G} cost."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)

    def _on_sac_chosen(state, sac_name):
        if sac_name is None:
            return  # begin_choose_permanent found no valid sacrifice target -- fizzle, shouldn't happen per legality, but don't crash if it somehow does
        sac_permanent = next(p for p in state.battlefield if p.card_def.name == sac_name)
        state.battlefield.remove(sac_permanent)
        state.graveyard.append(sac_permanent.card_def)

        def _on_fetch_chosen(state, land_name):
            found = find_and_remove_by_name(state, land_name)
            state.rng.shuffle(state.library)
            if found:
                enters_battlefield(state, found)

        resolution.begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, _on_fetch_chosen)

    resolution.begin_choose_permanent(
        state,
        lambda p: p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND,
        _on_sac_chosen,
    )


def begin_ancient_stirrings(state, revealed, on_complete):
    """The model picks at most one noncreature-colorless card among
    `revealed` to take, or declines -- a single decision, not a
    sequential walk like scry/surveil (Ancient Stirrings only ever takes
    one, if any). on_complete(state, chosen_card_or_None) runs once
    decided."""
    resolution.begin_resolution(state, "ancient_stirrings", on_complete, revealed=revealed)


def ancient_stirrings_options(state):
    revealed = state.pending_resolution["revealed"]
    eligible_names = sorted({c.name for c in revealed if is_noncreature_colorless(c)})
    return eligible_names + ["decline"]


def execute_ancient_stirrings_option(state, option):
    revealed = state.pending_resolution["revealed"]
    if option == "decline":
        chosen = None
    else:
        idx = next(i for i, c in enumerate(revealed) if c.name == option)
        chosen = revealed.pop(idx)
    state.rng.shuffle(revealed)  # whatever's left (all of it, if declined) goes to the bottom
    state.library.extend(revealed)
    resolution.complete_resolution(state, chosen)


def cast_ancient_stirrings(state, card_def):
    """{G}: look at top 5, may take one noncreature colorless card to hand
    -- the model's choice among eligible ones, or decline -- rest to
    bottom in random order."""
    state.hand.remove(card_def)
    state.graveyard.append(card_def)
    top = state.library[:5]
    del state.library[:5]

    def _on_chosen(state, chosen):
        if chosen is not None:
            state.hand.append(chosen)

    begin_ancient_stirrings(state, top, _on_chosen)


GREEN_EFFECT_REGISTRY = {
    EffectId.FOREST: {
        "mana": ("fixed", "G"),
    },
    EffectId.GENEROUS_ENT: {
        # Never hard-cast -- only forestcycled.
        "forestcycle": {
            "cost_key": "forestcycling_cost",
            "resolve": lambda state, card_def: forestcycle_generous_ent(state, card_def),
        },
    },
    EffectId.MASKED_VANDAL: {
        # No ability -- functionally a vanilla 1/3 for {1}{G} (P/T isn't
        # tracked anywhere in this engine; see design discussion).
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
    },
    EffectId.SARULI_CARETAKER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("flexible", set(COLORS)),
        "mana_extra_available": lambda state, permanent: _saruli_caretaker_extra_available(state, permanent),
        "on_tap": lambda state, permanent: _saruli_caretaker_on_tap(state, permanent),
        "on_tap_undo": lambda state, permanent: _saruli_caretaker_on_tap_undo(state, permanent),
    },
    EffectId.OVERGROWN_BATTLEMENT: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("count", "G", _is_defender),
    },
    EffectId.WALL_OF_ROOTS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("fixed", "G"),
        "on_tap": lambda state, permanent: _wall_of_roots_on_tap(state, permanent),
        "on_tap_undo": lambda state, permanent: _wall_of_roots_on_tap_undo(state, permanent),
    },
    EffectId.ROOST_SEEK: {
        "cast": {"resolve": lambda state, card_def: cast_roost_seek(state, card_def)},
        "pending_kinds": {"search_fetch"},
    },
    EffectId.GATECREEPER_VINE: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: gatecreeper_vine_etb(state),
        "pending_kinds": {"search_fetch"},
    },
    EffectId.NYXBORN_HYDRA: {
        # Cast as a fixed 0/1 for {G} -- X permanently 0, no Bestow, no
        # counters (a deliberate simplification per design discussion,
        # same treatment as Candy Trail's omitted lifegain).
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
    },
    EffectId.QUIRION_RANGER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "untap": {
                "legal": lambda state, permanent: quirion_ranger_untap_legal(state, permanent),
                "resolve": lambda state, permanent: quirion_ranger_untap_resolve(state, permanent),
            },
        },
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.WINDING_WAY: {
        "cast_modes": {
            "creature": {"resolve": lambda state, card_def: cast_winding_way_creature(state, card_def)},
            "land": {"resolve": lambda state, card_def: cast_winding_way_land(state, card_def)},
        },
    },
    EffectId.LEAD_THE_STAMPEDE: {
        "cast": {"resolve": lambda state, card_def: cast_lead_the_stampede(state, card_def)},
        "pending_kinds": {"select_to_hand"},
    },
    EffectId.LAND_GRANT: {
        "cast": {"resolve": lambda state, card_def: cast_land_grant(state, card_def)},
        "alt_cast": {
            "extra_legal": lambda state: land_grant_alt_cost_legal(state),
            "resolve": lambda state, card_def: cast_land_grant(state, card_def),
        },
    },
    EffectId.CROP_ROTATION: {
        "cast": {
            "resolve": lambda state, card_def: cast_crop_rotation(state, card_def),
            "extra_legal": lambda state: any(
                p.card_def.card_type == CardType.LAND and p.card_def.effect_id != EffectId.TRON_LAND
                for p in state.battlefield
            ),
        },
        "pending_kinds": {"choose_permanent", "search_fetch"},
    },
    EffectId.ANCIENT_STIRRINGS: {
        "cast": {"resolve": lambda state, card_def: cast_ancient_stirrings(state, card_def)},
        "pending_kinds": {"ancient_stirrings"},
    },
    # Bramble Wurm (filler): no entry needed, same EffectId.FILLER
    # single-canonical-entry precedent as Breath Weapon (red_cards.py).
}
