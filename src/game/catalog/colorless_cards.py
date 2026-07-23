"""Colorless-identity card catalog: lands/artifacts with no colored mana
symbol in their cost and no fixed-color mana output (an "any color"
ability grants no specific color, matching real Magic's own
color-identity rule -- e.g. Bonder's Ornament, Tron lands). Every card's
cost/type/oracle-text below is a direct Scryfall pull, except creature
power/toughness, which is a design choice, not Scryfall data. Rooftop
Percher/Boulderbranch Golem/Maelstrom Colossus/Pinnacle Kill-Ship (Tron
filler) verified colorless via Scryfall, not guessed -- Bramble Wurm and
Breath Weapon, the other two Tron filler names, turned out to be green
and red respectively and file there instead.

Rooftop Percher/Boulderbranch Golem/Maelstrom Colossus/Pinnacle Kill-Ship
are cast at their real default cost/stats, with whichever clause is a
real ETB life-gain effect wired for real. Each one's OWN complicating
clause is a deliberate, documented drop rather than a guess:
- Rooftop Percher: Changeling (every creature type) is a no-op -- no
  tribal-synergy card exists anywhere in this catalog to care. Its own
  "exile up to two target cards from graveyards" is also dropped: in this
  solitaire sim the only legal targets are this player's own graveyard,
  and nothing rewards emptying it, so a rational cast always chooses zero
  targets anyway -- unlike Relic of Progenitus' own repeatable graveyard-
  exile ability (this file, activate_relic_of_progenitus_exile), which IS
  implemented despite the identical "no real upside" reasoning, simply
  because it was worth building as its own always-available action
  rather than an ETB-only one-off choice bundled into a bigger creature
  spell.
- Boulderbranch Golem: Prototype ({3}{G} for a 3/3 instead) would need a
  second CardDef with its own power/toughness for the exact same card
  name -- real, but disproportionate machinery for one filler creature.
  Dropped, same category as Nyxborn Hydra's own Bestow/X drop
  (game.catalog.green_cards) -- always cast at its real default {7} 6/5.
- Maelstrom Colossus: Cascade is implemented for real (cast_maelstrom_
  colossus below) -- it invokes an ARBITRARY other catalog card's own
  "cast" resolve, which normally assumes the card is already in
  state.hand (discard_from_hand_to_graveyard's own not-in-hand
  RuntimeError), by temporarily inserting the hit card into state.hand
  first: CardDefs are shared/interned per name (registry.CARD_DEFS holds
  one per distinct name, not per physical copy), so every existing
  resolve's own hand-removal correctly finds and removes it, with no
  parallel "cast from library" implementation needed for any card. Its
  own extra_legal is still checked first (Cascade only waives the MANA
  cost, not other costs -- same distinction Plot's own docstring already
  draws), and if the hit's own resolve opens a further pending resolution
  of its own (Ancient Stirrings' take-one-or-decline), Maelstrom
  Colossus's own battlefield entry is chained onto that resolution's
  on_complete rather than happening immediately -- see the function's own
  docstring.
- Pinnacle Kill-Ship: Station (tap another creature: charge counters;
  becomes a creature with flying at 7+) is dropped entirely, along with
  its ETB "10 damage to up to one target creature" (no beneficial target
  exists here either, same reasoning as Rooftop Percher above) -- every
  Tron config runs with combat_enabled=False, so becoming a creature/
  gaining flying would be permanently unobservable regardless. Stays a
  plain, never-a-creature Artifact.

"mana" shapes: ("tron",) -- Tron's controls-all-three-doubling rule;
("fixed", symbol) -- always produces that one symbol; ("flexible",
{symbols}) -- caller chooses one of several. "filter_mana": {"colors":
{...}} marks Barrels of Blasting Jelly's and Conduit Pylons' colored-pip
filter ability (as opposed to Conduit Pylons' plain {T}: Add {C}, which
IS a "fixed" mana source below) -- offered by mana.tap_cost_options for
any of the 5 colors, same as a flexible source (its own {1} activation
cost is tracked separately, see mana.execute_tap_cost_option)."""

from .. import registry
from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_permanent_from_hand, enters_battlefield
from ..effects.shared import discard_from_hand_to_graveyard, find_to_hand
from ..effects.tokens import activate_blood_sac
from ..effects.win_check import gain_life
from ..mana import COLORS
from ..resolution import begin_choose_graveyard_card, begin_choose_target_player, begin_search_fetch, scry, surveil

COLORLESS_CARD_CATALOG = {
    "Urza's Mine": CardDef("Urza's Mine", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Mine"),
    "Urza's Power Plant": CardDef("Urza's Power Plant", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Power Plant"),
    "Urza's Tower": CardDef("Urza's Tower", CardType.LAND, None, EffectId.TRON_LAND, tron_type="Tower"),
    "Tocasia's Dig Site": CardDef(
        "Tocasia's Dig Site", CardType.LAND, None, EffectId.TOCASIA_DIG_SITE,
        surveil_ability_cost={"generic": 3},
    ),
    "Conduit Pylons": CardDef("Conduit Pylons", CardType.LAND, None, EffectId.CONDUIT_PYLONS),
    "Expedition Map": CardDef(
        "Expedition Map", CardType.ARTIFACT, {"generic": 1}, EffectId.EXPEDITION_MAP, ability_cost={"generic": 2},
    ),
    "Bonder's Ornament": CardDef(
        "Bonder's Ornament", CardType.ARTIFACT, {"generic": 3}, EffectId.BONDERS_ORNAMENT,
        draw_ability_cost={"generic": 4},
    ),
    "Candy Trail": CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ),
    "Barrels of Blasting Jelly": CardDef(
        "Barrels of Blasting Jelly", CardType.ARTIFACT, {"generic": 1}, EffectId.BARRELS_OF_BLASTING_JELLY,
        mana_ability_cost={"generic": 1},
    ),
    "Relic of Progenitus": CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ),
    "Lotus Petal": CardDef("Lotus Petal", CardType.ARTIFACT, {}, EffectId.LOTUS_PETAL),
    "Rooftop Percher": CardDef(
        "Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3,
    ),
    "Boulderbranch Golem": CardDef(
        "Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5,
    ),
    "Maelstrom Colossus": CardDef(
        "Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7,
    ),
    "Pinnacle Kill-Ship": CardDef("Pinnacle Kill-Ship", CardType.ARTIFACT, {"generic": 7}, EffectId.PINNACLE_KILL_SHIP),

    # --- boggles deck ---
    "Ash Barrens": CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1}),
}


def activate_tocasia_dig_site_surveil(state, permanent):
    """{3}, T: Surveil 1 (shares the tap cost with its plain {T}: Add {C})."""
    permanent.tapped = True
    surveil(state, 1)


def activate_expedition_map(state, permanent):
    """{2}, T, Sacrifice: search library for a land -- the model's choice.
    Caller has already paid the {1} cost."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    begin_search_fetch(state, lambda c: c.card_type == CardType.LAND, find_to_hand)


def activate_bonders_ornament_draw(state, permanent):
    """{4}, T: draw a card (shares the tap cost with its plain mana ability)."""
    permanent.tapped = True
    state.draw(1)


def activate_candy_trail_sac(state, permanent):
    """{2}, T, Sacrifice: gain 3 life and draw a card."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    gain_life(state, 3)
    state.draw(1)


def activate_relic_of_progenitus_draw(state, permanent):
    """{1}, Exile this artifact: exile all graveyards, draw a card. "All
    graveyards" loops every PlayerState (not just the active one) --
    correct even in a hypothetical future 2-player config that plays this
    card, and costs nothing extra in the 1-player configs that actually
    do today. Exile itself untracked, same convention as this same
    artifact's own repeatable {T} ability below -- just clears each
    graveyard to nothing rather than tracking a real exile pile."""
    state.battlefield.remove(permanent)  # exiled, not graveyard; exile is untracked
    for player in state.players:
        player.graveyard.clear()
    state.draw(1)


def activate_relic_of_progenitus_exile(state, permanent):
    """{T}: Target player exiles a card from their graveyard -- a REAL
    target-player choice (resolution.begin_choose_target_player), not a
    self-target assumed on the ability's behalf: "yourself" is always one
    legal choice (true in real Magic even alone), "the opponent" becomes
    a second one the moment a real one exists, and the model picks
    explicitly either way via drl_env's own fixed "Target: yourself"/
    "Target: opponent" actions. Whichever player is targeted, then chooses
    (simplified to the ACTIVATING player's own choice, same "no observable
    difference in this solitaire sim" precedent begin_choose_graveyard_
    card's own docstring already documents) one of THAT player's
    graveyard cards to exile -- untracked, same convention as this same
    artifact's other, one-shot exile-self ability above. Repeatable (no
    mana cost, {T} only), independent of that other ability. An empty
    target graveyard -> nothing to choose, the same empty-options safety
    net begin_choose_graveyard_card already provides."""
    permanent.tapped = True

    def _on_player_chosen(state, idx):
        target_player = state.players[idx]

        def _on_card_chosen(state, name):
            if name is None:
                return
            found = next(c for c in target_player.graveyard if c.name == name)
            target_player.graveyard.remove(found)  # exiled, not removed-to-nowhere; exile is untracked

        begin_choose_graveyard_card(state, lambda c: True, _on_card_chosen, graveyard=target_player.graveyard)

    begin_choose_target_player(state, _on_player_chosen)


def _lotus_petal_on_tap(state, permanent):
    """{T}, Sacrifice: add one mana of any color -- consumed, not just
    tapped, unlike every other mana source in this engine."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)


def _lotus_petal_on_tap_undo(state, permanent):
    state.graveyard.remove(permanent.card_def)
    state.battlefield.append(permanent)


def _basic_land(card_def):
    return card_def.extra.get("basic", False)


def cycle_ash_barrens(state, card_def):
    """Basic landcycling {1}: discard this card from hand, search library
    for a basic land, put it into hand, shuffle. No draw-a-card rider (a
    plain Cycling ability would have one; Basic Landcycling doesn't --
    verified via Scryfall, not guessed), and the found land goes to hand,
    not the battlefield -- this is exactly Generous Ent's own forestcycle
    shape (game.catalog.green_cards), just with a real model choice of
    WHICH basic land (this decklist runs both Forest and Plains, unlike
    Generous Ent's single fixed "Forest" target)."""
    discard_from_hand_to_graveyard(state, card_def)
    begin_search_fetch(state, _basic_land, find_to_hand)


def _mana_value(cast_cost):
    """Total converted cost of a cast_cost dict ({"generic": 2, "G": 1} ->
    3) -- every symbol, colored or generic, contributes 1 per point. None
    (a land) is never passed in here; {} (Lotus Petal) correctly gives 0."""
    return sum(cast_cost.values())


def cast_maelstrom_colossus(state, card_def):
    """Cascade: exile cards from the top of the library until a nonland
    card with mana value LESS than this card's own (8) is exiled; cast it
    without paying its mana cost, then put the rest on the bottom in a
    random order (real text: "the exiled cards," i.e. every OTHER card
    seen along the way, not the hit itself).

    "Cast it for free" reuses the hit card's own registry "cast" spec
    completely unchanged -- CardDefs are shared/interned per name
    (registry.CARD_DEFS holds one per distinct name, not per physical
    copy), so temporarily inserting it into state.hand first makes every
    existing resolve's own hand-removal (discard_from_hand_to_graveyard's
    universal convention) correctly find and remove it, with no parallel
    "cast from library" implementation needed for any card, current or
    future. Skipped (no cast, straight to the bottom with everything
    else) if the hit has no "cast" spec at all (Generous Ent's own
    "never hard-cast" precedent -- forestcycle-only cards have none) or
    its own extra_legal fails (Cascade only waives the MANA cost, not any
    other cost or precondition a card's normal cast still gates on --
    same distinction Plot's own docstring already draws for Highway
    Robbery; Crop Rotation's "sacrifice a non-Tron land" extra_legal is
    this catalog's own live example).

    If the hit's own resolve opens a further pending resolution of its
    own (Ancient Stirrings' take-one-or-decline is this catalog's only
    such "cast" spec), Maelstrom Colossus's own battlefield entry can't
    happen yet -- chained onto that resolution's own on_complete instead
    of running immediately, so it lands only once the cascaded card's
    entire effect, decisions included, is actually done (matching real
    Magic: the Cascade trigger resolves completely before Colossus, still
    the next thing up on the stack, ever does)."""
    discard_from_hand_to_graveyard(state, card_def)
    exiled = []
    hit = None
    while state.library:
        card = state.library.pop(0)
        exiled.append(card)
        if card.card_type != CardType.LAND and _mana_value(card.cast_cost) < 8:
            hit = card
            break

    cast_spec = registry.EFFECT_REGISTRY.get(hit.effect_id, {}).get("cast") if hit is not None else None
    extra_legal = cast_spec.get("extra_legal") if cast_spec is not None else None
    can_cast = cast_spec is not None and (extra_legal is None or extra_legal(state))

    # Every exiled card goes to the bottom EXCEPT the hit, and only if
    # it's actually being cast -- a whiff, a hit with no "cast" spec, or
    # one whose own extra_legal fails all leave it right here among the
    # rest (real text: "Put THE EXILED CARDS on the bottom" -- a card
    # that was never cast is still one of them).
    remaining = [c for c in exiled if not (can_cast and c is hit)]
    state.rng.shuffle(remaining)
    state.library.extend(remaining)

    def _enter_colossus(state):
        enters_battlefield(state, card_def)

    if can_cast:
        state.hand.append(hit)
        cast_spec["resolve"](state, hit)
        if state.pending_resolution is not None:
            pending = state.pending_resolution
            inner_on_complete = pending["on_complete"]

            def _finish_cascade(state, *args):
                inner_on_complete(state, *args)
                _enter_colossus(state)

            pending["on_complete"] = _finish_cascade
            return
    _enter_colossus(state)


COLORLESS_EFFECT_REGISTRY = {
    EffectId.TRON_LAND: {
        "mana": ("tron",),
    },
    EffectId.TOCASIA_DIG_SITE: {
        "mana": ("fixed", "C"),
        "activated_abilities": {
            "surveil": {
                "cost_key": "surveil_ability_cost",
                "resolve": lambda state, permanent: activate_tocasia_dig_site_surveil(state, permanent),
            },
        },
        "pending_kinds": {"surveil"},
    },
    EffectId.CONDUIT_PYLONS: {
        "mana": ("fixed", "C"),
        "etb_trigger": lambda state: surveil(state, 1),
        "filter_mana": {"colors": set(COLORS)},
        "pending_kinds": {"surveil"},
    },
    EffectId.EXPEDITION_MAP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "activate": {
                "cost_key": "ability_cost",
                "resolve": lambda state, permanent: activate_expedition_map(state, permanent),
            },
        },
        "pending_kinds": {"search_fetch"},
    },
    EffectId.BONDERS_ORNAMENT: {
        "mana": ("flexible", set(COLORS)),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_bonders_ornament_draw(state, permanent),
            },
        },
    },
    EffectId.CANDY_TRAIL: {
        "etb_trigger": lambda state: scry(state, 2),
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_candy_trail_sac(state, permanent),
            },
        },
        "pending_kinds": {"scry"},
    },
    EffectId.BLOOD_TOKEN: {
        "activated_abilities": {
            "sac": {
                "cost_key": "sac_ability_cost",
                "resolve": lambda state, permanent: activate_blood_sac(state, permanent),
            },
        },
    },
    EffectId.BARRELS_OF_BLASTING_JELLY: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "filter_mana": {"colors": set(COLORS)},
    },
    EffectId.RELIC_OF_PROGENITUS: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "activated_abilities": {
            "draw": {
                "cost_key": "draw_ability_cost",
                "resolve": lambda state, permanent: activate_relic_of_progenitus_draw(state, permanent),
            },
            "exile": {
                "cost_key": "graveyard_exile_ability_cost",
                "resolve": lambda state, permanent: activate_relic_of_progenitus_exile(state, permanent),
            },
        },
        "pending_kinds": {"choose_target_player", "choose_graveyard_card"},
    },
    EffectId.LOTUS_PETAL: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "mana": ("flexible", set(COLORS)),
        "on_tap": lambda state, permanent: _lotus_petal_on_tap(state, permanent),
        "on_tap_undo": lambda state, permanent: _lotus_petal_on_tap_undo(state, permanent),
    },
    # EffectId.FILLER's single canonical registry entry -- every reader
    # consults it via EFFECT_REGISTRY.get(effect_id, {}), which already
    # defaults a missing key to {} the same way -- kept explicit here
    # (rather than omitted entirely) only because several
    # game/effects/*.py self-checks temporarily reassign
    # registry.EFFECT_REGISTRY[EffectId.FILLER] via direct bracket
    # indexing, which requires the key to already exist.
    EffectId.FILLER: {},
    EffectId.ROOFTOP_PERCHER: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        "etb_trigger": lambda state: gain_life(state, 3),
        "keywords": {"flying"},
    },
    EffectId.BOULDERBRANCH_GOLEM: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
        # Real text is "gain life equal to its power" -- power is fixed at
        # 6 in this simplified (no-Prototype) model, so this is that same
        # value, not a dynamic read.
        "etb_trigger": lambda state: gain_life(state, 6),
    },
    EffectId.MAELSTROM_COLOSSUS: {
        "cast": {"resolve": lambda state, card_def: cast_maelstrom_colossus(state, card_def)},
    },
    EffectId.PINNACLE_KILL_SHIP: {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
    },

    # --- boggles deck ---
    EffectId.ASH_BARRENS: {
        "mana": ("fixed", "C"),
        "forestcycle": {
            "cost_key": "cycling_cost",
            "resolve": lambda state, card_def: cycle_ash_barrens(state, card_def),
        },
        "pending_kinds": {"search_fetch"},
    },
}


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via
    # `python -m game.catalog.colorless_cards` from src/.
    from ..resolution import choose_graveyard_card_options, execute_choose_graveyard_card_option
    from ..state import GameState, Permanent

    # Basic landcycling {1}: discard this card from hand, search for a
    # basic land -- a real model choice between Forest and Plains (unlike
    # Generous Ent's own forestcycle, which always searches "Forest"
    # specifically), put into hand, shuffle. No draw-a-card rider (unlike
    # a plain Cycling ability) -- verified via Scryfall, not guessed.
    state = GameState(on_the_play=True)
    ash_barrens = CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1})
    state.hand = [ash_barrens]
    state.library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
        CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1}),  # not basic -- ineligible
    ]
    cycle_ash_barrens(state, ash_barrens)
    assert state.pending_resolution["kind"] == "search_fetch"
    from ..resolution import search_fetch_options, execute_search_fetch_option
    assert search_fetch_options(state) == ["Forest", "Plains"]  # the 2nd Ash Barrens is correctly excluded
    execute_search_fetch_option(state, "Plains")
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["Plains"]
    assert sorted(c.name for c in state.graveyard) == ["Ash Barrens"]  # discarded itself, not the fetched land
    assert sorted(c.name for c in state.library) == ["Ash Barrens", "Forest"]  # shuffled; the unchosen basic stays

    print("colorless_cards.py Ash Barrens self-check: OK")

    # Candy Trail's sac ability: gain 3 life AND draw a card -- both
    # halves of its real text, not just the draw.
    state = GameState(on_the_play=True)
    candy_trail = Permanent(CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ))
    state.battlefield = [candy_trail]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_candy_trail_sac(state, candy_trail)
    assert state.battlefield == []
    assert state.life_total == 23  # STARTING_LIFE (20) + 3
    assert [c.name for c in state.hand] == ["Forest"]
    print("colorless_cards.py Candy Trail self-check: OK")

    # Relic of Progenitus: two independent abilities now. The repeatable
    # {T} one is a REAL target-player choice, not a self-target assumed
    # on its behalf -- "yourself" chosen explicitly still exiles a real
    # card (Test A), and targeting a genuine opponent reaches into THEIR
    # graveyard instead (Test B), never state.graveyard (the active
    # player's own). The one-shot {1}+exile-self draw ability now also
    # clears every graveyard first (real text: "exile ALL graveyards"),
    # not just draw (Test C).
    from ..resolution import execute_choose_target_player_option

    # Test A: explicit self-target.
    state = GameState(on_the_play=True)
    relic = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state.battlefield = [relic]
    state.graveyard = [
        CardDef("Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM),
        CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON),
    ]
    activate_relic_of_progenitus_exile(state, relic)
    assert relic.tapped
    assert state.pending_resolution["kind"] == "choose_target_player"  # a real choice, not skipped
    execute_choose_target_player_option(state, 0)  # explicitly target yourself
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    assert choose_graveyard_card_options(state) == ["Bramble Wurm", "Breath Weapon"]
    execute_choose_graveyard_card_option(state, "Bramble Wurm")
    assert state.pending_resolution is None
    assert [c.name for c in state.graveyard] == ["Breath Weapon"]  # only the chosen one removed

    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_relic_of_progenitus_draw(state, relic)
    assert state.battlefield == []
    assert state.graveyard == []  # "exile ALL graveyards" -- the untouched "Breath Weapon" is gone too
    assert [c.name for c in state.hand] == ["Forest"]
    print("colorless_cards.py Relic of Progenitus (self-target + exile-all) self-check: OK")

    # Test B: a real opponent exists -- targeting them reaches into THEIR
    # graveyard, never the active player's own.
    from ..state import PlayerState

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    relic2 = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state2.players[0].battlefield = [relic2]
    state2.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state2.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    activate_relic_of_progenitus_exile(state2, relic2)
    execute_choose_target_player_option(state2, 1)  # target the opponent
    assert choose_graveyard_card_options(state2) == ["Theirs"]  # THEIR graveyard, not "Mine"
    execute_choose_graveyard_card_option(state2, "Theirs")
    assert state2.players[1].graveyard == []
    assert [c.name for c in state2.players[0].graveyard] == ["Mine"]  # own graveyard untouched
    print("colorless_cards.py Relic of Progenitus (real opponent target) self-check: OK")

    # Rooftop Percher / Boulderbranch Golem: real ETB gain-life triggers,
    # the one piece of genuinely new logic these two add now that they're
    # no longer inert EffectId.FILLER entries (cast_permanent_from_hand
    # and enters_battlefield's own etb_trigger dispatch are already
    # self-checked elsewhere -- casting.py, this just confirms these two
    # cards' own specific gain amounts are wired to the right effect_id).
    state = GameState(on_the_play=True)
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    state.hand = [percher]
    cast_permanent_from_hand(state, percher)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3

    golem = CardDef("Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5)
    state.hand = [golem]
    cast_permanent_from_hand(state, golem)
    assert state.life_total == 29  # +6 on top of the 23 above

    print("colorless_cards.py Tron filler creature self-check: OK")

    # Maelstrom Colossus's real Cascade -- four cases: a hit that's cast
    # for free, a whiff (nothing eligible), a hit whose own extra_legal
    # fails (skipped, same as real Magic), and a hit whose own resolve
    # opens a further pending resolution of its own (Colossus's entry
    # deferred until that completes).
    from .. import resolution
    from ..state import GameState as _GS
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        # Hit: a plain permanent (cast_permanent_from_hand), mana value 2
        # (< 8) -- exiled cards after it go to the bottom (this engine's
        # library is already a shuffled abstraction, so "bottom" just
        # means "back in state.library"), Colossus itself enters last.
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)}}
        state = _GS(on_the_play=True)
        colossus = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        a_land = CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True)
        hit = CardDef("Free Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        after = CardDef("Never Seen", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER)
        state.hand = [colossus]
        state.library = [a_land, hit, after]
        cast_maelstrom_colossus(state, colossus)
        assert state.graveyard == [colossus]
        assert sorted(p.card_def.name for p in state.battlefield) == ["Free Hit", "Maelstrom Colossus"]
        # Real Cascade only exiles cards UP TO AND INCLUDING the hit --
        # "Never Seen" was still sitting below it, never revealed/exiled
        # at all, so it stays exactly where it was; only what was
        # actually exiled alongside the hit ("A Land") gets shuffled and
        # returned, appended after it.
        assert [c.name for c in state.library] == ["Never Seen", "A Land"]
        print("colorless_cards.py Maelstrom Colossus Cascade (hit) self-check: OK")

        # Whiff: nothing eligible (everything's either a land or costs 8+)
        # -- Colossus still enters, nothing else does.
        state2 = _GS(on_the_play=True)
        colossus2 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        too_expensive = CardDef("Too Expensive", CardType.ARTIFACT, {"generic": 8}, EffectId.FILLER)
        state2.hand = [colossus2]
        state2.library = [a_land, too_expensive]
        cast_maelstrom_colossus(state2, colossus2)
        assert [p.card_def.name for p in state2.battlefield] == ["Maelstrom Colossus"]
        assert sorted(c.name for c in state2.library) == ["A Land", "Too Expensive"]

        # extra_legal fails: Cascade only waives the MANA cost, not other
        # preconditions -- skipped, straight to the bottom with the rest,
        # same as a genuine whiff.
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def), "extra_legal": lambda state: False},
        }
        state3 = _GS(on_the_play=True)
        colossus3 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        ineligible_hit = CardDef("Ineligible Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state3.hand = [colossus3]
        state3.library = [ineligible_hit]
        cast_maelstrom_colossus(state3, colossus3)
        assert [p.card_def.name for p in state3.battlefield] == ["Maelstrom Colossus"]
        assert [c.name for c in state3.library] == ["Ineligible Hit"]  # never cast, just shuffled back in

        # The hit's own resolve opens a further pending resolution (an
        # Ancient-Stirrings-like take-it-or-decline) -- Colossus must NOT
        # enter until that resolution is actually completed.
        entered_before_decision = []

        def _opens_pending(state, card_def):
            state.hand.remove(card_def)

            def _on_complete(state, taken):
                if taken:
                    state.graveyard.append(card_def)
                entered_before_decision.append(any(p.card_def.name == "Maelstrom Colossus" for p in state.battlefield))

            resolution.begin_resolution(state, "fake_choice", _on_complete)

        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": _opens_pending}}
        state4 = _GS(on_the_play=True)
        colossus4 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        decision_hit = CardDef("Decision Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state4.hand = [colossus4]
        state4.library = [decision_hit]
        cast_maelstrom_colossus(state4, colossus4)
        assert state4.pending_resolution["kind"] == "fake_choice"  # Colossus genuinely hasn't entered yet
        assert not any(p.card_def.name == "Maelstrom Colossus" for p in state4.battlefield)
        resolution.complete_resolution(state4, True)
        assert entered_before_decision == [False]  # Colossus hadn't entered DURING the decision's own on_complete either
        assert sorted(p.card_def.name for p in state4.battlefield) == ["Maelstrom Colossus"]  # decision_hit chose graveyard, not battlefield, in this fake resolve
        assert state4.pending_resolution is None

        print("colorless_cards.py Maelstrom Colossus Cascade (whiff/extra_legal/chained-resolution) self-check: OK")
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
