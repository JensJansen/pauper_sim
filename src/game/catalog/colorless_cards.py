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
  targets anyway (same "no beneficial target exists" reasoning as Relic
  of Progenitus' own graveyard-hate ability).
- Boulderbranch Golem: Prototype ({3}{G} for a 3/3 instead) would need a
  second CardDef with its own power/toughness for the exact same card
  name -- real, but disproportionate machinery for one filler creature.
  Dropped, same category as Nyxborn Hydra's own Bestow/X drop
  (game.catalog.green_cards) -- always cast at its real default {7} 6/5.
- Maelstrom Colossus: Cascade would mean invoking an ARBITRARY other
  catalog card's own "cast" resolve outside its normal hand-cast
  convention (nearly every resolve in this catalog assumes the card is
  already in state.hand, e.g. discard_from_hand_to_graveyard's own
  not-in-hand RuntimeError) -- a correctness liability against every
  CURRENT AND FUTURE card, not a one-off. Dropped; cast as a vanilla 7/7.
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

from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_permanent_from_hand
from ..effects.shared import discard_from_hand_to_graveyard, find_to_hand
from ..effects.tokens import activate_blood_sac
from ..effects.win_check import gain_life
from ..mana import COLORS
from ..resolution import begin_search_fetch, scry, surveil

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
        draw_ability_cost={"generic": 1},
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
    """{2}, T, Sacrifice: draw a card (lifegain omitted, see plan)."""
    state.battlefield.remove(permanent)
    state.graveyard.append(permanent.card_def)
    state.draw(1)


def activate_relic_of_progenitus(state, permanent):
    """{1}, Exile this artifact: draw a card (graveyard-exile tap ability
    omitted -- no-op with no opposing graveyard, see plan)."""
    state.battlefield.remove(permanent)  # exiled, not graveyard; exile is untracked
    state.draw(1)


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
                "resolve": lambda state, permanent: activate_relic_of_progenitus(state, permanent),
            },
        },
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
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
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
    from ..state import GameState

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
