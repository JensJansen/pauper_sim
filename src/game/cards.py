"""Card data model shared by every deck: EffectId, CardType, CardDef.

A CardDef is the definition shared by every physical copy of a named card;
quantity is handled separately, at deck-construction time, by parsing a
plain decklist file (see game.decklist) against the shared catalog each
logic module contributes to game.registry.CARD_DEFS.
"""

from enum import Enum, auto


class EffectId(Enum):
    TRON_LAND = auto()
    FOREST = auto()
    WOODED_RIDGELINE = auto()
    BOJUKA_BOG = auto()
    TOCASIA_DIG_SITE = auto()
    CONDUIT_PYLONS = auto()
    EXPEDITION_MAP = auto()
    CROP_ROTATION = auto()
    ANCIENT_STIRRINGS = auto()
    BONDERS_ORNAMENT = auto()
    CANDY_TRAIL = auto()
    BARRELS_OF_BLASTING_JELLY = auto()
    RELIC_OF_PROGENITUS = auto()
    GENEROUS_ENT = auto()
    FILLER = auto()

    # --- spy_combo deck ---
    SWAMP = auto()
    MASKED_VANDAL = auto()
    BALUSTRADE_SPY = auto()
    SARULI_CARETAKER = auto()
    OVERGROWN_BATTLEMENT = auto()
    WALL_OF_ROOTS = auto()
    LOTLETH_GIANT = auto()
    ROOST_SEEK = auto()  # Sagu Wildling, implemented as its Adventure sorcery half only -- see decklist comment
    GATECREEPER_VINE = auto()
    NYXBORN_HYDRA = auto()
    QUIRION_RANGER = auto()
    MESMERIC_FIEND = auto()
    LOTUS_PETAL = auto()
    WINDING_WAY = auto()
    LEAD_THE_STAMPEDE = auto()
    LAND_GRANT = auto()
    DREAD_RETURN = auto()

    # --- shared tokens (mono_red_madness / rakdos_madness engine work,
    # docs/MADNESS_DECKS_PLAN.md item 8) -- not decklist cards, never in
    # CARD_DEFS/any decklist quantity; see effects_common.create_token.
    BLOOD_TOKEN = auto()
    ROBOT_TOKEN = auto()
    WARRIOR_TOKEN = auto()  # Cartouche of Solidarity
    ELDRAZI_SPAWN_TOKEN = auto()  # Malevolent Rumble

    # --- shared between rakdos_madness and mono_red_madness (both decklists
    # play these) -- registry entries live in game.catalog.red_cards
    # (Mountain/Sneaky Snacker/Voldaren Epicure/etc. are red or
    # multicolor identity, not deck-scoped) ---
    MOUNTAIN = auto()
    SNEAKY_SNACKER = auto()
    VOLDAREN_EPICURE = auto()
    FAITHLESS_LOOTING = auto()
    HIGHWAY_ROBBERY = auto()
    LIGHTNING_BOLT = auto()
    GRAB_THE_PRIZE = auto()
    FIERY_TEMPER = auto()

    # --- rakdos_madness only ---
    KITCHEN_IMP = auto()
    VAMPIRES_KISS = auto()
    ALMS_OF_THE_VEIN = auto()
    END_THE_FESTIVITIES = auto()
    RAKDOS_CARNARIUM = auto()
    JAGGED_BARRENS = auto()

    # --- mono_red_madness only ---
    MELDED_MOXITE = auto()
    GUTTERSNIPE = auto()
    FIREBLAST = auto()
    LAVA_DART = auto()

    # --- boggles deck ---
    PLAINS = auto()
    GLADECOVER_SCOUT = auto()
    SILHANA_LEDGEWALKER = auto()
    SLIPPERY_BOGLE = auto()
    RANCOR = auto()
    ARMADILLO_CLOAK = auto()
    ANCESTRAL_MASK = auto()
    ETHEREAL_ARMOR = auto()
    CARTOUCHE_OF_SOLIDARITY = auto()
    ASH_BARRENS = auto()
    MALEVOLENT_RUMBLE = auto()
    RAM_THROUGH = auto()  # functional blank -- see green_cards.py comment
    UTOPIA_SPRAWL = auto()
    ABUNDANT_GROWTH = auto()


class CardType(Enum):
    LAND = auto()
    ARTIFACT = auto()
    SORCERY = auto()
    INSTANT = auto()
    CREATURE = auto()
    ENCHANTMENT = auto()
    FILLER = auto()


class CardDef:
    """Definition shared by every physical copy of a named card."""

    def __init__(self, name, card_type, cast_cost, effect_id, **extra):
        self.name = name
        self.card_type = card_type
        self.cast_cost = cast_cost  # dict like {"generic": 1, "G": 1}, or None
        self.effect_id = effect_id
        self.extra = extra  # e.g. tron_type="Mine" for the three Tron lands

    def __repr__(self):
        return f"CardDef({self.name!r})"
