"""Card data model shared by every deck: EffectId, CardType, CardDef.

A CardDef is the definition shared by every physical copy of a named card;
quantity is handled separately, at deck-construction time, by parsing a
plain decklist file (see game.decklist) against the shared catalog each
logic module contributes to game.registry.CARD_DEFS.

Also the home of the pure CardDef predicates every catalog reads (is_artifact,
card_colors, card_subtypes) -- they take only a card_def, so they live next to
the class they inspect rather than in effects.shared (state manipulation).
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

    # --- monster_tron "filler" creatures, each with its own real cost/
    # stats/ETB (not EffectId.FILLER's inert single canonical entry) ---
    ROOFTOP_PERCHER = auto()
    BOULDERBRANCH_GOLEM = auto()
    BOULDERBRANCH_GOLEM_PROTOTYPE = auto()  # Prototype {3}{G} 3/3 mode -- own ETB gain amount (3, = its power)
    MAELSTROM_COLOSSUS = auto()
    PINNACLE_KILL_SHIP = auto()
    BRAMBLE_WURM = auto()

    # --- spy_combo deck ---
    SWAMP = auto()
    MASKED_VANDAL = auto()
    BALUSTRADE_SPY = auto()
    SARULI_CARETAKER = auto()
    OVERGROWN_BATTLEMENT = auto()
    WALL_OF_ROOTS = auto()
    LOTLETH_GIANT = auto()
    ROOST_SEEK = auto()  # Sagu Wildling's Omen sorcery half -- see decklist comment
    SAGU_WILDLING = auto()  # Sagu Wildling's own creature half, cast from exile via ROOST_SEEK's "adventure" spec
    GATECREEPER_VINE = auto()
    NYXBORN_HYDRA = auto()
    QUIRION_RANGER = auto()
    MESMERIC_FIEND = auto()
    LOTUS_PETAL = auto()
    WINDING_WAY = auto()
    LEAD_THE_STAMPEDE = auto()
    LAND_GRANT = auto()
    DREAD_RETURN = auto()

    # --- shared tokens (created by mono_red_madness/rakdos_madness engine
    # effects) -- not decklist cards, never in CARD_DEFS/any decklist
    # quantity; see game.effects.tokens.create_token.
    BLOOD_TOKEN = auto()
    ROBOT_TOKEN = auto()
    WARRIOR_TOKEN = auto()  # Cartouche of Solidarity
    ELDRAZI_SPAWN_TOKEN = auto()  # Malevolent Rumble
    FOOD_TOKEN = auto()  # Generous Ent

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
    BREATH_WEAPON = auto()  # monster_tron filler creature -- see decklist comment

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
    RAM_THROUGH = auto()  # castable only in a 2-player game -- see green_cards.py comment
    UTOPIA_SPRAWL = auto()
    ABUNDANT_GROWTH = auto()

    # --- new decks (dmir_terror / elves / grixis_affinity / jund_wildfire /
    # mono_blue_terror / mono_red_rally): lands (G1). Artifact lands carry
    # extra["artifact"]=True (both LAND and ARTIFACT -- matters for affinity/
    # metalcraft/artifact-sac); the four "Bridge" lands also carry
    # extra["indestructible"]=True. Filed by mana-output color, per each
    # catalog's own no-cost-land convention. ---
    ISLAND = auto()
    SEAT_OF_THE_SYNOD = auto()  # artifact land, {T}: U
    GREAT_FURNACE = auto()  # artifact land, {T}: R
    VAULT_OF_WHISPERS = auto()  # artifact land, {T}: B
    DROSSFORGE_BRIDGE = auto()  # artifact land, tapped, indestructible, {T}: B/R
    MISTVAULT_BRIDGE = auto()  # artifact land, tapped, indestructible, {T}: U/B
    SILVERBLUFF_BRIDGE = auto()  # artifact land, tapped, indestructible, {T}: U/R
    SLAGWOODS_BRIDGE = auto()  # artifact land, tapped, indestructible, {T}: R/G
    CONTAMINATED_AQUIFER = auto()  # tapped dual, {T}: U/B, subtypes Island Swamp
    ICE_TUNNEL = auto()  # tapped (snow) dual, {T}: U/B, subtypes Island Swamp
    GINGERBREAD_CABIN = auto()  # Forest, tapped unless 3+ other Forests, ETB Food if untapped
    TWISTED_LANDSCAPE = auto()  # {T}: C; {T},Sac: fetch basic S/M/F tapped; Cycling {B}{R}{G} (G2)

    # --- G2 card-flow spells (blue) ---
    MENTAL_NOTE = auto()  # {U} instant: mill 2, draw 1
    THOUGHT_SCOUR = auto()  # {U} instant: target player mills 2, draw 1
    LORIEN_REVEALED = auto()  # {3}{U}{U} sorcery: draw 3; Islandcycling {1}
    BRAINSTORM = auto()  # {U} instant: draw 3, put 2 from hand on top
    PONDER = auto()  # {U} sorcery: look top 3, reorder or shuffle, draw 1
    DEEP_ANALYSIS = auto()  # {3}{U} sorcery: target player draws 2; Flashback {1}{U}+3 life

    # --- G3 targeted removal & tricks ---
    CAST_DOWN = auto()  # {1}{B} instant: destroy target nonlegendary creature
    TERMINATE = auto()  # {B}{R} instant: destroy target creature, no regen
    SNUFF_OUT = auto()  # {3}{B} instant: destroy nonblack; alt-cost pay 4 life if control a Swamp
    AGONY_WARP = auto()  # {U}{B} instant: -3/-0 and -0/-3 (until EOT), 1-2 targets
    UNEXPECTED_FANGS = auto()  # {1}{B} instant: +1/+1 counter + lifelink counter on target creature
    TOXIN_ANALYSIS = auto()  # {B} instant: target gains deathtouch+lifelink until EOT; Investigate
    PULSE_OF_MURASA = auto()  # {2}{G} instant: return creature/land from gy to hand, gain 6
    CLEANSING_WILDFIRE = auto()  # {1}{R} sorcery: destroy target land, controller fetches basic tapped, draw
    CLUE_TOKEN = auto()  # Investigate token: "{2}, Sacrifice: Draw a card"

    # --- G4 counters & permission (blue) ---
    COUNTERSPELL = auto()  # {U}{U} instant: counter target spell
    DISPEL = auto()  # {U} instant: counter target instant spell
    SPELL_PIERCE = auto()  # {U} instant: counter target noncreature spell unless controller pays {2}

    # --- G5 impulse / kicker / escape ---
    RECKLESS_IMPULSE = auto()  # {1}{R} sorcery: impulse-exile top 2 until end of next turn
    GOBLIN_BUSHWHACKER = auto()  # {R} 1/1; Kicker {R}: if kicked, team +1/+0 & haste until EOT
    ABANDON_ATTACHMENTS = auto()  # {1}{U/R} (->{1}{U} in dmir) instant: may discard a card; if so, draw 2
    SLEEP_OF_THE_DEAD = auto()  # {U} sorcery: tap target creature, skip its next untap; Escape {2}{U}, exile 3

    # --- G6 tokens & go-wide red ---
    HUMAN_SOLDIER_TOKEN = auto()  # 1/1 white Human Soldier (Rally at the Hornburg)
    TREASURE_TOKEN = auto()  # artifact: {T}, Sacrifice: add one mana of any color
    BIRD_ILLUSION_TOKEN = auto()  # 1/1 blue Bird Illusion with flying (Murmuring Mystic)
    RALLY_AT_THE_HORNBURG = auto()  # {1}{R} sorcery: two 1/1 Human Soldiers; Humans gain haste
    RECKLESS_LACKEY = auto()  # {R} 1/2 first strike, haste; {2}{R},Sac: draw + Treasure
    GOBLIN_TOMB_RAIDER = auto()  # {R} 1/2; while you control an artifact, +1/+0 & haste
    MURMURING_MYSTIC = auto()  # {3}{U} 1/5; on cast I/S -> 1/1 flying Bird Illusion
    BURNING_TREE_EMISSARY = auto()  # {R/G}{R/G} (->{R}{R}) 2/2; ETB add {R}{G} (->{R}{R})

    # --- G7 cost reduction & threats ---
    MYR_ENFORCER = auto()  # {7} 4/4 artifact creature; Affinity for artifacts
    REFURBISHED_FAMILIAR = auto()  # {3}{B} 2/1 artifact flyer; Affinity; ETB each opp discards / else draw
    UTROM_MONITOR = auto()  # {4}{U} 3/3 artifact flyer; Affinity
    THOUGHTCAST = auto()  # {4}{U} sorcery; Affinity; draw 2
    GALVANIC_BLAST = auto()  # {R} instant; deal 2 (4 with Metalcraft) to any target
    GURMAG_ANGLER = auto()  # {6}{B} 5/5; Delve
    TOLARIAN_TERROR = auto()  # {6}{U} 5/5; costs 1 less per I/S in gy; Ward {2}
    CRYPTIC_SERPENT = auto()  # {5}{U}{U} 6/5; costs 1 less per I/S in gy
    DEEM_INFERIOR = auto()  # {3}{U} sorcery; costs 1 less per card drawn this turn; tuck target nonland permanent

    # --- G10 transform / DFC ---
    DELVER_OF_SECRETS = auto()  # {U} 1/1: upkeep, look at top; if instant/sorcery, may transform -> 3/2 flyer

    # --- G11 spell copy ---
    CHAIN_LIGHTNING = auto()  # {R} sorcery: 3 dmg any target; affected player may pay {R}{R} to copy + retarget (recursive)

    # --- G8 artifact value & sac engines ---
    ICHOR_WELLSPRING = auto()  # {2} artifact: ETB draw; dies draw
    CHROMATIC_STAR = auto()  # {1} artifact: {1}{T}Sac add any color; dies draw
    NIHIL_SPELLBOMB = auto()  # {1} artifact: {T}Sac exile a graveyard; dies may pay {B} draw
    LEMBAS = auto()  # {2} Food artifact: ETB scry1+draw; {2}{T}Sac gain 3; dies shuffle into library
    BLOOD_FOUNTAIN = auto()  # {B} artifact: ETB Blood; {3}{B}{T}Sac return up to 2 creature cards from gy
    SEWER_VEILLANCE_CAM = auto()  # {U} artifact, Flash: ETB/LTB may tap/untap target creature; {3}{U}Sac draw 2
    KRARK_CLAN_SHAMAN = auto()  # {R} 1/1: Sac an artifact: 1 dmg to each nonflying creature
    MAKESHIFT_MUNITIONS = auto()  # {1}{R} enchantment: {1},Sac artifact/creature: 1 dmg any target
    GIXIAN_INFILTRATOR = auto()  # {1}{B} 2/1: whenever you sac another permanent, +1/+1 counter
    WRITHING_CHRYSALIS = auto()  # {2}{R}{G} 2/3 Devoid Eldrazi, Reach: cast->2 Eldrazi Spawn; sac Eldrazi->+1/+1
    FANATICAL_OFFERING = auto()  # {1}{B} instant: sac artifact/creature; draw 2 + Map token
    EVISCERATORS_INSIGHT = auto()  # {1}{B} instant: sac artifact/creature; draw 2; Flashback {4}{B}
    RECKONERS_BARGAIN = auto()  # {1}{B} instant: sac artifact/creature; gain life=MV; draw 2
    EXPERIMENTAL_SYNTHESIZER = auto()  # {R} artifact: ETB/LTB impulse 1; {2}{R}Sac Samurai token
    CLOCKWORK_PERCUSSIONIST = auto()  # {R} 1/1 artifact creature, haste: dies impulse 1 (next turn)
    SAMURAI_TOKEN = auto()  # 2/2 white Samurai with vigilance (Experimental Synthesizer)
    MAP_TOKEN = auto()  # artifact: {1}{T}Sac target creature you control explores (Fanatical Offering)

    # --- G9 elves ---
    LLANOWAR_ELVES = auto()  # {G} 1/1 Elf: {T} add {G}
    FYNDHORN_ELVES = auto()  # {G} 1/1 Elf: {T} add {G}
    PRIEST_OF_TITANIA = auto()  # {1}{G} 1/1 Elf: {T} add {G} per Elf on the battlefield
    WELLWISHER = auto()  # {1}{G} 1/1 Elf: {T} gain 1 life per Elf on the battlefield
    TIMBERWATCH_ELF = auto()  # {2}{G} 1/2 Elf: {T} target creature +X/+X (X = # Elves) until EOT

    # --- G12 initiative / Undercity ---
    AVENGING_HUNTER = auto()  # {4}{G} 5/4 Trample: ETB you take the initiative (venture into Undercity)
    SKELETON_TOKEN = auto()  # 4/1 black Skeleton with menace (Undercity Catacombs)


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


_COLOR_PIPS = ("W", "U", "B", "R", "G")


def is_artifact(card_def):
    """Whether a card is an artifact -- a plain artifact (card_type ARTIFACT),
    an artifact land (card_type LAND + extra["artifact"]), or an artifact
    creature (card_type CREATURE + extra["artifact"], e.g. Myr Enforcer). The
    single predicate every "artifacts you control" reader uses: Goblin Tomb
    Raider's static boost, affinity, metalcraft, Krark-Clan/Makeshift's
    artifact-sacrifice."""
    return card_def.card_type == CardType.ARTIFACT or card_def.extra.get("artifact", False)


def card_colors(card_def):
    """The card's colors, from the colored pips in its mana cost. A Devoid
    card (extra["devoid"], e.g. Writhing Chrysalis) is colorless regardless
    of its cost's colored pips; a land / cost-less card (cast_cost None) is
    colorless too. Used by "nonblack"/etc. targeting restrictions (Snuff
    Out)."""
    if card_def.extra.get("devoid") or card_def.cast_cost is None:
        return set()
    return {c for c in card_def.cast_cost if c in _COLOR_PIPS}


def card_subtypes(card_def):
    """A card's subtypes (extra["subtypes"], e.g. ("Island","Swamp") or
    ("Elf",)), () if none declared. Only cards that need one for a rules
    check carry it -- basic Island/Forest (for Islandcycling / Gingerbread
    Cabin's Forest count), the U/B dual lands (Island subtype), the Elves
    (tribal counts). Not a full Scryfall subtype line; just the ones the
    engine actually reads."""
    return card_def.extra.get("subtypes", ())
