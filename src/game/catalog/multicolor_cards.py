"""Multicolor-identity card catalog: any card whose real cost or mana
output touches 2+ colors (e.g. Rakdos Carnarium's {T}: Add {B}{R}), one
shared bucket regardless of which pair for now -- split into per-guild
files later only if this one gets crowded. Every card's cost/type/
oracle-text below is a direct Scryfall pull. Sneaky Snacker's real cost
is {U}{B} -- never actually cast in either deck that plays it (no "cast"
spec at all: it's discarded, then returned by its own on_draw_count
trigger), kept here only as accurate catalog metadata."""

from ..cards import CardDef, CardType, EffectId
from ..effects_common import bounce_land_etb

MULTICOLOR_CARD_CATALOG = {
    "Wooded Ridgeline": CardDef("Wooded Ridgeline", CardType.LAND, None, EffectId.WOODED_RIDGELINE),
    "Rakdos Carnarium": CardDef("Rakdos Carnarium", CardType.LAND, None, EffectId.RAKDOS_CARNARIUM),
    "Jagged Barrens": CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS),
    "Sneaky Snacker": CardDef("Sneaky Snacker", CardType.CREATURE, {"U": 1, "B": 1}, EffectId.SNEAKY_SNACKER, power=2),
}


def jagged_barrens_etb(state):
    """Deals 1 damage to target opponent."""
    state.damage_dealt += 1


MULTICOLOR_EFFECT_REGISTRY = {
    EffectId.WOODED_RIDGELINE: {
        "mana": ("flexible", {"R", "G"}),
        "enters_tapped": True,
    },
    EffectId.RAKDOS_CARNARIUM: {
        "mana": ("fixed_multi", ("B", "R")),
        "enters_tapped": True,
        "etb_trigger": lambda state: bounce_land_etb(state),
        "pending_kinds": {"choose_permanent"},
    },
    EffectId.JAGGED_BARRENS: {
        "mana": ("flexible", {"B", "R"}),
        "enters_tapped": True,
        "etb_trigger": lambda state: jagged_barrens_etb(state),
    },
    # Never actually cast (real cost is {U}{B} -- off-color for both
    # rakdos_madness and mono_red_madness, by design): always discarded,
    # then returned by its own on_draw_count trigger. No "cast" spec at
    # all -- matches Generous Ent's own "never hard-cast" precedent.
    EffectId.SNEAKY_SNACKER: {
        "on_draw_count": {"count": 3},
    },
}
