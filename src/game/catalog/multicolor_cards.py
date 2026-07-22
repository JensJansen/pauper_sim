"""Multicolor-identity card catalog: any card whose real cost or mana
output touches 2+ colors (e.g. Rakdos Carnarium's {T}: Add {B}{R}), one
shared bucket regardless of which pair for now -- split into per-guild
files later only if this one gets crowded. Every card's cost/type/
oracle-text below is a direct Scryfall pull. Sneaky Snacker's real cost
is {U}{B} -- never actually cast in either deck that plays it (no "cast"
spec at all: it's discarded, then returned by its own on_draw_count
trigger), kept here only as accurate catalog metadata.

Slippery Bogle's real cost is the hybrid {G/U} -- multicolor for color
identity purposes (a hybrid symbol counts as both colors), same reasoning
that puts it here rather than green_cards.py despite boggles.txt never
touching blue. cast_cost below is modeled as plain {G}: boggles runs no
blue mana sources at all, so the hybrid's blue half is unreachable
regardless of how it's represented, and this engine has no general
"pay with either of these colors" cost representation to build for a
single unreachable branch on one card -- a deliberate simplification, not
a guess (real cost verified via Scryfall)."""

from ..cards import CardDef, CardType, EffectId
from ..effects_common import (
    any_creature_on_battlefield,
    bounce_land_etb,
    cast_aura,
    cast_permanent_from_hand,
    deal_damage_to_opponent,
)

MULTICOLOR_CARD_CATALOG = {
    "Wooded Ridgeline": CardDef("Wooded Ridgeline", CardType.LAND, None, EffectId.WOODED_RIDGELINE),
    "Rakdos Carnarium": CardDef("Rakdos Carnarium", CardType.LAND, None, EffectId.RAKDOS_CARNARIUM),
    "Jagged Barrens": CardDef("Jagged Barrens", CardType.LAND, None, EffectId.JAGGED_BARRENS),
    "Sneaky Snacker": CardDef(
        "Sneaky Snacker", CardType.CREATURE, {"U": 1, "B": 1}, EffectId.SNEAKY_SNACKER, power=2, toughness=1,
    ),
    "Slippery Bogle": CardDef(
        "Slippery Bogle", CardType.CREATURE, {"G": 1}, EffectId.SLIPPERY_BOGLE, power=1, toughness=1,
    ),
    "Armadillo Cloak": CardDef(
        "Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK,
    ),
}


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
        # Deals 1 damage to target opponent.
        "etb_trigger": lambda state: deal_damage_to_opponent(state, 1),
    },
    # Never actually cast (real cost is {U}{B} -- off-color for both
    # rakdos_madness and mono_red_madness, by design): always discarded,
    # then returned by its own on_draw_count trigger. No "cast" spec at
    # all -- matches Generous Ent's own "never hard-cast" precedent.
    EffectId.SNEAKY_SNACKER: {
        "on_draw_count": {"count": 3},
        # order_triggers (docs/PRIORITY_PLAN.md item 1): reachable the
        # instant 2+ copies both cross their own draw-count trigger on
        # the same draw -- a real placement-order choice, not fixed
        # queue order, even though this trigger is otherwise "automatic"
        # (no cast-or-decline choice of its own).
        "pending_kinds": {"order_triggers"},
    },
    EffectId.SLIPPERY_BOGLE: {
        # No ability -- functionally a vanilla 1/1 hexproof for {G};
        # hexproof is a documented no-op, same treatment as every other
        # bogle/hexproof creature here -- no opposing spells/abilities
        # exist in this solitaire simulator to be hexproof against.
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)},
    },
    EffectId.ARMADILLO_CLOAK: {
        # Real text: enchanted creature also gets trample (docs/COMBAT_
        # PLAN.md step 7 -- combat_damage_step's own trample-aware damage
        # assignment) and "whenever enchanted creature deals damage, you
        # gain that much life" -- the life-gain half is still a documented
        # no-op: not wired to the new per-player life_total
        # (docs/MULTIPLAYER_ENGINE_PLAN.md) -- fleshing out stubbed card
        # effects is separate future work.
        "cast": {
            "resolve": lambda state, card_def: cast_aura(
                state, card_def, lambda p: p.card_def.card_type == CardType.CREATURE,
            ),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
        },
        "pending_kinds": {"choose_permanent"},
        "pt_bonus": lambda state, aura: 2,
        "toughness_bonus": lambda state, aura: 2,  # real text is +2/+2
        "keywords": {"trample"},
    },
}
