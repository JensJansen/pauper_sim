"""White-identity card catalog: every card whose real mana cost is
mono-white (or, for lands with no cost, whose only mana output is white).
Every card's cost/type/oracle-text below is a direct Scryfall pull, except
creature power/toughness, which is a design choice, not Scryfall data.

Same shape as every other color file: a WHITE_CARD_CATALOG dict (name ->
CardDef) and a WHITE_EFFECT_REGISTRY dict (EffectId -> spec), unioned into
game.CARD_DEFS/EFFECT_REGISTRY by game/registry.py."""

from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_aura
from ..effects.shared import any_creature_on_battlefield
from ..effects.stats import enchantment_count
from ..effects.tokens import WARRIOR_TOKEN_CARD_DEF, create_token

WHITE_CARD_CATALOG = {
    "Plains": CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
    "Cartouche of Solidarity": CardDef(
        "Cartouche of Solidarity", CardType.ENCHANTMENT, {"W": 1}, EffectId.CARTOUCHE_OF_SOLIDARITY,
    ),
    "Ethereal Armor": CardDef("Ethereal Armor", CardType.ENCHANTMENT, {"W": 1}, EffectId.ETHEREAL_ARMOR),
}


def cartouche_of_solidarity_attach(state, aura):
    """ETB: create a 1/1 white Warrior creature token with vigilance --
    see EffectId.WARRIOR_TOKEN's own registry entry below (docs/
    COMBAT_PLAN.md step 7: attacking no longer taps it, combat.
    declare_attacker)."""
    create_token(state, WARRIOR_TOKEN_CARD_DEF)


def cast_cartouche_of_solidarity(state, card_def):
    cast_aura(
        state, card_def, lambda p: p.card_def.card_type == CardType.CREATURE,
        on_attached=cartouche_of_solidarity_attach,
    )


def cast_ethereal_armor(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_def.card_type == CardType.CREATURE)


WHITE_EFFECT_REGISTRY = {
    EffectId.PLAINS: {
        "mana": ("fixed", "W"),
    },
    EffectId.CARTOUCHE_OF_SOLIDARITY: {
        # Real text: enchanted creature also gets +1/+1 (both pt_bonus and
        # toughness_bonus below -- docs/COMBAT_PLAN.md's full-stats pass)
        # and has first strike (docs/COMBAT_PLAN.md step 7 -- combat_
        # damage_step's own first-strike sub-step).
        "cast": {
            "resolve": lambda state, card_def: cast_cartouche_of_solidarity(state, card_def),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
            "precast_choice": True,  # real MTG "enchant target creature" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_permanent"},
        "pt_bonus": lambda state, aura: 1,
        "toughness_bonus": lambda state, aura: 1,
        "keywords": {"first_strike"},
    },
    EffectId.ETHEREAL_ARMOR: {
        # Real text: +1/+1 for each enchantment you control -- INCLUDING
        # itself (unlike Ancestral Mask's "each OTHER enchantment";
        # verified via Scryfall, not guessed) -- and has first strike
        # (docs/COMBAT_PLAN.md step 7, same as Cartouche of Solidarity
        # above).
        "cast": {
            "resolve": lambda state, card_def: cast_ethereal_armor(state, card_def),
            "extra_legal": lambda state: any_creature_on_battlefield(state),
            "precast_choice": True,  # real MTG "enchant target creature" -- must be chosen before the stack, see drl_env._precast_choice_execute
        },
        "pending_kinds": {"choose_permanent"},
        "pt_bonus": lambda state, aura: enchantment_count(state, aura),
        "toughness_bonus": lambda state, aura: enchantment_count(state, aura),
        "keywords": {"first_strike"},
    },
    EffectId.WARRIOR_TOKEN: {
        # Cartouche of Solidarity's own ETB token (cartouche_of_solidarity_
        # attach above) -- vigilance means attacking never taps it
        # (combat.declare_attacker).
        "keywords": {"vigilance"},
    },
}
