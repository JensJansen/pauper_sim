"""White-identity card catalog: WHITE_CARD_CATALOG (name -> CardDef) and
WHITE_EFFECT_REGISTRY (EffectId -> spec), unioned into game.CARD_DEFS /
EFFECT_REGISTRY by game/registry.py. Cost/type/oracle text is from
Scryfall; power/toughness is a design choice."""

from ..cards import CardDef, CardType, EffectId
from ..effects.casting import cast_aura
from ..effects.shared import any_creature_on_battlefield, any_creature_on_either_battlefield
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
    """ETB: creates a 1/1 white Warrior token with vigilance."""
    create_token(state, WARRIOR_TOKEN_CARD_DEF)


def cast_cartouche_of_solidarity(state, card_def):
    """Enchant creature YOU CONTROL only (not any creature)."""
    cast_aura(
        state, card_def, lambda p: p.card_type == CardType.CREATURE and p in state.battlefield,
        on_attached=cartouche_of_solidarity_attach,
    )


def cast_ethereal_armor(state, card_def):
    cast_aura(state, card_def, lambda p: p.card_type == CardType.CREATURE)


WHITE_EFFECT_REGISTRY = {
    EffectId.PLAINS: {
        "mana": ("fixed", "W"),
    },
    EffectId.CARTOUCHE_OF_SOLIDARITY: {
        # Enchanted creature gets +1/+1 and first strike.
        "cast": {
            "resolve": lambda state, card_def: cast_cartouche_of_solidarity(state, card_def),
            "extra_legal": lambda state: any_creature_on_battlefield(state),  # caster's own creatures only
            "precast_choice": True,  # target chosen before the stack (drl_env._precast_choice_execute)
        },
        "pending_kinds": {"choose_any_target"},  # targets a creature you control
        "pt_bonus": lambda state, aura: 1,
        "toughness_bonus": lambda state, aura: 1,
        "keywords": {"first_strike"},
    },
    EffectId.ETHEREAL_ARMOR: {
        # +1/+1 for each enchantment you control, including itself; first strike.
        "cast": {
            "resolve": lambda state, card_def: cast_ethereal_armor(state, card_def),
            "extra_legal": lambda state: any_creature_on_either_battlefield(state),
            "precast_choice": True,  # target chosen before the stack (drl_env._precast_choice_execute)
        },
        "pending_kinds": {"choose_any_target"},  # targets any creature, either side (hexproof-aware)
        "pt_bonus": lambda state, aura: enchantment_count(state, aura),
        "toughness_bonus": lambda state, aura: enchantment_count(state, aura),
        "keywords": {"first_strike"},
    },
    EffectId.SAMURAI_TOKEN: {"keywords": {"vigilance"}},  # 2/2 white Samurai (Experimental Synthesizer)
    EffectId.WARRIOR_TOKEN: {"keywords": {"vigilance"}},  # Cartouche of Solidarity's ETB token
}
