"""White-identity card catalog: every card whose real mana cost is
mono-white (or, for lands with no cost, whose only mana output is white).
Every card's cost/type/oracle-text below is a direct Scryfall pull, except
creature power/toughness, which is a design choice, not Scryfall data.

Same shape as every other color file: a WHITE_CARD_CATALOG dict (name ->
CardDef) and a WHITE_EFFECT_REGISTRY dict (EffectId -> spec), unioned into
game.CARD_DEFS/EFFECT_REGISTRY by game/registry.py."""

from ..cards import CardDef, CardType, EffectId
from ..effects_common import WARRIOR_TOKEN_CARD_DEF, cast_aura, create_token, enchantment_count

WHITE_CARD_CATALOG = {
    "Plains": CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
    "Cartouche of Solidarity": CardDef(
        "Cartouche of Solidarity", CardType.ENCHANTMENT, {"W": 1}, EffectId.CARTOUCHE_OF_SOLIDARITY,
    ),
    "Ethereal Armor": CardDef("Ethereal Armor", CardType.ENCHANTMENT, {"W": 1}, EffectId.ETHEREAL_ARMOR),
}


def _aura_creature_extra_legal(state):
    return any(p.card_def.card_type == CardType.CREATURE for p in state.battlefield)


def cartouche_of_solidarity_attach(state, aura):
    """ETB: create a 1/1 white Warrior creature token with vigilance.
    Vigilance is a documented no-op here -- no combat granularity for it
    without an opponent/blockers, same treatment as every other
    keyword-only-relevant-against-an-opponent effect in this engine."""
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
        # Real text: enchanted creature also gets +1/+1 and has first
        # strike -- first strike is a documented no-op (no combat
        # granularity for it without blockers).
        "cast": {
            "resolve": lambda state, card_def: cast_cartouche_of_solidarity(state, card_def),
            "extra_legal": lambda state: _aura_creature_extra_legal(state),
        },
        "pending_kinds": {"choose_permanent"},
        "pt_bonus": lambda state, aura: 1,
    },
    EffectId.ETHEREAL_ARMOR: {
        # Real text: +1/+1 for each enchantment you control -- INCLUDING
        # itself (unlike Ancestral Mask's "each OTHER enchantment";
        # verified via Scryfall, not guessed) -- and has first strike
        # (no-op, same reasoning as Cartouche of Solidarity above).
        "cast": {
            "resolve": lambda state, card_def: cast_ethereal_armor(state, card_def),
            "extra_legal": lambda state: _aura_creature_extra_legal(state),
        },
        "pending_kinds": {"choose_permanent"},
        "pt_bonus": lambda state, aura: enchantment_count(state),
    },
}
