"""Card catalog + effect registry, merged across every color identity.

CARD_DEFS: shared name->CardDef catalog, union of each color catalog's
XXX_CARD_CATALOG. EFFECT_REGISTRY: shared EffectId->spec dict, union of
each color catalog's XXX_EFFECT_REGISTRY.

Importing this module loads every catalog and, transitively, most of
game/effects, mana, resolution, state, cards. Those submodules reference
`registry.EFFECT_REGISTRY`/`CARD_DEFS` only inside function bodies, since
this module's dicts aren't built yet when they first load.
"""

from .catalog import black_cards, blue_cards, colorless_cards, green_cards, multicolor_cards, red_cards, white_cards

CARD_DEFS = {
    **white_cards.WHITE_CARD_CATALOG,
    **blue_cards.BLUE_CARD_CATALOG,
    **black_cards.BLACK_CARD_CATALOG,
    **red_cards.RED_CARD_CATALOG,
    **green_cards.GREEN_CARD_CATALOG,
    **colorless_cards.COLORLESS_CARD_CATALOG,
    **multicolor_cards.MULTICOLOR_CARD_CATALOG,
}

EFFECT_REGISTRY = {
    **white_cards.WHITE_EFFECT_REGISTRY,
    **blue_cards.BLUE_EFFECT_REGISTRY,
    **black_cards.BLACK_EFFECT_REGISTRY,
    **red_cards.RED_EFFECT_REGISTRY,
    **green_cards.GREEN_EFFECT_REGISTRY,
    **colorless_cards.COLORLESS_EFFECT_REGISTRY,
    **multicolor_cards.MULTICOLOR_EFFECT_REGISTRY,
}

# Derived views: kept as module-level names for the callers that consult them.
_FLEXIBLE_SOURCE_CHOICES = {
    effect_id: spec["mana"][1]
    for effect_id, spec in EFFECT_REGISTRY.items()
    if spec.get("mana", (None,))[0] == "flexible"
}
ENTERS_TAPPED_EFFECTS = {
    effect_id for effect_id, spec in EFFECT_REGISTRY.items() if spec.get("enters_tapped")
}


def derive_pending_kinds(decklist):
    """Union of each card's own "pending_kinds" registry annotation across
    this decklist, beyond the universal baseline ("none"/"pay_cost").

    Not inferred from other spec keys ("cast"/"madness"/"flashback"/...):
    those call into resolution primitives from hand-written functions,
    which isn't statically inspectable, so each EffectId declares its own
    pending_kinds directly instead."""
    kinds = set()
    for name, _qty in decklist:
        effect_id = CARD_DEFS[name].effect_id
        kinds |= EFFECT_REGISTRY.get(effect_id, {}).get("pending_kinds", set())
    return tuple(sorted(kinds))
