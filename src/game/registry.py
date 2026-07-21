"""Card catalog + effect registry, merged across every color identity.

CARD_DEFS: one shared name->CardDef catalog, the union of every color
catalog's own XXX_CARD_CATALOG fragment (a card's definition -- type,
cost, effect, extra -- is fixed metadata, defined exactly once regardless
of how many decks play it). This is the single source of truth for "what
is card X" -- a deck's own decklist (parsed from data/*.txt by
game.decklist) supplies only names and quantities, looking up everything
else here.

EFFECT_REGISTRY: one EffectId->spec dict, the union of all 7 color
catalogs' own XXX_EFFECT_REGISTRY fragments. An EffectId present here is
"already implemented" -- this is what makes a card reusable by a future
deck without new code. Cards are no longer deck-scoped at all
(DECK_REGISTRY_REFRESH_PLAN.md) -- a card lives exactly once, in the one
catalog file matching its real color identity, regardless of how many
decklists name it.

Importing this module is what actually triggers loading every catalog
module (and, transitively, effects_common/mana/resolution/state/cards) --
see effects_common.py's module docstring for why those modules only ever
reference `registry.EFFECT_REGISTRY` / `registry.CARD_DEFS` etc. lazily,
from inside function bodies, instead of importing the names directly.
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

# Derived views: kept as module-level names for backward compatibility with
# every existing caller (game.mana, rewards.py's resource_quality_components).
SIMPLE_MANA_SOURCE_EFFECTS = {
    effect_id for effect_id, spec in EFFECT_REGISTRY.items() if spec.get("mana") is not None
}
# _FIXED_SOURCE_COLOR used to live here (a single-symbol-per-effect_id
# approximation mana.choose_taps_for_cost's legality solver consulted for
# "fixed"/"count" sources). Deleted -- MADNESS_DECKS_PLAN.md item 9's
# solver rewrite calls mana_output(p, state) directly instead, which
# handles any real output (including a multi-symbol "fixed_multi" tap or
# count's genuinely variable total) correctly, so the approximation (and
# its documented undercount) has no remaining reader.
_FLEXIBLE_SOURCE_CHOICES = {
    effect_id: spec["mana"][1]
    for effect_id, spec in EFFECT_REGISTRY.items()
    if spec.get("mana", (None,))[0] == "flexible"
}
ENTERS_TAPPED_EFFECTS = {
    effect_id for effect_id, spec in EFFECT_REGISTRY.items() if spec.get("enters_tapped")
}


def derive_pending_kinds(decklist):
    """Which pending-resolution kinds this decklist's own cards can
    actually produce, beyond the universal baseline ("none"/"pay_cost")
    every deck needs regardless -- the union of each distinct card's own
    explicit "pending_kinds" registry annotation.

    Not inferred from other spec keys ("cast"/"madness"/"flashback"/
    "alt_cast"/"plot"): several of those call into resolution.py's generic
    primitives (begin_discard, begin_sacrifice, ...) from inside a
    hand-written Python function, which isn't statically inspectable --
    e.g. a "flashback" spec might or might not itself need "sacrifice"
    depending on what its own resolve function does, so the key's mere
    presence doesn't tell you. Each EffectId instead declares directly
    what it needs, so this stays a plain, robust union with no
    special-casing. Means swapping which cards are in a decklist (not
    just their quantities) never needs a hand-maintained pending_kinds
    tuple updated to match -- this is it."""
    kinds = set()
    for name, _qty in decklist:
        effect_id = CARD_DEFS[name].effect_id
        kinds |= EFFECT_REGISTRY.get(effect_id, {}).get("pending_kinds", set())
    return tuple(sorted(kinds))
