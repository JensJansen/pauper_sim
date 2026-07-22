"""A creature's effective power/toughness/keywords -- its own base stats
plus every Aura currently enchanting it. Owner-agnostic throughout:
combat.py's combat_damage_step always runs with state.active_idx on the
ATTACKER, but still needs a BLOCKER's (the defender's own creature)
correct effective stats, so every function here searches state.players
directly rather than the active-player-proxied state.battlefield.

References game.registry.EFFECT_REGISTRY only from inside function
bodies, via `registry.EFFECT_REGISTRY` -- registry.py imports the catalog
modules, which import this module (via casting.py/combat.py), so a
`from .registry import EFFECT_REGISTRY` here would try to bind a name
that doesn't exist yet. Deferring the lookup to call time breaks the
cycle. See game/registry.py's own module docstring.
"""

from .. import registry
from ..cards import CardType


def _enchanting_auras(state, permanent):
    """Every Aura currently enchanting `permanent`, searched across BOTH
    players' own battlefields (state.players directly, NOT the
    active-player-proxied state.battlefield). An Aura and whatever it
    enchants are always on the SAME side (casting.cast_aura's own target
    predicate only ever matches the caster's own battlefield at cast
    time) -- but the caller here has no guarantee state.active_idx
    currently points at that side. In particular, combat.combat_damage_step
    always runs with active_idx on the ATTACKER, yet needs a BLOCKER's
    correct effective power/toughness (the defender's own creature) --
    reading state.battlefield there would silently search the wrong
    player's board and miss every Aura bonus on the blocker entirely."""
    for player in state.players:
        for aura in player.battlefield:
            if aura.flags.get("enchanting") is permanent:
                yield aura


def enchantment_count(state, aura):
    """How many ENCHANTMENT-type permanents `aura`'s OWN controller has on
    the battlefield -- shared by every "for each [other] enchantment you
    control" pt_bonus (Ancestral Mask/Ethereal Armor differ only in
    whether the caller subtracts 1 for itself). Takes `aura` (found via
    _enchanting_auras above) so the right controller can be found by
    membership, same reasoning as that function's own docstring -- reading
    state.battlefield instead would count whichever player is currently
    active, not the enchantment's own controller, wrongly conflating the
    two the instant a Blocker's own Ancestral Mask/Ethereal Armor is read
    during combat (attacker active, blocker's Auras on the defender's
    side)."""
    owner = next(player for player in state.players if aura in player.battlefield)
    return sum(1 for p in owner.battlefield if p.card_def.card_type == CardType.ENCHANTMENT)


def permanent_power(state, permanent):
    """A creature's effective power for combat.combat_damage_step (and Ram
    Through, once it's more than a functional blank): its own base power
    (card_def.extra["power"], 0 if absent -- no creature is absent one
    anymore, docs/COMBAT_PLAN.md's full-stats pass, but the default stays
    for FILLER/synthetic self-check permanents) plus every Aura currently
    enchanting it (_enchanting_auras above -- owner-agnostic, correct
    regardless of state.active_idx). Each Aura's registry entry supplies
    its own "pt_bonus" (state, aura_permanent) -> int -- a constant for a
    static bonus (Rancor's +2), a battlefield-wide count for a dynamic one
    (Ancestral Mask/Ethereal Armor's "for each [other] enchantment")."""
    base = permanent.card_def.extra.get("power", 0)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("pt_bonus", lambda *_a: 0)(state, aura)
        for aura in _enchanting_auras(state, permanent)
    )
    return base + bonus


def permanent_toughness(state, permanent):
    """A creature's effective toughness -- its own base toughness
    (card_def.extra["toughness"], 0 if absent, same convention as
    permanent_power) plus every Aura currently enchanting it. Deliberately
    NOT the same registry key as permanent_power's own "pt_bonus": real
    Rancor is +2/+0 (power only), so reusing pt_bonus here would wrongly
    also buff toughness. A separate, optional "toughness_bonus" key
    (defaulting to 0, same as pt_bonus's own default) covers the Auras
    that genuinely are symmetric in real Magic (Ancestral Mask/Ethereal
    Armor/Cartouche of Solidarity/Armadillo Cloak are each +X/+X) without
    touching permanent_power's own already-tested logic at all."""
    base = permanent.card_def.extra.get("toughness", 0)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("toughness_bonus", lambda *_a: 0)(state, aura)
        for aura in _enchanting_auras(state, permanent)
    )
    return base + bonus


# Real Magic keyword strings this engine models (docs/COMBAT_PLAN.md's
# confirmed scope -- only these four, only on the specific cards that
# already grant one): "vigilance" (Cartouche of Solidarity's Warrior
# token), "flying" (Kitchen Imp's real flying; also used for Silhana
# Ledgewalker's "can't be blocked except by creatures with flying" --
# functionally the identical blocking restriction in a ruleset with no
# reach, so one flag covers both rather than a second near-duplicate),
# "trample" (Rancor, Armadillo Cloak), "first_strike" (Cartouche of
# Solidarity, Ethereal Armor). Deathtouch/double strike/menace/reach:
# no card grants any of them -- not modeled, not a registry key.
def creature_keywords(state, permanent):
    """Union of this permanent's own intrinsic registry "keywords" set
    (a creature's own EFFECT_REGISTRY entry) plus every Aura currently
    enchanting it own GRANTED "keywords" set (an Aura's own EFFECT_REGISTRY
    entry) -- same "own base fact plus every enchanting Aura's own
    contribution" shape as permanent_power/permanent_toughness, reusing
    the same owner-agnostic _enchanting_auras (correct regardless of
    state.active_idx, e.g. reading a blocker's keywords from inside
    combat_damage_step, which always runs with active_idx on the
    attacker)."""
    keywords = set(registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("keywords", ()))
    for aura in _enchanting_auras(state, permanent):
        keywords |= set(registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("keywords", ()))
    return keywords


def has_keyword(state, permanent, keyword):
    return keyword in creature_keywords(state, permanent)


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.effects.stats`
    # from src/. Cross-player Aura reads (a real bug found while building
    # mutual combat damage): permanent_power/permanent_toughness/
    # enchantment_count used to read state.battlefield (active-player-
    # proxied) to find enchanting Auras -- silently wrong for a BLOCKER
    # (the defender's own creature), since combat_damage_step always runs
    # with active_idx on the ATTACKER.
    from ..cards import CardDef, EffectId
    from ..state import GameState, Permanent, PlayerState

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    defenders_creature = Permanent(CardDef("Defender's Creature", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    rancor_on_defender = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
    rancor_on_defender.flags["enchanting"] = defenders_creature
    mask_on_defender = Permanent(CardDef("Ancestral Mask", CardType.ENCHANTMENT, {"generic": 2, "G": 1}, EffectId.ANCESTRAL_MASK))
    mask_on_defender.flags["enchanting"] = defenders_creature
    state.players[1].battlefield = [defenders_creature, rancor_on_defender, mask_on_defender]
    state.active_idx = 0  # the ATTACKER's own perspective -- defender's battlefield is NOT state.battlefield right now

    # power: 1 base + 2 (Rancor) + 2 (Ancestral Mask, 1 OTHER enchantment
    # -- Rancor -- so +2, not +4). toughness: 1 base + 0 (Rancor, power
    # only) + 2 (Ancestral Mask, symmetric).
    assert permanent_power(state, defenders_creature) == 5
    assert permanent_toughness(state, defenders_creature) == 3
    assert has_keyword(state, defenders_creature, "flying") is False

    print("stats.py cross-player Aura-read self-check: OK")
