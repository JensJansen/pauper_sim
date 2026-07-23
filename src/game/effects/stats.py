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

# Counter kind -> (power, toughness) granted PER counter of that kind.
# Kinds absent here (Pinnacle Kill-Ship's own "charge") contribute nothing --
# they exist purely as a threshold Permanent.type_override/_animate_spec
# below reads, not a stat bonus. No +1/+1-vs--1/-1 annihilation (real Magic
# rule 122.3): no card in this pool ever puts both kinds on one permanent.
_COUNTER_PT = {"+1/+1": (1, 1), "-0/-1": (0, -1)}


def _animate_spec(permanent):
    """Pinnacle Kill-Ship's own Station: an "animate" registry spec (which
    counter kind, what threshold, and the power/toughness/keywords it grants
    once reached) -- returns that spec once permanent.counters[kind] >=
    threshold, else None. Pure per-permanent computation (one registry
    lookup plus this permanent's own counters dict) -- no battlefield scan,
    same "lives directly on the permanent" reasoning _COUNTER_PT above
    piggybacks on. Station's own resolve (colorless_cards.py) separately
    sets permanent.type_override = CardType.CREATURE the instant the
    threshold is first crossed, so combat eligibility/state-based death
    (which read permanent.card_type) see it too; this covers the STATS half
    (power/toughness/flying) that type_override alone can't express."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("animate")
    if spec is None or permanent.counters.get(spec["counter"], 0) < spec["threshold"]:
        return None
    return spec


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
    return sum(1 for p in owner.battlefield if p.card_type == CardType.ENCHANTMENT)


def permanent_power(state, permanent, enchanting_auras=None):
    """A creature's effective power for combat.combat_damage_step (and Ram
    Through, once it's more than a functional blank): its own base power
    (card_def.extra["power"], 0 if absent -- no creature is absent one
    anymore, docs/COMBAT_PLAN.md's full-stats pass, but the default stays
    for FILLER/synthetic self-check permanents) plus every Aura currently
    enchanting it (_enchanting_auras above -- owner-agnostic, correct
    regardless of state.active_idx). Each Aura's registry entry supplies
    its own "pt_bonus" (state, aura_permanent) -> int -- a constant for a
    static bonus (Rancor's +2), a battlefield-wide count for a dynamic one
    (Ancestral Mask/Ethereal Armor's "for each [other] enchantment").

    enchanting_auras: optional pre-fetched result of _enchanting_auras(state,
    permanent), for a caller that already needs the same list for multiple
    creatures in one pass (drl_env._creature_slot_block, which calls this
    AND permanent_toughness for every occupied creature slot in an
    observation -- profiled: _enchanting_auras's own battlefield scan was a
    real, measurable cost, repeated redundantly per call). None (every
    existing caller) means "compute it myself," identical to before this
    parameter existed -- purely additive, no behavior change for anyone who
    doesn't pass it.

    base also folds in _animate_spec (Pinnacle Kill-Ship's own animated
    power, once Station's threshold is met, overriding card_def.extra
    entirely) and _COUNTER_PT (a flat sum over this permanent's OWN
    counters -- Nyxborn Hydra's "+1/+1"s). Neither needs its own
    enchanting_auras-style pre-fetch/cache: both read only data already
    sitting on `permanent` itself (its own counters dict, one registry
    lookup) with no battlefield scan, so they're already exactly as cheap
    as the single per-creature call this function already gets."""
    animate = _animate_spec(permanent)
    base = animate["power"] if animate is not None else permanent.card_def.extra.get("power", 0)
    base += sum(_COUNTER_PT.get(kind, (0, 0))[0] * n for kind, n in permanent.counters.items())
    auras = enchanting_auras if enchanting_auras is not None else _enchanting_auras(state, permanent)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("pt_bonus", lambda *_a: 0)(state, aura)
        for aura in auras
    )
    return base + bonus


def permanent_toughness(state, permanent, enchanting_auras=None):
    """A creature's effective toughness -- its own base toughness
    (card_def.extra["toughness"], 0 if absent, same convention as
    permanent_power) plus every Aura currently enchanting it. Deliberately
    NOT the same registry key as permanent_power's own "pt_bonus": real
    Rancor is +2/+0 (power only), so reusing pt_bonus here would wrongly
    also buff toughness. A separate, optional "toughness_bonus" key
    (defaulting to 0, same as pt_bonus's own default) covers the Auras
    that genuinely are symmetric in real Magic (Ancestral Mask/Ethereal
    Armor/Cartouche of Solidarity/Armadillo Cloak are each +X/+X) without
    touching permanent_power's own already-tested logic at all.

    enchanting_auras: see permanent_power's own docstring -- same optional
    pre-fetch, same reasoning. base folds in _animate_spec/_COUNTER_PT the
    same way permanent_power's own base does -- see that function's
    docstring for why neither needs a battlefield-scan cache of its own."""
    animate = _animate_spec(permanent)
    base = animate["toughness"] if animate is not None else permanent.card_def.extra.get("toughness", 0)
    base += sum(_COUNTER_PT.get(kind, (0, 0))[1] * n for kind, n in permanent.counters.items())
    auras = enchanting_auras if enchanting_auras is not None else _enchanting_auras(state, permanent)
    bonus = sum(
        registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("toughness_bonus", lambda *_a: 0)(state, aura)
        for aura in auras
    )
    return base + bonus


def lifelink_count(state, permanent, enchanting_auras=None):
    """How many independent "whenever this deals damage, you gain that
    much life" triggers this creature's current damage carries -- summed
    across every enchanting Aura whose own registry entry sets "lifelink":
    True (Armadillo Cloak), same summed shape as pt_bonus/toughness_bonus
    above, NOT creature_keywords' own set union below.

    This deliberately does NOT belong in the boolean "keywords" set: real
    lifelink is a static ability (redundant copies are genuinely
    irrelevant -- a creature either has it or doesn't, so vigilance/
    flying/trample/first_strike's set-union dedup is correct for those).
    Armadillo Cloak's clause is a distinct TRIGGERED ability instead --
    two Cloaks on the same creature really do trigger twice, each for the
    full damage dealt (2x total life gained, not 1x), which a boolean
    "lifelink" in creature_keywords would silently dedup down to one
    trigger regardless of how many Cloaks are attached. See
    game.effects.combat's own call sites for how this count multiplies
    the life gained.

    enchanting_auras: see permanent_power's own docstring -- same optional
    pre-fetch, same reasoning. Added alongside creature_keywords' own
    identical parameter so combat.py's per-creature stats (power,
    toughness, keywords, lifelink_count) can all share ONE _enchanting_
    auras fetch per creature per combat instead of each independently
    re-scanning state.players."""
    auras = enchanting_auras if enchanting_auras is not None else _enchanting_auras(state, permanent)
    return sum(
        1 for aura in auras
        if registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("lifelink", False)
    )


# Real Magic keyword strings this engine models as a boolean set
# (docs/COMBAT_PLAN.md's confirmed scope -- only these four, only on the
# specific cards that already grant one): "vigilance" (Cartouche of
# Solidarity's Warrior token), "flying" (Kitchen Imp's real flying; also
# used for Silhana Ledgewalker's "can't be blocked except by creatures
# with flying" -- functionally the identical blocking restriction in a
# ruleset with no reach, so one flag covers both rather than a second
# near-duplicate), "trample" (Rancor, Armadillo Cloak), "first_strike"
# (Cartouche of Solidarity, Ethereal Armor). Deathtouch/double strike/
# menace/reach: no card grants any of them -- not modeled, not a registry
# key. Armadillo Cloak's own lifegain clause is NOT here -- see
# lifelink_count above for why a boolean keyword is the wrong model for
# a triggered, stacking ability.
def creature_keywords(state, permanent, enchanting_auras=None):
    """Union of this permanent's own intrinsic registry "keywords" set
    (a creature's own EFFECT_REGISTRY entry) plus every Aura currently
    enchanting it own GRANTED "keywords" set (an Aura's own EFFECT_REGISTRY
    entry) -- same "own base fact plus every enchanting Aura's own
    contribution" shape as permanent_power/permanent_toughness, reusing
    the same owner-agnostic _enchanting_auras (correct regardless of
    state.active_idx, e.g. reading a blocker's keywords from inside
    combat_damage_step, which always runs with active_idx on the
    attacker).

    enchanting_auras: see permanent_power's own docstring -- same optional
    pre-fetch, same reasoning. Also folds in _animate_spec's own granted
    keywords (Pinnacle Kill-Ship's flying, once animated) -- same no-scan
    reasoning as permanent_power/permanent_toughness."""
    auras = enchanting_auras if enchanting_auras is not None else _enchanting_auras(state, permanent)
    keywords = set(registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("keywords", ()))
    animate = _animate_spec(permanent)
    if animate is not None:
        keywords |= set(animate.get("keywords", ()))
    for aura in auras:
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

    # Counters: a flat per-permanent bonus, no battlefield scan involved --
    # "+1/+1" (Nyxborn Hydra) and "-0/-1" (Wall of Roots) both fold straight
    # into base power/toughness alongside the existing Aura bonus.
    hydra = Permanent(CardDef("Some Hydra", CardType.CREATURE, None, EffectId.FILLER, power=0, toughness=1))
    hydra.counters["+1/+1"] = 3
    assert permanent_power(state, hydra) == 3
    assert permanent_toughness(state, hydra) == 4

    wall = Permanent(CardDef("Some Wall", CardType.CREATURE, None, EffectId.FILLER, power=0, toughness=5))
    wall.counters["-0/-1"] = 5
    assert permanent_power(state, wall) == 0
    assert permanent_toughness(state, wall) == 0  # 5 counters against toughness 5 -- lethal via the ordinary SBA check, not special-cased here

    # _animate_spec: Pinnacle Kill-Ship's own Station -- below its charge
    # threshold, ordinary card_def stats/keywords apply; at/above it,
    # animate's own power/toughness/keywords fully override, unrelated to
    # any counter's own _COUNTER_PT contribution (charge isn't listed there
    # at all, so it never double-counts as a stat bonus on its own).
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {
        "animate": {"counter": "charge", "threshold": 7, "power": 7, "toughness": 7, "keywords": {"flying"}},
    }
    try:
        ship = Permanent(CardDef("Some Ship", CardType.ARTIFACT, None, EffectId.FILLER))
        ship.counters["charge"] = 6
        assert permanent_power(state, ship) == 0 and permanent_toughness(state, ship) == 0
        assert has_keyword(state, ship, "flying") is False
        ship.counters["charge"] = 7
        assert permanent_power(state, ship) == 7 and permanent_toughness(state, ship) == 7
        assert has_keyword(state, ship, "flying") is True
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("stats.py counters/animate self-check: OK")
