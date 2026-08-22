"""A creature's effective power/toughness/keywords -- base stats plus every
enchanting Aura. Owner-agnostic: combat_damage_step runs with active_idx on
the ATTACKER but needs a BLOCKER's stats too, so functions here search
state.players directly, not the active-player-proxied state.battlefield.

References registry.EFFECT_REGISTRY only inside function bodies -- see
game/registry.py's docstring."""

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


def _transform_spec(permanent):
    """A transformed double-faced permanent's back-face stats (Delver of
    Secrets -> Insectile Aberration 3/2 flying). Returns the registry
    "transform" spec ({"power","toughness","keywords"}) once
    permanent.flags["transformed"] is set, else None. Flag-gated (a real
    one-way transform, set by the upkeep trigger), unlike _animate_spec's
    counter threshold; both override the card_def's own base stats the same
    way, with counters/temp/static bonuses still added on top."""
    if not permanent.flags.get("transformed"):
        return None
    return registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("transform")


def _static_self(state, permanent):
    """A creature's OWN conditional static boost (Goblin Tomb Raider: "as
    long as you control an artifact, this creature gets +1/+0 and has
    haste"). Returns (power, toughness, keywords) from the registry
    "static_self" spec when its condition(state, permanent) holds, else
    (0, 0, empty). Distinct from Aura pt_bonus (that's granted BY another
    permanent); this is the creature's own always-on-when-condition-met
    ability, read directly off itself with one registry lookup + the spec's
    own condition."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("static_self")
    if spec is None or not spec["condition"](state, permanent):
        return 0, 0, set()
    return spec.get("power", 0), spec.get("toughness", 0), set(spec.get("keywords", ()))


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


def enchanting_by_target(state):
    """Every Aura on either battlefield, bucketed by id(enchanted permanent)
    -- one full scan producing a dict a caller can then read many
    permanents' auras from via `.get(id(permanent), ())`, instead of paying
    _enchanting_auras' own per-permanent scan once per permanent checked.
    Shared by every caller that reads MANY permanents' auras from the SAME
    state in one pass: combat.combat_damage_step, state_based.
    check_state_based_actions, rl.model.features.build_token_set.

    CAUTION -- each caller takes this snapshot once and reuses it for every
    permanent it checks in that one call. Safe today because no Aura in this
    pool has flash (though this engine already models flash on other card
    types) and nothing in any current caller casts or (de)attaches an Aura
    mid-call. If a flash/instant-speed Aura is ever added, re-verify it
    can't attach/detach inside a caller's own scan-to-read window before
    trusting this snapshot there."""
    result = {}
    for player in state.players:
        for aura in player.battlefield:
            target = aura.flags.get("enchanting")
            if target is not None:
                result.setdefault(id(target), []).append(aura)
    return result


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
    owner_idx = controller_idx(state, aura)
    assert owner_idx is not None, "enchantment_count: aura not found on any battlefield"
    owner = state.players[owner_idx]
    return sum(1 for p in owner.battlefield if p.card_type == CardType.ENCHANTMENT)


def permanent_power(state, permanent, enchanting_auras=None):
    """A creature's effective power for combat.combat_damage_step and Ram
    Through's fight damage: its own base power (card_def.extra["power"], 0 if
    absent -- every real creature CardDef carries its own power/toughness, so
    the 0 default only ever matters for FILLER/synthetic self-check
    permanents) plus every Aura currently enchanting it (_enchanting_auras
    above -- owner-agnostic, correct regardless of state.active_idx). Each
    Aura's registry entry supplies its own "pt_bonus" (state, aura_permanent)
    -> int -- a constant for a static bonus (Rancor's +2), a battlefield-wide
    count for a dynamic one (Ancestral Mask/Ethereal Armor's "for each
    [other] enchantment").

    enchanting_auras: optional pre-fetched result of _enchanting_auras(state,
    permanent), for a caller that already needs the same list for multiple
    creatures in one pass (the per-token feature builder does, calling this
    AND permanent_toughness for every battlefield creature in one
    observation -- _enchanting_auras's own battlefield scan is a real,
    measurable cost, so this avoids repeating it redundantly per call). None
    (every existing caller) means "compute it myself."

    base also folds in _animate_spec (Pinnacle Kill-Ship's own animated
    power, once Station's threshold is met, overriding card_def.extra
    entirely) and _COUNTER_PT (a flat sum over this permanent's OWN
    counters -- Nyxborn Hydra's "+1/+1"s). Neither needs its own
    enchanting_auras-style pre-fetch/cache: both read only data already
    sitting on `permanent` itself (its own counters dict, one registry
    lookup) with no battlefield scan, so they're already exactly as cheap
    as the single per-creature call this function already gets."""
    transform = _transform_spec(permanent)
    animate = _animate_spec(permanent)
    if transform is not None:
        base = transform["power"]
    elif animate is not None:
        base = animate["power"]
    else:
        base = permanent.card_def.extra.get("power", 0)
    base += sum(_COUNTER_PT.get(kind, (0, 0))[0] * n for kind, n in permanent.counters.items())
    base += permanent.temp_power  # until-EOT modifier (Agony Warp's -3/-0), cleared at cleanup_step
    base += _static_self(state, permanent)[0]  # conditional static self-boost (Goblin Tomb Raider)
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
    transform = _transform_spec(permanent)
    animate = _animate_spec(permanent)
    if transform is not None:
        base = transform["toughness"]
    elif animate is not None:
        base = animate["toughness"]
    else:
        base = permanent.card_def.extra.get("toughness", 0)
    base += sum(_COUNTER_PT.get(kind, (0, 0))[1] * n for kind, n in permanent.counters.items())
    base += permanent.temp_toughness  # until-EOT modifier (Agony Warp's -0/-3), cleared at cleanup_step
    base += _static_self(state, permanent)[1]  # conditional static self-boost (Goblin Tomb Raider)
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
    count = sum(
        1 for aura in auras
        if registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("lifelink", False)
    )
    # Real (keyword) lifelink -- from a lifelink counter (Unexpected Fangs) or
    # an until-EOT grant (Toxin Analysis) -- adds ONE trigger and, unlike
    # Armadillo Cloak's stacking triggered ability above, does NOT stack with
    # itself: two lifelink sources on one creature still gain life only once.
    # So it contributes +1 iff "lifelink" is in the creature's keywords,
    # regardless of how many keyword-lifelink sources there are.
    if "lifelink" in creature_keywords(state, permanent, enchanting_auras=auras):
        count += 1
    return count


# Real Magic keyword strings this engine models as a boolean set: "vigilance"
# (Cartouche of Solidarity's Warrior token), "flying" (Kitchen Imp's real
# flying; also used for Silhana Ledgewalker's "can't be blocked except by
# creatures with flying" -- functionally the identical blocking restriction in
# a ruleset with no reach, so one flag covers both rather than a second
# near-duplicate), "trample" (Rancor, Armadillo Cloak), "first_strike"
# (Cartouche of Solidarity, Ethereal Armor). "hexproof" (Slippery Bogle,
# Gladecover Scout, Silhana Ledgewalker) and "shroud": the targeting
# restriction (can_be_targeted below) -- once opponents can target across
# sides (faithful burn/removal) these stop being no-ops and gate legal
# targets. "deathtouch" is modeled (Toxin Analysis grants it until end of
# turn via permanent.temp_keywords): combat.py marks a creature dealt damage
# by a deathtouch source and state_based.check_state_based_actions treats any
# such marked damage as lethal. "menace" is modeled (the Undercity Catacombs
# Skeleton token has it -- an intrinsic registry keyword, read here;
# game.effects.combat enforces the 2+-blocker rule at block finalization).
# Double strike: no card in this pool grants it, so it isn't modeled. (Reach
# is modeled per-card for Bramble Wurm.) Armadillo Cloak's own lifegain clause
# is NOT here -- see lifelink_count above for why a boolean keyword is the
# wrong model for a triggered, stacking ability (real, non-stacking keyword
# lifelink -- a lifelink counter / an until-EOT grant -- is folded into
# lifelink_count instead, as a single non-stacking +1).
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
    transform = _transform_spec(permanent)
    if transform is not None:
        keywords |= set(transform.get("keywords", ()))  # Insectile Aberration's flying
    animate = _animate_spec(permanent)
    if animate is not None:
        keywords |= set(animate.get("keywords", ()))
    for aura in auras:
        keywords |= set(registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("keywords", ()))
    keywords |= permanent.temp_keywords  # until-EOT granted keywords (Toxin Analysis' deathtouch/lifelink)
    keywords |= _static_self(state, permanent)[2]  # conditional static self-boost keywords (Goblin Tomb Raider's haste)
    if permanent.counters.get("lifelink", 0) > 0:
        keywords.add("lifelink")  # a lifelink counter grants lifelink (Unexpected Fangs)
    if permanent.flags.get("throne_hexproof"):
        keywords.add("hexproof")  # Undercity's Throne of the Dead Three: hexproof until your next turn
    return keywords


def has_keyword(state, permanent, keyword):
    return keyword in creature_keywords(state, permanent)


def has_haste(state, permanent):
    """Haste from either representation a registry entry can use: a flat
    "haste": True boolean (Kitchen Imp) or "haste" being a member of the
    union creature_keywords computes (an intrinsic "keywords" entry, a
    static-granted keyword like Goblin Tomb Raider's, or an until-EOT
    temp_keywords grant like Rally at the Hornburg's/Goblin Bushwhacker's
    team haste). THE canonical haste check -- both mana.py's
    tap_summoning_locked and combat.py's creature_attack_eligible must call
    this rather than reimplementing it, or they silently diverge again (as
    combat.py did until 2026-08: checking only the flat boolean meant a
    plain intrinsic-keyword haste creature like Reckless Lackey could never
    attack the turn it was cast)."""
    if registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("haste", False):
        return True
    return "haste" in creature_keywords(state, permanent)


def controller_idx(state, permanent):
    """The index of the player whose battlefield `permanent` is on (its
    controller), or None if it's on no battlefield (already left)."""
    for idx, player in enumerate(state.players):
        if permanent in player.battlefield:
            return idx
    return None


def can_be_targeted(state, permanent, by_player_idx):
    """Real Magic hexproof/shroud targeting restriction. Shroud: can't be
    the target of ANY spell or ability. Hexproof: can't be the target of a
    spell or ability an OPPONENT of its controller controls -- its own
    controller may still target it. `by_player_idx` is the player choosing
    the target (the caster/activator). Everything else is freely targetable.
    Every targeted effect's candidate predicate ANDs this in, so the
    restriction is enforced once, uniformly, instead of re-derived per card
    (the whole boggles deck -- Slippery Bogle/Gladecover Scout/Silhana
    Ledgewalker -- depends on opponents being unable to target it)."""
    kws = creature_keywords(state, permanent)
    if "shroud" in kws:
        return False
    if "hexproof" in kws and controller_idx(state, permanent) != by_player_idx:
        return False
    return True
