"""A creature's effective power/toughness/keywords -- base stats plus every
enchanting Aura. Owner-agnostic: combat_damage_step runs with active_idx on
the attacker but needs a blocker's stats too, so functions here search
state.players directly, not the active-player-proxied state.battlefield.

References registry.EFFECT_REGISTRY only inside function bodies -- see
game/registry.py's docstring."""

from .. import registry
from ..cards import CardType

# Counter kind -> (power, toughness) granted per counter. Kinds absent here
# (Pinnacle Kill-Ship's "charge") are a threshold marker, not a stat bonus.
_COUNTER_PT = {"+1/+1": (1, 1), "-0/-1": (0, -1)}


def _animate_spec(permanent):
    """Pinnacle Kill-Ship's Station: the "animate" registry spec (counter
    kind, threshold, power/toughness/keywords granted) once
    permanent.counters[kind] >= threshold, else None. Station's own resolve
    separately sets type_override = CREATURE once the threshold is first
    crossed; this covers the stats half type_override can't express."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("animate")
    if spec is None or permanent.counters.get(spec["counter"], 0) < spec["threshold"]:
        return None
    return spec


def _transform_spec(permanent):
    """A transformed DFC's back-face stats (Delver of Secrets -> Insectile
    Aberration). Returns the registry "transform" spec once
    permanent.flags["transformed"] is set, else None."""
    if not permanent.flags.get("transformed"):
        return None
    return registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("transform")


def _static_self(state, permanent):
    """A creature's own conditional static boost (Goblin Tomb Raider: "as
    long as you control an artifact, +1/+0 and haste"). Returns (power,
    toughness, keywords) from the registry "static_self" spec when its
    condition holds, else (0, 0, empty set)."""
    spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("static_self")
    if spec is None or not spec["condition"](state, permanent):
        return 0, 0, set()
    return spec.get("power", 0), spec.get("toughness", 0), set(spec.get("keywords", ()))


def _enchanting_auras(state, permanent):
    """Every Aura currently enchanting `permanent`, searched across both
    players' battlefields. Owner-agnostic on purpose: combat_damage_step
    runs with active_idx on the attacker but needs a blocker's correct
    stats too, and reading state.battlefield there would search the wrong
    player's board."""
    for player in state.players:
        for aura in player.battlefield:
            if aura.flags.get("enchanting") is permanent:
                yield aura


def enchanting_by_target(state):
    """Every Aura on either battlefield, bucketed by id(enchanted
    permanent) -- one full scan producing a dict a caller reads many
    permanents' auras from via `.get(id(permanent), ())`, instead of
    re-paying _enchanting_auras' own scan per permanent. Shared by every
    caller reading many permanents' auras from the same state in one pass.

    Caution: each caller takes this snapshot once and reuses it for the
    whole call. Safe only because no Aura in this pool has flash and
    nothing casts/attaches an Aura mid-call -- re-verify before trusting
    this snapshot if a flash Aura is ever added."""
    result = {}
    for player in state.players:
        for aura in player.battlefield:
            target = aura.flags.get("enchanting")
            if target is not None:
                result.setdefault(id(target), []).append(aura)
    return result


def enchantment_count(state, aura):
    """How many ENCHANTMENT-type permanents `aura`'s own controller has on
    the battlefield -- shared by every "for each [other] enchantment you
    control" pt_bonus (Ancestral Mask/Ethereal Armor differ only in whether
    the caller subtracts 1 for itself). Finds the controller by membership,
    not state.battlefield, so a blocker's own Aura is counted against its
    actual controller during combat, not whoever is currently active."""
    owner_idx = controller_idx(state, aura)
    assert owner_idx is not None, "enchantment_count: aura not found on any battlefield"
    owner = state.players[owner_idx]
    return sum(1 for p in owner.battlefield if p.card_type == CardType.ENCHANTMENT)


def permanent_power(state, permanent, enchanting_auras=None):
    """A creature's effective power: base power (card_def.extra["power"],
    0 if absent) plus every enchanting Aura's own "pt_bonus" (state,
    aura_permanent) -> int -- a constant for a static bonus (Rancor's +2),
    a battlefield-wide count for a dynamic one (Ancestral Mask/Ethereal
    Armor).

    enchanting_auras: optional pre-fetched _enchanting_auras(state,
    permanent) result, for a caller reading many creatures' stats in one
    pass. None means "compute it myself."

    base also folds in _animate_spec and _COUNTER_PT (this permanent's own
    counters) -- both read data already on `permanent` itself, no
    battlefield scan needed."""
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
    """A creature's effective toughness, mirroring permanent_power. Uses a
    separate "toughness_bonus" registry key rather than reusing "pt_bonus":
    Rancor is +2/+0 (power only), so a shared key would wrongly buff
    toughness too; the Auras that are genuinely +X/+X (Ancestral Mask,
    Ethereal Armor, Cartouche of Solidarity, Armadillo Cloak) set both.

    enchanting_auras: see permanent_power's docstring."""
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
    much life" triggers this creature's damage carries -- summed across
    every enchanting Aura whose registry entry sets "lifelink": True
    (Armadillo Cloak). Not in the boolean creature_keywords set: real
    lifelink is a static, non-stacking ability, but Armadillo Cloak's is a
    distinct triggered ability -- two Cloaks really do trigger twice (2x
    life gained), which a boolean keyword would wrongly dedup to one.

    enchanting_auras: see permanent_power's docstring."""
    auras = enchanting_auras if enchanting_auras is not None else _enchanting_auras(state, permanent)
    count = sum(
        1 for aura in auras
        if registry.EFFECT_REGISTRY.get(aura.card_def.effect_id, {}).get("lifelink", False)
    )
    # Real keyword lifelink (a counter or until-EOT grant) adds one trigger
    # and does NOT stack with itself, unlike Armadillo Cloak above.
    if "lifelink" in creature_keywords(state, permanent, enchanting_auras=auras):
        count += 1
    return count


# Keyword strings modeled as a boolean set: vigilance, flying (also covers
# "can't be blocked except by flying"), trample, first_strike, hexproof,
# shroud, deathtouch (temp_keywords grant; marked damage from it is lethal
# per state_based.check_state_based_actions), menace (2+-blocker rule
# enforced by game.effects.combat). Reach is modeled per-card. Double strike
# isn't modeled (no card grants it). Armadillo Cloak's lifegain is NOT a
# keyword here -- see lifelink_count for why it's a stacking trigger instead.
def creature_keywords(state, permanent, enchanting_auras=None):
    """Union of this permanent's own intrinsic registry "keywords" plus
    every enchanting Aura's own granted "keywords" -- same shape as
    permanent_power/permanent_toughness, owner-agnostic via
    _enchanting_auras.

    enchanting_auras: see permanent_power's docstring. Also folds in
    _animate_spec's granted keywords (Pinnacle Kill-Ship's flying)."""
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
    "haste": True boolean, or "haste" in the creature_keywords union
    (intrinsic keyword, static grant, or until-EOT temp_keywords grant).
    The canonical haste check -- mana.py's tap_summoning_locked and
    combat.py's creature_attack_eligible must both call this rather than
    reimplementing it."""
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
    """Hexproof/shroud targeting restriction. Shroud: can't be targeted by
    any spell/ability. Hexproof: can't be targeted by an opponent of its
    controller (its own controller still can). `by_player_idx` is the
    player choosing the target. Every targeted effect's candidate predicate
    ANDs this in, so the restriction is enforced once, uniformly."""
    kws = creature_keywords(state, permanent)
    if "shroud" in kws:
        return False
    if "hexproof" in kws and controller_idx(state, permanent) != by_player_idx:
        return False
    return True
