"""Attack/block declaration, eligibility, and combat damage. Depends on
stats.py for effective power/toughness/keywords, state_based.py for the
state-based-action check combat damage triggers, and win_check.py for the
opponent-facing damage effect (and, for "lifelink", the life-gaining one)."""

from . import stats, state_based
from .win_check import deal_damage_to_opponent, gain_life
from .. import registry
from ..cards import CardType


def creature_attack_eligible(state, permanent):
    """Untapped, not a Defender (Wall of Roots/Overgrown Battlement/Saruli
    Caretaker/Gatecreeper Vine -- real Magic's own rule: a Defender can
    never attack, full stop, regardless of tapped/summoning-sick status),
    and not summoning sick unless it has a registry "haste": True spec
    (Kitchen Imp) -- the only other place that flag is ever read, so this
    is the only place haste needs to matter. Checked per creature (drl_env's
    "Attack: <name>" actions) so a model can declare SOME eligible
    creatures as attackers and hold others back (as blockers once those
    exist, or as mana sources) instead of the old everyone-eligible-attacks
    wholesale rule.

    Also excludes anything already in state.attackers -- ordinarily
    redundant with the tapped check above (declare_attacker taps its
    permanent), but vigilance (Cartouche of Solidarity's own Warrior
    token) deliberately skips that tap, so without this explicit guard a
    vigilant creature would stay "eligible" forever within the same
    combat and could be declared an attacker repeatedly, each declaration
    appending a duplicate entry to state.attackers and multiplying its
    power in combat_damage_step's unblocked-damage total. Mirrors the
    explicit, tapped-independent guard creature_block_eligible already
    has for the identical reason (blocking never taps anyone either)."""
    return (
        permanent.card_def.card_type == CardType.CREATURE and not permanent.tapped
        and not permanent.card_def.extra.get("defender", False)
        and permanent not in state.attackers
        and (not permanent.summoning_sick or registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("haste", False))
    )


def creature_block_eligible(state, permanent):
    """Untapped and not already assigned to block something else this
    combat -- no gang-blocking/menace modeled, docs/COMBAT_PLAN.md. Reads
    state.opponent.blocked_by, NOT state.blocked_by: this is only ever
    called with state.active_idx already flipped to the defender
    (game.turn._declare_blockers_gen), and PlayerState.blocked_by is keyed
    by the ATTACKING player's own attacker permanents (see its own
    docstring) -- from the flipped-to-defender perspective, that dict
    lives on state.opponent, not on the (active, defending) player
    state.blocked_by itself would read. Deliberately NOT the same
    eligibility as creature_attack_eligible: real Magic lets a Defender
    block (that's its whole point) and lets a summoning-sick creature
    block (summoning sickness only restricts attacking and {T}
    abilities) -- so neither check belongs here."""
    return (
        permanent.card_def.card_type == CardType.CREATURE and not permanent.tapped
        and permanent not in state.opponent.blocked_by.values()
    )


def declare_attacker(state, permanent):
    """Model chose to attack with this specific creature -- addressed by
    (name, slot) at the drl_env.py action-table layer
    (docs/COMBAT_PLAN.md's permanent-identity design), so the caller has
    already picked the exact physical copy it means, not an arbitrary
    same-named match. Tapped here, at declaration, same as real Magic --
    an attacking creature is unavailable for a mana ability etc. for the
    rest of combat, not just tapped as a side effect of dealing damage
    later -- UNLESS it has vigilance (Cartouche of Solidarity's own
    Warrior token), which is real Magic's entire point of the keyword:
    attacking doesn't tap it at all."""
    if not stats.has_keyword(state, permanent, "vigilance"):
        permanent.tapped = True
    state.attackers.append(permanent)


def declare_attackers_step(state):
    """game.turn.Phase.DECLARE_ATTACKERS phase-entry reset (rakdos
    madness / mono red madness / boggles -- gated by combat_enabled, same
    as combat always was): clears last turn's attackers AND blocks (both
    reset together -- a fresh combat has neither yet) so the model's own
    "Attack: <name> (slot k)" actions (drl_env.build_action_table, each
    one checking creature_attack_eligible and calling declare_attacker)
    start this turn's declaration fresh, one creature at a time, rather
    than the old wholesale auto-attack."""
    state.attackers = []
    state.blocked_by = {}


def _is_alive(state, permanent):
    return any(permanent in player.battlefield for player in state.players)


def _attacker_deal_damage(state, attacker, blocker, attacker_facts, blocker_facts):
    """Attacker deals its combat damage to its blocker -- trample-aware
    (Rancor, Armadillo Cloak): assigns only enough to be lethal (the
    blocker's own remaining toughness), letting any excess spill over to
    the DEFENDING player via deal_damage_to_opponent (state.opponent, from
    the attacker's own active perspective, IS the defender throughout
    combat_damage_step) instead of being wasted on an already-dead
    blocker. Real Magic lets the attacker choose to overkill the blocker
    instead -- never correct without deathtouch (not modeled in this card
    pool), so this always assigns the minimum lethal, no extra decision or
    action needed.

    "lifelink" (Armadillo Cloak's own "whenever enchanted creature deals
    damage, you gain that much life" -- a TRIGGERED ability, unlike real
    lifelink: stats.lifelink_count returns however many Cloaks are
    attached, each an independent trigger for the FULL damage dealt, so
    two Cloaks means 2x life gained, not 1x -- see that function's own
    docstring): the attacker's controller is always the currently ACTIVE
    player throughout combat_damage_step, so gain_life's own active-
    player-proxied state.life_total is already correct here -- total
    damage dealt is always `power` regardless of trample (lethal-to-
    blocker + spillover-to-player always sums back to power).

    attacker_facts/blocker_facts: combat_damage_step's own pre-fetched
    {power, toughness, first_strike, trample, lifelink_count} dict for
    each creature (see that function's own docstring) -- power/trample/
    lifelink_count read here are ALWAYS the attacker's own, toughness read
    here is ALWAYS the blocker's own; each of these used to be its own
    independent stats.py call (permanent_power/has_keyword("trample")/
    permanent_toughness/lifelink_count), each independently re-scanning
    state.players for the same creature's own Auras -- profiled: up to
    ~8-10 of those redundant scans per blocked pair, for what's really
    only 8 distinct facts (4 per creature)."""
    power = attacker_facts["power"]
    if attacker_facts["trample"]:
        lethal = min(power, max(blocker_facts["toughness"] - blocker.damage_marked, 0))
        blocker.damage_marked += lethal
        excess = power - lethal
        if excess > 0:
            deal_damage_to_opponent(state, excess)
    else:
        blocker.damage_marked += power
    lifelink_count = attacker_facts["lifelink_count"]
    if lifelink_count:
        gain_life(state, power * lifelink_count)


def _blocker_deal_damage(state, blocker, attacker, blocker_facts):
    """Blocker deals its combat damage to the attacker it's blocking --
    never tramples through to a player: trample is an attacking-creature
    keyword only, nothing in this card pool grants a blocker-side
    equivalent, and this engine doesn't model one.

    "lifelink": unlike the attacker-side case above, the blocker's
    controller is the DEFENDING player -- state.opponent from the
    currently-active attacker's own perspective, not state.life_total
    (which would wrongly credit the attacker) -- so this credits
    state.opponent.life_total directly instead of going through
    gain_life. Multiplied by stats.lifelink_count the same stacking way
    as the attacker-side case above (2 Cloaks on a blocker also trigger
    twice).

    blocker_facts: see _attacker_deal_damage's own docstring -- same
    pre-fetched dict, just the blocker's own this time (no attacker_facts
    needed here: a blocker never tramples, so nothing about the attacker's
    own stats is read in this direction)."""
    power = blocker_facts["power"]
    attacker.damage_marked += power
    lifelink_count = blocker_facts["lifelink_count"]
    if lifelink_count:
        state.opponent.life_total += power * lifelink_count


def combat_damage_step(state):
    """game.turn.Phase.COMBAT_DAMAGE: total power (stats.permanent_power(
    state, p) -- base card_def.extra["power"] plus any attached Auras' own
    "pt_bonus") of state.attackers NOT present in state.blocked_by
    (declared in DECLARE_ATTACKERS via declare_attacker; assigned a
    blocker, if any, during the defending player's own consult --
    docs/COMBAT_PLAN.md) hits the opponent via deal_damage_to_opponent
    once; a creature with neither power nor an Aura set (e.g. an
    untracked-stats vanilla from another deck) contributes 0.

    Every blocked attacker/blocker pair fights in up to two sub-steps,
    real Magic's own first-strike ordering: first, whichever side(s) of
    each pair have first_strike (Cartouche of Solidarity, Ethereal Armor)
    deal their damage and a state-based-action check runs -- a first-
    strike kill here means the victim is already gone before the regular
    sub-step, so it never deals damage back. Then, whichever side(s)
    DON'T have first strike deal their damage (their only shot -- a
    first-strike side already had its one shot above), but only if both
    it and its target are still alive, followed by a second SBA check.
    With no first strike anywhere in a given combat this collapses to the
    exact same single simultaneous exchange as before first strike
    existed (every pair's damage all lands in the "regular" sub-step).

    "lifelink" on an unblocked attacker: gained all at once here (summed
    across every unblocked lifelinker, same batching deal_damage_to_
    opponent's own unblocked_total already does) rather than inside
    _attacker_deal_damage -- there's no blocker/trample math to share per-
    creature here, unblocked damage is already just `power` per attacker.
    Each attacker's own power is multiplied by its own stats.lifelink_count
    (2 Cloaks on one attacker = 2x that attacker's own contribution), not
    just added once per lifelinking attacker.

    Every combatant's own {power, toughness, first_strike, trample,
    lifelink_count} is fetched exactly ONCE here, up front, and reused
    everywhere below -- profiled: this used to call stats.permanent_power
    twice for the same unblocked attacker (unblocked_total and
    lifelink_total, back to back), and stats.has_keyword(..., "first_
    strike") twice per blocked pair (once per sub-step loop, even though
    first-strike status can't change between them -- nothing runs in
    between except a state-based-action check, which only removes dead
    creatures). Each of those was its own independent stats.py call, each
    independently re-scanning state.players for that creature's own Auras
    via stats._enchanting_auras -- up to ~8-10 redundant scans per blocked
    pair for what's really only 8 distinct facts (4 per creature). Safe to
    prefetch once for the whole call: nothing in this function casts a new
    spell or attaches/detaches an Aura mid-resolution, so no combatant's
    own facts can change during it (damage_marked does change, but that's
    read fresh off the permanent itself everywhere, never cached here)."""
    unblocked = [p for p in state.attackers if p not in state.blocked_by]
    pairs = list(state.blocked_by.items())
    all_combatants = set(state.attackers) | {p for _a, p in pairs}

    enchanting_by_target = {}
    if all_combatants:
        for player in state.players:
            for aura in player.battlefield:
                target = aura.flags.get("enchanting")
                if target is not None:
                    enchanting_by_target.setdefault(id(target), []).append(aura)

    def _facts(permanent):
        auras = enchanting_by_target.get(id(permanent), ())
        keywords = stats.creature_keywords(state, permanent, enchanting_auras=auras)
        return {
            "power": stats.permanent_power(state, permanent, enchanting_auras=auras),
            "toughness": stats.permanent_toughness(state, permanent, enchanting_auras=auras),
            "first_strike": "first_strike" in keywords,
            "trample": "trample" in keywords,
            "lifelink_count": stats.lifelink_count(state, permanent, enchanting_auras=auras),
        }

    creature_facts = {id(p): _facts(p) for p in all_combatants}

    unblocked_total = sum(creature_facts[id(p)]["power"] for p in unblocked)
    lifelink_total = sum(
        creature_facts[id(p)]["power"] * creature_facts[id(p)]["lifelink_count"] for p in unblocked
    )
    state.attackers = []
    deal_damage_to_opponent(state, unblocked_total)
    if lifelink_total:
        gain_life(state, lifelink_total)

    for attacker, blocker in pairs:
        if creature_facts[id(attacker)]["first_strike"]:
            _attacker_deal_damage(state, attacker, blocker, creature_facts[id(attacker)], creature_facts[id(blocker)])
        if creature_facts[id(blocker)]["first_strike"]:
            _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)

    for attacker, blocker in pairs:
        attacker_alive, blocker_alive = _is_alive(state, attacker), _is_alive(state, blocker)
        if not creature_facts[id(attacker)]["first_strike"] and attacker_alive and blocker_alive:
            _attacker_deal_damage(state, attacker, blocker, creature_facts[id(attacker)], creature_facts[id(blocker)])
        if not creature_facts[id(blocker)]["first_strike"] and blocker_alive and attacker_alive:
            _blocker_deal_damage(state, blocker, attacker, creature_facts[id(blocker)])
    state_based.check_state_based_actions(state)


if __name__ == "__main__":
    # ponytail self-check: run via `python -m game.effects.combat` from
    # src/. Attack eligibility + declaration + damage, then the keyword
    # trio (vigilance/trample/first strike) -- everything specific to THIS
    # module. The combat+SBA creature-death handoff lives in
    # effects/integration_check.py instead (it exercises state_based.py
    # just as much as this module).
    from ..cards import CardDef, EffectId
    from ..state import GameState, Permanent

    state = GameState(on_the_play=True, terminated_fn=lambda s: s.damage_dealt >= 5)
    attacker = Permanent(CardDef("Attacker", CardType.CREATURE, None, EffectId.FILLER, power=3))
    attacker.summoning_sick = False
    sick = Permanent(CardDef("Sick", CardType.CREATURE, None, EffectId.FILLER, power=10))  # summoning_sick=True by construction -- never cleared here (that's untap_step's job)
    already_tapped = Permanent(CardDef("Tapped Out", CardType.CREATURE, None, EffectId.FILLER, power=10), tapped=True)
    already_tapped.summoning_sick = False
    vanilla = Permanent(CardDef("No Stats", CardType.CREATURE, None, EffectId.FILLER))  # no "power" key at all -- untracked-stats precedent (Masked Vandal, Mesmeric Fiend)
    vanilla.summoning_sick = False
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER, power=10))
    not_a_creature.summoning_sick = False
    defender = Permanent(CardDef("Turtle Wall", CardType.CREATURE, None, EffectId.FILLER, power=10, defender=True))
    defender.summoning_sick = False
    state.battlefield = [attacker, sick, already_tapped, vanilla, not_a_creature, defender]

    declare_attackers_step(state)
    assert state.attackers == []  # phase-entry reset, no auto-population anymore
    assert creature_attack_eligible(state, attacker)
    assert creature_attack_eligible(state, vanilla)  # 0 power still eligible, same as a real 0-power creature
    assert not creature_attack_eligible(state, sick)
    assert not creature_attack_eligible(state, already_tapped)
    assert not creature_attack_eligible(state, not_a_creature)
    assert not creature_attack_eligible(state, defender)  # every other rule satisfied, but can never attack

    declare_attacker(state, attacker)
    assert attacker.tapped and attacker in state.attackers
    assert not vanilla.tapped and vanilla not in state.attackers  # partial declaration -- vanilla deliberately left back
    combat_damage_step(state)
    assert state.damage_dealt == 3
    assert state.attackers == []
    assert state.turn_won is None

    print("combat.py eligibility + damage self-check: OK")

    # Haste (Kitchen Imp): a "haste": True registry spec lets a summoning-
    # sick creature be attack-eligible anyway -- the only place that spec
    # is ever read.
    state = GameState(on_the_play=True)
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"haste": True}
    try:
        hasty = Permanent(CardDef("Hasty", CardType.CREATURE, None, EffectId.FILLER, power=2))
        assert hasty.summoning_sick
        state.battlefield = [hasty]
        declare_attackers_step(state)
        assert creature_attack_eligible(state, hasty)
        declare_attacker(state, hasty)
        combat_damage_step(state)
        assert state.damage_dealt == 2 and hasty.tapped
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup

    print("combat.py haste self-check: OK")

    # Vigilance (docs/COMBAT_PLAN.md step 7): attacking a vigilant
    # creature never taps it, unlike an ordinary attacker.
    from .tokens import WARRIOR_TOKEN_CARD_DEF  # the real EffectId.WARRIOR_TOKEN registry entry (white_cards.py) grants vigilance

    state = GameState(on_the_play=True)
    vigilant = Permanent(WARRIOR_TOKEN_CARD_DEF)
    vigilant.summoning_sick = False
    ordinary = Permanent(CardDef("Ordinary Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ordinary.summoning_sick = False
    state.battlefield = [vigilant, ordinary]
    declare_attackers_step(state)
    declare_attacker(state, vigilant)
    declare_attacker(state, ordinary)
    assert not vigilant.tapped and vigilant in state.attackers
    assert ordinary.tapped and ordinary in state.attackers

    # Regression: a vigilant creature staying untapped must NOT make it
    # re-declarable -- it already attacked this combat, tapped or not.
    # Before creature_attack_eligible's own state.attackers guard existed,
    # this stayed "eligible" forever (vigilance skips the only other
    # exclusion, tapped) and repeated declare_attacker calls silently
    # duplicated it in state.attackers, multiplying its power in
    # combat_damage_step's unblocked-damage total.
    assert not creature_attack_eligible(state, vigilant)
    assert state.attackers.count(vigilant) == 1
    combat_damage_step(state)
    assert state.damage_dealt == 1 + 1  # vigilant's own power once, ordinary's power once -- not doubled

    print("combat.py vigilance self-check: OK")

    # Trample (the real EffectId.RANCOR registry entry): a blocked
    # attacker with trample assigns only enough damage to be lethal to its
    # blocker, letting the rest spill over to the DEFENDING player.
    from ..state import PlayerState

    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Trampler", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
        trampler.summoning_sick = False
        registry.CARD_DEFS["Trampler"] = trampler.card_def
        rancor_on_trampler = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
        rancor_on_trampler.flags["enchanting"] = trampler
        weak_blocker = Permanent(CardDef("Weak Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        registry.CARD_DEFS["Weak Blocker"] = weak_blocker.card_def
        state.players[0].battlefield = [trampler, rancor_on_trampler]
        state.players[1].battlefield = [weak_blocker]

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = weak_blocker
        combat_damage_step(state)

        # Effective power 7 (5 base + Rancor's +2): 2 assigned as lethal
        # (weak_blocker's own toughness), 5 tramples through.
        assert weak_blocker not in state.players[1].battlefield
        assert state.damage_dealt == 5
        assert trampler in state.players[0].battlefield and trampler.damage_marked == 1
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py trample self-check: OK")

    # First strike (the real EffectId.CARTOUCHE_OF_SOLIDARITY registry
    # entry): a blocked attacker with first strike deals its damage BEFORE
    # the blocker gets a chance to.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        fs_attacker = Permanent(CardDef("First Striker", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=1))
        fs_attacker.summoning_sick = False
        registry.CARD_DEFS["First Striker"] = fs_attacker.card_def
        cartouche_on_attacker = Permanent(CardDef("Cartouche of Solidarity", CardType.ENCHANTMENT, {"W": 1}, EffectId.CARTOUCHE_OF_SOLIDARITY))
        cartouche_on_attacker.flags["enchanting"] = fs_attacker
        lethal_blocker = Permanent(CardDef("Would-Be Killer", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        registry.CARD_DEFS["Would-Be Killer"] = lethal_blocker.card_def
        state.players[0].battlefield = [fs_attacker, cartouche_on_attacker]
        state.players[1].battlefield = [lethal_blocker]

        declare_attackers_step(state)
        declare_attacker(state, fs_attacker)
        state.blocked_by[fs_attacker] = lethal_blocker
        combat_damage_step(state)

        # Effective power 5 (4 base + Cartouche's +1) >= lethal_blocker's
        # toughness 3 -- dies in the FIRST STRIKE sub-step, before it ever
        # deals its own power-3 damage back.
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py first-strike self-check: OK")

    # Lifelink (the real EffectId.ARMADILLO_CLOAK registry entry):
    # "whenever enchanted creature deals damage, you gain that much life"
    # -- credited to whichever side actually controls the damage-dealing
    # creature, both for an unblocked attacker and for a BLOCKING creature
    # (the defending player, never the currently-active attacker).
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        lifelinker = Permanent(CardDef("Lifelinker", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        lifelinker.summoning_sick = False
        registry.CARD_DEFS["Lifelinker"] = lifelinker.card_def
        cloak = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak.flags["enchanting"] = lifelinker
        state.players[0].battlefield = [lifelinker, cloak]

        declare_attackers_step(state)
        declare_attacker(state, lifelinker)
        combat_damage_step(state)  # unblocked

        # Effective power 5 (3 base + Cloak's own +2, same Aura bonus
        # permanent_power always applies) -- both the damage AND the
        # lifelink gain use this effective total, not the base 3.
        assert state.damage_dealt == 5
        assert state.players[0].life_total == 25  # STARTING_LIFE (20) + 5, unblocked lifelink

        # STACKING: two Armadillo Cloaks on the SAME creature -- two
        # independent triggers, not a boolean that dedups to one. Real
        # rule (unlike real lifelink, which never stacks): each Cloak's
        # own "whenever enchanted creature deals damage, you gain that
        # much life" fires separately, each for the FULL damage dealt.
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        double_cloaked = Permanent(CardDef("Double Cloaked", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        double_cloaked.summoning_sick = False
        registry.CARD_DEFS["Double Cloaked"] = double_cloaked.card_def
        cloak_a = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak_a.flags["enchanting"] = double_cloaked
        cloak_b = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak_b.flags["enchanting"] = double_cloaked
        state.players[0].battlefield = [double_cloaked, cloak_a, cloak_b]

        declare_attackers_step(state)
        declare_attacker(state, double_cloaked)
        combat_damage_step(state)  # unblocked

        # Effective power 7 (3 base + 2+2 from both Cloaks). Life gained:
        # 7 * 2 (one trigger per Cloak) = 14, NOT just 7 (what a boolean
        # "lifelink" keyword would wrongly give, deduped to one trigger).
        assert state.damage_dealt == 7
        assert state.players[0].life_total == 34  # STARTING_LIFE (20) + 14

        # A blocked lifelinker with trample: effective power 5 (3 base +
        # Cloak's own +2) vs a 2-toughness blocker -- 2 assigned as
        # lethal, 3 tramples through, but the FULL 5 is gained as life --
        # Armadillo Cloak's own text has no "excess damage only" carve-out,
        # unlike trample's own player-damage rule.
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        lifelinker2 = Permanent(CardDef("Lifelinker2", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        lifelinker2.summoning_sick = False
        registry.CARD_DEFS["Lifelinker2"] = lifelinker2.card_def
        cloak2 = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak2.flags["enchanting"] = lifelinker2
        blocker = Permanent(CardDef("Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        registry.CARD_DEFS["Blocker"] = blocker.card_def
        state.players[0].battlefield = [lifelinker2, cloak2]
        state.players[1].battlefield = [blocker]

        declare_attackers_step(state)
        declare_attacker(state, lifelinker2)
        state.blocked_by[lifelinker2] = blocker
        combat_damage_step(state)

        assert state.damage_dealt == 3  # trampled-through excess (5 power - 2 lethal)
        assert state.players[0].life_total == 25  # +5, the FULL effective power, not just the excess

        # A BLOCKING lifelinker: life goes to the DEFENDING player (index
        # 1 here, since player 0 is the attacker/active side throughout
        # combat_damage_step), never state.life_total (which would
        # silently credit the wrong side).
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker3 = Permanent(CardDef("Attacker3", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=5))
        attacker3.summoning_sick = False
        registry.CARD_DEFS["Attacker3"] = attacker3.card_def
        blocker_lifelinker = Permanent(CardDef("BlockerLifelinker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        registry.CARD_DEFS["BlockerLifelinker"] = blocker_lifelinker.card_def
        cloak3 = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"generic": 1, "G": 1, "W": 1}, EffectId.ARMADILLO_CLOAK))
        cloak3.flags["enchanting"] = blocker_lifelinker
        state.players[0].battlefield = [attacker3]
        state.players[1].battlefield = [blocker_lifelinker, cloak3]

        declare_attackers_step(state)
        declare_attacker(state, attacker3)
        state.blocked_by[attacker3] = blocker_lifelinker
        combat_damage_step(state)

        assert state.players[1].life_total == 24  # +4 (2 base + Cloak's +2) -- the DEFENDING player
        assert state.players[0].life_total == 20  # unaffected -- attacker3 itself has no lifelink
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)

    print("combat.py lifelink self-check: OK")
