"""Attack eligibility + declaration + damage, then the keyword trio
(vigilance/trample/first strike) -- everything specific to THIS module. The
combat+SBA creature-death handoff lives in
tests/game/effects/test_integration_check.py instead (it exercises
state_based.py just as much as this module)."""
import pytest

from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.combat import (
    can_block,
    combat_damage_step,
    creature_attack_eligible,
    declare_attacker,
    declare_attackers_step,
    enforce_menace,
    has_unfulfilled_goad,
    menace_block_incomplete,
)
from game.effects.stack import resolve_top_of_stack
from game.effects.tokens import WARRIOR_TOKEN_CARD_DEF
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def test_creature_attack_eligibility_and_damage():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
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
    assert state.attackers == []  # phase-entry reset -- a fresh combat starts with no attackers declared
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
    assert state.players[1].life_total == 17  # 20 - the lone eligible attacker's power (3)
    assert state.attackers == []
    assert state.turn_won is None


@pytest.mark.parametrize(
    "spec",
    [{"haste": True}, {"keywords": {"haste"}}],
    ids=["flat_bool", "keyword_set"],
)
def test_haste_registry_spec(spec):
    # Haste, both registry spec forms: a flat "haste": True (Kitchen Imp) and
    # haste granted via the "keywords" set (Reckless Lackey, Clockwork
    # Percussionist) must both let a summoning-sick creature be
    # attack-eligible anyway. Regression (2026-08): creature_attack_eligible
    # used to check ONLY the flat boolean, so a keyword-set haste creature
    # could never attack the turn it was cast. Both are real branches of
    # stats.has_haste -- the one canonical haste check creature_attack_eligible
    # and mana.py's tap_summoning_locked both share.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = spec
    try:
        hasty = Permanent(CardDef("Hasty", CardType.CREATURE, None, EffectId.FILLER, power=2))
        assert hasty.summoning_sick
        state.battlefield = [hasty]
        declare_attackers_step(state)
        assert creature_attack_eligible(state, hasty)
        declare_attacker(state, hasty)
        combat_damage_step(state)
        assert state.players[1].life_total == 18 and hasty.tapped  # 20 - hasty's power (2)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_vigilance_no_tap_and_no_redeclare():
    # Vigilance: attacking with a vigilant creature never taps it, unlike an
    # ordinary attacker.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    vigilant = Permanent(WARRIOR_TOKEN_CARD_DEF)  # the real EffectId.WARRIOR_TOKEN registry entry (white_cards.py) grants vigilance
    vigilant.summoning_sick = False
    ordinary = Permanent(CardDef("Ordinary Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ordinary.summoning_sick = False
    state.battlefield = [vigilant, ordinary]
    declare_attackers_step(state)
    declare_attacker(state, vigilant)
    declare_attacker(state, ordinary)
    assert not vigilant.tapped and vigilant in state.attackers
    assert ordinary.tapped and ordinary in state.attackers

    # A vigilant creature staying untapped must NOT make it re-declarable --
    # it already attacked this combat, tapped or not. Without
    # creature_attack_eligible's own state.attackers guard, a vigilant
    # creature (which skips the only other exclusion, tapped) would stay
    # "eligible" forever, and repeated declare_attacker calls would silently
    # duplicate it in state.attackers, multiplying its power in
    # combat_damage_step's unblocked-damage total.
    assert not creature_attack_eligible(state, vigilant)
    assert state.attackers.count(vigilant) == 1
    combat_damage_step(state)
    assert state.players[1].life_total == 18  # 20 - (vigilant's power once + ordinary's power once) -- not doubled


def test_trample_spills_excess_to_defending_player():
    # Trample (the real EffectId.RANCOR registry entry): a blocked
    # attacker with trample assigns only enough damage to be lethal to its
    # blocker, letting the rest spill over to the DEFENDING player.
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
        state.blocked_by[trampler] = [weak_blocker]
        combat_damage_step(state)

        # Effective power 7 (5 base + Rancor's +2): 2 assigned as lethal
        # (weak_blocker's own toughness), 5 tramples through.
        assert weak_blocker not in state.players[1].battlefield
        assert state.players[1].life_total == 15  # 20 - the 5 that trampled through
        assert trampler in state.players[0].battlefield and trampler.damage_marked == 1
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_trample_all_through_when_every_blocker_already_gone():
    # A blocked trampler whose only blocker died BEFORE combat_damage_step
    # ran (e.g. an instant-speed removal spell resolved in the real priority
    # round game.turn gives both players right after blocks + damage
    # assignment, still within Phase.DECLARE_BLOCKERS, before Phase.
    # COMBAT_DAMAGE even starts) is still "blocked" -- 702.19e/510.1c: with
    # no living blocker left to assign any lethal damage to, its FULL power
    # tramples through to the defending player, not zero. Modeled directly
    # here (state.blocked_by records the block; the blocker was simply never
    # put on any battlefield, matching what "already gone by the time combat
    # damage is dealt" looks like to combat_damage_step's own _is_alive checks).
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Trampler2", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
        trampler.summoning_sick = False
        registry.CARD_DEFS["Trampler2"] = trampler.card_def
        rancor_on_trampler = Permanent(CardDef("Rancor", CardType.ENCHANTMENT, {"G": 1}, EffectId.RANCOR))
        rancor_on_trampler.flags["enchanting"] = trampler
        gone_blocker = Permanent(CardDef("Gone Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
        state.players[0].battlefield = [trampler, rancor_on_trampler]
        # gone_blocker deliberately NOT added to either battlefield -- already
        # dead by the time combat damage is dealt, same as _is_alive would see
        # a real mid-combat removal target.

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = [gone_blocker]
        combat_damage_step(state)

        # Effective power 7 (5 base + Rancor's +2), ALL of it through -- no
        # living blocker left to assign any of it to.
        assert state.players[1].life_total == 13  # 20 - 7
        assert trampler.damage_marked == 0  # a dead blocker deals no damage back
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_first_strike_kills_before_blocker_deals_damage():
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
        state.blocked_by[fs_attacker] = [lethal_blocker]
        combat_damage_step(state)

        # Effective power 5 (4 base + Cartouche's +1) >= lethal_blocker's
        # toughness 3 -- dies in the FIRST STRIKE sub-step, before it ever
        # deals its own power-3 damage back.
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_unblocked_attacker():
    # Lifelink (the real EffectId.ARMADILLO_CLOAK registry entry):
    # "whenever enchanted creature deals damage, you gain that much life"
    # -- credited to whichever side actually controls the damage-dealing
    # creature.
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
        assert state.players[1].life_total == 15  # 20 - the unblocked lifelinker's power (5)
        assert state.players[0].life_total == 25  # STARTING_LIFE (20) + 5, unblocked lifelink
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_stacking_two_cloaks():
    # STACKING: two Armadillo Cloaks on the SAME creature -- two
    # independent triggers, not a boolean that dedups to one. Real
    # rule (unlike real lifelink, which never stacks): each Cloak's
    # own "whenever enchanted creature deals damage, you gain that
    # much life" fires separately, each for the FULL damage dealt.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
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
        assert state.players[1].life_total == 13  # 20 - the double-cloaked lifelinker's power (7)
        assert state.players[0].life_total == 34  # STARTING_LIFE (20) + 14
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_blocked_trample_gains_full_effective_power():
    # A blocked lifelinker with trample: effective power 5 (3 base +
    # Cloak's own +2) vs a 2-toughness blocker -- 2 assigned as
    # lethal, 3 tramples through, but the FULL 5 is gained as life --
    # Armadillo Cloak's own text has no "excess damage only" carve-out,
    # unlike trample's own player-damage rule.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
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
        state.blocked_by[lifelinker2] = [blocker]
        combat_damage_step(state)

        assert state.players[1].life_total == 17  # 20 - the trampled-through excess (5 power - 2 lethal)
        assert state.players[0].life_total == 25  # +5, the FULL effective power, not just the excess
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_on_blocking_creature_credits_defender():
    # A BLOCKING lifelinker: life goes to the DEFENDING player (index
    # 1 here, since player 0 is the attacker/active side throughout
    # combat_damage_step), never state.life_total (which would
    # silently credit the wrong side).
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
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
        state.blocked_by[attacker3] = [blocker_lifelinker]
        combat_damage_step(state)

        assert state.players[1].life_total == 24  # +4 (2 base + Cloak's +2) -- the DEFENDING player
        assert state.players[0].life_total == 20  # unaffected -- attacker3 itself has no lifelink
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_gang_blocking_damage_split():
    # --- GANG-BLOCKING: one attacker, two blockers, model-decided split ---
    # A 3/3 attacker gang-blocked by two 2/2s. The attacker's controller
    # assigns 2 damage to the first blocker (lethal -> dies) and only 1 to
    # the second (survives) -- an ARBITRARY, non-lethal-to-all split, the
    # whole point of the model decision. Both blockers deal 2 back, so the
    # attacker takes 4 >= 3 and dies. The split is stashed on the
    # attacker's own flags exactly as resolution.begin_assign_combat_damage
    # records it, and must be consumed (popped) by the damage step.
    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        gang_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        gb_atk = Permanent(CardDef("GBAttacker", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        gb_atk.summoning_sick = False
        gb_b1 = Permanent(CardDef("GBBlocker1", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        gb_b2 = Permanent(CardDef("GBBlocker2", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
        for _p in (gb_atk, gb_b1, gb_b2):
            registry.CARD_DEFS[_p.card_def.name] = _p.card_def
        gang_state.players[0].battlefield = [gb_atk]
        gang_state.players[1].battlefield = [gb_b1, gb_b2]
        declare_attackers_step(gang_state)
        declare_attacker(gang_state, gb_atk)
        gang_state.blocked_by[gb_atk] = [gb_b1, gb_b2]
        gb_atk.flags["combat_damage_split"] = ({gb_b1: 2, gb_b2: 1}, 0)  # attacker's own arbitrary choice
        combat_damage_step(gang_state)
        assert gb_b1 not in gang_state.players[1].battlefield  # took 2 (lethal) -> dead
        assert gb_b2 in gang_state.players[1].battlefield       # took only 1 -> survives
        assert gb_atk not in gang_state.players[0].battlefield  # took 2+2=4 >= 3 -> dead
        assert "combat_damage_split" not in gb_atk.flags        # consumed (popped) by the damage step
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_can_block_evasion_and_reach():
    # can_block: evasion + reach. A real flier (Kitchen Imp) and
    # Silhana's "can't be blocked except by flying" both demand a flying or
    # reach blocker; reach (Bramble Wurm) satisfies that, a vanilla creature
    # doesn't, and -- crucially -- Silhana itself (evasion, NOT real flying)
    # can NOT block a flier.
    kb_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    imp = Permanent(CardDef("Imp", CardType.CREATURE, None, EffectId.KITCHEN_IMP, power=2, toughness=2))
    silhana = Permanent(CardDef("Silhana", CardType.CREATURE, None, EffectId.SILHANA_LEDGEWALKER, power=1, toughness=1))
    wurm = Permanent(CardDef("Wurm", CardType.CREATURE, None, EffectId.BRAMBLE_WURM, power=7, toughness=6))
    vanilla_c = Permanent(CardDef("Vanilla", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    assert not can_block(kb_state, vanilla_c, imp)      # vanilla can't block a flier
    assert can_block(kb_state, wurm, imp)               # reach blocks a flier
    assert can_block(kb_state, imp, imp)                # flying blocks flying
    assert not can_block(kb_state, vanilla_c, silhana)  # Silhana's evasion needs a flying/reach blocker
    assert can_block(kb_state, wurm, silhana)           # reach satisfies Silhana's evasion too
    assert not can_block(kb_state, silhana, imp)        # Silhana is NOT flying -> can't block a flier
    assert can_block(kb_state, vanilla_c, vanilla_c)    # no restriction -> anyone blocks


def test_menace_block_incomplete_and_enforce_menace():
    # Menace (509.1c): a declaration leaving a menace attacker with exactly ONE
    # blocker is illegal. menace_block_incomplete flags it (so drl_env forbids
    # "Done" until fixed -- 0 or 2+); enforce_menace is the cap-abandon backstop
    # that drops a stray lone block. Player 0 attacks, player 1 (the defender,
    # active during blocking) assigns blockers.
    _fb = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"menace"}}
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        menacer = Permanent(CardDef("Menacer", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=1))
        menacer.summoning_sick = False
        state.players[0].battlefield = [menacer]
        state.players[0].attackers = [menacer]
        lone = Permanent(CardDef("Lone", CardType.CREATURE, None, None, power=1, toughness=1))
        state.active_idx = 1  # the DEFENDER is active during declare-blockers

        state.players[0].blocked_by = {menacer: [lone]}  # exactly one -> incomplete/illegal
        assert menace_block_incomplete(state)  # "Done" would be forbidden here
        b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=1))
        state.players[0].blocked_by = {menacer: [lone, b2]}  # two -> a legal block
        assert not menace_block_incomplete(state)
        state.players[0].blocked_by = {}  # zero -> also legal (unblocked)
        assert not menace_block_incomplete(state)

        # Backstop: enforce_menace (active back on the attacker) drops a stray
        # lone menace-block, keeping a two-block one.
        state.active_idx = 0
        state.players[0].blocked_by = {menacer: [lone]}
        enforce_menace(state)
        assert menacer not in state.players[0].blocked_by  # unblocked
        state.players[0].blocked_by = {menacer: [lone, b2]}
        enforce_menace(state)
        assert state.players[0].blocked_by == {menacer: [lone, b2]}
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _fb


def test_goad_forces_declaration_and_excludes_non_turn_player():
    # Goad: a goaded creature that can attack blocks its controller's Pass
    # (has_unfulfilled_goad) until it's declared.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    goaded = Permanent(CardDef("Goaded", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    goaded.summoning_sick = False
    goaded.flags["goaded_by"] = 1
    state.players[0].battlefield = [goaded]
    declare_attackers_step(state)
    assert has_unfulfilled_goad(state)  # able + undeclared -> must attack
    declare_attacker(state, goaded)
    assert not has_unfulfilled_goad(state)  # now declared -> satisfied
    # a goaded creature that CAN'T attack (tapped) never forces the issue
    state.players[0].attackers = []
    goaded.tapped = True
    assert not has_unfulfilled_goad(state)

    # Goad binds the turn player during THEIR own declare step, not a NON-turn
    # player who merely holds priority during DECLARE_ATTACKERS
    # (game.turn._run_priority_round_gen flips active_idx to them). A forcing
    # goaded creature under the non-turn player must NOT block their priority-
    # Pass -- they cannot declare an attacker at all (_attack_legal needs
    # active_idx == turn_player_idx), so blocking it would leave an all-False
    # action mask (the rl.agent._seat_step crash this guards against).
    # turn_player_idx stays 0.
    nonturn_goaded = Permanent(CardDef("NonturnGoaded", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    nonturn_goaded.summoning_sick = False
    nonturn_goaded.flags["goaded_by"] = 0  # goaded by the turn player (idx 0)
    state.players[1].battlefield = [nonturn_goaded]
    state.active_idx = 1  # non-turn player now holds priority
    assert not has_unfulfilled_goad(state), "goad must never block a NON-turn player's priority-Pass"


def test_initiative_transfer_on_combat_damage():
    # Initiative transfer: the holder taking combat damage queues the
    # attacker's own "take_initiative" triggered ability (CR 722.2) -- a real
    # trigger, not an instant effect, so it doesn't flip state.initiative_idx
    # until IT resolves off the stack, which then queues the venture.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.initiative_idx = 1  # the DEFENDER holds it
    hitter = Permanent(CardDef("Hitter", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    hitter.summoning_sick = False
    state.players[0].battlefield = [hitter]
    state.players[0].attackers = [hitter]
    combat_damage_step(state)
    assert state.players[1].life_total == 17  # 3 unblocked to the defender
    assert state.initiative_idx == 1  # not yet -- the trigger hasn't resolved
    assert any(e["type"] == "take_initiative" for e in state.players[0].trigger_queue)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # the take_initiative trigger itself
    assert state.initiative_idx == 0  # NOW stolen by the attacker
    assert any(e["type"] == "venture" for e in state.players[0].trigger_queue)


def test_initiative_transfer_not_masked_by_simultaneous_lifelink_gain():
    # A lifelink blocker's life GAIN for the defender (routed to the
    # defending player -- see test_lifelink_on_blocking_creature_credits_
    # defender above) must not mask combat damage a separate unblocked
    # attacker actually dealt to the initiative holder that same step. The
    # transfer is keyed on damage dealt, not net life-total change (real
    # Magic: "whenever one or more creatures a player controls deal combat
    # damage to the player who has the initiative...").
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.initiative_idx = 1  # the DEFENDER holds it
    hitter = Permanent(CardDef("Hitter", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    hitter.summoning_sick = False
    weakling = Permanent(CardDef("Weakling", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    weakling.summoning_sick = False
    state.players[0].battlefield = [hitter, weakling]
    state.players[0].attackers = [hitter, weakling]
    blocker = Permanent(CardDef("BlockerLifelinker", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    blocker.counters["lifelink"] = 1
    state.players[1].battlefield = [blocker]
    state.players[0].blocked_by = {weakling: [blocker]}

    combat_damage_step(state)

    # Net life change is positive (+5 lifelink, -3 unblocked) -- a net-life
    # check would wrongly conclude no combat damage reached the defender.
    assert state.players[1].life_total == 22  # 20 - 3 (Hitter, unblocked) + 5 (blocker's lifelink)
    assert any(e["type"] == "take_initiative" for e in state.players[0].trigger_queue)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # the take_initiative trigger itself
    assert state.initiative_idx == 0  # still stolen -- Hitter DID deal combat damage to the holder
    assert any(e["type"] == "venture" for e in state.players[0].trigger_queue)


def test_free_damage_assignment_skips_blockers_that_already_left():
    """Rule 510.1a: an attacker assigns combat damage only among creatures
    CURRENTLY blocking it.

    combat_damage_step's auto path already handled dead blockers (see
    test_trample_all_through_when_every_blocker_already_gone). The FREE
    assignment path -- 2+ blockers, where the attacker's controller chooses the
    split -- did not: it captured state.blocked_by at declaration and offered
    the whole list, including any blocker killed by removal in the priority
    round that game.turn gives both players right after blocks.

    That was a hard crash, not a rules nicety. A dead blocker is not in
    build_token_set (which walks the battlefield), so rl.action_bridge's
    pointer mask -- an identity match against the pending's blocker list --
    could not address it, and with no trample option either the whole action
    mask came back all-False. Hit on turn 79 of an 11-deck league game,
    2026-08-16.
    """
    from game.turn import _assign_combat_damage_gen

    _card_defs_backup = dict(registry.CARD_DEFS)
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker = Permanent(CardDef("Ganged2", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=4))
        attacker.summoning_sick = False
        registry.CARD_DEFS["Ganged2"] = attacker.card_def
        state.players[0].battlefield = [attacker]

        def blocker(name):
            return Permanent(CardDef(name, CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))

        alive, dead = blocker("Alive"), blocker("Dead")
        state.players[1].battlefield = [alive]  # `dead` deliberately on NO battlefield

        declare_attackers_step(state)
        declare_attacker(state, attacker)
        state.blocked_by[attacker] = [alive, dead]

        # Two blockers declared, but only one still alive -> no free choice
        # remains, so the generator must finish WITHOUT asking for a decision.
        # It previously opened a pending whose only addressable option was gone.
        assert list(_assign_combat_damage_gen(state)) == []
        assert state.pending_resolution is None

        # And with two survivors out of three, the decision IS offered -- over
        # the living pair only, never the corpse.
        alive2 = blocker("Alive2")
        state.players[1].battlefield = [alive, alive2]
        state.blocked_by[attacker] = [alive, dead, alive2]
        gen = _assign_combat_damage_gen(state)
        assert next(gen) is None, "should pause for the controller's assignment"
        assert state.pending_resolution["kind"] == "assign_combat_damage"
        assert state.pending_resolution["blockers"] == [alive, alive2]
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_blocked_attacker_without_trample_deals_nothing_when_its_blocker_is_gone():
    # 509.1h: a creature that was blocked REMAINS blocked even if every
    # creature blocking it leaves combat. Without trample it therefore deals
    # no combat damage at all -- it does NOT "become unblocked" and hit the
    # player. The trample counterpart of this is
    # test_trample_all_through_when_every_blocker_already_gone above; this is
    # the case that must NOT spill through.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Grounded", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
    attacker.summoning_sick = False
    gone_blocker = Permanent(CardDef("Gone Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    state.players[0].battlefield = [attacker]  # blocker deliberately on no battlefield -- already dead

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [gone_blocker]
    combat_damage_step(state)

    assert state.players[1].life_total == 20, "a blocked non-trampler must not hit the player when its blocker dies"
    assert attacker.damage_marked == 0, "a dead blocker deals no damage back"


def test_attacker_removed_after_blocks_deals_and_takes_no_damage():
    # 506.4: an attacker removed from the battlefield after blockers were
    # declared stops being an attacking creature -- it deals no combat damage
    # to its blocker or the player, and its blocker deals none back to it.
    # remove_from_combat prunes state.attackers but deliberately leaves the
    # blocked_by ENTRY (509.1: its blockers stay declared and must not be
    # freed to block again), so combat_damage_step -- which iterates
    # blocked_by.items() -- has to skip the group on its own.
    from game.effects.combat import remove_from_combat

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Doomed", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=3))
    attacker.summoning_sick = False
    blocker = Permanent(CardDef("Blocker", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [blocker]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [blocker]

    # The removal spell resolves after blocks: the attacker leaves.
    state.players[0].battlefield.remove(attacker)
    remove_from_combat(state, attacker)
    assert attacker not in state.attackers

    combat_damage_step(state)
    assert state.players[1].life_total == 20, "a removed attacker deals no damage to the player"
    assert blocker.damage_marked == 0, "a removed attacker deals no damage to its blocker"
    assert attacker.damage_marked == 0, "a blocker deals no damage to an attacker that left combat"


def test_multi_blocker_damage_splits_over_living_blockers_only():
    # Gang block, then one blocker dies before combat damage. The attacker's
    # damage is assigned across the LIVING blockers only -- the dead one
    # absorbs nothing and deals nothing back.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Big", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=6))
    attacker.summoning_sick = False
    dead_blocker = Permanent(CardDef("Dead Blocker", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=1))
    live_blocker = Permanent(CardDef("Live Blocker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=9))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [live_blocker]  # dead_blocker deliberately absent

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [dead_blocker, live_blocker]
    combat_damage_step(state)

    assert state.players[1].life_total == 20, "a blocked non-trampler never reaches the player"
    assert live_blocker.damage_marked == 4, "all 4 power goes to the one living blocker"
    assert dead_blocker.damage_marked == 0
    assert attacker.damage_marked == 1, "only the living blocker deals damage back"


def test_menace_attacker_legally_blocked_stays_blocked_when_one_blocker_dies():
    # 509.1h vs 509.1c. A menace attacker legally blocked by TWO creatures,
    # one of which then dies in the post-declare-blockers priority window,
    # REMAINS blocked -- it does not revert to unblocked and hit the player.
    # This works because game.turn calls enforce_menace once, right after the
    # declaration is finalized and BEFORE that priority window (turn.py's own
    # ordering comment); enforce_menace's lone-blocker drop is only ever the
    # abandoned-declaration backstop, never a re-check after blocks are legal.
    _fb = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"menace"}}
    try:
        from game.effects.combat import remove_from_combat

        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        menacer = Permanent(CardDef("Menacer", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=5))
        menacer.summoning_sick = False
        b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=1))
        b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=1))
        state.players[0].battlefield = [menacer]
        state.players[1].battlefield = [b1, b2]

        declare_attackers_step(state)
        declare_attacker(state, menacer)
        state.blocked_by[menacer] = [b1, b2]
        enforce_menace(state)  # runs here, while the block is still legal
        assert state.blocked_by[menacer] == [b1, b2]

        # Now b1 dies, during the priority window that follows.
        state.players[1].battlefield.remove(b1)
        remove_from_combat(state, b1)
        assert state.blocked_by[menacer] == [b2], "506.4: the dead blocker stops being a blocking creature"

        combat_damage_step(state)
        assert state.players[1].life_total == 20, "509.1h: still blocked -- it must not hit the player"
        assert b2.damage_marked == 4
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _fb


def test_sacrificed_attacker_is_removed_from_combat():
    # 506.4 via the sacrifice exit path (sacrifice_to_graveyard), the choke
    # point every "Sacrifice this" ability and artifact-sac cost routes
    # through -- distinct from the SBA-death path below.
    from game.effects.state_based import sacrifice_to_graveyard

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Fling Fodder", CardType.CREATURE, None, EffectId.FILLER, power=6, toughness=1))
    attacker.summoning_sick = False
    state.players[0].battlefield = [attacker]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    assert attacker in state.attackers

    sacrifice_to_graveyard(state, attacker)
    assert attacker not in state.attackers, "506.4: a sacrificed attacker leaves combat"

    combat_damage_step(state)
    assert state.players[1].life_total == 20, "a sacrificed attacker deals no combat damage"


def test_attacker_killed_by_state_based_actions_is_removed_from_combat():
    # 506.4 via the SBA-death exit path (check_state_based_actions ->
    # _destroy_creature) -- the most common way an attacker leaves combat, and
    # the one the second infinite declare-blockers loop was reached through.
    from game.effects.state_based import check_state_based_actions

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Shocked", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=2))
    attacker.summoning_sick = False
    survivor = Permanent(CardDef("Survivor", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    survivor.summoning_sick = False
    state.players[0].battlefield = [attacker, survivor]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    declare_attacker(state, survivor)

    attacker.damage_marked = 2  # a burn spell resolved -- lethal
    check_state_based_actions(state)
    assert attacker not in state.players[0].battlefield
    assert attacker not in state.attackers, "506.4: an attacker that dies leaves combat"
    assert survivor in state.attackers, "the survivor is untouched"

    combat_damage_step(state)
    assert state.players[1].life_total == 18, "only the survivor's 2 power connects"
