"""Attack eligibility, declaration, damage, and the keyword trio
(vigilance/trample/first strike). The combat+SBA creature-death handoff
lives in test_integration_check.py instead."""
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
    sick = Permanent(CardDef("Sick", CardType.CREATURE, None, EffectId.FILLER, power=10))  # summoning_sick=True by construction
    already_tapped = Permanent(CardDef("Tapped Out", CardType.CREATURE, None, EffectId.FILLER, power=10), tapped=True)
    already_tapped.summoning_sick = False
    vanilla = Permanent(CardDef("No Stats", CardType.CREATURE, None, EffectId.FILLER))  # no "power" key at all
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
    # both registry spec forms -- flat "haste": True and haste via the
    # "keywords" set -- must let a summoning-sick creature attack anyway
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
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    vigilant = Permanent(WARRIOR_TOKEN_CARD_DEF)  # WARRIOR_TOKEN's registry entry grants vigilance
    vigilant.summoning_sick = False
    ordinary = Permanent(CardDef("Ordinary Attacker", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ordinary.summoning_sick = False
    state.battlefield = [vigilant, ordinary]
    declare_attackers_step(state)
    declare_attacker(state, vigilant)
    declare_attacker(state, ordinary)
    assert not vigilant.tapped and vigilant in state.attackers
    assert ordinary.tapped and ordinary in state.attackers

    # staying untapped must not make vigilant re-declarable -- it already
    # attacked this combat; creature_attack_eligible's state.attackers guard
    # prevents duplicate declare_attacker calls from doubling its power
    assert not creature_attack_eligible(state, vigilant)
    assert state.attackers.count(vigilant) == 1
    combat_damage_step(state)
    assert state.players[1].life_total == 18  # 20 - (vigilant's power once + ordinary's power once) -- not doubled


def test_trample_spills_excess_to_defending_player():
    # a blocked trampler assigns only enough damage to be lethal to its
    # blocker; the rest spills over to the defending player
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

        # effective power 7 (5 base + Rancor +2): 2 lethal to weak_blocker, 5 tramples through
        assert weak_blocker not in state.players[1].battlefield
        assert state.players[1].life_total == 15  # 20 - the 5 that trampled through
        assert trampler in state.players[0].battlefield and trampler.damage_marked == 1
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_trample_all_through_when_every_blocker_already_gone():
    # 702.19e/510.1c: a blocked trampler whose only blocker died before
    # combat_damage_step ran is still "blocked", but with no living blocker
    # to assign lethal damage to, its full power tramples through.
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
        # gone_blocker deliberately not on any battlefield -- already dead by combat damage

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = [gone_blocker]
        combat_damage_step(state)

        # effective power 7 (5 base + Rancor +2), all of it through
        assert state.players[1].life_total == 13  # 20 - 7
        assert trampler.damage_marked == 0  # a dead blocker deals no damage back
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_first_strike_kills_before_blocker_deals_damage():
    # a blocked first-strike attacker deals its damage before the blocker gets a chance to
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

        # effective power 5 (4 base + Cartouche +1) >= toughness 3 -- dies in the
        # first-strike sub-step, before dealing its own damage back
        assert lethal_blocker not in state.players[1].battlefield
        assert fs_attacker in state.players[0].battlefield and fs_attacker.damage_marked == 0
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_unblocked_attacker():
    # lifelink credits whichever side actually controls the damage-dealing creature
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

        # effective power 5 (3 base + Cloak +2) -- both damage and lifelink gain use this total
        assert state.players[1].life_total == 15  # 20 - the unblocked lifelinker's power (5)
        assert state.players[0].life_total == 25  # STARTING_LIFE (20) + 5, unblocked lifelink
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_stacking_two_cloaks():
    # two Armadillo Cloaks on the same creature: two independent triggers,
    # each for the full damage dealt, not a boolean that dedups to one
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

        # effective power 7 (3 base + 2+2); life gained is 7*2=14, one trigger per Cloak
        assert state.players[1].life_total == 13  # 20 - the double-cloaked lifelinker's power (7)
        assert state.players[0].life_total == 34  # STARTING_LIFE (20) + 14
    finally:
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup)


def test_lifelink_blocked_trample_gains_full_effective_power():
    # a blocked lifelink trampler gains the FULL effective power as life,
    # not just the excess that tramples through (no such carve-out on Cloak)
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
    # a blocking lifelinker's life goes to the defending player (idx 1),
    # never state.life_total, which would credit the attacker instead
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
    # a 3/3 attacker gang-blocked by two 2/2s: controller assigns 2 (lethal)
    # to the first, 1 to the second (survives); both deal 2 back so the
    # attacker takes 4 and dies. The split lives on the attacker's flags as
    # resolution.begin_assign_combat_damage records it, consumed (popped)
    # by the damage step.
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
    # a real flier and Silhana's evasion both demand a flying/reach blocker;
    # Silhana itself (evasion, not real flying) cannot block a flier
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
    # 509.1c: a declaration leaving a menace attacker with exactly one
    # blocker is illegal. menace_block_incomplete flags it (0 or 2+ only);
    # enforce_menace is the backstop that drops a stray lone block.
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

        # enforce_menace drops a stray lone menace-block, keeps a two-block one
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
    # a goaded creature that can attack blocks its controller's Pass
    # (has_unfulfilled_goad) until it's declared
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
    # a goaded creature that can't attack (tapped) never forces the issue
    state.players[0].attackers = []
    goaded.tapped = True
    assert not has_unfulfilled_goad(state)

    # goad binds the turn player during their own declare step; a non-turn
    # player merely holding priority during DECLARE_ATTACKERS cannot declare
    # an attacker at all (_attack_legal needs active_idx == turn_player_idx),
    # so goad must not block their Pass either. turn_player_idx stays 0.
    nonturn_goaded = Permanent(CardDef("NonturnGoaded", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    nonturn_goaded.summoning_sick = False
    nonturn_goaded.flags["goaded_by"] = 0  # goaded by the turn player (idx 0)
    state.players[1].battlefield = [nonturn_goaded]
    state.active_idx = 1  # non-turn player now holds priority
    assert not has_unfulfilled_goad(state), "goad must never block a NON-turn player's priority-Pass"


def test_initiative_transfer_on_combat_damage():
    # dealing combat damage to the initiative holder queues a
    # "take_initiative" trigger (CR 722.2), which only flips
    # state.initiative_idx once it resolves off the stack, then queues venture
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
    # a lifelink blocker's life gain for the defender must not mask combat
    # damage a separate unblocked attacker dealt that same step -- the
    # transfer is keyed on damage dealt, not net life-total change
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

    # net life change is positive (+5 lifelink, -3 unblocked) -- a net-life
    # check would wrongly conclude no combat damage reached the defender
    assert state.players[1].life_total == 22  # 20 - 3 (Hitter, unblocked) + 5 (blocker's lifelink)
    assert any(e["type"] == "take_initiative" for e in state.players[0].trigger_queue)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)  # the take_initiative trigger itself
    assert state.initiative_idx == 0  # still stolen -- Hitter DID deal combat damage to the holder
    assert any(e["type"] == "venture" for e in state.players[0].trigger_queue)


def test_free_damage_assignment_skips_blockers_that_already_left():
    """510.1a: an attacker assigns combat damage only among creatures
    currently blocking it. The free-assignment path (2+ blockers, controller
    chooses the split) must exclude any blocker killed by removal during the
    priority window after blocks, not just the auto-assignment path covered
    by test_trample_all_through_when_every_blocker_already_gone."""
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

        # two blockers declared, only one alive -> no free choice remains,
        # generator must finish without asking for a decision
        assert list(_assign_combat_damage_gen(state)) == []
        assert state.pending_resolution is None

        # two survivors out of three: the decision is offered over the living pair only
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
    # 509.1h: a blocked creature remains blocked even if every blocker
    # leaves combat; without trample it deals no damage at all (never
    # "becomes unblocked"). Trample counterpart: test_trample_all_through_when_every_blocker_already_gone.
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
    # 506.4: an attacker removed after blockers were declared stops being an
    # attacking creature -- no damage either direction. remove_from_combat
    # prunes state.attackers but leaves the blocked_by entry (509.1: its
    # blockers stay declared), so combat_damage_step must skip that group itself.
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
    # a gang-blocker that dies before combat damage absorbs nothing and deals nothing back
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
    # 509.1h vs 509.1c: a menace attacker legally blocked by two creatures
    # remains blocked if one later dies -- it never reverts to unblocked.
    # enforce_menace runs once, right after declaration and before that
    # priority window, so it never re-checks after blocks are already legal.
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
    # 506.4 via the sacrifice exit path (sacrifice_to_graveyard), distinct from the SBA-death path below
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
    # 506.4 via the SBA-death exit path (check_state_based_actions -> _destroy_creature)
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
