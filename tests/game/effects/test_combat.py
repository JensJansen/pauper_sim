"""Attack eligibility + declaration + damage, then the keyword trio
(vigilance/trample/first strike) -- everything specific to THIS module. The
combat+SBA creature-death handoff lives in
tests/game/effects/test_integration_check.py instead (it exercises
state_based.py just as much as this module)."""
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
from game.effects.tokens import WARRIOR_TOKEN_CARD_DEF
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


def test_haste_registry_spec():
    # Haste (Kitchen Imp): a flat "haste": True registry spec lets a
    # summoning-sick creature be attack-eligible anyway, via stats.has_haste
    # (the one canonical haste check creature_attack_eligible and mana.py's
    # tap_summoning_locked both share).
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
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
        assert state.players[1].life_total == 18 and hasty.tapped  # 20 - hasty's power (2)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_haste_intrinsic_keyword_spec_regression():
    # Regression (2026-08): creature_attack_eligible used to check ONLY the
    # flat "haste": True boolean, missing haste granted via the "keywords"
    # set (Reckless Lackey, Clockwork Percussionist both use this form) --
    # such a creature could never attack the turn it was cast. Confirms
    # stats.has_haste's other branch (creature_keywords) now gates
    # attack-eligibility too, not just mana.py's tap_summoning_locked.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"haste"}}
    try:
        hasty = Permanent(CardDef("Hasty Keyword", CardType.CREATURE, None, EffectId.FILLER, power=2))
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
    # Initiative transfer: the holder taking combat damage passes it to the
    # attacker, who then has a venture queued.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = state.turn_player_idx = 0
    state.initiative_idx = 1  # the DEFENDER holds it
    hitter = Permanent(CardDef("Hitter", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    hitter.summoning_sick = False
    state.players[0].battlefield = [hitter]
    state.players[0].attackers = [hitter]
    combat_damage_step(state)
    assert state.players[1].life_total == 17  # 3 unblocked to the defender
    assert state.initiative_idx == 0  # stolen by the attacker
    assert any(e["type"] == "venture" for e in state.players[0].trigger_queue)
