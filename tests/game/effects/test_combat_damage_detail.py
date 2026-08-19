"""Combat-damage detail: deathtouch through combat, the attacking player's own
free damage split, Aura-modified lethal calculation, and the first-strike
sub-step in the direction test_combat.py does not already cover.

The broad combat cases (attack eligibility, trample spill, lifelink, menace,
goad, The Initiative, gang-block splits) live in test_combat.py; this file is
the Phase 3 gap fill from the 2026-08-19 combat review.
"""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.combat import combat_damage_step, declare_attacker, declare_attackers_step
from game.effects.state_based import check_state_based_actions
from game.state import GameState, Permanent, PlayerState


def _with_filler_keywords(keywords):
    """Swap EffectId.FILLER's registry entry for one granting `keywords`,
    returning the original so the caller can restore it. Same trick
    test_combat.py's menace test uses."""
    original = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": set(keywords)}
    return original


def test_deathtouch_attacker_kills_a_larger_blocker_through_combat_damage_step():
    # End-to-end deathtouch through combat: _attacker_deal_damage stamps
    # flags["deathtouched"] on whatever it damages, and the SBA check then
    # treats that marked damage as lethal regardless of toughness. The SBA half
    # is covered by test_state_based's own deathtouch test; this is the combat
    # half that feeds it.
    original = _with_filler_keywords({"deathtouch"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        assassin = Permanent(CardDef("Assassin", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        assassin.summoning_sick = False
        giant = Permanent(CardDef("Giant", CardType.CREATURE, None, None, power=1, toughness=6))
        state.players[0].battlefield = [assassin]
        state.players[1].battlefield = [giant]

        declare_attackers_step(state)
        declare_attacker(state, assassin)
        state.blocked_by[assassin] = [giant]
        combat_damage_step(state)

        assert giant.damage_marked == 1
        assert giant.flags.get("deathtouched"), "a deathtouch combat hit must be stamped for the SBA"
        check_state_based_actions(state)
        assert giant not in state.players[1].battlefield, "1 damage from deathtouch is lethal to a 1/6"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_deathtouch_trample_default_split_assigns_full_toughness_not_one():
    # OPEN RULES QUESTION. This pins CURRENT behavior so it cannot drift
    # silently; it is deliberately NOT a claim about what real Magic requires.
    #
    # 702.2b + 510.1a: when ASSIGNING combat damage, any nonzero damage from a
    # deathtouch source counts as lethal. So a 5-power deathtouch trampler
    # blocked by one 4-toughness creature may legally assign 1 and trample 4.
    # _default_damage_assignment computes lethal as (toughness - damage_marked)
    # and is deathtouch-blind, so it assigns 4 and tramples 1.
    #
    # Over-assigning to a blocker is itself LEGAL in real Magic, so this is not
    # an illegal-state bug. The substantive issue is that a SINGLE-blocked
    # attacker never gets an assign_combat_damage decision at all
    # (attackers_needing_damage_assignment fires only for 2+ blockers), so the
    # attacking player is denied a choice real Magic gives them. Fixing that
    # widens the action space, so it is raised for the owner, not changed here.
    original = _with_filler_keywords({"deathtouch", "trample"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        killer = Permanent(CardDef("Killer", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
        killer.summoning_sick = False
        wall = Permanent(CardDef("Wall", CardType.CREATURE, None, None, power=0, toughness=4))
        state.players[0].battlefield = [killer]
        state.players[1].battlefield = [wall]

        declare_attackers_step(state)
        declare_attacker(state, killer)
        state.blocked_by[killer] = [wall]
        combat_damage_step(state)

        assert wall.damage_marked == 4, "current: lethal from toughness, deathtouch ignored in assignment"
        assert state.players[1].life_total == 19, "current: only the leftover 1 tramples over"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_agent_chosen_multi_blocker_split_is_honored():
    # A 2+-blocked attacker's controller freely assigns its damage
    # (begin_assign_combat_damage stashes the choice on
    # flags["combat_damage_split"]). combat_damage_step must consume that split
    # verbatim rather than recomputing the lethal-in-order default -- including
    # a deliberately lopsided, non-lethal assignment.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Chooser", CardType.CREATURE, None, EffectId.FILLER, power=6, toughness=6))
    attacker.summoning_sick = False
    b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=3))
    b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=3))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [b1, b2]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [b1, b2]
    # All 6 onto b1, none to b2 -- allowed by the engine's free-assignment
    # spec, and NOT what the lethal-in-order default (3/3) would produce.
    attacker.flags["combat_damage_split"] = ({b1: 6, b2: 0}, 0)
    combat_damage_step(state)

    assert b1.damage_marked == 6 and b2.damage_marked == 0, "the recorded split must beat the default"
    assert attacker.damage_marked == 2, "both blockers still deal their power back"
    assert state.players[1].life_total == 20, "no trample -- nothing reaches the player"


def test_lethal_calculation_uses_aura_modified_toughness():
    # The default lethal-in-order split must read EFFECTIVE toughness, not the
    # printed value: an Armadillo Cloak (+2/+2) on the blocker raises the bar,
    # so a trampler spills less over.
    original = _with_filler_keywords({"trample"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Stomper", CardType.CREATURE, None, EffectId.FILLER, power=6, toughness=6))
        trampler.summoning_sick = False
        blocker = Permanent(CardDef("Cloaked", CardType.CREATURE, None, None, power=1, toughness=2))
        cloak = Permanent(CardDef("Armadillo Cloak", CardType.ENCHANTMENT, {"G": 1}, EffectId.ARMADILLO_CLOAK))
        cloak.flags["enchanting"] = blocker
        state.players[0].battlefield = [trampler]
        state.players[1].battlefield = [blocker, cloak]

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = [blocker]
        combat_damage_step(state)

        # Blocker is effectively 3/4 under the Cloak: 4 lethal, 2 tramples over.
        assert blocker.damage_marked == 4, "lethal must use Aura-modified toughness, not the printed 2"
        # The Cloak also grants its own lifelink, and it is on the DEFENDER's
        # creature, so the defender gains the 3 their boosted blocker dealt:
        # 20 - 2 trample + 3 lifelink. Asserted together deliberately -- the
        # +2/+2 and the lifelink come from one Aura and must both apply.
        assert trampler.damage_marked == 3, "the Cloak's +2/+0 applies to the damage dealt back"
        assert state.players[1].life_total == 21, "20 - 2 trample excess + 3 Cloak lifelink"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_first_strike_blocker_kills_attacker_before_it_deals_damage():
    # The untested direction of the first-strike sub-step. test_combat.py
    # covers a first-striking ATTACKER; this is a first-striking BLOCKER that
    # kills the attacker outright, so the SBA check between sub-steps removes
    # it before it ever deals its own regular combat damage.
    original = _with_filler_keywords({"first_strike"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        attacker = Permanent(CardDef("Fragile", CardType.CREATURE, None, None, power=5, toughness=2))
        attacker.summoning_sick = False
        fs_blocker = Permanent(CardDef("Quickdraw", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
        state.players[0].battlefield = [attacker]
        state.players[1].battlefield = [fs_blocker]

        declare_attackers_step(state)
        declare_attacker(state, attacker)
        state.blocked_by[attacker] = [fs_blocker]
        combat_damage_step(state)

        assert attacker not in state.players[0].battlefield, "killed in the first-strike sub-step"
        assert fs_blocker.damage_marked == 0, "a dead attacker never deals its regular combat damage"
        assert state.players[1].life_total == 20
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original
