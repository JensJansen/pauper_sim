"""Combat-damage detail: deathtouch through combat, the attacking player's own
free damage split, Aura-modified lethal calculation, and the first-strike
sub-step in the direction test_combat.py does not already cover.

The broad combat cases (attack eligibility, trample spill, lifelink, menace,
goad, The Initiative, gang-block splits) live in test_combat.py; this file is
the Phase 3 gap fill from the 2026-08-19 combat review.
"""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.combat import blocker_lethal_capacities, combat_damage_step, declare_attacker, declare_attackers_step
from game.effects.state_based import check_state_based_actions
from game.resolution import assign_combat_damage_options, begin_assign_combat_damage, execute_assign_combat_damage_option
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


def test_deathtouch_trample_default_split_assigns_one_per_blocker():
    # 702.2b + 510.1a: when ASSIGNING combat damage, any nonzero damage from a
    # deathtouch source counts as lethal, so a 5-power deathtouch trampler
    # blocked by one 4-toughness creature assigns 1 and tramples 4.
    #
    # _default_damage_assignment used to compute lethal as (toughness -
    # damage_marked) with no deathtouch case, assigning 4 and spilling only 1.
    # That was a LEGAL assignment -- over-assigning to a blocker is allowed --
    # so it was not an illegal-state bug. It was the wrong DEFAULT: a
    # single-blocked attacker never gets an assign_combat_damage decision
    # (attackers_needing_damage_assignment fires only for 2+ blockers), so the
    # auto split is the entire decision and should be the attacker-optimal
    # legal one. Minimum-lethal is dominant -- over-assigning to a blocker
    # never helps the attacker -- so automating it needs no new action.
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

        assert wall.damage_marked == 1, "deathtouch makes 1 damage lethal for assignment purposes"
        assert state.players[1].life_total == 16, "the other 4 trample over (20 - 4)"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_multi_blocker_split_caps_each_blocker_at_lethal_and_spares_the_rest_only_if_power_runs_out():
    # A 2+-blocked attacker's controller assigns its damage one point at a
    # time (begin_assign_combat_damage/execute_assign_combat_damage_option),
    # but never past a blocker's own lethal cap (blocker_lethal_capacities)
    # -- no overkill. Once a blocker is capped it drops out of assign_
    # combat_damage_options entirely. The only real discretion left is which
    # blocker(s) get lethal first when total power can't cover every
    # blocker's cap -- real Magic's own damage-assignment-order rule
    # (510.1c) -- so this is the one case a blocker can still end up
    # non-lethally damaged: power ran out, not because the controller chose
    # to leave a capped-eligible blocker alone.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Chooser", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=6))
    attacker.summoning_sick = False
    b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=3))
    b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=3))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [b1, b2]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [b1, b2]

    lethal_by_blocker = blocker_lethal_capacities(state, attacker, [b1, b2])
    assert lethal_by_blocker == {b1: 3, b2: 3}
    begin_assign_combat_damage(
        state, attacker, [b1, b2], power=4, has_trample=False,
        lethal_by_blocker=lethal_by_blocker, on_complete=lambda s: None,
    )

    for _ in range(3):  # fill b1 to its own cap
        execute_assign_combat_damage_option(state, "B1", b1.slot)
    assert ("B1", b1.slot) not in assign_combat_damage_options(state), "capped -- no more overkill onto b1"
    assert state.pending_resolution is not None, "1 point still remains and b2 is still open"

    execute_assign_combat_damage_option(state, "B2", b2.slot)  # the 4th and last point -- only b2 is legal
    assert state.pending_resolution is None, "remaining hit 0 -- resolves without forcing b2 to its own cap"

    combat_damage_step(state)
    assert b1.damage_marked == 3 and b1 not in state.players[1].battlefield
    assert b2.damage_marked == 1 and b2 in state.players[1].battlefield, "spared -- power simply ran out first"
    assert state.players[1].life_total == 20, "no trample -- nothing reaches the player"


def test_multi_blocker_trample_spills_to_player_automatically_once_every_blocker_is_capped():
    # 702.19e/510.1c: once every blocker in a multi-blocked trampler's split
    # is at its own lethal cap, whatever combat damage remains is a FORCED
    # outcome, never an agent choice -- there is no "assign to player"
    # action anymore (drl_env._assign_damage_to_opponent_legal is now
    # permanently False). The spillover happens as a direct side effect of
    # the point that caps the LAST still-open blocker.
    original = _with_filler_keywords({"trample"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Stomper2", CardType.CREATURE, None, EffectId.FILLER, power=8, toughness=8))
        trampler.summoning_sick = False
        b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=3))
        b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=3))
        state.players[0].battlefield = [trampler]
        state.players[1].battlefield = [b1, b2]

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = [b1, b2]

        lethal_by_blocker = blocker_lethal_capacities(state, trampler, [b1, b2])
        begin_assign_combat_damage(
            state, trampler, [b1, b2], power=8, has_trample=True,
            lethal_by_blocker=lethal_by_blocker, on_complete=lambda s: None,
        )

        for _ in range(3):
            execute_assign_combat_damage_option(state, "B1", b1.slot)
        assert state.pending_resolution is not None, "b2 still open -- 5 points left, nothing forced yet"
        for _ in range(2):
            execute_assign_combat_damage_option(state, "B2", b2.slot)
        assert state.pending_resolution is not None, "b2 not capped yet"
        execute_assign_combat_damage_option(state, "B2", b2.slot)  # caps b2 -- the LAST open blocker

        assert state.pending_resolution is None, "both blockers capped -- the remaining 2 must auto-spill"
        assert trampler.flags["combat_damage_split"] == ({b1: 3, b2: 3}, 2)

        combat_damage_step(state)
        assert state.players[1].life_total == 18, "20 - the 2 that spilled over with no explicit action taken"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_multi_blocker_non_trample_excess_piles_onto_last_blocker_when_all_are_capped():
    # RULES EXCEPTION (owner-approved 2026-08-20): without trample, real
    # Magic still requires leftover combat damage be assigned to a blocking
    # creature even once every blocker is already at its own lethal cap --
    # it just does nothing further (that blocker is already dead). Rather
    # than model exactly which blocker "in reality" absorbs it, this engine
    # piles all of it onto the last blocker in the split. Inert: nothing in
    # this card pool reads how much *excess* damage an already-dead blocker
    # received, and the defending player takes none of it either way.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Chooser2", CardType.CREATURE, None, EffectId.FILLER, power=8, toughness=8))
    attacker.summoning_sick = False
    b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=3))
    b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=3))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [b1, b2]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [b1, b2]

    lethal_by_blocker = blocker_lethal_capacities(state, attacker, [b1, b2])
    begin_assign_combat_damage(
        state, attacker, [b1, b2], power=8, has_trample=False,
        lethal_by_blocker=lethal_by_blocker, on_complete=lambda s: None,
    )
    for _ in range(3):
        execute_assign_combat_damage_option(state, "B1", b1.slot)
    for _ in range(3):
        execute_assign_combat_damage_option(state, "B2", b2.slot)

    assert state.pending_resolution is None, "both blockers capped -- the remaining 2 must auto-pile, not wait"
    assert attacker.flags["combat_damage_split"] == ({b1: 3, b2: 5}, 0), "the last blocker absorbs the dead-letter 2"

    combat_damage_step(state)
    assert state.players[1].life_total == 20, "no trample -- none of the overkill reaches the player"


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
