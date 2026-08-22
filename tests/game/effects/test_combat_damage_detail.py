"""Combat-damage detail: deathtouch through combat, the attacker's free
damage split, Aura-modified lethal calculation, and the first-strike
sub-step in the direction test_combat.py does not already cover. The broad
combat cases (attack eligibility, trample spill, lifelink, menace, goad, The
Initiative, gang-block splits) live in test_combat.py."""
from game import registry
from game.cards import CardDef, CardType, EffectId
from game.effects.combat import blocker_lethal_capacities, combat_damage_step, declare_attacker, declare_attackers_step
from game.effects.state_based import check_state_based_actions, destroy_permanent
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
    # _attacker_deal_damage stamps flags["deathtouched"] on whatever it
    # damages; check_state_based_actions then treats that as lethal regardless of toughness
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
    # 702.2b + 510.1a: when assigning combat damage, any nonzero damage from
    # a deathtouch source counts as lethal, so a 5-power deathtouch trampler
    # blocked by one 4-toughness creature assigns 1 and tramples 4. A
    # single-blocked attacker gets no assign_combat_damage decision, so the
    # auto split must already be this minimum-lethal, attacker-optimal one.
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
    # the controller assigns damage one point at a time, never past a
    # blocker's lethal cap (blocker_lethal_capacities); a capped blocker
    # drops out of assign_combat_damage_options. A blocker ends up
    # non-lethally damaged only when power runs out (510.1c), never by choice.
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
    # 702.19e/510.1c: once every blocker in a trampler's split is at its
    # lethal cap, the remaining damage spills to the player as a forced
    # side effect of the point that caps the last open blocker -- never an
    # explicit "assign to player" action.
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


def test_trample_blocker_departing_after_split_recorded_spills_its_share_to_player():
    # the free-assignment split is recorded during DECLARE_BLOCKERS but not
    # dealt until COMBAT_DAMAGE, a phase later; a blocker earmarked damage
    # can die to an instant in between. With trample, its earmarked share
    # must flow through to the player instead (702.19b), not vanish.
    original = _with_filler_keywords({"trample"})
    try:
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        trampler = Permanent(CardDef("Stomper3", CardType.CREATURE, None, EffectId.FILLER, power=6, toughness=6))
        trampler.summoning_sick = False
        b1 = Permanent(CardDef("B1", CardType.CREATURE, None, None, power=1, toughness=2))
        b2 = Permanent(CardDef("B2", CardType.CREATURE, None, None, power=1, toughness=2))
        state.players[0].battlefield = [trampler]
        state.players[1].battlefield = [b1, b2]

        declare_attackers_step(state)
        declare_attacker(state, trampler)
        state.blocked_by[trampler] = [b1, b2]

        lethal_by_blocker = blocker_lethal_capacities(state, trampler, [b1, b2])
        begin_assign_combat_damage(
            state, trampler, [b1, b2], power=6, has_trample=True,
            lethal_by_blocker=lethal_by_blocker, on_complete=lambda s: None,
        )
        for _ in range(2):
            execute_assign_combat_damage_option(state, "B1", b1.slot)
        for _ in range(2):
            execute_assign_combat_damage_option(state, "B2", b2.slot)
        assert state.pending_resolution is None
        assert trampler.flags["combat_damage_split"] == ({b1: 2, b2: 2}, 2), "6 power = 2+2 lethal, 2 excess tramples"

        # an instant kills b1 before damage is dealt
        destroy_permanent(state, b1)
        assert b1 not in state.players[1].battlefield

        combat_damage_step(state)
        assert b2.damage_marked == 2, "the surviving blocker still takes its own recorded share"
        assert state.players[1].life_total == 16, "20 - (2 original trample + 2 orphaned from b1)"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_non_trample_blocker_departing_after_split_recorded_does_not_spill():
    # same departing-blocker scenario as the trample test above, but without
    # trample: 509.1h keeps the attacker "blocked," so the orphaned share is
    # just never dealt, not routed anywhere
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    attacker = Permanent(CardDef("Chooser3", CardType.CREATURE, None, EffectId.FILLER, power=4, toughness=4))
    attacker.summoning_sick = False
    c1 = Permanent(CardDef("C1", CardType.CREATURE, None, None, power=1, toughness=2))
    c2 = Permanent(CardDef("C2", CardType.CREATURE, None, None, power=1, toughness=2))
    state.players[0].battlefield = [attacker]
    state.players[1].battlefield = [c1, c2]

    declare_attackers_step(state)
    declare_attacker(state, attacker)
    state.blocked_by[attacker] = [c1, c2]

    lethal_by_blocker = blocker_lethal_capacities(state, attacker, [c1, c2])
    begin_assign_combat_damage(
        state, attacker, [c1, c2], power=4, has_trample=False,
        lethal_by_blocker=lethal_by_blocker, on_complete=lambda s: None,
    )
    for _ in range(2):
        execute_assign_combat_damage_option(state, "C1", c1.slot)
    for _ in range(2):
        execute_assign_combat_damage_option(state, "C2", c2.slot)
    assert state.pending_resolution is None
    assert attacker.flags["combat_damage_split"] == ({c1: 2, c2: 2}, 0)

    destroy_permanent(state, c1)
    combat_damage_step(state)
    assert c2.damage_marked == 2
    assert state.players[1].life_total == 20, "no trample -- c1's orphaned 2 is dropped, never spilled to the player"


def test_lethal_calculation_uses_aura_modified_toughness():
    # the default lethal-in-order split must read effective toughness, not
    # printed: an Armadillo Cloak (+2/+2) raises the bar, so less tramples over
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

        # blocker is effectively 3/4 under the Cloak: 4 lethal, 2 tramples over
        assert blocker.damage_marked == 4, "lethal must use Aura-modified toughness, not the printed 2"
        # the Cloak also grants lifelink on the defender's creature, so the
        # defender gains the 3 their boosted blocker dealt -- +2/+2 and
        # lifelink both come from the one Aura
        assert trampler.damage_marked == 3, "the Cloak's +2/+0 applies to the damage dealt back"
        assert state.players[1].life_total == 21, "20 - 2 trample excess + 3 Cloak lifelink"
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original


def test_first_strike_blocker_kills_attacker_before_it_deals_damage():
    # test_combat.py covers a first-striking attacker; this is a
    # first-striking blocker that kills the attacker outright, so the SBA
    # check between sub-steps removes it before its regular damage step
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
