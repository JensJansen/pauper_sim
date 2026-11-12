"""Tests for game.catalog.red_cards."""

import drl_env
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.red_cards import (
    _fireblast_alt_extra_legal,
    _goblin_bushwhacker_kicked,
    _grab_the_prize_extra_legal,
    _makeshift_munitions_legal,
    activate_melded_moxite_sac,
    activate_reckless_lackey_sac,
    cast_breath_weapon,
    cast_chain_lightning,
    cast_cleansing_wildfire,
    cast_end_the_festivities,
    cast_faithless_looting,
    cast_fiery_temper,
    cast_fireblast,
    cast_fireblast_alt,
    cast_galvanic_blast,
    cast_grab_the_prize,
    cast_highway_robbery,
    cast_highway_robbery_from_exile,
    cast_lava_dart,
    cast_lightning_bolt,
    cast_rally_at_the_hornburg,
    cast_reckless_impulse,
    experimental_synthesizer_sac,
    flashback_faithless_looting,
    flashback_lava_dart,
    krark_clan_shaman_activate,
    makeshift_munitions_activate,
    melded_moxite_etb,
    voldaren_epicure_etb,
)
from game.effects.casting import cast_permanent_from_hand
from game.effects.combat import combat_damage_step, declare_attacker, declare_attackers_step
from game.effects.madness_and_plot import execute_madness_cast, plot_to_exile
from game.effects.stack import on_cast_trigger, resolve_top_of_stack
from game.effects.state_based import sacrifice_to_graveyard
from game.effects.stats import has_keyword, permanent_power
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, begin_pay_cost, execute_pool_spend, pool_spend_options
from game.state import GameState, Permanent, PlayerState
from game.turn import Phase, untap_step


def test_breath_weapon_symmetric_two_damage_wipe():
    """Breath Weapon deals 2 damage to every non-Dragon creature on both
    battlefields, including the caster's own."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    breath_weapon = CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)
    state.hand = [breath_weapon]
    mine_dies = Permanent(CardDef("Mine (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    mine_survives = Permanent(CardDef("Mine (survives)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    theirs_dies = Permanent(CardDef("Theirs (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    not_a_creature = Permanent(CardDef("Some Land", CardType.LAND, None, EffectId.FILLER))
    state.players[0].battlefield = [mine_dies, mine_survives, not_a_creature]
    state.players[1].battlefield = [theirs_dies]

    cast_breath_weapon(state, breath_weapon)
    assert state.hand == [] and any(c.name == breath_weapon.name for c in state.graveyard)
    assert mine_dies not in state.players[0].battlefield
    assert mine_survives in state.players[0].battlefield and mine_survives.damage_marked == 2
    assert theirs_dies not in state.players[1].battlefield
    assert not_a_creature in state.players[0].battlefield  # a land is never a valid target


def test_breath_weapon_spares_dragons_avenging_hunter_regression():
    """Breath Weapon's non-Dragon filter is live: a real Dragon takes zero
    damage while a non-Dragon on the same battlefield takes the full 2."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    breath_weapon = CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)
    state.hand = [breath_weapon]
    dragon = Permanent(registry.CARD_DEFS["Avenging Hunter"])  # a real Dragon
    dragon.slot = 1
    non_dragon = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    non_dragon.slot = 1
    state.players[1].battlefield = [dragon, non_dragon]

    cast_breath_weapon(state, breath_weapon)
    assert dragon in state.players[1].battlefield and dragon.damage_marked == 0
    assert dragon.damage_marked != 2, "old symmetric behavior would have marked 2 damage on the Dragon"
    assert non_dragon.damage_marked == 2


def test_end_the_festivities_hits_only_the_opponent_not_the_caster():
    # End the Festivities deals 1 damage to the opponent and each creature
    # they control; the caster's own life and board are untouched.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    festivities = CardDef("End the Festivities", CardType.SORCERY, {"R": 1}, EffectId.END_THE_FESTIVITIES)
    state.hand = [festivities]
    mine_untouched = Permanent(CardDef("Mine (untouched)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs_dies = Permanent(CardDef("Theirs (dies)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    theirs_survives = Permanent(CardDef("Theirs (survives)", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=2))
    their_land = Permanent(CardDef("Their Land", CardType.LAND, None, EffectId.FILLER))
    state.players[0].battlefield = [mine_untouched]
    state.players[1].battlefield = [theirs_dies, theirs_survives, their_land]
    opponent_life_before = state.players[1].life_total

    cast_end_the_festivities(state, festivities)
    assert state.hand == [] and any(c.name == festivities.name for c in state.graveyard)
    assert state.players[1].life_total == opponent_life_before - 1
    assert state.players[0].life_total == 20  # caster is never damaged
    assert mine_untouched in state.players[0].battlefield and mine_untouched.damage_marked == 0
    assert theirs_dies not in state.players[1].battlefield
    assert theirs_survives in state.players[1].battlefield and theirs_survives.damage_marked == 1
    assert their_land in state.players[1].battlefield


def test_highway_robbery_discard_path_triggers_madness():
    """Highway Robbery's discard cost draws 2, and discarding a Madness card
    still triggers Madness's exile-not-graveyard replacement."""
    state = GameState(on_the_play=True)
    hr = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    fiery_temper = CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER)
    state.hand = [hr, fiery_temper]
    state.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state, hr)
    assert state.pending_resolution["kind"] == "discard_or_sacrifice"
    resolution.execute_discard_or_sacrifice_discard(state, "Fiery Temper")
    assert len(state.hand) == 2  # drew 2
    assert state.exile and state.exile[0][0].name == "Fiery Temper"  # exiled, not graveyarded
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"


def test_highway_robbery_sacrifice_land_path():
    """Highway Robbery's sacrifice-a-land path opens a choose_permanent
    sub-decision for which land pays it, and fires sacrifice triggers."""
    state2 = GameState(on_the_play=True)
    hr2 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    mountain = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    state2.hand = [hr2]
    state2.battlefield = [mountain, gix]
    state2.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery(state2, hr2)
    resolution.execute_discard_or_sacrifice_trigger_sacrifice(state2)
    assert state2.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state2, "Mountain", mountain.slot)
    assert state2.battlefield == [gix]
    assert sorted(c.name for c in state2.graveyard) == ["Highway Robbery", "Mountain"]
    assert len(state2.hand) == 2
    queued = [e for e in state2.trigger_queue if e["type"] == "on_sacrifice"]
    assert queued and queued[0]["permanent"] is gix


def test_highway_robbery_decline_no_draw():
    """Highway Robbery's discard-or-sacrifice is genuinely optional, even
    with something payable on hand; declining draws nothing."""
    state3 = GameState(on_the_play=True)
    hr3 = CardDef("Highway Robbery", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.HIGHWAY_ROBBERY)
    spare_card = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)
    spare_land = Permanent(CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN))
    state3.hand = [hr3, spare_card]
    state3.battlefield = [spare_land]
    cast_highway_robbery(state3, hr3)
    assert state3.pending_resolution["kind"] == "discard_or_sacrifice"  # genuinely offered, not auto-completed
    resolution.execute_discard_or_sacrifice_decline(state3)
    assert [c.name for c in state3.hand] == ["Lightning Bolt"]  # untouched, no draw
    assert spare_land in state3.battlefield  # untouched
    assert state3.pending_resolution is None


def _fresh_bolt_state():
    bolt = CardDef("Lightning Bolt", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT)
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.hand = [bolt]
    return s, bolt


def test_lightning_bolt_hits_opponent_creature():
    """Lightning Bolt deals 3 damage to a targeted opponent creature."""
    state, bolt = _fresh_bolt_state()
    opp_creature = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    opp_creature.slot = 1
    state.players[1].battlefield = [opp_creature]
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)
    assert state.hand == []
    resolve_top_of_stack(state)
    assert opp_creature.damage_marked == 3 and any(c.name == bolt.name for c in state.graveyard)


def test_lightning_bolt_hits_opponent_face():
    """Lightning Bolt deals 3 damage to the opponent's face."""
    state, bolt = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17


def test_lightning_bolt_can_target_self():
    """Lightning Bolt can legally target the caster's own face."""
    state, bolt = _fresh_bolt_state()
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 0)
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 17


def test_lightning_bolt_fizzles_if_target_removed():
    """Lightning Bolt fizzles with no effect if its target leaves before resolution."""
    state, bolt = _fresh_bolt_state()
    doomed = Permanent(CardDef("Grizzly Bears", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=4))
    doomed.slot = 1
    state.players[1].battlefield = [doomed]
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)
    state.players[1].battlefield = []  # target leaves before the bolt resolves
    resolve_top_of_stack(state)
    assert any(c.name == bolt.name for c in state.graveyard) and state.players[1].life_total == 20


def test_lightning_bolt_hexproof_targeting():
    """An opponent's hexproof creature is not a legal Bolt target, but the
    caster's own hexproof creature still is."""
    original_filler = registry.EFFECT_REGISTRY[EffectId.FILLER]
    try:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = {"keywords": {"hexproof"}}
        state, bolt = _fresh_bolt_state()
        opp_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        mine_hex = Permanent(CardDef("Hex Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        opp_hex.slot = mine_hex.slot = 1
        state.players[1].battlefield = [opp_hex]
        state.players[0].battlefield = [mine_hex]
        cast_lightning_bolt(state, bolt)
        creature_opts = resolution.choose_any_target_creature_options(state)
        assert (0, "Hex Bear", 1) in creature_opts
        assert (1, "Hex Bear", 1) not in creature_opts
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = original_filler


def _drain_pool_pay(state):
    """Drain a pending pool-spend pay_cost to completion."""
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 12
        execute_pool_spend(state, pool_spend_options(state)[0])


def test_chain_lightning_copy_rider_pay_copy_retarget_decline():
    """Chain Lightning deals 3, then its rider lets the target pay {R}{R}
    to copy and retarget it back at the original caster."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].mana_pool = {"R": 2}
    chain = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain]
    cast_chain_lightning(state, chain)
    resolution.execute_choose_any_target_player(state, 1)
    assert chain not in state.players[0].hand
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17 and any(c.name == chain.name for c in state.players[0].graveyard)
    # rider's first "may": the affected player (opp, idx 1) may pay {R}{R}
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    resolution.pay_unless_pay(state)
    _drain_pool_pay(state)
    # rider's second, independent "may": having paid, opp may copy
    assert state.pending_resolution["kind"] == "may_copy" and state.active_idx == 1
    resolution.execute_may_copy(state, True)
    # the opponent now chooses the copy's new target
    assert state.pending_resolution["kind"] == "choose_any_target" and state.active_idx == 1
    resolution.execute_choose_any_target_player(state, 0)  # retarget the original caster
    assert state.active_idx == 0
    assert len(state.stack) == 1 and state.stack[0]["controller"] == 1  # opp controls the copy
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 17
    # the copy's own rider: the caster (idx 0) declines to pay, so the chain stops
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 0
    resolution.pay_unless_decline(state)
    assert state.pending_resolution is None and state.stack == []


def test_chain_lightning_pay_but_decline_copy():
    """Chain Lightning's rider: paying {R}{R} but declining to copy is
    legal -- the mana is spent, no copy is made."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].mana_pool = {"R": 2}
    chain2 = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain2]
    cast_chain_lightning(state, chain2)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17
    resolution.pay_unless_pay(state)
    _drain_pool_pay(state)
    assert state.pending_resolution["kind"] == "may_copy" and state.active_idx == 1
    resolution.execute_may_copy(state, False)
    assert state.pending_resolution is None and state.stack == []
    assert state.active_idx == 0
    assert sum(state.players[1].mana_pool.values()) == 0  # {R}{R} was still spent


def test_chain_lightning_rider_decline_no_copy():
    """Chain Lightning's rider: the affected player declines to pay {R}{R}
    at all -- no copy, no retarget, chain stops, no mana spent."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    chain = CardDef("Chain Lightning", CardType.SORCERY, {"R": 1}, EffectId.CHAIN_LIGHTNING)
    state.players[0].hand = [chain]
    cast_chain_lightning(state, chain)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    resolution.pay_unless_decline(state)
    assert state.pending_resolution is None and state.stack == []
    assert state.active_idx == 0
    assert sum(state.players[1].mana_pool.values()) == 0


def test_cleansing_wildfire_own_land():
    """Cleansing Wildfire destroys a targeted land; its controller fetches
    a tapped basic, and the caster draws. Here, the caster's own land."""
    state = GameState(on_the_play=True)
    dual = Permanent(CardDef("Twisted Landscape", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE))
    dual.slot = 1
    dual.tapped = True
    state.battlefield = [dual]
    cw = CardDef("Cleansing Wildfire", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.CLEANSING_WILDFIRE)
    state.hand = [cw]
    state.library = [
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
        CardDef("Guttersnipe", CardType.CREATURE, {"generic": 2, "R": 1}, EffectId.GUTTERSNIPE, power=2, toughness=2),  # nonbasic
    ]
    cast_cleansing_wildfire(state, cw)
    assert state.pending_resolution["kind"] == "choose_any_target"
    assert (0, "Twisted Landscape", 1) in resolution.choose_any_target_creature_options(state)
    resolution.execute_choose_any_target_creature(state, 0, "Twisted Landscape", 1)
    resolve_top_of_stack(state)
    assert dual not in state.battlefield
    assert state.pending_resolution["kind"] == "search_fetch"
    assert resolution.search_fetch_options(state) == ["Mountain"]  # only the basic is eligible
    resolution.execute_search_fetch_option(state, "Mountain")
    fetched = [p for p in state.battlefield if p.card_def.name == "Mountain"]
    assert len(fetched) == 1 and fetched[0].tapped
    assert len(state.hand) == 1


def test_cleansing_wildfire_opponent_land():
    """Targeting an opponent's land: it's destroyed, the opponent searches
    their own library, and the caster still draws."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp_land = Permanent(CardDef("Opp Dual", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE))
    opp_land.slot = 1
    state.players[1].battlefield = [opp_land]
    state.players[1].library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",))]
    cw2 = CardDef("Cleansing Wildfire", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.CLEANSING_WILDFIRE)
    state.players[0].hand = [cw2]
    state.players[0].library = [CardDef("CasterDraw", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",))]
    cast_cleansing_wildfire(state, cw2)
    resolution.execute_choose_any_target_creature(state, 1, "Opp Dual", 1)
    resolve_top_of_stack(state)
    assert opp_land not in state.players[1].battlefield
    assert state.pending_resolution["kind"] == "search_fetch" and state.active_idx == 1  # land's controller searches
    resolution.execute_search_fetch_option(state, "Forest")
    opp_basics = [p for p in state.players[1].battlefield if p.card_def.name == "Forest"]
    assert len(opp_basics) == 1 and opp_basics[0].tapped
    assert state.active_idx == 0
    assert len(state.players[0].hand) == 1 and state.players[0].hand[0].name == "CasterDraw"


def test_reckless_impulse_exile_and_expiry():
    """Reckless Impulse exiles the top 2 cards into the impulse zone
    (playable until the player's next turn); untap prunes them on expiry."""
    state = GameState(on_the_play=True)
    state.turn_number = 3
    ri = CardDef("Reckless Impulse", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RECKLESS_IMPULSE)
    state.hand = [ri]
    state.library = [CardDef("A", CardType.LAND, None, EffectId.MOUNTAIN), CardDef("B", CardType.INSTANT, {"R": 1}, EffectId.LIGHTNING_BOLT), CardDef("C", CardType.LAND, None, EffectId.MOUNTAIN)]
    cast_reckless_impulse(state, ri)
    assert len(state.impulse) == 2 and all(u == 3 + len(state.players) for _cd, u in state.impulse)  # until next turn
    assert [c.name for c in state.graveyard] == ["Reckless Impulse"]
    state.turn_number = 3 + len(state.players) + 1  # past the deadline
    untap_step(state)
    assert state.impulse == []  # expired, pruned


def test_goblin_bushwhacker_kicked_team_pump():
    """Goblin Bushwhacker's kicked mode gives the team (including itself)
    +1/+0 and haste until end of turn."""
    state = GameState(on_the_play=True)
    ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ally.slot = 1
    state.battlefield = [ally]
    bw = CardDef("Goblin Bushwhacker", CardType.CREATURE, {"R": 1}, EffectId.GOBLIN_BUSHWHACKER, power=1, toughness=1)
    state.hand = [bw]
    _goblin_bushwhacker_kicked(state, bw)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    gob = next(p for p in state.battlefield if p.card_def.name == "Goblin Bushwhacker")
    assert permanent_power(state, gob) == 2 and has_keyword(state, gob, "haste")
    assert permanent_power(state, ally) == 2 and has_keyword(state, ally, "haste")


def test_goblin_bushwhacker_via_action_table_cost_override_mode():
    """Casting Goblin Bushwhacker via the real action table and choosing
    the kicked mode charges that mode's own {R}{R} cost override, not the
    card's base {R}."""
    bw_dl = [("Goblin Bushwhacker", 4), ("Mountain", 8)]
    bw_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(bw_dl, registry.EFFECT_REGISTRY)}
    bw_state = GameState(on_the_play=True)
    bw_state.phase = Phase.MAIN1
    bw_state.turn_player_idx = 0
    bw_state.active_idx = 0
    bw_state.hand = [registry.CARD_DEFS["Goblin Bushwhacker"]]
    bw_ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    bw_ally.slot = 1
    bw_state.battlefield = [bw_ally, Permanent(registry.CARD_DEFS["Mountain"]), Permanent(registry.CARD_DEFS["Mountain"])]
    for p in bw_state.battlefield:
        if p.card_def.name == "Mountain":
            activate_mana_source(bw_state, p)
    assert bw_state.mana_pool.get("R", 0) == 2
    cast_legal, cast_execute = bw_byname["Cast Goblin Bushwhacker"]
    assert cast_legal(bw_state)
    cast_execute(bw_state)
    assert bw_state.pending_resolution["kind"] == "choose_cast_mode"
    mode2_legal, mode2_execute = bw_byname["Mode 2"]  # kicked mode
    assert mode2_legal(bw_state)
    mode2_execute(bw_state)
    assert bw_state.pending_resolution["kind"] == "pay_cost"
    assert bw_state.pending_resolution["remaining"] == {"R": 2}, (
        "must owe the kicked mode's OWN {R}{R} cost override, not the card's base {R}"
    )
    guard = 0
    while bw_state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(bw_state, pool_spend_options(bw_state)[0])
    while bw_state.stack:  # resolve the spell, creating the permanent and queuing its ETB
        resolve_top_of_stack(bw_state)
    promote_triggers_to_stack(bw_state)  # then promote the ETB trigger
    while bw_state.stack:
        resolve_top_of_stack(bw_state)
    assert bw_state.pending_resolution is None and bw_state.stack == []


def test_rally_at_the_hornburg_haste_only_humans():
    """Rally at the Hornburg makes two 1/1 Human Soldiers and gives haste
    to Humans you control, including ones already on the battlefield."""
    state = GameState(on_the_play=True)
    human = Permanent(CardDef("Human Ally", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1, subtypes=("Human",)))
    human.slot = 1
    goblin = Permanent(CardDef("Goblin Ally", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1, subtypes=("Goblin",)))
    goblin.slot = 1
    state.battlefield = [human, goblin]
    rally = CardDef("Rally at the Hornburg", CardType.SORCERY, {"generic": 1, "R": 1}, EffectId.RALLY_AT_THE_HORNBURG)
    state.hand = [rally]
    cast_rally_at_the_hornburg(state, rally)
    assert sum(1 for p in state.battlefield if p.card_def.name == "Human Soldier") == 2
    assert has_keyword(state, human, "haste")
    assert not has_keyword(state, goblin, "haste")


def test_reckless_lackey_sac_draws_and_makes_treasure():
    """Reckless Lackey has first strike and haste; its sac ability draws
    a card and makes a Treasure."""
    state = GameState(on_the_play=True)
    lackey = Permanent(registry.CARD_DEFS["Reckless Lackey"])
    lackey.slot = 1
    state.battlefield = [lackey]
    state.library = [CardDef("Top", CardType.LAND, None, EffectId.MOUNTAIN)]
    assert has_keyword(state, lackey, "first_strike") and has_keyword(state, lackey, "haste")
    activate_reckless_lackey_sac(state, lackey)
    assert lackey not in state.battlefield and any(c.name == "Reckless Lackey" for c in state.graveyard)
    resolve_top_of_stack(state)
    assert len(state.hand) == 1 and any(p.card_def.name == "Treasure" for p in state.battlefield)


def test_reckless_lackey_sac_mid_combat_removes_it_from_combat():
    """Sacrificing an attacking Reckless Lackey mid-combat must remove it
    from state.attackers, not just the battlefield, so it deals no combat
    damage."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    lackey = Permanent(registry.CARD_DEFS["Reckless Lackey"])
    lackey.slot = 1
    lackey.summoning_sick = False
    state.players[0].battlefield = [lackey]
    state.library = [CardDef("Top", CardType.LAND, None, EffectId.MOUNTAIN)]

    declare_attackers_step(state)
    declare_attacker(state, lackey)
    assert lackey in state.attackers

    activate_reckless_lackey_sac(state, lackey)
    assert lackey not in state.attackers, "506.4: a sacrificed attacker leaves combat"

    combat_damage_step(state)
    assert state.players[1].life_total == 20, "a sacrificed attacker deals no combat damage"


def test_goblin_tomb_raider_conditional_static():
    """Goblin Tomb Raider gets +1/+0 and haste only while its controller
    controls an artifact."""
    state = GameState(on_the_play=True)
    gtr = Permanent(registry.CARD_DEFS["Goblin Tomb Raider"])
    gtr.slot = 1
    state.battlefield = [gtr]
    assert permanent_power(state, gtr) == 1 and not has_keyword(state, gtr, "haste")
    state.battlefield.append(Permanent(registry.CARD_DEFS["Great Furnace"]))
    assert permanent_power(state, gtr) == 2 and has_keyword(state, gtr, "haste")


def test_burning_tree_emissary_etb_adds_rr():
    """Burning-Tree Emissary: ETB adds {R}{R} to the pool (authorized simplification)."""
    state = GameState(on_the_play=True)
    bte = registry.CARD_DEFS["Burning-Tree Emissary"]
    state.hand = [bte]
    cast_permanent_from_hand(state, bte)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    assert state.mana_pool.get("R") == 2
    assert state.mana_pool_single_pip == {}  # 2-symbol event, never single-pip-tagged


def _drive_stack(state):
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)


def test_krark_clan_shaman_sac_artifact_sweeps_nonflyers():
    """Krark-Clan Shaman's sac-an-artifact ability deals 1 damage to each
    nonflying creature; flyers are spared, and it kills itself as a 1/1."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    krark = Permanent(registry.CARD_DEFS["Krark-Clan Shaman"])
    krark.slot = 1
    art = Permanent(registry.CARD_DEFS["Great Furnace"])
    state.players[0].battlefield = [krark, art]
    ground = Permanent(registry.CARD_DEFS["Gurmag Angler"])  # nonflying
    ground.slot = 1
    flyer = Permanent(registry.CARD_DEFS["Utrom Monitor"])  # flying
    flyer.slot = 1
    state.players[1].battlefield = [ground, flyer]
    krark_clan_shaman_activate(state, krark)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Great Furnace", art.slot)
    _drive_stack(state)
    assert ground.damage_marked == 1 and flyer.damage_marked == 0
    assert krark not in state.players[0].battlefield


def test_experimental_synthesizer_impulse_and_samurai():
    """Experimental Synthesizer's ETB triggers impulse draw; its {2}{R}
    sac ability makes a Samurai and triggers another impulse on leaving."""
    state = GameState(on_the_play=True)
    state.turn_number = 1
    state.hand = [registry.CARD_DEFS["Experimental Synthesizer"]]
    state.library = [CardDef(f"t{i}", CardType.LAND, None, EffectId.MOUNTAIN, basic=True) for i in range(3)]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Experimental Synthesizer"])
    _drive_stack(state)
    assert len(state.impulse) == 1
    es = next(p for p in state.battlefield if p.card_def.name == "Experimental Synthesizer")
    experimental_synthesizer_sac(state, es)
    _drive_stack(state)
    assert any(p.card_def.name == "Samurai" for p in state.battlefield) and len(state.impulse) == 2
    samurai = next(p for p in state.battlefield if p.card_def.name == "Samurai")
    assert has_keyword(state, samurai, "vigilance")


def test_clockwork_percussionist_dies_into_impulse():
    """Clockwork Percussionist has haste and triggers impulse draw on death."""
    state = GameState(on_the_play=True)
    state.turn_number = 1
    state.library = [CardDef("q", CardType.LAND, None, EffectId.MOUNTAIN, basic=True)]
    cp = Permanent(registry.CARD_DEFS["Clockwork Percussionist"])
    cp.slot = 1
    state.battlefield = [cp]
    assert has_keyword(state, cp, "haste")
    sacrifice_to_graveyard(state, cp)
    _drive_stack(state)
    assert len(state.impulse) == 1


def test_galvanic_blast_no_metalcraft():
    """Galvanic Blast deals 2 damage without Metalcraft."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=9))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    gb = registry.CARD_DEFS["Galvanic Blast"]
    state.players[0].hand = [gb]
    cast_galvanic_blast(state, gb)
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.damage_marked == 2


def test_galvanic_blast_metalcraft_four_damage():
    """Galvanic Blast deals 4 damage with Metalcraft (3+ artifacts controlled)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]) for _ in range(3)]
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=9))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    gb = registry.CARD_DEFS["Galvanic Blast"]
    state.players[0].hand = [gb]
    cast_galvanic_blast(state, gb)
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.damage_marked == 4


def test_voldaren_epicure_etb_damages_opponent_and_creates_blood():
    """Voldaren Epicure's ETB deals 1 damage to each opponent and creates
    a Blood token."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    voldaren_epicure_etb(state)
    assert state.players[1].life_total == 19
    assert any(p.card_def.name == "Blood" for p in state.players[0].battlefield)


def test_fiery_temper_baseline_cast_hits_any_target():
    """Fiery Temper's baseline {1}{R}{R} cast deals 3 damage to any target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ft = CardDef("Fiery Temper", CardType.INSTANT, {"generic": 1, "R": 2}, EffectId.FIERY_TEMPER)
    state.players[0].hand = [ft]
    cast_fiery_temper(state, ft)
    resolution.execute_choose_any_target_player(state, 1)
    assert ft not in state.players[0].hand
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17 and any(c.name == "Fiery Temper" for c in state.players[0].graveyard)


def test_fiery_temper_madness_cast_via_real_discard_chain():
    """Discarding Fiery Temper (via Faithless Looting) exiles it under
    Madness rather than graveyarding it; casting it for {R} deals 3 damage
    and it ends up in the graveyard without ever returning to hand."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    fl = registry.CARD_DEFS["Faithless Looting"]
    ft = registry.CARD_DEFS["Fiery Temper"]
    state.players[0].hand = [fl, ft]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mountain]

    cast_faithless_looting(state, fl)
    assert len(state.players[0].hand) == 3  # Fiery Temper + 2 drawn
    resolution.execute_discard_option(state, "Fiery Temper")  # Madness: exile, not graveyard
    resolution.execute_discard_option(state, "D0")
    assert ft not in state.players[0].hand
    assert any(cd.name == "Fiery Temper" for cd, _stamp in state.exile)
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"

    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "madness_decision"

    activate_mana_source(state, mountain)
    execute_madness_cast(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 17
    assert any(c.name == "Fiery Temper" for c in state.players[0].graveyard)
    assert not any(cd.name == "Fiery Temper" for cd, _stamp in state.exile)


def test_faithless_looting_cast_draws_two_discards_two():
    """Faithless Looting's baseline {R} cast draws 2, then discards 2
    (mandatory)."""
    state = GameState(on_the_play=True)
    fl = CardDef("Faithless Looting", CardType.SORCERY, {"R": 1}, EffectId.FAITHLESS_LOOTING)
    spare_a = CardDef("Spare A", CardType.LAND, None, EffectId.MOUNTAIN)
    spare_b = CardDef("Spare B", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [fl, spare_a, spare_b]
    state.library = [CardDef(f"Draw{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_faithless_looting(state, fl)
    assert fl not in state.hand and any(c.name == "Faithless Looting" for c in state.graveyard)
    assert len(state.hand) == 4  # Spare A/B + 2 drawn
    resolution.execute_discard_option(state, "Spare A")
    resolution.execute_discard_option(state, "Spare B")
    assert state.pending_resolution is None
    assert len(state.hand) == 2 and all(c.name.startswith("Draw") for c in state.hand)
    assert sorted(c.name for c in state.graveyard) == ["Faithless Looting", "Spare A", "Spare B"]


def test_faithless_looting_flashback_effect_resolves_and_exiles():
    """Faithless Looting's Flashback draws 2/discards 2, and the flashed-
    back card ends up exiled, never back in the graveyard."""
    state = GameState(on_the_play=True)
    fl_inst = state.new_instance(registry.CARD_DEFS["Faithless Looting"])
    state.graveyard = [fl_inst]
    state.hand = [CardDef("Spare A", CardType.LAND, None, EffectId.MOUNTAIN), CardDef("Spare B", CardType.LAND, None, EffectId.MOUNTAIN)]
    state.library = [CardDef(f"Draw{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    flashback_faithless_looting(state, fl_inst)
    assert fl_inst not in state.graveyard  # left the graveyard the moment Flashback was chosen
    assert len(state.stack) == 1
    resolve_top_of_stack(state)
    assert len(state.hand) == 4  # Spare A/B + 2 drawn
    resolution.execute_discard_option(state, "Spare A")
    resolution.execute_discard_option(state, "Spare B")
    assert len(state.hand) == 2 and all(c.name.startswith("Draw") for c in state.hand)
    assert not any(c.name == "Faithless Looting" for c in state.graveyard)


def test_highway_robbery_plot_cost_and_cast_from_exile():
    """Highway Robbery's Plot costs {1}{R}, and casting it from exile runs
    the same discard-or-sacrifice as a normal cast, moving it exile ->
    graveyard without ever touching hand again."""
    plot_spec = registry.EFFECT_REGISTRY[EffectId.HIGHWAY_ROBBERY]["plot"]
    assert plot_spec["cost"] == {"generic": 1, "R": 1}  # real Plot cost

    state = GameState(on_the_play=True)
    hr = registry.CARD_DEFS["Highway Robbery"]
    spare = CardDef("Spare Card", CardType.INSTANT, {"R": 1}, EffectId.FILLER)
    state.hand = [hr, spare]
    plot_to_exile(state, hr)
    assert state.hand == [spare]
    assert state.exile == [(hr, state.turn_number)]

    state.library = [CardDef(f"Filler {i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    cast_highway_robbery_from_exile(state, hr)
    assert any(c.name == "Highway Robbery" for c in state.graveyard)  # exile -> graveyard, never hand
    assert hr not in state.hand
    assert state.pending_resolution["kind"] == "discard_or_sacrifice"
    resolution.execute_discard_or_sacrifice_discard(state, "Spare Card")
    assert any(c.name == "Spare Card" for c in state.graveyard)
    assert len(state.hand) == 2 and all(c.name.startswith("Filler") for c in state.hand)  # drew 2


def test_grab_the_prize_extra_legal_requires_second_hand_card():
    """Grab the Prize's discard is an additional cost requiring a card in
    hand besides itself; with nothing else in hand it's illegal to cast."""
    state = GameState(on_the_play=True)
    gtp = registry.CARD_DEFS["Grab the Prize"]
    state.hand = [gtp]
    assert not _grab_the_prize_extra_legal(state)
    state.hand = [gtp, CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)]
    assert _grab_the_prize_extra_legal(state)


def test_grab_the_prize_discard_nonland_deals_damage():
    """Grab the Prize draws 2; discarding a nonland deals 2 damage to
    each opponent."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    gtp = registry.CARD_DEFS["Grab the Prize"]
    nonland = CardDef("Spare Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER)
    state.players[0].hand = [gtp, nonland]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_grab_the_prize(state, gtp)
    assert state.pending_resolution["kind"] == "discard"
    resolution.execute_discard_option(state, "Spare Bolt")
    assert len(state.players[0].hand) == 2  # drew 2
    assert state.players[1].life_total == 18
    assert any(c.name == "Grab the Prize" for c in state.players[0].graveyard)


def test_grab_the_prize_discard_land_no_damage():
    """Grab the Prize: discarding a land still draws 2, but deals no damage."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    gtp = registry.CARD_DEFS["Grab the Prize"]
    land = CardDef("Spare Land", CardType.LAND, None, EffectId.MOUNTAIN)
    state.players[0].hand = [gtp, land]
    state.players[0].library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]

    cast_grab_the_prize(state, gtp)
    resolution.execute_discard_option(state, "Spare Land")
    assert len(state.players[0].hand) == 2  # drew 2
    assert state.players[1].life_total == 20


def test_melded_moxite_etb_discard_draws_two():
    """Melded Moxite's ETB: discarding a card (may) draws two."""
    state = GameState(on_the_play=True)
    spare = CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [spare]
    state.library = [CardDef(f"D{i}", CardType.LAND, None, EffectId.MOUNTAIN) for i in range(2)]
    melded_moxite_etb(state)
    assert state.pending_resolution["kind"] == "discard" and state.pending_resolution["optional"] is True
    resolution.execute_discard_option(state, "Spare")
    assert len(state.hand) == 2


def test_melded_moxite_etb_decline_no_draw():
    """Melded Moxite's ETB discard is genuinely optional; declining draws nothing."""
    state = GameState(on_the_play=True)
    spare = CardDef("Spare", CardType.LAND, None, EffectId.MOUNTAIN)
    state.hand = [spare]
    melded_moxite_etb(state)
    resolution.execute_discard_decline(state)
    assert state.hand == [spare]
    assert state.pending_resolution is None


def test_melded_moxite_sac_pays_three_and_creates_tapped_robot():
    """Melded Moxite's {3} sac ability creates a tapped 2/2 Robot token."""
    state = GameState(on_the_play=True)
    mm = Permanent(registry.CARD_DEFS["Melded Moxite"])
    mm.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [mm] + mountains
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3

    begin_pay_cost(state, mm.card_def.extra["sac_ability_cost"], on_complete=lambda s: activate_melded_moxite_sac(s, mm))
    assert state.pending_resolution["kind"] == "pay_cost" and state.pending_resolution["remaining"] == {"generic": 3}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert mm not in state.battlefield and any(c.name == "Melded Moxite" for c in state.graveyard)
    resolve_top_of_stack(state)
    robot = next(p for p in state.battlefield if p.card_def.name == "Robot")
    assert robot.tapped is True and permanent_power(state, robot) == 2


def test_fireblast_hard_cast_deals_four():
    """Fireblast's baseline {4}{R}{R} cast deals 4 damage to any target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = CardDef("Fireblast", CardType.INSTANT, {"generic": 4, "R": 2}, EffectId.FIREBLAST)
    state.players[0].hand = [fb]
    cast_fireblast(state, fb)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 16  # 20 - 4
    assert any(c.name == "Fireblast" for c in state.players[0].graveyard)


def test_fireblast_alt_cost_sacrifices_two_mountains_no_mana():
    """Fireblast's alt cost sacrifices 2 Mountains instead of paying mana,
    for the same 4 damage."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = registry.CARD_DEFS["Fireblast"]
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(2)]
    mountains[0].slot = 1
    mountains[1].slot = 2
    state.players[0].hand = [fb]
    state.players[0].battlefield = mountains
    assert _fireblast_alt_extra_legal(state)

    cast_fireblast_alt(state, fb)
    assert fb not in state.players[0].hand and any(c.name == "Fireblast" for c in state.players[0].graveyard)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 2)
    assert state.players[0].battlefield == []
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 16
    assert sum(state.players[0].mana_pool.values()) == 0  # no mana ever paid


def test_fireblast_alt_cost_discounts_tag_from_a_tapped_mountain():
    """Tapping one of two Mountains for {R} before sacrificing both via
    Fireblast's alt cost discounts the tagged R pip by exactly 1, not 2 --
    only that one Mountain produced tagged mana."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    fb = registry.CARD_DEFS["Fireblast"]
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(2)]
    mountains[0].slot = 1
    mountains[1].slot = 2
    state.players[0].hand = [fb]
    state.players[0].battlefield = mountains
    activate_mana_source(state, mountains[0])  # float+tag {R} before sacrificing it
    assert state.players[0].mana_pool_single_pip == {"R": 1}

    cast_fireblast_alt(state, fb)
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    resolution.execute_choose_permanent_option(state, "Mountain", 2)
    assert state.players[0].mana_pool_single_pip == {}  # the tagged R pip is excused
    assert state.players[0].mana_pool == {"R": 1}  # floated mana is still there, unspent


def test_fireblast_alt_illegal_with_fewer_than_two_mountains():
    """Fireblast's alt cost is illegal with fewer than 2 Mountains controlled."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Mountain"])]  # only 1
    assert not _fireblast_alt_extra_legal(state)
    state.battlefield = []
    assert not _fireblast_alt_extra_legal(state)


def test_guttersnipe_on_cast_deals_two_to_opponent():
    """Guttersnipe deals 2 damage to each opponent whenever its controller
    casts an instant or sorcery."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    guttersnipe = Permanent(registry.CARD_DEFS["Guttersnipe"])
    guttersnipe.slot = 1
    state.players[0].battlefield = [guttersnipe]
    bolt = registry.CARD_DEFS["Lightning Bolt"]
    state.players[0].hand = [bolt]

    on_cast_trigger(state, bolt)  # what every real cast path calls once the cost is paid
    cast_lightning_bolt(state, bolt)
    resolution.execute_choose_any_target_player(state, 1)
    _drive_stack(state)
    assert state.players[1].life_total == 20 - 3 - 2


def test_lava_dart_cast_deals_one():
    """Lava Dart's baseline {R} cast deals 1 damage to any target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ld = CardDef("Lava Dart", CardType.INSTANT, {"R": 1}, EffectId.LAVA_DART)
    state.players[0].hand = [ld]
    cast_lava_dart(state, ld)
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19
    assert any(c.name == "Lava Dart" for c in state.players[0].graveyard)


def test_lava_dart_flashback_sacrifices_mountain_no_mana():
    """Lava Dart's Flashback sacrifices a Mountain instead of paying mana,
    for the same 1 damage."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    ld_inst = state.new_instance(registry.CARD_DEFS["Lava Dart"])
    state.players[0].graveyard = [ld_inst]
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    mountain.slot = 1
    state.players[0].battlefield = [mountain]

    flashback_lava_dart(state, ld_inst)
    assert ld_inst not in state.players[0].graveyard  # left the graveyard the moment Flashback was chosen
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mountain", 1)
    assert mountain not in state.players[0].battlefield
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19
    assert sum(state.players[0].mana_pool.values()) == 0  # no mana spent


def test_goblin_bushwhacker_unkicked_via_action_table_only_charges_r():
    """Goblin Bushwhacker's unkicked mode charges only its base {R}, and
    its ETB is a no-op -- no team pump -- when not kicked."""
    bw_dl = [("Goblin Bushwhacker", 4), ("Mountain", 8)]
    bw_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(bw_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Goblin Bushwhacker"]]
    ally = Permanent(CardDef("Ally", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    ally.slot = 1
    state.battlefield = [ally, Permanent(registry.CARD_DEFS["Mountain"])]
    activate_mana_source(state, state.battlefield[1])
    assert state.mana_pool.get("R", 0) == 1
    cast_legal, cast_execute = bw_byname["Cast Goblin Bushwhacker"]
    assert cast_legal(state)
    cast_execute(state)
    assert state.pending_resolution["kind"] == "choose_cast_mode"
    mode1_legal, mode1_execute = bw_byname["Mode 1"]  # unkicked mode
    assert mode1_legal(state)
    mode1_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"R": 1}, (
        "unkicked mode must charge only the card's own base {R}, no override"
    )
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 30
        execute_pool_spend(state, pool_spend_options(state)[0])
    while state.stack:
        resolve_top_of_stack(state)
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    gob = next(p for p in state.battlefield if p.card_def.name == "Goblin Bushwhacker")
    assert permanent_power(state, gob) == 1 and not has_keyword(state, gob, "haste"), (
        "unkicked -- goblin_bushwhacker_etb must no-op, no team pump"
    )
    assert permanent_power(state, ally) == 1 and not has_keyword(state, ally, "haste")


def test_reckless_lackey_sac_via_action_table_pays_generic_and_r():
    """Reckless Lackey's sac ability, activated via the real action table,
    charges its {2}{R} cost."""
    rl_dl = [("Reckless Lackey", 1), ("Mountain", 8)]
    rl_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(rl_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    lackey = Permanent(registry.CARD_DEFS["Reckless Lackey"])
    lackey.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [lackey] + mountains
    state.library = [CardDef("Top", CardType.LAND, None, EffectId.MOUNTAIN)]
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3
    legal, execute = rl_byname["Activate Reckless Lackey (sac)"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 2, "R": 1}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert lackey not in state.battlefield and any(c.name == "Reckless Lackey" for c in state.graveyard)
    resolve_top_of_stack(state)
    assert state.pending_resolution is None and state.stack == []


def test_krark_clan_shaman_illegal_with_no_artifact():
    """Krark-Clan Shaman's sac-artifact ability is illegal with zero
    artifacts controlled."""
    state = GameState(on_the_play=True)
    krark = Permanent(registry.CARD_DEFS["Krark-Clan Shaman"])
    krark.slot = 1
    state.battlefield = [krark]
    ability = registry.EFFECT_REGISTRY[EffectId.KRARK_CLAN_SHAMAN]["activated_abilities"]["sweep"]
    assert ability["legal"](state, krark) is False


def test_makeshift_munitions_activate_pays_one_sacrifices_and_deals_damage():
    """Makeshift Munitions' ability pays {1}, sacrifices an artifact or
    creature, then deals 1 damage to a chosen target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    mm = Permanent(registry.CARD_DEFS["Makeshift Munitions"])
    creature = Permanent(CardDef("Fodder", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    creature.slot = 1
    mountain = Permanent(registry.CARD_DEFS["Mountain"])
    state.players[0].battlefield = [mm, creature, mountain]
    activate_mana_source(state, mountain)
    assert state.mana_pool.get("R", 0) == 1
    assert _makeshift_munitions_legal(state, mm)

    makeshift_munitions_activate(state, mm)
    assert state.pending_resolution["kind"] == "pay_cost"
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.pending_resolution["kind"] == "choose_permanent"  # which artifact/creature pays
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    assert creature not in state.players[0].battlefield
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_player(state, 1)
    resolve_top_of_stack(state)
    assert state.players[1].life_total == 19


def test_experimental_synthesizer_sac_via_action_table_sorcery_speed_and_cost():
    """Experimental Synthesizer's sac ability charges {2}{R} and is
    "Activate only as a sorcery" -- illegal outside a main phase."""
    es_dl = [("Experimental Synthesizer", 1), ("Mountain", 8)]
    es_byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(es_dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    es = Permanent(registry.CARD_DEFS["Experimental Synthesizer"])
    es.slot = 1
    mountains = [Permanent(registry.CARD_DEFS["Mountain"]) for _ in range(3)]
    state.battlefield = [es] + mountains
    for m in mountains:
        activate_mana_source(state, m)
    assert state.mana_pool.get("R", 0) == 3
    legal, execute = es_byname["Activate Experimental Synthesizer (make_samurai)"]

    state.phase = Phase.DECLARE_ATTACKERS  # a real phase, but not a main phase
    assert not legal(state), "\"Activate only as a sorcery\" must be illegal outside MAIN1/MAIN2"

    state.phase = Phase.MAIN1
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 2, "R": 1}
    guard = 0
    while state.pending_resolution is not None:
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert es not in state.battlefield  # sacrificed as part of the effect's resolve
    _drive_stack(state)
    assert state.pending_resolution is None and state.stack == []
