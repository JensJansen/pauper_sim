"""Tests for game.catalog.colorless_cards."""

import drl_env
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.colorless_cards import (
    BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF,
    _pinnacle_kill_ship_station_legal,
    _pinnacle_kill_ship_station_resolve,
    activate_barrels_of_blasting_jelly_burn,
    activate_bonders_ornament_draw,
    activate_candy_trail_sac,
    activate_expedition_map,
    activate_relic_of_progenitus_draw,
    activate_relic_of_progenitus_exile,
    activate_tocasia_dig_site_surveil,
    activate_twisted_landscape_fetch,
    cast_boulderbranch_prototype,
    cast_maelstrom_colossus,
    chromatic_star_mana,
    cycle_ash_barrens,
    cycle_twisted_landscape,
    lembas_sac,
    nihil_spellbomb_dies,
    nihil_spellbomb_sac,
    pinnacle_kill_ship_etb,
)
from game.effects.casting import cast_permanent_from_hand, play_land_from_hand
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import sacrifice_to_graveyard
from game.effects.stats import has_keyword, permanent_power, permanent_toughness
from game.effects.tokens import (
    CLUE_TOKEN_CARD_DEF, MAP_TOKEN_CARD_DEF, TREASURE_TOKEN_CARD_DEF, activate_clue_sac, activate_map_sac,
)
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, execute_pool_spend, pool_spend_options
from game.state import GameState, Permanent, PlayerState
from game.turn import Phase, Speed


def _resolve_etb(state):
    """Promote a queued ETB trigger to the stack and resolve it."""
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)


def _drive(state):
    """Promote queued triggers and resolve the stack until empty."""
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)


def test_ash_barrens_basic_landcycling():
    """Basic landcycling {1}: discard from hand, search library for a
    basic land, put into hand, shuffle. No draw rider."""
    state = GameState(on_the_play=True)
    ash_barrens = CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1})
    state.hand = [ash_barrens]
    state.library = [
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Plains", CardType.LAND, None, EffectId.PLAINS, basic=True),
        CardDef("Ash Barrens", CardType.LAND, None, EffectId.ASH_BARRENS, cycling_cost={"generic": 1}),  # not basic -- ineligible
    ]
    cycle_ash_barrens(state, ash_barrens)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert resolution.search_fetch_options(state) == ["Forest", "Plains"]  # the 2nd Ash Barrens is correctly excluded
    resolution.execute_search_fetch_option(state, "Plains")
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["Plains"]
    assert sorted(c.name for c in state.graveyard) == ["Ash Barrens"]  # discarded itself, not the fetched land
    assert sorted(c.name for c in state.library) == ["Ash Barrens", "Forest"]  # shuffled; the unchosen basic stays


def test_barrels_of_blasting_jelly_blast_deals_5_to_target_creature():
    """{5}, {T}, Sacrifice: deals 5 damage to target creature. The
    sacrifice is a cost paid immediately; the damage waits on the stack
    and fizzles if the target is gone by resolution."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    barrels = Permanent(registry.CARD_DEFS["Barrels of Blasting Jelly"])
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=6))
    victim.slot = 1
    state.players[0].battlefield = [barrels]
    state.players[1].battlefield = [victim]
    activate_barrels_of_blasting_jelly_burn(state, barrels)
    assert barrels not in state.players[0].battlefield  # sacrificed -- a cost, paid immediately
    assert state.pending_resolution["kind"] == "choose_any_target"
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.damage_marked == 5


def test_barrels_of_blasting_jelly_blast_needs_a_legal_creature_target():
    """The ability can't be activated with zero legal creature targets on
    board, even though the {5} mana would otherwise be payable."""
    state = GameState(on_the_play=True)
    barrels = Permanent(registry.CARD_DEFS["Barrels of Blasting Jelly"])
    state.battlefield = [barrels]
    extra_legal = registry.EFFECT_REGISTRY[EffectId.BARRELS_OF_BLASTING_JELLY]["activated_abilities"]["blast"]["extra_legal"]
    assert not extra_legal(state)


def test_candy_trail_sac_gain_life_and_draw():
    """Candy Trail's sac ability gains 3 life and draws a card."""
    state = GameState(on_the_play=True)
    candy_trail = Permanent(CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ))
    state.battlefield = [candy_trail]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_candy_trail_sac(state, candy_trail)
    assert state.battlefield == []  # sacrificed -- a cost, paid immediately on activation
    # gain+draw are the effect, on the stack -- not applied immediately
    assert len(state.stack) == 1 and state.life_total == 20 and state.hand == []
    resolve_top_of_stack(state)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3, once the effect resolves
    assert [c.name for c in state.hand] == ["Forest"]


def _relic_of_progenitus_state():
    """Shared setup for Relic of Progenitus's two abilities."""
    state = GameState(on_the_play=True)
    relic = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state.battlefield = [relic]
    state.graveyard = [
        CardDef("Bramble Wurm", CardType.CREATURE, {"generic": 6, "G": 1}, EffectId.BRAMBLE_WURM),
        CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON),
    ]
    return state, relic


def test_relic_of_progenitus_exile_targeting_yourself():
    """The repeatable {T} ability targets a player (including yourself);
    the targeted player then chooses their own graveyard card to exile.
    Also checks the {T} cost is logged as a tap_or_untap event."""
    state, relic = _relic_of_progenitus_state()
    state.event_log = []
    activate_relic_of_progenitus_exile(state, relic)
    assert relic.tapped  # {T} -- a cost, paid immediately on activation
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Relic of Progenitus", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == state.active_idx
    assert state.pending_resolution["kind"] == "choose_target_player"  # target chosen at activation
    resolution.execute_choose_target_player_option(state, 0)  # explicitly target yourself
    # The effect (the targeted player exiles a graveyard card) is now on the
    # stack -- it opens the graveyard-card choice only once it resolves.
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    relic_opts = resolution.choose_graveyard_card_options(state)
    assert [c.name for c in relic_opts] == ["Bramble Wurm", "Breath Weapon"]
    resolution.execute_choose_graveyard_card_option(state, next(o for o in relic_opts if o.name == "Bramble Wurm"))
    assert state.pending_resolution is None
    assert [c.name for c in state.graveyard] == ["Breath Weapon"]  # only the chosen one removed


def test_relic_of_progenitus_draw_exiles_all_graveyards_and_draws():
    """The one-shot {1}+exile-self ability exiles every player's
    graveyard, then draws a card."""
    state, relic = _relic_of_progenitus_state()
    state.graveyard = [CardDef("Breath Weapon", CardType.INSTANT, {"generic": 2, "R": 1}, EffectId.BREATH_WEAPON)]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_relic_of_progenitus_draw(state, relic)
    assert state.battlefield == []  # exile-self -- a cost, paid immediately
    # "exile ALL graveyards" + draw are the effect -- on the stack now.
    assert len(state.stack) == 1 and [c.name for c in state.graveyard] == ["Breath Weapon"]
    resolve_top_of_stack(state)
    assert state.graveyard == []  # the untouched "Breath Weapon" is gone too, once resolved
    assert [c.name for c in state.hand] == ["Forest"]


def test_relic_of_progenitus_exile_targeting_a_real_opponent():
    """Targeting an opponent exiles from their graveyard, not the active
    player's own."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    relic = Permanent(CardDef(
        "Relic of Progenitus", CardType.ARTIFACT, {"generic": 1}, EffectId.RELIC_OF_PROGENITUS,
        draw_ability_cost={"generic": 1}, graveyard_exile_ability_cost={},
    ))
    state.players[0].battlefield = [relic]
    state.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    activate_relic_of_progenitus_exile(state, relic)
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent (locked at activation)
    resolve_top_of_stack(state)  # effect resolves -> the TARGETED player picks the card
    assert state.active_idx == 1  # active_idx flipped to the targeted player for their choice
    theirs_opts = resolution.choose_graveyard_card_options(state)
    assert [c.name for c in theirs_opts] == ["Theirs"]  # THEIR graveyard, not "Mine"
    resolution.execute_choose_graveyard_card_option(state, theirs_opts[0])
    assert state.active_idx == 0  # restored to the activator once the choice is made
    assert state.players[1].graveyard == []
    assert [c.name for c in state.players[0].graveyard] == ["Mine"]  # own graveyard untouched


def test_rooftop_percher_etb_with_empty_graveyard_gains_life_only():
    """With an empty graveyard, the ETB's up-to-two exile auto-completes
    and only the 3 life gain happens."""
    state = GameState(on_the_play=True)
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    state.hand = [percher]
    cast_permanent_from_hand(state, percher)
    _resolve_etb(state)  # ETB (target-at-promotion): empty graveyard -> no exile targets, gain 3
    assert state.life_total == 23  # STARTING_LIFE (20) + 3


def test_boulderbranch_golem_etb_gains_life_equal_to_power():
    """Boulderbranch Golem's ETB gains life equal to its power (6, at
    normal {7} cost)."""
    state = GameState(on_the_play=True)
    golem = CardDef("Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5)
    state.hand = [golem]
    cast_permanent_from_hand(state, golem)
    _resolve_etb(state)
    assert state.life_total == 26  # STARTING_LIFE (20) + 6


def _rp_state():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    return state


def _mk_gy(name):
    return CardDef(name, CardType.INSTANT, {"R": 1}, EffectId.FILLER)


def test_rooftop_percher_etb_exiles_two_distinct_target_instances():
    """ETB exiles up to two target cards from any graveyard and gains 3
    life; same-named copies are tracked as distinct instances."""
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    st = _rp_state()
    b1 = st.new_instance(_mk_gy("Bolt")); b2 = st.new_instance(_mk_gy("Bolt"))
    st.players[0].graveyard = [b1, b2]
    opp = st.new_instance(_mk_gy("Opp Card")); st.players[1].graveyard = [opp]
    st.players[0].hand = [percher]
    cast_permanent_from_hand(st, percher)
    promote_triggers_to_stack(st)  # target-at-promotion opens the up-to-2 pick
    assert st.pending_resolution["kind"] == "choose_graveyard_card"
    assert sorted(o.name for o in resolution.choose_graveyard_card_options(st)) == ["Bolt", "Bolt", "Opp Card"]  # both same-named copies offered
    resolution.execute_choose_graveyard_card_option(st, b1)
    assert b1 not in resolution.choose_graveyard_card_options(st) and b2 in resolution.choose_graveyard_card_options(st)  # a1 excluded BY IDENTITY, its twin still choosable
    resolution.execute_choose_graveyard_card_option(st, opp)  # 2nd pick -> max reached, ability's effect pushed
    assert st.pending_resolution is None and len(st.stack) == 1
    resolve_top_of_stack(st)
    assert st.players[0].life_total == 23  # +3 (always)
    assert b1 not in st.players[0].graveyard and b2 in st.players[0].graveyard  # b1 exiled, unpicked b2 stays
    assert opp not in st.players[1].graveyard  # opp's card exiled from ITS graveyard


def test_rooftop_percher_etb_partial_fizzle_survivor_only_exiled():
    """If one chosen target leaves before resolution, only the surviving
    target is exiled; life is still gained."""
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    st = _rp_state()
    p1 = st.new_instance(_mk_gy("X")); p2 = st.new_instance(_mk_gy("Y")); st.players[0].graveyard = [p1, p2]
    st.players[0].hand = [percher]
    cast_permanent_from_hand(st, percher); promote_triggers_to_stack(st)
    resolution.execute_choose_graveyard_card_option(st, p1); resolution.execute_choose_graveyard_card_option(st, p2)
    st.players[0].graveyard.remove(p1)  # p1 exiled by other hate before this resolves
    resolve_top_of_stack(st)
    assert st.players[0].life_total == 23 and p2 not in st.players[0].graveyard  # survivor exiled, life gained


def test_rooftop_percher_etb_all_targets_gone_life_still_gained():
    """All exile targets gone before resolution -- exile does nothing, but
    life gain still happens.
    # AUTHORIZED SIMPLIFICATION (owner, 2026-07-29): the ability never
    # wholesale-fizzles on all-targets-illegal (strict 608.2b would
    # counter it, dropping the life gain too); the life gain is
    # unconditional here."""
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    st = _rp_state()
    z1 = st.new_instance(_mk_gy("Z")); st.players[0].graveyard = [z1]
    st.players[0].hand = [percher]
    cast_permanent_from_hand(st, percher); promote_triggers_to_stack(st)
    resolution.execute_choose_graveyard_card_option(st, z1)  # only eligible card -> selection auto-ends
    st.players[0].graveyard.remove(z1)  # its only target leaves
    resolve_top_of_stack(st)
    assert st.players[0].life_total == 23  # life ALWAYS gained even when the exile fully fizzles


def test_rooftop_percher_etb_explicit_decline_after_one_pick():
    """With two targets available, the "up to two" pick can stop after
    taking just one."""
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    st = _rp_state()
    d1 = st.new_instance(_mk_gy("P")); d2 = st.new_instance(_mk_gy("Q")); st.players[0].graveyard = [d1, d2]
    st.players[0].hand = [percher]
    cast_permanent_from_hand(st, percher); promote_triggers_to_stack(st)
    resolution.execute_choose_graveyard_card_option(st, d1)
    assert st.pending_resolution["kind"] == "choose_graveyard_card"  # a 2nd pick is offered...
    resolution.execute_choose_graveyard_card_decline(st)  # ...but the agent stops at one
    resolve_top_of_stack(st)
    assert d1 not in st.players[0].graveyard and d2 in st.players[0].graveyard and st.players[0].life_total == 23


def test_boulderbranch_golem_prototype_cast_and_etb():
    """Boulderbranch Golem Prototype is a cheaper {3}{G} 3/3 cast option;
    its ETB gains life equal to its power (3), not the normal mode's 6."""
    state = GameState(on_the_play=True)
    golem_hand = CardDef("Boulderbranch Golem", CardType.CREATURE, {"generic": 7}, EffectId.BOULDERBRANCH_GOLEM, power=6, toughness=5)
    state.hand = [golem_hand]
    cast_boulderbranch_prototype(state, BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF)
    assert state.hand == []  # the {7} hand card was consumed by the prototype cast
    _resolve_etb(state)
    assert state.life_total == 23  # 20 + 3 (the 3/3's power), not +6
    proto = next(p for p in state.battlefield if p.card_def.name == "Boulderbranch Golem")
    assert proto.card_def is BOULDERBRANCH_GOLEM_PROTOTYPE_CARD_DEF
    assert permanent_power(state, proto) == 3 and permanent_toughness(state, proto) == 3


def test_maelstrom_colossus_cascade_casts_hit_for_free():
    """Cascade exiles cards from the library until a nonland with mana
    value less than 8 is found, casts it for free, then bottoms the rest
    in random order."""
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)}}
    try:
        state = GameState(on_the_play=True)
        colossus = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        a_land = CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True)
        hit = CardDef("Free Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        after = CardDef("Never Seen", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER)
        state.hand = [colossus]
        state.library = [a_land, hit, after]
        cast_maelstrom_colossus(state, colossus)
        assert state.pending_resolution["kind"] == "may_cast"  # Cascade offers the caster a may-cast
        resolution.execute_may_cast(state, True)               # choose to cast the hit for free
        assert [c.name for c in state.graveyard] == [colossus.name]
        assert sorted(p.card_def.name for p in state.battlefield) == ["Free Hit", "Maelstrom Colossus"]
        # only cards exiled alongside the hit get shuffled back; cards below the hit are untouched
        assert [c.name for c in state.library] == ["Never Seen", "A Land"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_maelstrom_colossus_cascade_may_cast_decline_bottoms_hit():
    """Declining the may-cast bottoms the hit instead of casting it."""
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)}}
    try:
        state1b = GameState(on_the_play=True)
        colossus1b = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        a_land = CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True)
        hit1b = CardDef("Declined Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        after = CardDef("Never Seen", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER)
        state1b.hand = [colossus1b]
        state1b.library = [a_land, hit1b, after]
        cast_maelstrom_colossus(state1b, colossus1b)
        assert state1b.pending_resolution["kind"] == "may_cast"
        resolution.execute_may_cast(state1b, False)            # decline -> hit stays in the library
        assert [p.card_def.name for p in state1b.battlefield] == ["Maelstrom Colossus"]  # hit NOT cast
        assert state1b.library[0].name == "Never Seen"  # below the hit, never touched
        assert sorted(c.name for c in state1b.library[1:]) == ["A Land", "Declined Hit"]  # both bottomed
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_maelstrom_colossus_cascade_whiff_nothing_eligible():
    """If nothing is eligible for Cascade, only Colossus enters."""
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def)}}
    try:
        state2 = GameState(on_the_play=True)
        colossus2 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        a_land = CardDef("A Land", CardType.LAND, None, EffectId.FOREST, basic=True)
        too_expensive = CardDef("Too Expensive", CardType.ARTIFACT, {"generic": 8}, EffectId.FILLER)
        state2.hand = [colossus2]
        state2.library = [a_land, too_expensive]
        cast_maelstrom_colossus(state2, colossus2)
        assert [p.card_def.name for p in state2.battlefield] == ["Maelstrom Colossus"]
        assert sorted(c.name for c in state2.library) == ["A Land", "Too Expensive"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_maelstrom_colossus_cascade_extra_legal_fails_skips_hit():
    """A hit that fails extra_legal is skipped, same as a genuine whiff --
    Cascade only waives the mana cost, not other preconditions."""
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {
        "cast": {"resolve": lambda state, card_def: cast_permanent_from_hand(state, card_def), "extra_legal": lambda state: False},
    }
    try:
        state3 = GameState(on_the_play=True)
        colossus3 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        ineligible_hit = CardDef("Ineligible Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state3.hand = [colossus3]
        state3.library = [ineligible_hit]
        cast_maelstrom_colossus(state3, colossus3)
        assert [p.card_def.name for p in state3.battlefield] == ["Maelstrom Colossus"]
        assert [c.name for c in state3.library] == ["Ineligible Hit"]  # never cast, just shuffled back in
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_maelstrom_colossus_cascade_defers_entry_until_chained_resolution_completes():
    """Colossus doesn't enter the battlefield until the cascaded card's
    own resolution, including any further decisions, fully completes."""
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    entered_before_decision = []

    def _opens_pending(state, card_def):
        state.hand.remove(card_def)

        def _on_complete(state, taken):
            if taken:
                state.move_card(card_def, state.graveyard)
            entered_before_decision.append(any(p.card_def.name == "Maelstrom Colossus" for p in state.battlefield))

        resolution.begin_resolution(state, "fake_choice", _on_complete)

    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": _opens_pending}}
    try:
        state4 = GameState(on_the_play=True)
        colossus4 = CardDef("Maelstrom Colossus", CardType.CREATURE, {"generic": 8}, EffectId.MAELSTROM_COLOSSUS, power=7, toughness=7)
        decision_hit = CardDef("Decision Hit", CardType.ARTIFACT, {"generic": 2}, EffectId.FILLER)
        state4.hand = [colossus4]
        state4.library = [decision_hit]
        cast_maelstrom_colossus(state4, colossus4)
        assert state4.pending_resolution["kind"] == "may_cast"  # the may-cast comes first
        resolution.execute_may_cast(state4, True)               # cast it -> its own resolve opens fake_choice
        assert state4.pending_resolution["kind"] == "fake_choice"  # Colossus genuinely hasn't entered yet
        assert not any(p.card_def.name == "Maelstrom Colossus" for p in state4.battlefield)
        resolution.complete_resolution(state4, True)
        assert entered_before_decision == [False]  # Colossus hadn't entered DURING the decision's own on_complete either
        assert sorted(p.card_def.name for p in state4.battlefield) == ["Maelstrom Colossus"]  # decision_hit chose graveyard, not battlefield, in this fake resolve
        assert state4.pending_resolution is None
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_pinnacle_kill_ship_station_charges_and_animates_at_threshold():
    """Station taps another creature, adding charge counters equal to its
    power; at 7+ counters Kill-Ship animates into a 7/7 flier. Also checks
    the {T} cost is logged as a tap_or_untap event."""
    state = GameState(on_the_play=True, event_log=[])
    kill_ship = Permanent(CardDef("Pinnacle Kill-Ship", CardType.ARTIFACT, {"generic": 7}, EffectId.PINNACLE_KILL_SHIP))
    weak = Permanent(CardDef("Weak Tapper", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    strong = Permanent(CardDef("Strong Tapper", CardType.CREATURE, None, EffectId.FILLER, power=5, toughness=5))
    state.battlefield = [kill_ship, weak, strong]

    assert _pinnacle_kill_ship_station_legal(state, kill_ship) is True
    assert kill_ship.card_type == CardType.ARTIFACT  # not yet animated
    assert permanent_power(state, kill_ship) == 0 and permanent_toughness(state, kill_ship) == 0

    _pinnacle_kill_ship_station_resolve(state, kill_ship)
    assert resolution.choose_permanent_options(state) == [("Strong Tapper", 1), ("Weak Tapper", 1)]  # Kill-Ship itself never offered -- "another creature"
    resolution.execute_choose_permanent_option(state, "Strong Tapper", 1)
    assert strong.tapped is True  # tapped -- a cost, paid immediately on activation
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Strong Tapper", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == state.active_idx
    # charge counters are the effect, applied on resolution, not immediately
    assert len(state.stack) == 1 and kill_ship.counters.get("charge", 0) == 0
    resolve_top_of_stack(state)
    assert kill_ship.counters["charge"] == 5  # the TAPPED creature's own power
    assert kill_ship.card_type == CardType.ARTIFACT  # still below the 7-counter threshold

    _pinnacle_kill_ship_station_resolve(state, kill_ship)
    resolution.execute_choose_permanent_option(state, "Weak Tapper", 1)
    resolve_top_of_stack(state)
    assert kill_ship.counters["charge"] == 8  # 5 + 3, now >= 7
    assert kill_ship.card_type == CardType.CREATURE  # animated
    assert permanent_power(state, kill_ship) == 7 and permanent_toughness(state, kill_ship) == 7
    assert has_keyword(state, kill_ship, "flying") is True
    assert not _pinnacle_kill_ship_station_legal(state, kill_ship)  # both other creatures already tapped


def test_pinnacle_kill_ship_etb_hits_opponent_creature_lethal():
    """ETB deals 10 damage to up to one target creature on either
    battlefield, chosen when the trigger is put on the stack; targeting an
    opponent's creature kills it."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    pinnacle_kill_ship_etb(state)
    assert state.pending_resolution["kind"] == "choose_any_target" and state.pending_resolution["optional"]
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    assert len(state.stack) == 1  # the ETB effect waits on the stack
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield  # 10 vs 3 -> lethal via SBA


def test_pinnacle_kill_ship_etb_decline_resolves_doing_nothing():
    """Declining the optional target resolves the ability doing nothing."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    bystander = Permanent(CardDef("Bystander", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    bystander.slot = 1
    state.players[1].battlefield = [bystander]
    pinnacle_kill_ship_etb(state)
    resolution.execute_choose_any_target_decline(state)
    resolve_top_of_stack(state)
    assert bystander in state.players[1].battlefield and bystander.damage_marked == 0


def test_pinnacle_kill_ship_etb_fizzles_if_target_leaves_before_resolution():
    """The ability fizzles if the chosen target leaves before it
    resolves."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    doomed = Permanent(CardDef("Doomed", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    doomed.slot = 1
    bystander = Permanent(CardDef("Bystander", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    bystander.slot = 2
    state.players[1].battlefield = [doomed, bystander]
    pinnacle_kill_ship_etb(state)
    resolution.execute_choose_any_target_creature(state, 1, "Doomed", 1)
    state.players[1].battlefield.remove(doomed)  # exiled/bounced before resolution
    resolve_top_of_stack(state)
    assert state.pending_resolution is None and state.stack == []  # nothing left waiting
    assert bystander in state.players[1].battlefield and bystander.damage_marked == 0  # untouched by the fizzle


def test_twisted_landscape_sacrifice_fetch_basic_tapped():
    """Sacrificing Twisted Landscape searches for a basic Swamp, Mountain,
    or Forest and puts it onto the battlefield tapped."""
    state = GameState(on_the_play=True)
    twisted = Permanent(CardDef(
        "Twisted Landscape", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE,
        fetch_ability_cost={}, cycling_cost={"B": 1, "R": 1, "G": 1},
    ))
    state.battlefield = [twisted]
    state.library = [
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True, subtypes=("Forest",)),
        CardDef("Island", CardType.LAND, None, EffectId.ISLAND, basic=True, subtypes=("Island",)),  # ineligible
    ]
    activate_twisted_landscape_fetch(state, twisted)
    assert twisted not in state.battlefield  # sacrificed -- a cost, paid immediately
    assert state.graveyard[-1].name == "Twisted Landscape"
    assert len(state.stack) == 1  # the search is the EFFECT, on the stack
    resolve_top_of_stack(state)
    assert sorted(resolution.search_fetch_options(state)) == ["Forest", "Mountain"]  # Island excluded (not S/M/F)
    resolution.execute_search_fetch_option(state, "Mountain")
    fetched = [p for p in state.battlefield if p.card_def.name == "Mountain"]
    assert len(fetched) == 1 and fetched[0].tapped  # onto the battlefield TAPPED
    assert not any(c.name == "Mountain" for c in state.library)  # removed from library


def test_myr_enforcer_affinity_reduces_generic_cost_by_artifacts_controlled():
    """Affinity reduces Myr Enforcer's {7} cost by the number of artifacts
    controlled."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(CardDef("Great Furnace", CardType.LAND, None, EffectId.GREAT_FURNACE, artifact=True)) for _ in range(4)]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Myr Enforcer"])
    assert eff["generic"] == 3, eff  # 7 - 4 artifacts
    state.battlefield = []
    assert drl_env._effective_cast_cost(state, registry.CARD_DEFS["Myr Enforcer"])["generic"] == 7  # no artifacts -> full


def test_ichor_wellspring_etb_and_dies_both_draw():
    """Ichor Wellspring draws a card on both ETB and death."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Ichor Wellspring"]]
    state.library = [CardDef(f"c{i}", CardType.LAND, None, EffectId.ISLAND, basic=True) for i in range(4)]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Ichor Wellspring"])
    _drive(state)
    assert len(state.hand) == 1  # ETB
    ichor = next(p for p in state.battlefield if p.card_def.name == "Ichor Wellspring")
    sacrifice_to_graveyard(state, ichor)
    _drive(state)
    assert len(state.hand) == 2  # dies draw


def test_nihil_spellbomb_sac_exiles_graveyard_then_dies_may_pay_b_draw():
    """Sacrificing Nihil Spellbomb exiles a target player's graveyard; the
    resulting dies trigger lets the controller pay {B} to draw a card."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].graveyard = [CardDef("g", CardType.INSTANT, {"U": 1}, EffectId.FILLER)]
    nihil = Permanent(registry.CARD_DEFS["Nihil Spellbomb"])
    nihil.slot = 1
    state.players[0].battlefield = [nihil]
    state.players[0].library = [CardDef("d", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    state.players[0].mana_pool = {"B": 1}
    nihil_spellbomb_sac(state, nihil)
    resolution.execute_choose_target_player_option(state, 1)
    _drive(state)
    assert state.players[1].graveyard == []  # exiled
    assert state.pending_resolution["kind"] == "pay_unless"  # dies: may pay {B}
    resolution.pay_unless_pay(state)
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert len(state.players[0].hand) == 1  # paid, drew


def test_nihil_spellbomb_dies_trigger_belongs_to_its_own_owner_not_whoever_is_active():
    """Regression: the dies trigger's "may pay {B}, draw" choice belongs
    to the permanent's owner, not whoever is currently the active
    player."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0  # active player is 0 ...
    nihil = Permanent(registry.CARD_DEFS["Nihil Spellbomb"])
    nihil.slot = 1
    nihil.flags["owner_idx"] = 1  # ... but the Spellbomb's real owner/controller is player 1
    state.players[1].library = [CardDef("d", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    state.players[1].mana_pool = {"B": 1}
    state.players[0].library = [CardDef("decoy", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    state.players[0].mana_pool = {"B": 1}  # the OLD bug would find this and wrongly spend via player 0 instead

    nihil_spellbomb_dies(state, nihil)
    assert state.pending_resolution["kind"] == "pay_unless"
    assert state.active_idx == 1  # active_idx flips to the payer -- the owner (1), not player 0
    resolution.pay_unless_pay(state)
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.players[1].mana_pool == {} and state.players[0].mana_pool == {"B": 1}  # PLAYER 1's own B spent; player 0's untouched


def test_lembas_dies_shuffles_itself_into_library():
    """Sacrificing Lembas triggers it shuffling itself back into its
    owner's library, alongside the sac ability's own +3 life."""
    state = GameState(on_the_play=True)
    lembas = Permanent(registry.CARD_DEFS["Lembas"])
    lembas.slot = 1
    state.battlefield = [lembas]
    lembas_sac(state, lembas)
    _drive(state)
    assert registry.CARD_DEFS["Lembas"] in state.library and not any(c.name == "Lembas" for c in state.graveyard)
    assert state.life_total == 23  # the sac ability's own +3 life


def test_lembas_dies_does_not_shuffle_a_different_same_named_graveyard_copy():
    """A different, already-in-graveyard copy of Lembas is not the one
    shuffled back -- only the exact dying instance is."""
    state = GameState(on_the_play=True)
    other_lembas_in_gy = state.new_instance(registry.CARD_DEFS["Lembas"])
    state.graveyard = [other_lembas_in_gy]
    lembas2 = Permanent(registry.CARD_DEFS["Lembas"])
    lembas2.slot = 1
    state.battlefield = [lembas2]
    lembas_sac(state, lembas2)
    _drive(state)
    assert registry.CARD_DEFS["Lembas"] in state.library
    assert other_lembas_in_gy in state.graveyard  # untouched -- the OTHER copy stayed put


def test_chromatic_star_any_color_choice_and_dies_draw():
    """Chromatic Star's sac ability adds one mana of any color (never
    colorless); its dies trigger draws a card."""
    state = GameState(on_the_play=True)
    star = Permanent(registry.CARD_DEFS["Chromatic Star"])
    star.slot = 1
    state.battlefield = [star]
    state.library = [CardDef("Draw Me", CardType.LAND, None, EffectId.FILLER)]
    chromatic_star_mana(state, star)
    assert star not in state.battlefield  # sacrificed
    assert state.pending_resolution["kind"] == "choose_mana_color"
    assert set(resolution.choose_mana_color_options(state)) == {"W", "U", "B", "R", "G"}  # any color, never {C}
    resolution.execute_choose_mana_color(state, "U")
    assert state.mana_pool.get("U") == 1 and "C" not in state.mana_pool  # a real blue, not colorless
    assert state.mana_pool_single_pip == {"U": 1}  # a 1-symbol event -- tagged single-pip
    _drive(state)  # the dies-trigger (draw) was queued by the sacrifice
    assert len(state.hand) == 1  # dies-draw fired


def test_tron_lands_double_mana_when_all_three_controlled():
    """Controlling all three Tron lands makes each one tap for a doubled
    or tripled amount; controlling fewer produces the plain single {C}."""
    state = GameState(on_the_play=True)
    mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    plant = Permanent(registry.CARD_DEFS["Urza's Power Plant"])
    tower = Permanent(registry.CARD_DEFS["Urza's Tower"])
    state.battlefield = [mine, plant, tower]
    activate_mana_source(state, mine)
    assert state.mana_pool == {"C": 2}  # all three online -- Mine doubles
    assert state.mana_pool_single_pip == {}  # a 2-symbol event -- never single-pip-tagged
    activate_mana_source(state, tower)
    assert state.mana_pool == {"C": 5}  # + Tower's own tripled {C}{C}{C}
    assert state.mana_pool_single_pip == {}  # a 3-symbol event -- never single-pip-tagged

    solo_state = GameState(on_the_play=True)
    solo_mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    solo_state.battlefield = [solo_mine]
    activate_mana_source(solo_state, solo_mine)
    assert solo_state.mana_pool == {"C": 1}  # alone -- just a plain single C
    assert solo_state.mana_pool_single_pip == {"C": 1}  # a 1-symbol event -- tagged, same as any land

    two_state = GameState(on_the_play=True)
    two_mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    two_plant = Permanent(registry.CARD_DEFS["Urza's Power Plant"])
    two_state.battlefield = [two_mine, two_plant]
    activate_mana_source(two_state, two_mine)
    assert two_state.mana_pool == {"C": 1}  # only two of three -- still just one C
    # same source, tagged differently here vs above -- tag tracks the actual per-event yield
    assert two_state.mana_pool_single_pip == {"C": 1}


def test_tocasia_dig_site_taps_for_colorless():
    """Tocasia's Dig Site's plain ability taps for {C}."""
    state = GameState(on_the_play=True)
    dig_site = Permanent(registry.CARD_DEFS["Tocasia's Dig Site"])
    state.battlefield = [dig_site]
    activate_mana_source(state, dig_site)
    assert state.mana_pool == {"C": 1} and dig_site.tapped


def test_tocasia_dig_site_surveil_ability():
    """The {3}, T: Surveil 1 ability puts its effect on the stack. Also
    checks the {T} cost is logged as a tap_or_untap event."""
    state = GameState(on_the_play=True, event_log=[])
    dig_site = Permanent(registry.CARD_DEFS["Tocasia's Dig Site"])
    state.battlefield = [dig_site]
    state.library = [CardDef("Top Card", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_tocasia_dig_site_surveil(state, dig_site)
    assert dig_site.tapped  # {T} -- a cost, paid now
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Tocasia's Dig Site", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == state.active_idx
    assert len(state.stack) == 1  # the surveil is the effect, on the stack
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "surveil"


def test_conduit_pylons_etb_surveils_one():
    """Conduit Pylons' ETB surveils 1."""
    state = GameState(on_the_play=True)
    pylons = registry.CARD_DEFS["Conduit Pylons"]
    state.hand = [pylons]
    state.library = [CardDef("Top Card", CardType.LAND, None, EffectId.FOREST, basic=True)]
    play_land_from_hand(state, pylons)
    _resolve_etb(state)
    assert state.pending_resolution["kind"] == "surveil"


def test_conduit_pylons_taps_for_colorless():
    """Conduit Pylons' plain ability taps for {C}."""
    state = GameState(on_the_play=True)
    pylons = Permanent(registry.CARD_DEFS["Conduit Pylons"])
    state.battlefield = [pylons]
    activate_mana_source(state, pylons)
    assert state.mana_pool == {"C": 1} and pylons.tapped


def test_conduit_pylons_filter_any_color_gated_on_untapped():
    """The {1}: Add one mana of any color filter ability shares Pylons'
    tap with its plain {C} ability, so it's legal only while Pylons is
    untapped; paying its {1} taps Pylons immediately."""
    actions = drl_env.build_action_table([("Conduit Pylons", 1)], registry.EFFECT_REGISTRY)
    pylons = Permanent(registry.CARD_DEFS["Conduit Pylons"])
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1  # main-phase-only timing gate needs a phase set
    state.battlefield = [pylons]
    state.mana_pool = {"U": 1}
    _, legal, execute = next((nm, lg, ex) for nm, lg, ex in actions if nm == "Filter Conduit Pylons, paying U")
    assert legal(state)
    execute(state)
    assert state.mana_pool == {}  # U spent as the cost; no output produced yet
    assert pylons.tapped  # paying the {1} taps Pylons itself, immediately
    assert state.mana_subdecision is not None and state.mana_subdecision["stage"] == "choose_color"

    _, _produce_g_legal, produce_g_execute = next((nm, lg, ex) for nm, lg, ex in actions if nm == "Produce G")
    produce_g_execute(state)
    assert state.mana_pool == {"G": 1}
    assert state.mana_subdecision is None

    # reset the sweep-scoped filter cache by hand since this bypasses a real sweep
    drl_env._actions_mana._filter_source_cache = None
    _, legal_again, _ = next((nm, lg, ex) for nm, lg, ex in actions if nm == "Filter Conduit Pylons, paying G")
    assert not legal_again(state)  # now tapped -- the filter's own gate closes


def test_bonders_ornament_draw_ability():
    """The {4}, T: draw a card ability puts the draw on the stack. Also
    checks the {T} cost is logged as a tap_or_untap event."""
    state = GameState(on_the_play=True, event_log=[])
    ornament = Permanent(registry.CARD_DEFS["Bonder's Ornament"])
    state.battlefield = [ornament]
    state.library = [CardDef("Drawn", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_bonders_ornament_draw(state, ornament)
    assert ornament.tapped  # {T} -- a cost, paid immediately
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Bonder's Ornament", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == state.active_idx
    assert len(state.stack) == 1 and state.hand == []
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Drawn"]


def test_candy_trail_etb_scry_two():
    """Candy Trail's ETB scries 2."""
    state = GameState(on_the_play=True)
    candy_trail = CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    )
    state.hand = [candy_trail]
    state.library = [
        CardDef("Top", CardType.LAND, None, EffectId.FOREST, basic=True),
        CardDef("Second", CardType.LAND, None, EffectId.MOUNTAIN, basic=True),
    ]
    cast_permanent_from_hand(state, candy_trail)
    _resolve_etb(state)
    assert state.pending_resolution["kind"] == "scry"
    assert len(state.pending_resolution["remaining"]) == 2  # scry TWO, not scry one


def test_twisted_landscape_cycling_draws_a_card():
    """Cycling discards Twisted Landscape and draws a card."""
    state = GameState(on_the_play=True)
    twisted = CardDef(
        "Twisted Landscape", CardType.LAND, None, EffectId.TWISTED_LANDSCAPE,
        fetch_ability_cost={}, cycling_cost={"B": 1, "R": 1, "G": 1},
    )
    state.hand = [twisted]
    state.library = [CardDef("Drawn", CardType.LAND, None, EffectId.FOREST, basic=True)]
    cycle_twisted_landscape(state, twisted)
    assert [c.name for c in state.graveyard] == ["Twisted Landscape"]  # discarded itself
    assert [c.name for c in state.hand] == ["Drawn"]  # the draw rider


def test_lembas_etb_scry_then_draw():
    """Lembas's ETB scries 1, then draws a card."""
    state = GameState(on_the_play=True)
    lembas = registry.CARD_DEFS["Lembas"]
    state.hand = [lembas]
    state.library = [CardDef("Scried", CardType.LAND, None, EffectId.FOREST, basic=True)]
    cast_permanent_from_hand(state, lembas)
    _resolve_etb(state)
    assert state.pending_resolution["kind"] == "scry"
    resolution.execute_scry_surveil_option(state, "keep")  # keep it on top
    assert state.pending_resolution is None
    assert [c.name for c in state.hand] == ["Scried"]  # scry kept it on top -- the draw picked it right back up


def test_clue_token_sac_pay_and_draw():
    """Clue's sac ability draws a card; as a token it never visits the
    graveyard."""
    state = GameState(on_the_play=True)
    clue = Permanent(CLUE_TOKEN_CARD_DEF)
    state.battlefield = [clue]
    state.library = [CardDef("Drawn", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_clue_sac(state, clue)
    assert clue not in state.battlefield
    assert not any(c.name == "Clue" for c in state.graveyard)  # a TOKEN -- ceases to exist, never a graveyard trip
    assert len(state.stack) == 1 and state.hand == []
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Drawn"]


def test_expedition_map_activate_completes_the_search_to_hand():
    """Activating Expedition Map searches the library for a land and puts
    it into hand."""
    state = GameState(on_the_play=True)
    emap = Permanent(registry.CARD_DEFS["Expedition Map"])
    state.battlefield = [emap]
    state.library = [
        CardDef("Nonland", CardType.ARTIFACT, {"generic": 1}, EffectId.FILLER),
        CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True),
    ]
    activate_expedition_map(state, emap)
    assert emap not in state.battlefield  # sacrificed -- a cost, paid immediately
    assert state.graveyard[-1].name == "Expedition Map"
    assert len(state.stack) == 1  # the search is the effect, on the stack
    resolve_top_of_stack(state)
    assert resolution.search_fetch_options(state) == ["Forest"]  # only the land is offered
    resolution.execute_search_fetch_option(state, "Forest")
    assert [c.name for c in state.hand] == ["Forest"]  # actually landed in hand, not left mid-search


def test_lotus_petal_mana_any_color_choice_and_consumed():
    """Lotus Petal adds one mana of any color, chosen at activation, and
    is sacrificed to the graveyard (a real card)."""
    state = GameState(on_the_play=True)
    petal = Permanent(registry.CARD_DEFS["Lotus Petal"])
    state.battlefield = [petal]
    activate_mana_source(state, petal, color_choice="G")
    assert state.mana_pool.get("G") == 1 and "C" not in state.mana_pool  # a real green, not colorless
    assert petal not in state.battlefield  # consumed -- sacrificed, not merely tapped
    assert state.graveyard[-1].name == "Lotus Petal"


def test_pinnacle_kill_ship_cast_auto_fires_etb_trigger():
    """Casting Pinnacle Kill-Ship from hand fires its ETB trigger
    automatically."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    kill_ship_def = registry.CARD_DEFS["Pinnacle Kill-Ship"]
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.hand = [kill_ship_def]
    cast_permanent_from_hand(state, kill_ship_def)
    promote_triggers_to_stack(state)
    assert state.pending_resolution["kind"] == "choose_any_target"  # the ETB genuinely fired off the real cast/entry
    resolution.execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield  # 10 damage landed -- the real chain ran end to end


def test_treasure_token_mana_any_color_choice_and_ceases_to_exist():
    """Treasure adds one mana of any color and ceases to exist (a token)
    rather than visiting the graveyard."""
    state = GameState(on_the_play=True)
    treasure = Permanent(TREASURE_TOKEN_CARD_DEF)
    state.battlefield = [treasure]
    activate_mana_source(state, treasure, color_choice="R")
    assert state.mana_pool.get("R") == 1 and "C" not in state.mana_pool  # a real red, not colorless
    assert treasure not in state.battlefield
    assert not any(c.name == "Treasure" for c in state.graveyard)  # ceases to exist -- never a graveyard trip


def test_map_token_own_wiring_creature_choice_cost_and_sorcery_speed_gate():
    """Map's activated ability (Fanatical Offering): {1}+sac, target
    creature you control explores, sorcery-speed only."""
    state = GameState(on_the_play=True)
    creature = Permanent(CardDef("My Creature", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    map_token = Permanent(MAP_TOKEN_CARD_DEF)
    swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP))
    state.battlefield = [creature, map_token, swamp]
    state.library = [CardDef("Fetched Land", CardType.LAND, None, EffectId.FOREST, basic=True)]
    state.phase = Phase.MAIN1

    actions = drl_env.build_action_table([], registry.EFFECT_REGISTRY, token_card_defs=(MAP_TOKEN_CARD_DEF,))
    _, legal, execute = next((nm, lg, ex) for nm, lg, ex in actions if nm == "Activate Map (explore)")

    state.stack.append({})  # mid-resolution of ANYTHING -- Speed.SORCERY requires an empty stack
    assert not legal(state)
    state.stack.clear()
    # legal even with nothing floating -- payment is produced during cost-payment
    assert legal(state)  # own MAIN1, empty stack, a source that can still pay {1}
    activate_mana_source(state, swamp)
    assert legal(state)  # and still legal once it IS floating

    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    execute_pool_spend(state, pool_spend_options(state)[0])  # pay the {1}
    assert map_token not in state.battlefield  # sacrificed -- a cost
    assert not any(c.name == "Map" for c in state.graveyard)  # a TOKEN -- ceases, never a graveyard trip
    assert resolution.choose_permanent_options(state) == [("My Creature", 1)]  # the creature-choice step -- only the creature offered
    resolution.execute_choose_permanent_option(state, "My Creature", 1)
    assert len(state.stack) == 1  # explore is the effect, on the stack
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Fetched Land"]  # explore actually ran, against the chosen creature
