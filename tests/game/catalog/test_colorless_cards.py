"""Tests for game.catalog.colorless_cards. See the module under test for
the card-implementation rationale (real-rules citations, etc.) each test
below guards."""

import contextlib
import io

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
    """Test helper: drive a just-queued ETB (faithful timing queues it
    rather than firing inline) onto the stack and resolve it, the way
    game.turn's own priority round does in real play."""
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)


def _drive(state):
    """Test helper: promote any queued triggers to the stack and resolve
    everything until the stack is empty (chained ETB/dies triggers)."""
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)


def test_ash_barrens_basic_landcycling():
    """Basic landcycling {1}: discard this card from hand, search library
    for a basic land -- a real model choice between Forest and Plains
    (unlike Generous Ent's own forestcycle, which always searches "Forest"
    specifically), put into hand, shuffle. No draw-a-card rider (unlike a
    plain Cycling ability) -- verified via Scryfall, not guessed."""
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
    """{5}, {T}, Sacrifice this artifact: it deals 5 damage to target
    creature -- the ability's own real text was previously missing
    entirely (only the {1}: add mana filter ability existed). Sacrifice is
    a COST, paid immediately on activation; the 5 damage waits on the
    stack for its locked target, fizzling per 608.2b if that creature is
    gone by resolution."""
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
    """A MANDATORY "target creature": the ability can't be activated at
    all with zero legal creature targets on board (601.2c/602.2b), even
    though the {5} mana would otherwise be payable."""
    state = GameState(on_the_play=True)
    barrels = Permanent(registry.CARD_DEFS["Barrels of Blasting Jelly"])
    state.battlefield = [barrels]
    extra_legal = registry.EFFECT_REGISTRY[EffectId.BARRELS_OF_BLASTING_JELLY]["activated_abilities"]["blast"]["extra_legal"]
    assert not extra_legal(state)


def test_candy_trail_sac_gain_life_and_draw():
    """Candy Trail's sac ability: gain 3 life AND draw a card -- both
    halves of its real text, not just the draw."""
    state = GameState(on_the_play=True)
    candy_trail = Permanent(CardDef(
        "Candy Trail", CardType.ARTIFACT, {"generic": 1}, EffectId.CANDY_TRAIL, sac_ability_cost={"generic": 2},
    ))
    state.battlefield = [candy_trail]
    state.library = [CardDef("Forest", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_candy_trail_sac(state, candy_trail)
    assert state.battlefield == []  # sacrificed -- a cost, paid immediately on activation
    # gain 3 + draw are the ability's EFFECT -- now on the stack (faithful
    # timing), not applied the instant the cost was paid.
    assert len(state.stack) == 1 and state.life_total == 20 and state.hand == []
    resolve_top_of_stack(state)
    assert state.life_total == 23  # STARTING_LIFE (20) + 3, once the effect resolves
    assert [c.name for c in state.hand] == ["Forest"]


def _relic_of_progenitus_state():
    """Shared single-player setup for Relic of Progenitus's two
    independent abilities: the repeatable {T} target-player exile, and the
    one-shot {1}+exile-self "exile ALL graveyards, draw a card" ability."""
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
    """The repeatable {T} ability is a REAL target-player choice
    (resolution.begin_choose_target_player), not a self-target assumed on
    the ability's behalf: "yourself" is always one legal choice (true in
    real Magic even alone), chosen explicitly here via drl_env's own fixed
    "Target: yourself" action equivalent. Whichever player is targeted
    then chooses one of THEIR OWN graveyard cards to exile -- made by THAT
    player themselves via the active_idx flip.

    Faithful timing + cross-player choice: the {T} is a COST (paid now);
    the target player is chosen as the ability is put on the stack
    (targets lock at activation); only the EFFECT waits on the stack."""
    state, relic = _relic_of_progenitus_state()
    activate_relic_of_progenitus_exile(state, relic)
    assert relic.tapped  # {T} -- a cost, paid immediately on activation
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
    """The one-shot {1}+exile-self draw ability also clears every
    graveyard first (real text: "exile ALL graveyards"), not just draw --
    "all graveyards" loops every PlayerState (not just the active one),
    so a card left over from something else entirely (here, standing in
    for whatever the repeatable exile ability didn't already remove) is
    exiled too.

    Faithful timing: exiling this artifact is a COST (paid now); exiling
    all graveyards and drawing are the effect, so they go on the stack and
    resolve after a priority window. "Exile all graveyards" is measured at
    resolution (the effect reads state.players' graveyards then), matching
    real Magic."""
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
    """A real opponent exists -- targeting them reaches into THEIR
    graveyard, never the active player's own."""
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
    """Rooftop Percher's ETB ("exile up to two target cards from
    graveyards. You gain 3 life.") with NO graveyard cards at all -- the
    up-to-2 pick auto-completes empty and the ability just gains 3 (the
    unconditional co-effect). Percher's own full (a)-(d) targeting
    scenarios get a dedicated block below; Boulderbranch Golem gets its
    own dedicated ETB test next."""
    state = GameState(on_the_play=True)
    percher = CardDef("Rooftop Percher", CardType.CREATURE, {"generic": 5}, EffectId.ROOFTOP_PERCHER, power=3, toughness=3)
    state.hand = [percher]
    cast_permanent_from_hand(state, percher)
    _resolve_etb(state)  # ETB (target-at-promotion): empty graveyard -> no exile targets, gain 3
    assert state.life_total == 23  # STARTING_LIFE (20) + 3


def test_boulderbranch_golem_etb_gains_life_equal_to_power():
    """Boulderbranch Golem: real ETB gain-life trigger -- "gain life equal
    to its power" (6, in its normal {7} mode)."""
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
    """Rooftop Percher's ETB, in full: "exile up to two target cards from
    graveyards. You gain 3 life." Targets chosen at PROMOTION
    (target-at-promotion, project_targeted_triggered_abilities), from
    EITHER player's graveyard ("from graveyardS"). Exercises two
    same-named copies in the SAME graveyard (distinct instances, not a
    by-name pick) plus a card from the opponent's own graveyard."""
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
    """Partial fizzle (608.2c): pick both, one leaves before resolution --
    only the survivor is exiled, life is still gained."""
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
    """All exile targets gone -> exile does nothing, but +3 life STILL
    happens.
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
    """The "up to" slack: two available, take one, explicitly decline the
    second."""
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
    """Boulderbranch Golem Prototype {3}{G} -- 3/3: a second cast option
    (own cheaper cost, smaller stats, own ETB "gain life = its power" = 3).
    Cast the prototype from hand; the {7} hand card is consumed (found by
    NAME, not identity, same as Sagu Wildling's own Omen creature half), a
    3/3 enters, and life goes up by 3 (not the normal mode's 6)."""
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
    """Maelstrom Colossus's real Cascade: exile cards from the top of the
    library until a nonland card with mana value LESS than this card's own
    (8) is exiled; cast it without paying its mana cost, then put the rest
    on the bottom in a random order (real text: "the exiled cards," i.e.
    every OTHER card seen along the way, not the hit itself).

    "Cast it for free" reuses the hit card's own registry "cast" spec
    completely unchanged -- CardDefs are shared/interned per name
    (registry.CARD_DEFS holds one per distinct name, not per physical
    copy), so temporarily inserting it into state.hand first makes every
    existing resolve's own hand-removal correctly find and remove it, with
    no parallel "cast from library" implementation needed for any card.

    Hit: a plain permanent (cast_permanent_from_hand), mana value 2 (< 8)
    -- exiled cards after it go to the bottom (this engine's library is
    already a shuffled abstraction, so "bottom" just means "back in
    state.library"), Colossus itself enters last."""
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
        # Real Cascade only exiles cards UP TO AND INCLUDING the hit --
        # "Never Seen" was still sitting below it, never revealed/exiled
        # at all, so it stays exactly where it was; only what was actually
        # exiled alongside the hit ("A Land") gets shuffled and returned,
        # appended after it.
        assert [c.name for c in state.library] == ["Never Seen", "A Land"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_maelstrom_colossus_cascade_may_cast_decline_bottoms_hit():
    """DECLINE the may-cast: the hit is NOT cast; it bottoms with the
    rest. Real Cascade is "you MAY cast it.\""""
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
    """Whiff: nothing eligible (everything's either a land or costs 8+) --
    Colossus still enters, nothing else does."""
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
    """extra_legal fails: Cascade only waives the MANA cost, not other
    preconditions -- skipped, straight to the bottom with the rest, same
    as a genuine whiff (same distinction Plot's own docstring already
    draws for Highway Robbery)."""
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
    """The hit's own resolve opens a further pending resolution of its own
    (an Ancient-Stirrings-like take-it-or-decline) -- Colossus's own
    battlefield entry can't happen yet, chained onto that resolution's own
    on_complete instead of running immediately, so it lands only once the
    cascaded card's entire effect, decisions included, is actually done
    (matching real Magic: the Cascade trigger resolves completely before
    Colossus, still the next thing up on the stack, ever does)."""
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
    """Pinnacle Kill-Ship: Station is a REAL per-creature choice (unlike
    Saruli Caretaker's fungible-by-name tap) -- charge counters equal to
    whichever creature actually gets tapped (its own power matters, so
    this can't reuse Saruli's own "any untapped creature, arbitrarily"
    simplification), and the animate threshold
    (Permanent.type_override/stats._animate_spec) they feed once 7+ are
    reached.

    Faithful timing: tapping another creature is a COST, paid now on
    activation; putting the charge counters (and the animate check) is the
    effect, so it goes on the stack and resolves after a priority window."""
    state = GameState(on_the_play=True)
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
    # Placing the charge counters is the EFFECT -- on the stack now (faithful
    # timing), applied only on resolution.
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
    """ETB trigger: "deals 10 damage to up to one target creature." Faithful:
    the trigger's EFFECT goes on the stack -- up to one target creature on
    EITHER battlefield (hexproof/shroud aware) is chosen as the trigger is
    put on the stack (begin_choose_any_target, optional=True for "up to
    one"), the 10 damage waits on the stack, then hits that exact creature
    at resolution. Target selection at ETB time is when the trigger goes
    on the stack (603.3d) -- there's no state-based step between entering
    and this that could change the legal targets.

    Here: target an opponent creature -> 10 damage at resolution (lethal
    via SBA)."""
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
    """Up-to-one: decline -> the ability resolves doing nothing."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    bystander = Permanent(CardDef("Bystander", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    bystander.slot = 1
    state.players[1].battlefield = [bystander]
    pinnacle_kill_ship_etb(state)
    resolution.execute_choose_any_target_decline(state)
    resolve_top_of_stack(state)
    assert bystander in state.players[1].battlefield and bystander.damage_marked == 0


def test_pinnacle_kill_ship_etb_fizzles_if_target_leaves_before_resolution():
    """Fizzle: the chosen creature is gone before the ETB effect resolves
    (608.2b)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    doomed = Permanent(CardDef("Doomed", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3))
    doomed.slot = 1
    state.players[1].battlefield = [doomed]
    pinnacle_kill_ship_etb(state)
    resolution.execute_choose_any_target_creature(state, 1, "Doomed", 1)
    state.players[1].battlefield = []  # exiled/bounced before resolution
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        resolve_top_of_stack(state)
    assert "fizzle" in log.getvalue().lower()


def test_twisted_landscape_sacrifice_fetch_basic_tapped():
    """Twisted Landscape: "{T}, Sacrifice this land: search library for a
    basic Swamp, Mountain, or Forest, put it onto the battlefield TAPPED,
    then shuffle." The {T} (untapped requirement enforced generically by
    drl_env._activate_legal) and the sacrifice are COSTS, paid now; the
    search is the effect, so it goes on the stack and resolves after a
    priority window -- same shape as Expedition Map, except the fetch
    lands on the battlefield tapped (force_tapped) rather than in hand."""
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
    """G7: Myr Enforcer affinity -- {7} reduced by the number of artifacts
    you control."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(CardDef("Great Furnace", CardType.LAND, None, EffectId.GREAT_FURNACE, artifact=True)) for _ in range(4)]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Myr Enforcer"])
    assert eff["generic"] == 3, eff  # 7 - 4 artifacts
    state.battlefield = []
    assert drl_env._effective_cast_cost(state, registry.CARD_DEFS["Myr Enforcer"])["generic"] == 7  # no artifacts -> full


def test_ichor_wellspring_etb_and_dies_both_draw():
    """G8 artifact value engine: ETB and dies both "draw a card.\""""
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
    """Nihil Spellbomb: sac -> "{T}, Sacrifice: Exile target player's
    graveyard" (the sacrifice fires its dies-trigger; the exile of the
    chosen player's whole graveyard is the effect, on the stack); dies ->
    "When put into a graveyard from the battlefield, you may pay {B}. If
    you do, draw a card," resolved by the controller (active player at
    this trigger's resolution)."""
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
    """REGRESSION: nihil_spellbomb_dies's "may pay {B}, draw" choice belongs
    to Nihil Spellbomb's own CONTROLLER (its owner) -- NOT necessarily
    whoever happens to be state.active_idx when the trigger actually
    resolves. The OLD (buggy) implementation read `payer = state.active_idx`
    directly inside nihil_spellbomb_dies; the fix instead reads
    `permanent.flags["owner_idx"]`, stashed by state_based._destroy_creature/
    sacrifice_to_graveyard the instant the permanent actually left the
    battlefield -- see that function's own docstring.

    Called DIRECTLY (not through nihil_spellbomb_sac -> promote_triggers_
    to_stack -> resolve_top_of_stack): driving it through that real pipeline
    can't actually distinguish the two versions here, because resolve_top_
    of_stack always resets state.active_idx to the stack entry's own
    stamped "controller" -- and promote_triggers_to_stack stamps that
    controller while active_idx is briefly flipped to the trigger's true
    owner during promotion (game.effects.triggers._place_trigger_groups),
    so by the time any ltb_trigger hook actually runs, state.active_idx is
    ALREADY the correct owner regardless of which line nihil_spellbomb_dies
    itself reads. The bug is only reachable exactly the way it was filed:
    a direct call with active_idx pointing at the wrong player.

    Player 1 (NOT the active player, active_idx=0) owns the Spellbomb.
    Player 0 is ALSO given a floating {B} (and both players a library card,
    so a draw either way can't crash on an empty library), so the OLD
    buggy code -- which would open the "may pay" prompt against whoever
    state.active_idx happens to be, player 0 -- has something to wrongly
    pay with instead of correctly refusing/crashing; the assertions below
    would cleanly FAIL against it.

    (Only the "who gets asked / whose mana pays" half is asserted here --
    the fixed code's own resulting `state.draw(1)` runs from whatever
    active_idx resolution.pay_unless_pay's own payment-complete callback
    restores it to, which is state.active_idx as it stood BEFORE this
    direct call -- always the true owner in any real, un-bypassed
    resolve_top_of_stack-driven game, but genuinely a separate, narrower
    question from "who was offered the choice / whose mana was spent" in
    this deliberately pipeline-bypassing direct-call construction, so it's
    not asserted here -- see this file's own final report for the full
    note.)"""
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
    assert state.active_idx == 1  # begin_pay_unless flips active_idx to the PAYER -- must be the owner (1), not player 0
    resolution.pay_unless_pay(state)
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert state.players[1].mana_pool == {} and state.players[0].mana_pool == {"B": 1}  # PLAYER 1's own B spent; player 0's untouched


def test_lembas_dies_shuffles_itself_into_library():
    """Lembas: "When put into a graveyard from the battlefield, its owner
    shuffles it into their library." A 400.7 linked ability -- driven
    through the real sacrifice path (lembas_sac -> sacrifice_to_graveyard),
    so the exact-instance tracking (graveyard_instance stashed on the
    dying permanent's own flags) is exercised for real, not
    hand-simulated. Sacrificing fires its dies-trigger (shuffle itself
    back into the library) -- so Lembas recurs -- alongside the sac
    ability's own +3 life."""
    state = GameState(on_the_play=True)
    lembas = Permanent(registry.CARD_DEFS["Lembas"])
    lembas.slot = 1
    state.battlefield = [lembas]
    lembas_sac(state, lembas)
    _drive(state)
    assert registry.CARD_DEFS["Lembas"] in state.library and not any(c.name == "Lembas" for c in state.graveyard)
    assert state.life_total == 23  # the sac ability's own +3 life


def test_lembas_dies_does_not_shuffle_a_different_same_named_graveyard_copy():
    """A second same-named Lembas already sitting in the graveyard must NOT
    be the one shuffled back -- the EXACT dying instance is tracked (a
    400.7 linked ability), not any same-named graveyard card."""
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
    """Chromatic Star: "{1},{T},Sacrifice: Add one mana of ANY color" -- the
    activator picks the color (all five offered, never {C}); the dies-draw
    still fires. (The {1} is paid by drl_env's cost_key wiring in real
    play; here the resolve is called directly, post-cost.)"""
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
    _drive(state)  # the dies-trigger (draw) was queued by the sacrifice
    assert len(state.hand) == 1  # dies-draw fired


def test_tron_lands_double_mana_when_all_three_controlled():
    """Tron lands' ("tron",) mana shape: once you control all three of
    Urza's Mine/Power Plant/Tower, EACH one taps for its own doubled
    amount -- Mine/Power Plant for {C}{C}, Tower for {C}{C}{C} -- instead
    of the ordinary single {C}. Controlling only one or two of them
    produces just the plain single {C}, same as any other land."""
    state = GameState(on_the_play=True)
    mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    plant = Permanent(registry.CARD_DEFS["Urza's Power Plant"])
    tower = Permanent(registry.CARD_DEFS["Urza's Tower"])
    state.battlefield = [mine, plant, tower]
    activate_mana_source(state, mine)
    assert state.mana_pool == {"C": 2}  # all three online -- Mine doubles
    activate_mana_source(state, tower)
    assert state.mana_pool == {"C": 5}  # + Tower's own tripled {C}{C}{C}

    solo_state = GameState(on_the_play=True)
    solo_mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    solo_state.battlefield = [solo_mine]
    activate_mana_source(solo_state, solo_mine)
    assert solo_state.mana_pool == {"C": 1}  # alone -- just a plain single C

    two_state = GameState(on_the_play=True)
    two_mine = Permanent(registry.CARD_DEFS["Urza's Mine"])
    two_plant = Permanent(registry.CARD_DEFS["Urza's Power Plant"])
    two_state.battlefield = [two_mine, two_plant]
    activate_mana_source(two_state, two_mine)
    assert two_state.mana_pool == {"C": 1}  # only two of three -- still just one C


def test_tocasia_dig_site_taps_for_colorless():
    """Tocasia's Dig Site: plain "{T}: Add {C}" mana ability -- shares its
    {T} with the {3}, T: Surveil 1 ability below (verified via Scryfall:
    "{T}: Add {C}. {3}, {T}: Surveil 1.", no enters-tapped clause)."""
    state = GameState(on_the_play=True)
    dig_site = Permanent(registry.CARD_DEFS["Tocasia's Dig Site"])
    state.battlefield = [dig_site]
    activate_mana_source(state, dig_site)
    assert state.mana_pool == {"C": 1} and dig_site.tapped


def test_tocasia_dig_site_surveil_ability():
    """Tocasia's Dig Site's own second ability: "{3}, T: Surveil 1."
    Faithful timing: the {T} is a COST, paid now; the surveil is the
    effect, on the stack, resolving after a priority window."""
    state = GameState(on_the_play=True)
    dig_site = Permanent(registry.CARD_DEFS["Tocasia's Dig Site"])
    state.battlefield = [dig_site]
    state.library = [CardDef("Top Card", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_tocasia_dig_site_surveil(state, dig_site)
    assert dig_site.tapped  # {T} -- a cost, paid now
    assert len(state.stack) == 1  # the surveil is the effect, on the stack
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "surveil"


def test_conduit_pylons_etb_surveils_one():
    """Conduit Pylons ETB: "surveil 1" (a real Scryfall pull, per this
    module's own docstring)."""
    state = GameState(on_the_play=True)
    pylons = registry.CARD_DEFS["Conduit Pylons"]
    state.hand = [pylons]
    state.library = [CardDef("Top Card", CardType.LAND, None, EffectId.FOREST, basic=True)]
    play_land_from_hand(state, pylons)
    _resolve_etb(state)
    assert state.pending_resolution["kind"] == "surveil"


def test_conduit_pylons_taps_for_colorless():
    """Conduit Pylons: plain "{T}: Add {C}" -- a "fixed" mana source,
    distinct from its own {1} filter ability below."""
    state = GameState(on_the_play=True)
    pylons = Permanent(registry.CARD_DEFS["Conduit Pylons"])
    state.battlefield = [pylons]
    activate_mana_source(state, pylons)
    assert state.mana_pool == {"C": 1} and pylons.tapped


def test_conduit_pylons_filter_any_color_gated_on_untapped():
    """Conduit Pylons' own second ability -- "{1}: Add one mana of any
    color" (filter_mana): a two-step conversion (pay the {1} immediately,
    then choose the output color via the shared choose_color
    mana_subdecision stage -- see drl_env._actions._filter_mana_execute),
    legal only while Pylons itself is still UNTAPPED (real text shares the
    land's own {T} with its plain {T}: Add {C} ability, so only one of the
    two can be used a turn), and paying its {1} taps Pylons immediately --
    distinct from Barrels of Blasting Jelly's own once-per-turn
    used_this_turn flag gating the identical shape (see this module's own
    docstring)."""
    actions = drl_env.build_action_table([("Conduit Pylons", 1)], registry.EFFECT_REGISTRY)
    pylons = Permanent(registry.CARD_DEFS["Conduit Pylons"])
    state = GameState(on_the_play=True)
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

    # This test drives legal()/execute() directly rather than through a real
    # legal_action_mask sweep, so the sweep-scoped _filter_source_cache
    # (drl_env._actions, reset before/after every real sweep) must be reset
    # by hand here -- otherwise it would keep serving the stale, pre-tap
    # lookup from the "paying U" check above.
    drl_env._actions._filter_source_cache = None
    _, legal_again, _ = next((nm, lg, ex) for nm, lg, ex in actions if nm == "Filter Conduit Pylons, paying G")
    assert not legal_again(state)  # now tapped -- the filter's own gate closes


def test_bonders_ornament_draw_ability():
    """Bonder's Ornament's own "{4}, T: draw a card" -- shares its {T}
    with its own plain flexible mana ability (already tested elsewhere,
    not duplicated here). Faithful timing: the {T} is a COST, paid now;
    the draw is the effect, on the stack, resolving after a priority
    window."""
    state = GameState(on_the_play=True)
    ornament = Permanent(registry.CARD_DEFS["Bonder's Ornament"])
    state.battlefield = [ornament]
    state.library = [CardDef("Drawn", CardType.LAND, None, EffectId.FOREST, basic=True)]
    activate_bonders_ornament_draw(state, ornament)
    assert ornament.tapped  # {T} -- a cost, paid immediately
    assert len(state.stack) == 1 and state.hand == []
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Drawn"]


def test_candy_trail_etb_scry_two():
    """Candy Trail's ETB: scry 2 (its own separate sac ability -- gain 3
    life AND draw a card -- is already covered above; this only exercises
    the ETB half, never exercised anywhere else)."""
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
    """Cycling {B}{R}{G}: discard this card from hand, draw a card (plain
    Cycling has the draw rider, unlike Ash Barrens' own Basic Landcycling
    above). Its fetch ability is already covered elsewhere in this file --
    not duplicated here."""
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
    """Lembas ETB: scry 1, THEN draw a card (the draw chained onto the
    scry's own completion) -- driven through the real cast/ETB path,
    unlike the file's other two Lembas tests, which both invoke
    lembas_sac directly and never touch the ETB at all."""
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
    """Clue's own resolve: "{2}, Sacrifice this artifact: Draw a card"
    (Investigate makes one -- Toxin Analysis). The {2} mana is handled
    generically by drl_env's cost_key wiring in real play; activate_clue_
    sac is called directly here, post-cost, same convention as Chromatic
    Star's/Candy Trail's own sac-ability tests above."""
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
    """Expedition Map's own {2}, T, Sacrifice: search library for a land --
    the existing coverage elsewhere in this file only confirms the ability
    fires a Gixian-Infiltrator sacrifice trigger; this drives the search
    all the way to completion, landing the found card in hand."""
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
    """Lotus Petal: "{T}, Sacrifice this artifact: Add one mana of any
    color" -- a real color-fixing choice (mana.py's own "flexible" mana
    kind), made by the activator AT ACTIVATION (unlike Chromatic Star's
    own color choice above, which opens a resolution AFTER its
    sac-triggered dies-draw) -- mirrors that same test's own shape, for
    Lotus Petal specifically. Consumed -- sacrificed to the graveyard (a
    real card, unlike Treasure's own token version below)."""
    state = GameState(on_the_play=True)
    petal = Permanent(registry.CARD_DEFS["Lotus Petal"])
    state.battlefield = [petal]
    activate_mana_source(state, petal, color_choice="G")
    assert state.mana_pool.get("G") == 1 and "C" not in state.mana_pool  # a real green, not colorless
    assert petal not in state.battlefield  # consumed -- sacrificed, not merely tapped
    assert state.graveyard[-1].name == "Lotus Petal"


def test_pinnacle_kill_ship_cast_auto_fires_etb_trigger():
    """Cast Pinnacle Kill-Ship from hand through the real cast path -- the
    ETB trigger (pinnacle_kill_ship_etb) must fire as an automatic
    consequence of casting/entering, not merely when called directly (each
    half is already unit-tested in isolation elsewhere in this file)."""
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
    """Treasure: "{T}, Sacrifice this artifact: Add one mana of any color"
    -- consumed on tap like Lotus Petal, but a TOKEN: it ceases to exist
    entirely rather than making a graveyard trip."""
    state = GameState(on_the_play=True)
    treasure = Permanent(TREASURE_TOKEN_CARD_DEF)
    state.battlefield = [treasure]
    activate_mana_source(state, treasure, color_choice="R")
    assert state.mana_pool.get("R") == 1 and "C" not in state.mana_pool  # a real red, not colorless
    assert treasure not in state.battlefield
    assert not any(c.name == "Treasure" for c in state.graveyard)  # ceases to exist -- never a graveyard trip


def test_map_token_own_wiring_creature_choice_cost_and_sorcery_speed_gate():
    """Map's own "{1}, {T}, Sacrifice this token: Target creature you
    control explores. Activate only as a sorcery." (Fanatical Offering) --
    exercising Map's OWN wiring through the real drl_env action-table gate:
    its real {1}+sac cost (paid through the generic cost_key machinery,
    same as every other Activate action), the creature-choice step, and
    the Speed.SORCERY timing gate. explore()'s own land/nonland branching
    logic is already tested generically via direct calls elsewhere -- not
    duplicated here."""
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
    assert not legal(state)  # empty stack now, but nothing floated yet -- still not payable
    activate_mana_source(state, swamp)  # float {1}'s worth BEFORE activating (float-first)
    assert legal(state)  # own MAIN1, empty stack, {1} now floating -- legal

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
