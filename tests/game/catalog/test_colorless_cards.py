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
    activate_candy_trail_sac,
    activate_relic_of_progenitus_draw,
    activate_relic_of_progenitus_exile,
    activate_twisted_landscape_fetch,
    cast_boulderbranch_prototype,
    cast_maelstrom_colossus,
    chromatic_star_mana,
    cycle_ash_barrens,
    lembas_sac,
    nihil_spellbomb_sac,
    pinnacle_kill_ship_etb,
)
from game.effects.casting import cast_permanent_from_hand
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import sacrifice_to_graveyard
from game.effects.stats import has_keyword, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.mana import execute_pool_spend, pool_spend_options
from game.state import GameState, Permanent, PlayerState


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
