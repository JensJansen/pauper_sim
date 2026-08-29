"""Tests for game.catalog.black_cards."""

import pytest

import drl_env
import game

from game import registry
from game import resolution
from game.cards import CardDef, CardType, EffectId
from game.catalog.black_cards import (
    _nonblack,
    blood_fountain_return,
    cast_alms_of_the_vein,
    cast_cast_down,
    cast_dread_return,
    cast_eviscerators_insight,
    cast_fanatical_offering,
    cast_reckoners_bargain,
    cast_snuff_out,
    cast_snuff_out_alt,
    cast_toxin_analysis,
    cast_unexpected_fangs,
    cast_vampires_kiss,
    flashback_dread_return,
    flashback_eviscerators_insight,
    lotleth_giant_etb,
    refurbished_familiar_etb,
    snuff_out_alt_legal,
)
from game.effects import madness_and_plot
from game.effects.casting import enters_battlefield
from game.effects.combat import combat_damage_step, creature_attack_eligible, declare_attacker, declare_attackers_step
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import check_state_based_actions, cleanup_step, sacrifice_to_graveyard
from game.effects.stats import has_keyword, lifelink_count, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, execute_pool_spend, mana_output, pool_spend_options
from game.state import GameState, Permanent, PlayerState
from game.turn import Phase, _run_priority_round_gen


# --- Dread Return: target locked at cast; fizzles if the target leaves the
# graveyard before it resolves. ---

_DREAD_RETURN = CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN)
_GRIZZLY_BEARS = CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2)


def _dread_return_cast_and_target():
    """Hard-cast Dread Return and lock its reanimation target (a graveyard
    creature card, by object identity)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [_DREAD_RETURN]
    grizzly_inst = state.new_instance(_GRIZZLY_BEARS)
    state.graveyard = [grizzly_inst]
    cast_dread_return(state, _DREAD_RETURN)
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, grizzly_inst)
    return state, grizzly_inst


def test_dread_return_hard_cast_reanimates_target():
    """Hard cast reanimates the chosen creature card."""
    state, grizzly_inst = _dread_return_cast_and_target()
    assert state.hand == [] and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert any(c.name == "Dread Return" for c in state.graveyard)
    assert any(p.card_def is _GRIZZLY_BEARS for p in state.battlefield)  # reanimated
    assert all(c.name != "Grizzly Bears" for c in state.graveyard)


def test_dread_return_fizzles_if_target_leaves_graveyard():
    """Fizzles if the target leaves the graveyard before it resolves."""
    state, grizzly_inst = _dread_return_cast_and_target()
    state.graveyard.remove(grizzly_inst)
    state.event_log = []
    resolve_top_of_stack(state)
    assert any(e["kind"] == "target_fizzle" for e in state.event_log)
    assert any(c.name == "Dread Return" for c in state.graveyard)
    assert not any(p.card_def is _GRIZZLY_BEARS for p in state.battlefield)  # nothing reanimated


def test_bojuka_bog_etb_exiles_target_players_graveyard():
    """Bojuka Bog enters tapped; its ETB targets a player and exiles their
    graveyard, never the caster's own."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    bog = CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG)
    enters_battlefield(state, bog, from_zone="hand")
    bog_perm = next(p for p in state.battlefield if p.card_def.name == "Bojuka Bog")
    assert bog_perm.tapped  # enters tapped
    assert state.pending_resolution is None
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert state.players[1].graveyard == []  # their graveyard exiled
    assert [c.name for c in state.players[0].graveyard] == ["Mine"]  # own graveyard untouched


# --- Mesmeric Fiend: ETB exiles a nonland from the opponent's hand; the
# Fiend's leaves-the-battlefield trigger returns it, whether the Fiend dies
# or is sacrificed. ---

_MESMERIC_FIEND_DEF = CardDef(
    "Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1,
)


def _enter_fiend_and_exile():
    """The Fiend enters and its ETB exiles a nonland card from the
    opponent's hand, tracked and linked to this Fiend."""
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    a_spell = CardDef("Their Spell", CardType.SORCERY, {"B": 1}, EffectId.FILLER)
    a_land = CardDef("Their Land", CardType.LAND, None, EffectId.SWAMP)
    st.players[1].hand = [a_spell, a_land]
    fiend = enters_battlefield(st, _MESMERIC_FIEND_DEF, from_zone="hand")
    assert [e["type"] for e in st.trigger_queue] == ["etb"]
    promote_triggers_to_stack(st)
    resolve_top_of_stack(st)  # ETB resolves -> choose a nonland from the opponent's hand
    assert st.pending_resolution["kind"] == "choose_graveyard_card"
    fiend_opts = resolution.choose_graveyard_card_options(st)
    assert [c.name for c in fiend_opts] == ["Their Spell"]  # the LAND is excluded (nonland only)
    resolution.execute_choose_graveyard_card_option(st, fiend_opts[0])
    assert [c.name for c in st.players[1].hand] == ["Their Land"]  # nonland exiled from their hand
    assert fiend.flags["mesmeric_exiled"][0] is a_spell  # tracked, linked to this Fiend
    return st, fiend, a_spell


def test_mesmeric_fiend_dies_returns_exiled_card_to_owner():
    """The Fiend dying returns the exiled card to its owner's hand."""
    state, fiend, a_spell = _enter_fiend_and_exile()
    fiend.damage_marked = _MESMERIC_FIEND_DEF.extra["toughness"]  # lethal
    check_state_based_actions(state)
    assert fiend not in state.players[0].battlefield  # dead
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]  # LTB queued on leave
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand  # returned to its OWNER's hand
    assert "mesmeric_exiled" not in fiend.flags  # linkage consumed


def test_mesmeric_fiend_sacrificed_returns_exiled_card_to_owner():
    """Sacrificing the Fiend also returns the exiled card."""
    state, fiend, a_spell = _enter_fiend_and_exile()
    resolution.begin_sacrifice(state, lambda p: p.card_def.name == "Mesmeric Fiend", 1, on_complete=lambda s, ok: None)
    assert state.pending_resolution["kind"] == "choose_permanent"
    resolution.execute_choose_permanent_option(state, "Mesmeric Fiend", fiend.slot)
    assert fiend not in state.players[0].battlefield
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand


def test_mesmeric_fiend_solo_no_opponent_etb_is_noop():
    """No opponent to target -> ETB does nothing, no exile."""
    solo = GameState(on_the_play=True)
    solo_fiend = enters_battlefield(solo, _MESMERIC_FIEND_DEF, from_zone="hand")
    promote_triggers_to_stack(solo)
    assert solo.stack == [] and solo.pending_resolution is None and "mesmeric_exiled" not in solo_fiend.flags


def test_balustrade_spy_etb_mills_target_players_library():
    """ETB targets a player and mills their library until a land turns up
    (inclusive), never the caster's own."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].library = [CardDef("Own1", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].library = [
        CardDef("Their1", CardType.CREATURE, None, EffectId.FILLER),
        CardDef("Their2", CardType.LAND, None, EffectId.SWAMP, basic=True),
        CardDef("Their3", CardType.CREATURE, None, EffectId.FILLER),
    ]
    spy = CardDef("Balustrade Spy", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.BALUSTRADE_SPY, power=2, toughness=2)
    enters_battlefield(state, spy, from_zone="hand")
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent
    resolve_top_of_stack(state)
    assert [c.name for c in state.players[1].library] == ["Their3"]  # milled through the land, inclusive
    assert [c.name for c in state.players[1].graveyard] == ["Their1", "Their2"]
    assert [c.name for c in state.players[0].library] == ["Own1"]  # own library untouched


def test_blood_fountain_returns_up_to_two_targets_with_partial_fizzle():
    """{3}{B}, {T}, Sacrifice: return up to two target creature cards from
    the graveyard to hand, both targets locked at activation; if one leaves
    the graveyard in response, the other still comes back. Also checks the
    {T} cost is logged, tapping before the sacrifice's zone_move."""
    state = GameState(on_the_play=True, event_log=[])
    fountain = Permanent(CardDef(
        "Blood Fountain", CardType.ARTIFACT, {"B": 1}, EffectId.BLOOD_FOUNTAIN, sac_ability_cost={"generic": 3, "B": 1},
    ))
    fountain.slot = 1
    bear = state.new_instance(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    wolf = state.new_instance(CardDef("Wolf", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    state.battlefield = [fountain]
    state.graveyard = [bear, wolf]
    blood_fountain_return(state, fountain)
    assert state.battlefield == []  # sacrificed -- a cost, paid immediately on activation
    tap_idx = next(i for i, e in enumerate(state.event_log) if e["kind"] == "tap_or_untap")
    sac_idx = next(i for i, e in enumerate(state.event_log) if e["kind"] == "zone_move" and e.get("from_zone") == "battlefield")
    assert tap_idx < sac_idx  # tapping first, THEN sacrificing
    tap_event = state.event_log[tap_idx]
    assert tap_event["permanent"] == ["Blood Fountain", 1]
    assert tap_event["now_tapped"] is True
    assert tap_event["owner_idx"] == state.active_idx
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, bear)
    resolution.execute_choose_graveyard_card_option(state, wolf)
    assert state.pending_resolution is None and len(state.stack) == 1
    state.graveyard.remove(bear)  # bear becomes an illegal target before resolution
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Wolf"]  # only the surviving target returns


def test_alms_of_the_vein_target_opponent_loses_life_caster_gains():
    """Target opponent loses 3 life, caster gains 3."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].life_total = 20
    state.players[1].life_total = 20
    state.hand = [registry.CARD_DEFS["Alms of the Vein"]]
    assert registry.EFFECT_REGISTRY[EffectId.ALMS_OF_THE_VEIN]["cast"]["extra_legal"](state)
    cast_alms_of_the_vein(state, registry.CARD_DEFS["Alms of the Vein"])
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 23  # caster gained 3
    assert state.players[1].life_total == 17  # opponent lost 3


def test_alms_of_the_vein_uncastable_with_no_opponent():
    """No opponent -> no legal target -> can't be cast."""
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.ALMS_OF_THE_VEIN]["cast"]["extra_legal"](state)


def test_vampires_kiss_targets_any_player_and_grants_life():
    """Targets any player; caster gains 2 life and two Blood tokens, and the
    targeted player loses 2 life."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].life_total = 20
    state.players[1].life_total = 20
    state.hand = [registry.CARD_DEFS["Vampire's Kiss"]]
    cast_vampires_kiss(state, registry.CARD_DEFS["Vampire's Kiss"])
    assert state.pending_resolution["kind"] == "choose_target_player"
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent
    resolve_top_of_stack(state)
    assert state.players[0].life_total == 22  # +2 gained
    assert state.players[1].life_total == 18  # -2 lost
    assert sum(1 for p in state.battlefield if p.card_def.name == "Blood") == 2


# --- G3 removal & tricks (mono-black) ---

def _g3_two():
    return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])


def _pay_cost_fully(state):
    """Drain a "pay_cost" pending resolution via the first available
    pool-spend option each iteration."""
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        execute_pool_spend(state, pool_spend_options(state)[0])


def test_cast_down_destroys_target_creature():
    """Cast Down: {1}{B}, destroy target nonlegendary creature. No legendary
    creature exists in this pool, so "nonlegendary" is every creature (the
    default eligible set)."""
    state = _g3_two()
    victim = Permanent(CardDef("Kitchen Imp", CardType.CREATURE, {"generic": 3, "B": 1}, EffectId.KITCHEN_IMP, power=2, toughness=2))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[0].hand = [registry.CARD_DEFS["Cast Down"]]
    cast_cast_down(state, registry.CARD_DEFS["Cast Down"])
    resolution.execute_choose_any_target_creature(state, 1, "Kitchen Imp", 1)
    resolve_top_of_stack(state)
    assert victim not in state.players[1].battlefield


def test_snuff_out_nonblack_targeting_and_alt_cost():
    """Only targets nonblack creatures; the 4-life alt cost requires a
    Swamp, >= 4 life, and a legal nonblack target."""
    assert _nonblack(Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1)))
    assert not _nonblack(Permanent(CardDef("B Creature", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1)))
    state = _g3_two()
    swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)))
    swamp.slot = 1
    tgt = Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1))
    tgt.slot = 1
    state.players[0].battlefield = [swamp]
    state.players[1].battlefield = [tgt]
    state.players[0].hand = [registry.CARD_DEFS["Snuff Out"]]
    state.players[0].life_total = 20
    assert snuff_out_alt_legal(state)
    cast_snuff_out_alt(state, registry.CARD_DEFS["Snuff Out"])
    assert state.players[0].life_total == 16  # paid 4 life
    resolution.execute_choose_any_target_creature(state, 1, "R Creature", 1)
    resolve_top_of_stack(state)
    assert tgt not in state.players[1].battlefield


def test_unexpected_fangs_grants_counter_and_lifelink():
    """Puts a +1/+1 counter and a lifelink counter on the target."""
    state = _g3_two()
    c = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    c.slot = 1
    state.players[0].battlefield = [c]
    state.players[0].hand = [registry.CARD_DEFS["Unexpected Fangs"]]
    cast_unexpected_fangs(state, registry.CARD_DEFS["Unexpected Fangs"])
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    resolve_top_of_stack(state)
    assert permanent_power(state, c) == 2 and permanent_toughness(state, c) == 2
    assert has_keyword(state, c, "lifelink") and lifelink_count(state, c) == 1


def test_toxin_analysis_grants_deathtouch_lifelink_and_clue_then_cleanup_clears():
    """Target creature gains deathtouch and lifelink until end of turn, and
    a Clue token is created; the keywords clear at cleanup."""
    state = _g3_two()
    c = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    c.slot = 1
    state.players[0].battlefield = [c]
    state.players[0].hand = [registry.CARD_DEFS["Toxin Analysis"]]
    cast_toxin_analysis(state, registry.CARD_DEFS["Toxin Analysis"])
    resolution.execute_choose_any_target_creature(state, 0, "Bear", 1)
    resolve_top_of_stack(state)
    assert has_keyword(state, c, "deathtouch") and has_keyword(state, c, "lifelink")
    assert any(p.card_def.name == "Clue" for p in state.players[0].battlefield)
    cleanup_step(state)
    assert not has_keyword(state, c, "deathtouch") and not has_keyword(state, c, "lifelink")


# --- G7: grixis_affinity / jund_wildfire ---

def test_refurbished_familiar_affinity_cost_reduction():
    """Refurbished Familiar affinity ({3}{B}, -1 per artifact you control)."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]), Permanent(registry.CARD_DEFS["Great Furnace"])]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Refurbished Familiar"])
    assert eff["generic"] == 1, eff  # 3 - 2 artifacts


def test_refurbished_familiar_etb_opponent_empty_hand_draws_instead():
    """Opponent's hand is empty -> caster draws instead of them discarding."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].hand = []
    state.players[0].library = [CardDef("Top", CardType.LAND, None, EffectId.SWAMP)]
    refurbished_familiar_etb(state)
    assert len(state.players[0].hand) == 1  # drew (opponent couldn't discard)


def test_gurmag_angler_delve_exiles_graveyard_cards_to_pay_generic():
    """Delve N exiles N graveyard cards and pays {6-N}{B} for the rest."""
    dl = [("Gurmag Angler", 4), ("Swamp", 8)]
    byname = {
        a[0]: (a[1], a[2])
        for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)
    }
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Gurmag Angler"]]
    state.graveyard = [
        state.new_instance(CardDef("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
        state.new_instance(CardDef("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
    ]
    state.battlefield = [Permanent(registry.CARD_DEFS["Swamp"]) for _ in range(5)]  # {6}{B} minus delve 2 = 5 mana
    for sw in state.battlefield:
        activate_mana_source(state, sw)
    legal, execute = byname["Cast Gurmag Angler"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "choose_delve_amount"
    delve_legal, delve_execute = byname["Delve 2"]
    assert delve_legal(state)
    delve_execute(state)
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])
    _pay_cost_fully(state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []


def test_no_mana_tap_during_delve_exile_step():
    """No mana ability may be activated during delve's exile-to-graveyard
    sub-cost, which sits between announcing the cast and payment opening."""
    dl = [("Gurmag Angler", 4), ("Contaminated Aquifer", 8)]
    byname = {
        a[0]: (a[1], a[2])
        for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)
    }
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Gurmag Angler"]]
    state.graveyard = [
        state.new_instance(CardDef("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
        state.new_instance(CardDef("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
    ]
    # 5 Contaminated Aquifers (each U-or-B), none pre-floated: payment is
    # still wide open when delve starts.
    state.battlefield = [Permanent(registry.CARD_DEFS["Contaminated Aquifer"]) for _ in range(5)]

    tap_for_u_legal, _ = byname["Tap Contaminated Aquifer for U"]
    assert tap_for_u_legal(state), "tapping is an ordinary main-phase mana ability before any cast starts"

    legal, execute = byname["Cast Gurmag Angler"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "choose_delve_amount"
    assert not tap_for_u_legal(state), "601.2f: no mana ability is legal once a cast is announced, even before the exile step opens"

    delve_legal, delve_execute = byname["Delve 2"]
    assert delve_legal(state)
    delve_execute(state)
    assert state.pending_resolution["kind"] == "choose_graveyard_card" and state.pending_resolution["mid_cast"] is True
    assert not tap_for_u_legal(state), (
        "the actual bug: tapping a dual for U here (instead of B) can strand the delve-reduced B "
        "cost that opens next -- mid_cast must refuse every mana ability for this pending's duration"
    )

    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])
    assert state.pending_resolution["kind"] == "pay_cost"
    assert tap_for_u_legal(state), "payment is open now -- ordinary mana-ability timing (605.1a) resumes"

    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        options = pool_spend_options(state)
        if options:
            execute_pool_spend(state, options[0])
        else:
            # Nothing floated yet -- tap a source for whichever color is still owed.
            still_owed = state.pending_resolution["remaining"]
            color = "B" if still_owed.get("B", 0) > 0 else "U"
            byname[f"Tap Contaminated Aquifer for {color}"][1](state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []


# --- G8 sac engines (grixis_affinity / jund_wildfire) ---

def _drive8(s):
    promote_triggers_to_stack(s)
    # Simultaneous triggers for the same controller need an explicit order choice.
    while s.pending_resolution is not None and s.pending_resolution["kind"] == "order_triggers":
        resolution.execute_order_triggers_option(s, resolution.order_triggers_options(s)[0])
    while s.stack:
        resolve_top_of_stack(s)


def test_gixian_infiltrator_gains_counter_on_sacrifice():
    """"Whenever you sacrifice another permanent, put a +1/+1 counter on
    this creature" -- a real triggered ability, not applied immediately."""
    state = GameState(on_the_play=True)
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    fodder = Permanent(registry.CARD_DEFS["Ichor Wellspring"])
    fodder.slot = 1
    state.battlefield = [gix, fodder]
    state.library = [CardDef("x", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    sacrifice_to_graveyard(state, fodder)
    _drive8(state)
    assert gix.counters.get("+1/+1") == 1


def test_fanatical_offering_sac_draws_two_and_creates_map():
    """{1}{B}, sacrifice an artifact or creature: draw two, create a Map token."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Fanatical Offering"]]
    state.library = [CardDef(f"n{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    state.battlefield = [fodder]
    cast_fanatical_offering(state, registry.CARD_DEFS["Fanatical Offering"])
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    _drive8(state)
    assert any(p.card_def.name == "Map" for p in state.battlefield) and len(state.hand) == 2


def test_reckoners_bargain_gains_life_equal_to_mana_value_and_draws_two():
    """{1}{B}, sacrifice an artifact or creature: gain life equal to its
    mana value, draw two."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Reckoner's Bargain"]]
    state.library = [CardDef(f"m{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    big = Permanent(CardDef("Big", CardType.CREATURE, {"generic": 4, "B": 1}, EffectId.FILLER, power=3, toughness=3))  # MV 5
    big.slot = 1
    state.battlefield = [big]
    cast_reckoners_bargain(state, registry.CARD_DEFS["Reckoner's Bargain"])
    resolution.execute_choose_permanent_option(state, "Big", 1)
    _drive8(state)
    assert state.life_total == 25 and len(state.hand) == 2  # +5 MV, drew 2


def test_refurbished_familiar_cross_player_discard_routing():
    """"Each opponent discards a card" is the opponent's own choice: the ETB
    flips active_idx to the opponent for the forced discard, then restores it."""
    rf = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    rf.turn_player_idx = 0
    rf.active_idx = 0  # caster is the turn/active player
    rf.players[1].hand = [registry.CARD_DEFS["Swamp"], registry.CARD_DEFS["Gurmag Angler"]]
    refurbished_familiar_etb(rf)
    assert rf.active_idx == 1, "the OPPONENT (seat 1), not the caster, must be active for their own discard"
    assert rf.pending_resolution["kind"] == "discard"
    resolution.execute_discard_option(rf, "Swamp")  # answered as seat 1 -- the opponent picks their own discard
    assert rf.active_idx == 0, "active_idx restored to the caster once the opponent's forced choice is made"
    assert [c.name for c in rf.players[1].graveyard] == ["Swamp"]


def test_balustrade_spy_has_flying():
    """Balustrade Spy has flying (regression: registry entry once lacked it)."""
    state = GameState(on_the_play=True)
    spy = Permanent(registry.CARD_DEFS["Balustrade Spy"])
    state.battlefield = [spy]
    assert has_keyword(state, spy, "flying")


def test_vault_of_whispers_taps_for_black():
    """{T}: Add {B}."""
    state = GameState(on_the_play=True)
    vault = Permanent(registry.CARD_DEFS["Vault of Whispers"])
    state.battlefield = [vault]
    assert mana_output(vault, state) == ["B"]
    activate_mana_source(state, vault)
    assert state.mana_pool == {"B": 1} and vault.tapped
    assert state.mana_pool_single_pip == {"B": 1}  # a 1-symbol event -- tagged single-pip


def test_bojuka_bog_taps_for_black():
    """{T}: Add {B}."""
    state = GameState(on_the_play=True)
    bog = Permanent(registry.CARD_DEFS["Bojuka Bog"])
    bog.tapped = False  # real Bojuka Bog enters tapped; force untapped to isolate the mana ability
    state.battlefield = [bog]
    assert mana_output(bog, state) == ["B"]
    activate_mana_source(state, bog)
    assert state.mana_pool == {"B": 1} and bog.tapped
    assert state.mana_pool_single_pip == {"B": 1}  # a 1-symbol event -- tagged single-pip


def test_lotleth_giant_etb_damages_opponent_per_graveyard_creature():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[1].life_total = 20
    state.graveyard = [
        CardDef("c1", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1),
        CardDef("c2", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1),
        CardDef("c3", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1),
        CardDef("Not A Creature", CardType.SORCERY, {"generic": 1}, EffectId.FILLER),  # doesn't count
    ]
    lotleth_giant_etb(state)
    assert state.players[1].life_total == 17  # 3 creature cards -> 3 damage, sorcery excluded


def test_dread_return_flashback_sacrifices_three_creatures_and_reanimates():
    """Flashback: sacrifice three creatures instead of paying mana. The
    sacrificed creatures land in the graveyard before the reanimation
    target is chosen, so they're eligible targets too -- this picks the
    pre-existing Grizzly Bears instead, to keep the assertion unambiguous."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    dread_return_inst = state.new_instance(_DREAD_RETURN)
    grizzly_inst = state.new_instance(_GRIZZLY_BEARS)
    state.graveyard = [dread_return_inst, grizzly_inst]
    fodder = [
        Permanent(CardDef(f"Fodder{i}", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
        for i in range(3)
    ]
    state.battlefield = fodder
    assert registry.EFFECT_REGISTRY[EffectId.DREAD_RETURN]["flashback"]["legal"](state)  # >= 3 creatures

    flashback_dread_return(state, dread_return_inst)
    assert dread_return_inst not in state.graveyard  # left the graveyard the moment Flashback was chosen
    assert state.pending_resolution["kind"] == "choose_permanent"
    for _ in range(3):
        name, slot = resolution.choose_permanent_options(state)[0]
        resolution.execute_choose_permanent_option(state, name, slot)
    assert state.battlefield == []  # all 3 fodder sacrificed

    assert state.pending_resolution["kind"] == "choose_graveyard_card"  # reanimation target, now that the cost is paid
    resolution.execute_choose_graveyard_card_option(state, grizzly_inst)
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert any(p.card_def is _GRIZZLY_BEARS for p in state.battlefield)  # reanimated
    assert grizzly_inst not in state.graveyard
    assert all(c.name != "Dread Return" for c in state.graveyard)  # exiled by Flashback, never re-graveyarded


def test_dread_return_uncastable_with_no_creature_card_in_graveyard():
    """No creature card anywhere in the graveyard -> illegal to hard-cast."""
    state = GameState(on_the_play=True)
    state.graveyard = [CardDef("Not A Creature", CardType.SORCERY, {"generic": 1}, EffectId.FILLER)]
    assert not registry.EFFECT_REGISTRY[EffectId.DREAD_RETURN]["cast"]["extra_legal"](state)


def test_kitchen_imp_has_flying_and_haste():
    """Flying and haste; haste lets a summoning-sick Kitchen Imp attack and
    deal damage the turn it enters."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    imp = Permanent(registry.CARD_DEFS["Kitchen Imp"])
    assert imp.summoning_sick  # just entered
    state.battlefield = [imp]
    assert has_keyword(state, imp, "flying")
    declare_attackers_step(state)
    assert creature_attack_eligible(state, imp)  # haste overrides summoning sickness
    declare_attacker(state, imp)
    combat_damage_step(state)
    assert state.players[1].life_total == 18  # 20 - Kitchen Imp's power (2)


def test_kitchen_imp_madness_cast_from_discard():
    """Madness {B}: discarding Kitchen Imp exiles it and queues a
    cast-or-decline decision; casting puts it on the battlefield straight
    from exile, never touching hand or graveyard."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Kitchen Imp"]]
    state.battlefield = [Permanent(registry.CARD_DEFS["Swamp"])]
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: None)
    resolution.execute_discard_option(state, "Kitchen Imp")
    assert state.exile and state.exile[0][0].name == "Kitchen Imp"  # exiled, not graveyarded
    assert state.trigger_queue and state.trigger_queue[0]["kind"] == "madness"
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "madness_decision"
    madness_and_plot.execute_madness_cast(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    activate_mana_source(state, state.battlefield[0])  # float {B}
    execute_pool_spend(state, "B")
    assert state.pending_resolution is None and state.exile == []
    while state.stack:
        resolve_top_of_stack(state)
    assert any(p.card_def.name == "Kitchen Imp" for p in state.battlefield)


def test_alms_of_the_vein_madness_cast_from_discard():
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].life_total = 20
    state.players[1].life_total = 20
    state.hand = [registry.CARD_DEFS["Alms of the Vein"]]
    state.battlefield = [Permanent(registry.CARD_DEFS["Swamp"])]
    resolution.begin_discard(state, 1, optional=False, on_complete=lambda s, cards: None)
    resolution.execute_discard_option(state, "Alms of the Vein")
    assert state.exile and state.exile[0][0].name == "Alms of the Vein"
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "madness_decision"
    madness_and_plot.execute_madness_cast(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    activate_mana_source(state, state.battlefield[0])
    execute_pool_spend(state, "B")
    assert state.pending_resolution is None and state.exile == []
    while state.stack:
        resolve_top_of_stack(state)
    assert state.players[0].life_total == 23  # caster gained 3
    assert state.players[1].life_total == 17  # opponent lost 3
    assert any(c.name == "Alms of the Vein" for c in state.graveyard)  # a sorcery -> graveyard, not exiled


def test_blood_fountain_etb_creates_blood_token():
    state = GameState(on_the_play=True)
    fountain_def = CardDef(
        "Blood Fountain", CardType.ARTIFACT, {"B": 1}, EffectId.BLOOD_FOUNTAIN, sac_ability_cost={"generic": 3, "B": 1},
    )
    state.hand = [fountain_def]
    enters_battlefield(state, fountain_def, from_zone="hand")
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Blood" for p in state.battlefield)


def test_blood_fountain_activate_via_action_table_pays_real_mana_cost():
    """Activating through the real action table gates on and pays the
    ability's {3}{B} cost."""
    dl = [("Blood Fountain", 4), ("Swamp", 8)]
    byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    fountain = Permanent(registry.CARD_DEFS["Blood Fountain"])
    state.battlefield = [fountain] + [Permanent(registry.CARD_DEFS["Swamp"]) for _ in range(4)]
    for sw in state.battlefield:
        if sw.card_def.name == "Swamp":
            activate_mana_source(state, sw)  # pre-float 4 B before activating ({3}{B} = 4 mana)
    bear = state.new_instance(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    state.graveyard = [bear]

    legal, execute = byname["Activate Blood Fountain (return)"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 3, "B": 1}
    _pay_cost_fully(state)

    assert fountain not in state.battlefield  # {T} + Sacrifice paid on resolve
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, bear)
    assert state.pending_resolution is None and len(state.stack) == 1  # only 1 eligible target -> selection auto-closed
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Bear"]


def test_fanatical_offering_uncastable_with_no_sac_fodder():
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.FANATICAL_OFFERING]["cast"]["extra_legal"](state)


def test_reckoners_bargain_uncastable_with_no_sac_fodder():
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.RECKONERS_BARGAIN]["cast"]["extra_legal"](state)


def test_eviscerators_insight_sac_draws_two():
    """{1}{B}, sacrifice an artifact or creature: draw two, no other effect."""
    state = GameState(on_the_play=True)
    state.hand = [registry.CARD_DEFS["Eviscerator's Insight"]]
    state.library = [CardDef(f"n{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    state.battlefield = [fodder]
    cast_eviscerators_insight(state, registry.CARD_DEFS["Eviscerator's Insight"])
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    _drive8(state)
    assert len(state.hand) == 2
    assert not any(p.card_def.name == "Map" for p in state.battlefield)


def test_eviscerators_insight_extra_legal_both_branches():
    empty = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.EVISCERATORS_INSIGHT]["cast"]["extra_legal"](empty)
    fodder_present = GameState(on_the_play=True)
    fodder_present.battlefield = [Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))]
    assert registry.EFFECT_REGISTRY[EffectId.EVISCERATORS_INSIGHT]["cast"]["extra_legal"](fodder_present)


def test_sac_cost_fodder_dying_mid_payment_crashes_the_resolve():
    """PROOF of the 2026-08-27 production crash's underlying mechanism (see
    traceback at logs_main_league.log around the jund_wildfire iter-19
    failure), isolated from game.turn._run_priority_round_gen's own SBA
    scheduling (see test_sac_cost_fodder_survives_mid_payment_via_the_
    real_priority_loop below for the fix, driven through that real loop).

    game.begin_pay_cost's multi-step payment window pays one pip per agent
    action (game.mana.execute_pool_spend); production dispatches each pip
    through _run_priority_round_gen, whose loop calls
    check_state_based_actions(state) before every action. This test calls
    that same check_state_based_actions directly (bypassing the loop
    entirely, including its now-added state.casting_depth guard) to isolate
    the actual failure this repo's fix targets: a creature that's the
    caster's ONLY sac fodder when Eviscerator's Insight is cast (extra_legal
    passes) dying to a state-based action (lethal damage marked by
    combat/burn) in the window between the spell's own mana-payment
    sub-actions -- BEFORE the additional-cost sacrifice ever gets chosen.
    begin_choose_permanent then auto-fizzles with None per its own
    documented contract (handlers_targeting.py: "fizzles with None if
    nothing matches"), and _sac_artifact_or_creature's `_on_chosen` doesn't
    handle that documented case: `name, slot = choice` raises TypeError on
    None."""
    state = GameState(on_the_play=True)
    card_def = registry.CARD_DEFS["Eviscerator's Insight"]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    swamp1 = Permanent(registry.CARD_DEFS["Swamp"])
    swamp2 = Permanent(registry.CARD_DEFS["Swamp"])
    state.battlefield = [fodder, swamp1, swamp2]
    assert registry.EFFECT_REGISTRY[EffectId.EVISCERATORS_INSIGHT]["cast"]["extra_legal"](state)  # legal to cast

    activate_mana_source(state, swamp1)
    activate_mana_source(state, swamp2)
    assert state.mana_pool == {"B": 2}  # {1}{B} fully coverable

    def _after_pay(s):
        game.on_cast_trigger(s, card_def)
        cast_eviscerators_insight(s, card_def)
    game.begin_pay_cost(state, card_def.cast_cost, on_complete=_after_pay)

    execute_pool_spend(state, "B")  # pays the {B} pip; {1} generic still owed -- cost not yet satisfied
    assert state.pending_resolution["kind"] == "pay_cost"

    # The window real _run_priority_round_gen always opens between this pip
    # and the next: lethal combat/burn damage lands on the caster's only
    # creature, and (absent the casting_depth guard, bypassed here on
    # purpose) an unconditional SBA check would claim it before the
    # caster's next action.
    fodder.damage_marked = permanent_toughness(state, fodder)
    check_state_based_actions(state)
    assert fodder not in state.battlefield  # genuinely removed by the real SBA path, not test bookkeeping

    with pytest.raises(TypeError, match="cannot unpack non-iterable NoneType"):
        execute_pool_spend(state, "B")  # last pip -> cost satisfied -> resolve() -> crash


def test_sac_cost_fodder_survives_mid_payment_via_the_real_priority_loop():
    """Regression for the fix: driven through the REAL
    game.turn._run_priority_round_gen (not the isolated pieces the test
    above exercises), the same lethal-damage-mid-payment race must NOT
    crash, because state.casting_depth (incremented by begin_pay_cost,
    decremented by push_to_stack -- see both docstrings) keeps the loop's
    check_state_based_actions/refizzle_if_now_targetless suppressed for the
    spell's WHOLE cast, not just its pay_cost sub-window: mana payment,
    AND cast_eviscerators_insight's own follow-up sacrifice choice, which
    opens only after pay_cost completes and is exactly where an SBA
    suppressed only during "pay_cost" would still have struck (proven by
    hand before landing on the casting_depth-spanning fix below)."""
    state = GameState(on_the_play=True)
    state.turn_player_idx = 0
    card_def = registry.CARD_DEFS["Eviscerator's Insight"]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    swamp1 = Permanent(registry.CARD_DEFS["Swamp"])
    swamp2 = Permanent(registry.CARD_DEFS["Swamp"])
    state.battlefield = [fodder, swamp1, swamp2]
    activate_mana_source(state, swamp1)
    activate_mana_source(state, swamp2)

    def _cast():
        def _after_pay(s):
            game.on_cast_trigger(s, card_def)
            cast_eviscerators_insight(s, card_def)
        game.begin_pay_cost(state, card_def.cast_cost, on_complete=_after_pay)

    gen = _run_priority_round_gen(state)
    next(gen)  # primes the round: first (no-op) SBA check, first yield
    gen.send(_cast)  # announces the cast, opens "pay_cost" -- casting_depth == 1
    gen.send(lambda: execute_pool_spend(state, "B"))  # pays the {B} pip

    # Same production race as the crash repro above, but now inside the
    # real loop: lethal damage lands on the caster's only creature between
    # two actions of the SAME spell's cast.
    fodder.damage_marked = permanent_toughness(state, fodder)

    # Last pip -> cost satisfied -> resolve() -> _sac_artifact_or_creature
    # opens "choose_permanent" -- still inside this one send(), the loop
    # loops back and would (pre-fix) immediately re-check SBAs against the
    # freshly-opened choose_permanent and refizzle it to None. No crash.
    gen.send(lambda: execute_pool_spend(state, "B"))

    assert state.pending_resolution["kind"] == "choose_permanent"
    assert fodder in state.battlefield  # protected through the whole cast, not just pay_cost
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    assert fodder not in state.battlefield  # sacrificed as intended, not claimed by the SBA first
    assert state.stack and state.stack[-1]["card_def"] is card_def  # spell fully cast, sitting on the stack
    assert state.casting_depth == 0  # balanced back down once push_to_stack committed it


def test_eviscerators_insight_flashback_sac_draws_two_and_exiles():
    """Flashback {4}{B} + the sac-fodder cost: draw two, then exile the
    card instead of returning it to the graveyard."""
    state = GameState(on_the_play=True)
    insight_inst = state.new_instance(registry.CARD_DEFS["Eviscerator's Insight"])
    state.graveyard = [insight_inst]
    state.library = [CardDef(f"n{i}", CardType.LAND, None, EffectId.SWAMP, basic=True) for i in range(5)]
    fodder = Permanent(CardDef("Fodder", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    fodder.slot = 1
    state.battlefield = [fodder]
    assert registry.EFFECT_REGISTRY[EffectId.EVISCERATORS_INSIGHT]["flashback"]["legal"](state)  # sac fodder payable

    flashback_eviscerators_insight(state, insight_inst)
    assert insight_inst not in state.graveyard  # left the graveyard the moment Flashback was chosen
    resolution.execute_choose_permanent_option(state, "Fodder", 1)
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)
    assert len(state.hand) == 2
    assert all(c.name != "Eviscerator's Insight" for c in state.graveyard)  # exiled, not graveyarded


def test_snuff_out_ordinary_mana_paid_cast():
    """The ordinary {3}{B} mana-paid cast: no life paid."""
    state = _g3_two()
    tgt = Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1))
    tgt.slot = 1
    state.players[1].battlefield = [tgt]
    state.players[0].hand = [registry.CARD_DEFS["Snuff Out"]]
    state.players[0].life_total = 20
    cast_snuff_out(state, registry.CARD_DEFS["Snuff Out"])
    resolution.execute_choose_any_target_creature(state, 1, "R Creature", 1)
    resolve_top_of_stack(state)
    assert tgt not in state.players[1].battlefield
    assert state.players[0].life_total == 20  # no life paid


def test_snuff_out_alt_illegal_without_swamp():
    state = _g3_two()
    tgt = Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1))
    tgt.slot = 1
    state.players[1].battlefield = [tgt]
    state.players[0].life_total = 20
    assert not snuff_out_alt_legal(state)  # no Swamp controlled


def test_snuff_out_alt_illegal_with_less_than_4_life():
    state = _g3_two()
    swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)))
    swamp.slot = 1
    tgt = Permanent(CardDef("R Creature", CardType.CREATURE, {"R": 1}, EffectId.FILLER, power=1, toughness=1))
    tgt.slot = 1
    state.players[0].battlefield = [swamp]
    state.players[1].battlefield = [tgt]
    state.players[0].life_total = 3  # short of the 4-life alt cost
    assert not snuff_out_alt_legal(state)


def test_snuff_out_alt_illegal_without_nonblack_target():
    state = _g3_two()
    swamp = Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP, basic=True, subtypes=("Swamp",)))
    swamp.slot = 1
    black_tgt = Permanent(CardDef("B Creature", CardType.CREATURE, {"B": 1}, EffectId.FILLER, power=1, toughness=1))
    black_tgt.slot = 1
    state.players[0].battlefield = [swamp]
    state.players[1].battlefield = [black_tgt]  # the only creature on board is BLACK -- no legal target
    state.players[0].life_total = 20
    assert not snuff_out_alt_legal(state)
