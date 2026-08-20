"""Tests for game.catalog.black_cards.

black_cards.py's own module docstring: every card's cost/type/oracle-text is
a direct Scryfall pull, except creature power/toughness (a design choice,
not Scryfall data). Real Jagged Barrens/End the Festivities/Vampire's
Kiss/Voldaren Epicure/Alms of the Vein reference "each opponent"/"target
opponent" -- all of these route through win_check.deal_damage_to_opponent,
which hits the opponent's real per-player life_total.
"""

import contextlib
import io

import drl_env

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
from game.turn import Phase


# --- Dread Return: target locked at cast, effect on the stack, and the
# 608.2b fizzle when the chosen creature card leaves the graveyard before it
# resolves (reachable via opponent graveyard hate). ---

_DREAD_RETURN = CardDef("Dread Return", CardType.SORCERY, {"generic": 2, "B": 2}, EffectId.DREAD_RETURN)
_GRIZZLY_BEARS = CardDef("Grizzly Bears", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2)


def _dread_return_cast_and_target():
    """Shared setup for both Dread Return tests below: hard-cast Dread
    Return, which locks its reanimation target -- a creature card in the
    graveyard -- AT CAST, by object identity (the exact CardInstance), per
    real MTG 400.7/608.2b -- so two same-named copies are distinct (one can
    leave while the other stays)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.hand = [_DREAD_RETURN]
    grizzly_inst = state.new_instance(_GRIZZLY_BEARS)
    state.graveyard = [grizzly_inst]
    cast_dread_return(state, _DREAD_RETURN)  # precast: begins the graveyard-target choice
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, grizzly_inst)  # the exact instance
    return state, grizzly_inst


def test_dread_return_hard_cast_reanimates_target():
    """(a) hard cast reanimates the chosen creature card. The graveyard holds
    a CardInstance; the pick captures that EXACT instance."""
    state, grizzly_inst = _dread_return_cast_and_target()
    assert state.hand == [] and len(state.stack) == 1  # Dread Return LEFT hand at cast, now on the stack
    resolve_top_of_stack(state)
    assert any(c.name == "Dread Return" for c in state.graveyard)  # Dread Return resolved -> graveyard
    assert any(p.card_def is _GRIZZLY_BEARS for p in state.battlefield)  # reanimated
    assert all(c.name != "Grizzly Bears" for c in state.graveyard)  # left the graveyard


def test_dread_return_fizzles_if_target_leaves_graveyard():
    """(b) fizzle: the chosen instance leaves the graveyard before
    resolution (608.2b -- reachable via opponent graveyard hate, e.g. Relic
    of Progenitus exiling graveyards)."""
    state, grizzly_inst = _dread_return_cast_and_target()
    state.graveyard.remove(grizzly_inst)  # exiled by graveyard hate before Dread Return resolves
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        resolve_top_of_stack(state)
    assert "fizzle" in log.getvalue().lower()
    assert any(c.name == "Dread Return" for c in state.graveyard)  # Dread Return still goes to the graveyard
    assert not any(p.card_def is _GRIZZLY_BEARS for p in state.battlefield)  # nothing reanimated


def test_bojuka_bog_etb_exiles_target_players_graveyard():
    """Bojuka Bog: enters tapped, and its ETB "exile target player's
    graveyard" is a real target-player choice resolved off the stack (ETBs
    go through the trigger queue now). 2-player so targeting the OPPONENT
    empties THEIR graveyard, never the active player's own.

    Target-at-promotion (project_targeted_triggered_abilities): the target
    player is chosen as the ability goes on the stack, and the exile happens
    at resolution -- "target player" is always legal (no way to make a
    player an illegal target), so this never fizzles, but choosing at
    promotion still surfaces the decision one priority window earlier, which
    can inform play, so it's modeled faithfully rather than collapsed to
    resolution."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].graveyard = [CardDef("Mine", CardType.CREATURE, None, EffectId.FILLER)]
    state.players[1].graveyard = [CardDef("Theirs", CardType.CREATURE, None, EffectId.FILLER)]
    bog = CardDef("Bojuka Bog", CardType.LAND, None, EffectId.BOJUKA_BOG)
    enters_battlefield(state, bog, from_zone="hand")
    bog_perm = next(p for p in state.battlefield if p.card_def.name == "Bojuka Bog")
    assert bog_perm.tapped  # enters tapped
    # ETB queued (faithful timing), not run inline. Target-at-promotion: promote
    # opens the target-player choice; picking it pushes the exile effect.
    assert state.pending_resolution is None
    assert [e["type"] for e in state.trigger_queue] == ["etb"]
    promote_triggers_to_stack(state)  # opens the target-player choice at promotion
    assert state.pending_resolution["kind"] == "choose_target_player"
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent -> effect pushed onto the stack
    assert state.pending_resolution is None and len(state.stack) == 1
    resolve_top_of_stack(state)  # exile that player's graveyard
    assert state.players[1].graveyard == []  # their graveyard exiled
    assert [c.name for c in state.players[0].graveyard] == ["Mine"]  # own graveyard untouched


# --- Mesmeric Fiend: ETB exiles a nonland from target opponent's hand
# (tracked, linked to this Fiend); its leaves-the-battlefield trigger -- the
# engine's first -- returns that card when the Fiend leaves, whether it DIES
# (state-based) or is SACRIFICED (resolution.begin_sacrifice, via
# state_based.sacrifice_to_graveyard), the two ways a creature leaves in
# this pool. ---

_MESMERIC_FIEND_DEF = CardDef(
    "Mesmeric Fiend", CardType.CREATURE, {"generic": 1, "B": 1}, EffectId.MESMERIC_FIEND, power=1, toughness=1,
)


def _enter_fiend_and_exile():
    """Shared setup for the three Mesmeric Fiend tests below: the Fiend
    enters, and its ETB exiles a nonland card from the opponent's hand,
    tracked and linked to this exact Fiend permanent."""
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
    """(a) the Fiend DIES -> LTB returns the exiled card to its owner's hand."""
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
    """(b) the Fiend is SACRIFICED (e.g. to Dread Return's Flashback) -> same LTB."""
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
    """(c) 1-player (no opponent to target): ETB does nothing, no exile.
    Target-at-promotion (etb_targets: True) means the opponent is captured
    -- or, here, found not to exist -- AT PROMOTION, so with no legal
    target the ability never even reaches the stack (nothing for
    resolve_top_of_stack to pop), unlike a plain non-targeting ETB."""
    solo = GameState(on_the_play=True)
    solo_fiend = enters_battlefield(solo, _MESMERIC_FIEND_DEF, from_zone="hand")
    promote_triggers_to_stack(solo)
    assert solo.stack == [] and solo.pending_resolution is None and "mesmeric_exiled" not in solo_fiend.flags


def test_balustrade_spy_etb_mills_target_players_library():
    """Balustrade Spy: real "target player" (begin_choose_target_player),
    same shape as Bojuka Bog's own ETB -- targeting the OPPONENT mills
    THEIR library, never the caster's own, until a land turns up
    (inclusive)."""
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
    promote_triggers_to_stack(state)  # target-at-promotion opens the target-player choice
    assert state.pending_resolution["kind"] == "choose_target_player"
    resolution.execute_choose_target_player_option(state, 1)  # target the opponent
    resolve_top_of_stack(state)
    assert [c.name for c in state.players[1].library] == ["Their3"]  # milled through the land, inclusive
    assert [c.name for c in state.players[1].graveyard] == ["Their1", "Their2"]
    assert [c.name for c in state.players[0].library] == ["Own1"]  # own library untouched


def test_blood_fountain_returns_up_to_two_targets_with_partial_fizzle():
    """Blood Fountain: {3}{B}, {T}, Sacrifice: return up to two TARGET
    creature cards from your graveyard to hand -- both targets locked at
    ACTIVATION (before the ability goes on the stack), and 608.2c partial
    fizzle: if one target leaves the graveyard in response, the other
    still comes back."""
    state = GameState(on_the_play=True)
    fountain = Permanent(CardDef(
        "Blood Fountain", CardType.ARTIFACT, {"B": 1}, EffectId.BLOOD_FOUNTAIN, sac_ability_cost={"generic": 3, "B": 1},
    ))
    bear = state.new_instance(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    wolf = state.new_instance(CardDef("Wolf", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    state.battlefield = [fountain]
    state.graveyard = [bear, wolf]
    blood_fountain_return(state, fountain)
    assert state.battlefield == []  # sacrificed -- a cost, paid immediately on activation
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, bear)
    resolution.execute_choose_graveyard_card_option(state, wolf)
    assert state.pending_resolution is None and len(state.stack) == 1  # both targets locked, ability on the stack
    state.graveyard.remove(bear)  # bear reanimated/exiled in response -- an illegal target by resolution
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Wolf"]  # only the surviving target returns (bear silently skipped)


def test_alms_of_the_vein_target_opponent_loses_life_caster_gains():
    """Alms of the Vein: target opponent loses 3, caster gains 3 -- BOTH
    halves of the real text. Target (the opponent) is locked at CAST
    (precast_choice), and the spell can't be cast at all without one
    (extra_legal, since "target opponent" needs a legal target to exist)."""
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
    """No opponent -> the mandatory "target opponent" has no legal target,
    so the spell can't be cast at all (601.2c)."""
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.ALMS_OF_THE_VEIN]["cast"]["extra_legal"](state)


def test_vampires_kiss_targets_any_player_and_grants_life():
    """Vampire's Kiss: real "target player" -- ANY player, unlike Alms of
    the Vein's opponent-restricted version. Targeting the opponent drains
    them; the caster still gains 2 life and two Blood tokens either way
    (the gain-life half was previously dropped entirely)."""
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
    """Drain a "pay_cost" pending resolution by repeatedly taking the first
    available pool-spend option, the same guarded loop every mana-cost-paying
    test below drives to completion."""
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
    """Snuff Out: nonblack only ("nonblack creature" -- colorless, e.g. a
    Devoid creature, counts as nonblack), and the 4-life alt cost ("you may
    pay 4 life rather than pay this spell's mana cost", legal only with a
    Swamp, >= 4 life, and a legal nonblack target)."""
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
    """Unexpected Fangs: puts a +1/+1 counter AND a lifelink counter on the
    target (the lifelink counter grants lifelink)."""
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
    """Toxin Analysis: target creature gains deathtouch and lifelink until
    end of turn (temp_keywords, cleared at cleanup_step); then Investigate (a
    Clue token) -- Investigate is inside the target-legal branch: if the only
    target is gone, the spell doesn't resolve at all, so no Clue is made."""
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
    """"For each opponent who can't [discard], you draw a card." Opponent's
    hand is empty -> you draw instead of them discarding."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].hand = []
    state.players[0].library = [CardDef("Top", CardType.LAND, None, EffectId.SWAMP)]
    refurbished_familiar_etb(state)
    assert len(state.players[0].hand) == 1  # drew (opponent couldn't discard)


def test_gurmag_angler_delve_exiles_graveyard_cards_to_pay_generic():
    """Gurmag Angler Delve: delve N exiles N graveyard cards + pays
    {6-N}{B}. Delve-only cast: "Cast Gurmag Angler", then a generic "Delve N"
    (N in 0..6; delve 0 is the plain {6}{B} cast) choice exiles N graveyard
    cards to pay {N} of the generic (drl_env delve loop). 702.66/601.2f: the
    delve amount is its own generic sub-decision, chosen BEFORE the exile
    sub-cost opens -- "Delve 2" is a shared button (drl_env._choose_delve_
    amount_legal), not baked into the cast row. Graveyard holds distinct
    CardInstances (object-identity model); delve exiles the exact chosen
    instance, so seed + pick by object, not by name."""
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
    state.battlefield = [Permanent(registry.CARD_DEFS["Swamp"]) for _ in range(5)]  # {6}{B} minus delve 2 = {4}{B} = 5 mana
    for sw in state.battlefield:
        activate_mana_source(state, sw)  # pre-float (still valid, just no longer required): float 5 B into the pool BEFORE casting
    legal, execute = byname["Cast Gurmag Angler"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "choose_delve_amount"
    delve_legal, delve_execute = byname["Delve 2"]
    assert delve_legal(state)
    delve_execute(state)
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])  # exile 1st (delve 1)
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])  # exile 2nd (delve 2)
    _pay_cost_fully(state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []


def test_no_mana_tap_during_delve_exile_step():
    """774fe5b's actual fix: delve's exile-to-graveyard sub-cost sits BETWEEN
    announcing the cast and the payment opening, so no mana ability may be
    activated during it (601.2f, game.mana.mid_cast) -- begin_exile_n_from_
    graveyard passes mid_cast=True precisely so drl_env._actions_mana.
    _mana_timing_legal refuses every "Tap X" row for its duration. Found
    live, not hypothetical (that commit's own docstring): dmir_terror tapped
    both Contaminated Aquifers for {U} mid-exile, stranding the delve-reduced
    {B} with no way to pay it. This drives the SPECIFIC exile-step pending
    (kind="choose_graveyard_card", mid_cast stamped) that test_no_mana_
    ability_during_a_casting_step_before_601_2f does not reach -- that test
    only exercises the generic _CASTING_STEP_PENDING_KINDS path
    (choose_cast_copy), a different branch of game.mana.mid_cast."""
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
    # 5 Contaminated Aquifers (each U-or-B): {6}{B} minus delve 2 = {4}{B},
    # payable from these 5 sources ONLY if at least one taps for B -- none
    # pre-floated, so the payment is still wide open when delve starts.
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
            # Nothing floated yet -- tap a source for whichever color is
            # still owed (mirrors the live incident's fix: B must remain
            # reachable here, which is exactly what mid_cast protected).
            still_owed = state.pending_resolution["remaining"]
            color = "B" if still_owed.get("B", 0) > 0 else "U"
            byname[f"Tap Contaminated Aquifer for {color}"][1](state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []


# --- G8 sac engines (grixis_affinity / jund_wildfire) ---

def _drive8(s):
    promote_triggers_to_stack(s)
    # Ichor Wellspring's own "dies: draw" LTB trigger and Gixian Infiltrator's
    # "sacrifice: +1/+1" trigger can queue simultaneously for the same
    # controller -- real Magic (603.3b) lets that player choose the
    # placement order, opening an order_triggers pending resolution instead
    # of placing both automatically (resolution.begin_order_triggers).
    while s.pending_resolution is not None and s.pending_resolution["kind"] == "order_triggers":
        resolution.execute_order_triggers_option(s, resolution.order_triggers_options(s)[0])
    while s.stack:
        resolve_top_of_stack(s)


def test_gixian_infiltrator_gains_counter_on_sacrifice():
    """Gixian Infiltrator: "Whenever you sacrifice another permanent, put a
    +1/+1 counter on this creature." A real triggered ability: queued by
    shared.fire_sacrifice_triggers from every sacrifice path (it only
    iterates the sacrificer's OTHER battlefield permanents, so "another" is
    automatic) and placed on the stack at the next priority window, not
    applied immediately."""
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
    """Fanatical Offering: {1}{B}, sacrifice an artifact or creature: Draw
    two cards and create a Map token."""
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
    """Reckoner's Bargain: {1}{B}, sacrifice an artifact or creature: gain
    life equal to the sacrificed permanent's mana value, draw two."""
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
    """Cross-player routing: Refurbished Familiar's "each opponent discards a
    card" is the OPPONENT's own choice, not the caster's. The ETB flips
    active_idx to the opponent for the forced discard and restores it after --
    which is exactly what makes the training harness (keyed on
    state.active_idx, see rl.train.collect_rollout) query the OPPONENT's own
    net, never the active player deciding on their behalf."""
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


# --- Regression: Balustrade Spy was missing Flying entirely ---

def test_balustrade_spy_has_flying():
    """Regression for the source fix: real Balustrade Spy is {3}{B} 2/3
    Flying Vampire Rogue (Gatecrash #57) -- its registry entry previously
    had no "keywords" spec at all."""
    state = GameState(on_the_play=True)
    spy = Permanent(registry.CARD_DEFS["Balustrade Spy"])
    state.battlefield = [spy]
    assert has_keyword(state, spy, "flying")


# --- Vault of Whispers / Bojuka Bog: the {T}: Add {B} mana ability itself
# (Bojuka Bog's ETB is already tested above; this is just the tap). ---

def test_vault_of_whispers_taps_for_black():
    """Artifact land: {T}: Add {B} -- just the mana ability (its artifact-
    ness is what affinity/metalcraft reads elsewhere, out of scope here)."""
    state = GameState(on_the_play=True)
    vault = Permanent(registry.CARD_DEFS["Vault of Whispers"])
    state.battlefield = [vault]
    assert mana_output(vault, state) == ["B"]
    activate_mana_source(state, vault)
    assert state.mana_pool == {"B": 1} and vault.tapped
    assert state.mana_pool_single_pip == {"B": 1}  # a 1-symbol event -- tagged single-pip


def test_bojuka_bog_taps_for_black():
    """{T}: Add {B} -- just the mana ability (its ETB "exile target
    player's graveyard" is covered by test_bojuka_bog_etb_exiles_target_
    players_graveyard above)."""
    state = GameState(on_the_play=True)
    bog = Permanent(registry.CARD_DEFS["Bojuka Bog"])
    bog.tapped = False  # real Bojuka Bog enters tapped; force untapped to isolate the mana ability
    state.battlefield = [bog]
    assert mana_output(bog, state) == ["B"]
    activate_mana_source(state, bog)
    assert state.mana_pool == {"B": 1} and bog.tapped
    assert state.mana_pool_single_pip == {"B": 1}  # a 1-symbol event -- tagged single-pip


# --- Lotleth Giant: Undergrowth ETB, "1 damage to the opponent per
# creature card in your graveyard." ---

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


# --- Dread Return: Flashback (sacrifice 3 creatures instead of {2}{B}{B}),
# and extra_legal's false branch (no creature card in the graveyard). ---

def test_dread_return_flashback_sacrifices_three_creatures_and_reanimates():
    """Flashback -- sacrifice three creatures instead of paying mana. The
    Dread Return card leaves the graveyard the instant Flashback is chosen
    (exiled, untracked, after it resolves); the newly sacrificed creatures
    land in the graveyard BEFORE the reanimation target is chosen, so
    they're eligible targets too (a real interaction) -- this picks the
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
    """extra_legal's false branch: no creature card anywhere in the
    graveyard -> illegal to hard-cast (a mandatory target needs one to
    exist, 601.2c)."""
    state = GameState(on_the_play=True)
    state.graveyard = [CardDef("Not A Creature", CardType.SORCERY, {"generic": 1}, EffectId.FILLER)]
    assert not registry.EFFECT_REGISTRY[EffectId.DREAD_RETURN]["cast"]["extra_legal"](state)


# --- Kitchen Imp: its own registry Flying/haste, and a real Madness cast
# from discard. ---

def test_kitchen_imp_has_flying_and_haste():
    """Real Oracle text: Flying, haste. Both are registry-driven keyword
    flags -- the generic "haste": True mechanism itself is tested via a
    monkeypatched FILLER card in tests/game/effects/test_combat.py; this
    asserts it for Kitchen Imp's own real registry entry, together with
    Flying, and proves haste actually lets a summoning-sick Kitchen Imp
    attack and deal damage the turn it enters."""
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
    """Madness {B}: discarding Kitchen Imp exiles it instead of
    graveyarding (madness_and_plot's replacement effect, fires regardless
    of why it was discarded) and queues a cast-or-decline decision;
    choosing "cast" pays {B} and puts it straight onto the battlefield from
    exile -- never touching hand or graveyard."""
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


# --- Alms of the Vein: Madness {B}, same target-opponent shape as the
# hard cast, from exile. ---

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


# --- Blood Fountain: the ETB (Blood token), and the activated ability's
# {3}{B} mana-cost legality/payment driven through the real action table
# (existing coverage calls blood_fountain_return directly, skipping cost
# dispatch entirely). ---

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
    """"Activate Blood Fountain (return)" through the REAL action table:
    _activate_legal/_activate_execute (drl_env) gate on and pay the ability's
    {3}{B} cost via begin_pay_cost, same dispatch every other cost_key
    activated ability uses -- exercised end to end instead of jumping
    straight to blood_fountain_return like the existing partial-fizzle test
    does."""
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
            activate_mana_source(state, sw)  # pre-float (still valid, just no longer required): float 4 B BEFORE activating ({3}{B} = 4 mana)
    bear = state.new_instance(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=1, toughness=1))
    state.graveyard = [bear]

    legal, execute = byname["Activate Blood Fountain (return)"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    assert state.pending_resolution["remaining"] == {"generic": 3, "B": 1}
    _pay_cost_fully(state)

    # Cost paid -> resolve fires: {T} + Sacrifice paid here (fountain gone), then
    # the up-to-two graveyard targets are chosen.
    assert fountain not in state.battlefield
    assert state.pending_resolution["kind"] == "choose_graveyard_card"
    resolution.execute_choose_graveyard_card_option(state, bear)
    assert state.pending_resolution is None and len(state.stack) == 1  # only 1 eligible target -> selection auto-closed
    resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Bear"]


# --- Fanatical Offering / Reckoner's Bargain: extra_legal's false branch
# (no sacrificeable artifact or creature). ---

def test_fanatical_offering_uncastable_with_no_sac_fodder():
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.FANATICAL_OFFERING]["cast"]["extra_legal"](state)


def test_reckoners_bargain_uncastable_with_no_sac_fodder():
    state = GameState(on_the_play=True)
    assert not registry.EFFECT_REGISTRY[EffectId.RECKONERS_BARGAIN]["cast"]["extra_legal"](state)


# --- Eviscerator's Insight: completely untested before this. ---

def test_eviscerators_insight_sac_draws_two():
    """{1}{B}, sacrifice an artifact or creature: Draw two cards -- no Map
    token (unlike Fanatical Offering) and no life gain (unlike Reckoner's
    Bargain), just the draw."""
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


def test_eviscerators_insight_flashback_sac_draws_two_and_exiles():
    """Flashback {4}{B} (paid by the graveyard-cast machinery) + the same
    sac-fodder additional cost: draw two, then exile -- never returns to
    the graveyard."""
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


# --- Snuff Out: the ordinary mana-paid cast (never tested standalone --
# every existing test drives the 4-life alt cost instead), and each false
# branch of the alt cost's own legality gate. ---

def test_snuff_out_ordinary_mana_paid_cast():
    """cast_snuff_out (the {3}{B} mana path), not cast_snuff_out_alt -- no
    life paid, unlike every other Snuff Out test in this file."""
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
