"""pytest transcription of src/game/catalog/black_cards.py's former
`if __name__ == "__main__":` self-check block.

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
    cast_cast_down,
    cast_dread_return,
    cast_fanatical_offering,
    cast_reckoners_bargain,
    cast_snuff_out_alt,
    cast_toxin_analysis,
    cast_unexpected_fangs,
    refurbished_familiar_etb,
    snuff_out_alt_legal,
)
from game.effects.casting import enters_battlefield
from game.effects.stack import resolve_top_of_stack
from game.effects.state_based import check_state_based_actions, cleanup_step, sacrifice_to_graveyard
from game.effects.stats import has_keyword, lifelink_count, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, execute_pool_spend, pool_spend_options
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
# (state-based) or is SACRIFICED (resolution.execute_sacrifice_option), the
# two ways a creature leaves in this pool. ---

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
    resolution.execute_sacrifice_option(state, "Mesmeric Fiend")
    assert fiend not in state.players[0].battlefield
    assert [e["type"] for e in state.trigger_queue] == ["ltb"]
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert a_spell in state.players[1].hand


def test_mesmeric_fiend_solo_no_opponent_etb_is_noop():
    """(c) 1-player (no opponent to target): ETB does nothing, no exile."""
    solo = GameState(on_the_play=True)
    solo_fiend = enters_battlefield(solo, _MESMERIC_FIEND_DEF, from_zone="hand")
    promote_triggers_to_stack(solo)
    resolve_top_of_stack(solo)
    assert solo.pending_resolution is None and "mesmeric_exiled" not in solo_fiend.flags


# --- G3 removal & tricks (mono-black) ---

def _g3_two():
    return GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])


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


def test_refurbished_familiar_etb_opponent_discards_own_choice():
    """"When this creature enters, each opponent discards a card." 2-player:
    the one opponent discards a card of THEIR choice (active_idx flipped to
    them for the discard)."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[1].hand = [CardDef("Opp1", CardType.LAND, None, EffectId.SWAMP), CardDef("Opp2", CardType.LAND, None, EffectId.SWAMP)]
    refurbished_familiar_etb(state)
    assert state.pending_resolution["kind"] == "discard" and state.active_idx == 1
    resolution.execute_discard_option(state, "Opp1")
    assert len(state.players[1].hand) == 1 and state.active_idx == 0


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
        for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY, pending_kinds=registry.derive_pending_kinds(dl))
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
        activate_mana_source(state, sw)  # float-first: float 5 B into the pool BEFORE casting
    legal, execute = byname["Cast Gurmag Angler"]
    assert legal(state)
    execute(state)
    assert state.pending_resolution["kind"] == "choose_delve_amount"
    delve_legal, delve_execute = byname["Delve 2"]
    assert delve_legal(state)
    delve_execute(state)
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])  # exile 1st (delve 1)
    resolution.execute_choose_graveyard_card_option(state, resolution.choose_graveyard_card_options(state)[0])  # exile 2nd (delve 2)
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        execute_pool_spend(state, pool_spend_options(state)[0])
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Gurmag Angler" for p in state.battlefield) and state.graveyard == []


# --- G8 sac engines (grixis_affinity / jund_wildfire) ---

def _drive8(s):
    promote_triggers_to_stack(s)
    while s.stack:
        resolve_top_of_stack(s)


def test_gixian_infiltrator_gains_counter_on_sacrifice():
    """Gixian Infiltrator: "Whenever you sacrifice another permanent, put a
    +1/+1 counter on this creature." Fired by shared.fire_sacrifice_triggers
    from every sacrifice path (it only iterates the sacrificer's OTHER
    battlefield permanents, so "another" is automatic)."""
    state = GameState(on_the_play=True)
    gix = Permanent(registry.CARD_DEFS["Gixian Infiltrator"])
    gix.slot = 1
    fodder = Permanent(registry.CARD_DEFS["Ichor Wellspring"])
    fodder.slot = 1
    state.battlefield = [gix, fodder]
    state.library = [CardDef("x", CardType.LAND, None, EffectId.SWAMP, basic=True)]
    sacrifice_to_graveyard(state, fodder)
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
