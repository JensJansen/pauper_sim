"""Tests for game.catalog.blue_cards, transcribed from its former
`if __name__ == "__main__":` self-check block (faithful transcription --
every assertion and its rationale comment is preserved as-is)."""

import drl_env

from game import registry
from game.cards import CardDef, CardType, EffectId
from game.state import CardInstance, GameState, Permanent, PlayerState
from game.catalog.blue_cards import (
    _has_stack_spell,
    _sleep_escape_legal,
    cast_abandon_attachments,
    cast_brainstorm,
    cast_counterspell,
    cast_deem_inferior,
    cast_deep_analysis,
    cast_lorien_revealed,
    cast_mental_note,
    cast_ponder,
    cast_sleep_of_the_dead,
    cast_spell_pierce,
    cast_thought_scour,
    escape_sleep_of_the_dead,
    flashback_deep_analysis,
    islandcycle_lorien_revealed,
    sewer_cam_sac,
)
from game.effects.casting import cast_permanent_from_hand, cast_targeting_creature
from game.effects.stack import on_cast_trigger, push_to_stack, resolve_top_of_stack
from game.effects.state_based import check_state_based_actions, destroy_permanent
from game.effects.stats import creature_keywords, has_keyword, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.mana import execute_pool_spend, pool_spend_options
from game.resolution import (
    choose_any_target_creature_options,
    choose_stack_target_options,
    execute_choose_any_target_creature,
    execute_choose_any_target_decline,
    execute_choose_graveyard_card_option,
    execute_choose_stack_target_option,
    execute_choose_target_player_option,
    execute_discard_option,
    execute_may_transform,
    execute_ponder_option,
    execute_ponder_shuffle,
    execute_put_on_top_option,
    execute_search_fetch_option,
    execute_tuck_position,
    pay_unless_decline,
    pay_unless_pay,
    put_on_top_options,
    search_fetch_options,
)
from game.turn import untap_step, upkeep_step


def _card(name, ct=CardType.LAND, eid=EffectId.ISLAND):
    return CardDef(name, ct, None, eid)


def _lorien_def():
    return CardDef(
        "Lórien Revealed", CardType.SORCERY, {"generic": 3, "U": 2}, EffectId.LORIEN_REVEALED,
        islandcycling_cost={"generic": 1},
    )


def _stack_target_named(state, name):
    """choose_stack_target is pointer-addressed (the exact stack-entry
    object), not by name -- this finds the matching entry the same way a
    real pointer pick would, for this self-check's own convenience."""
    return next(e for e in choose_stack_target_options(state) if e["card_def"].name == name)


def _stack_spell(st, name, controller=1):
    """Put a real spell on the stack as if `controller` cast it (its card
    physically in their hand, reserved -- what a normal cast looks like)."""
    cd = CardDef(name, CardType.INSTANT if name != "Ponder" else CardType.SORCERY, {"U": 1}, EffectId.FILLER)
    st.players[controller].hand.append(cd)
    saved = st.active_idx
    st.active_idx = controller
    push_to_stack(st, cd, lambda s, c: None, reserves_hand_card=True, is_spell=True)
    st.active_idx = saved
    return cd


def _cast_zap_at_tt(pay):
    """G12 Ward {2}: an opponent targeting Tolarian Terror must pay {2} or
    the spell is countered. Casts a targeted removal spell ("Zap") at
    Tolarian Terror from the opponent's seat and drives the game up to the
    Ward pay-or-counter decision, returning the state for the caller to
    resolve either way."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tt = Permanent(registry.CARD_DEFS["Tolarian Terror"])
    tt.slot = 1
    state.players[0].battlefield = [tt]
    state.active_idx = 1  # the OPPONENT is casting
    if pay:
        state.players[1].mana_pool = {"U": 2}  # can afford Ward's {2}
    zap = CardDef("Zap", CardType.INSTANT, {"U": 1}, EffectId.FILLER)
    state.players[1].hand = [zap]
    cast_targeting_creature(state, zap, lambda s, perm: destroy_permanent(s, perm))
    execute_choose_any_target_creature(state, 0, "Tolarian Terror", 1)  # lock the target -> Ward triggers, Zap pushed
    # Ward is TOLARIAN TERROR'S OWN triggered ability -- it belongs to its
    # controller (idx 0), not the caster (idx 1, state.active_idx right
    # now): _maybe_trigger_ward writes into state.players[controller]
    # directly rather than the active-player proxy, same owner-threading
    # fix game.effects.state_based._queue_leave_triggers already needed.
    assert len(state.stack) == 1 and any(e["type"] == "ward" for e in state.players[0].trigger_queue)
    promote_triggers_to_stack(state)  # Ward goes ON TOP of Zap
    assert len(state.stack) == 2 and state.stack[-1]["card_def"].name == "Tolarian Terror"
    resolve_top_of_stack(state)  # Ward resolves -> pay_unless for the caster (idx 1)
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    return state, tt, zap


def test_mental_note():
    """Mental Note {U}: mill 2 (self), draw 1. The spell moves itself to the
    graveyard as it resolves, ahead of the mill/draw."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Mental Note", CardType.INSTANT, {"U": 1}, EffectId.MENTAL_NOTE)]
    state.library = [
        CardDef("A", CardType.LAND, None, EffectId.ISLAND), CardDef("B", CardType.LAND, None, EffectId.ISLAND),
        CardDef("C", CardType.LAND, None, EffectId.ISLAND),
    ]
    cast_mental_note(state, state.hand[0])
    assert [c.name for c in state.graveyard] == ["Mental Note", "A", "B"]  # itself + 2 milled
    assert [c.name for c in state.hand] == ["C"]  # drew the 3rd


def test_thought_scour_target_opponent():
    """Thought Scour {U}: target player mills 2, caster draws 1. Targeting the
    OPPONENT mills THEIR library; the active player still draws."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].hand = [CardDef("Thought Scour", CardType.INSTANT, {"U": 1}, EffectId.THOUGHT_SCOUR)]
    state.players[0].library = [CardDef("MyCard", CardType.LAND, None, EffectId.ISLAND)]
    state.players[1].library = [
        CardDef("Opp1", CardType.LAND, None, EffectId.ISLAND), CardDef("Opp2", CardType.LAND, None, EffectId.ISLAND),
        CardDef("Opp3", CardType.LAND, None, EffectId.ISLAND),
    ]
    cast_thought_scour(state, state.players[0].hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 1)  # target the opponent
    assert [c.name for c in state.players[1].graveyard] == ["Opp1", "Opp2"]  # THEIR top 2 milled
    assert [c.name for c in state.players[0].hand] == ["MyCard"]  # the caster drew, not the opponent
    assert state.active_idx == 0  # no stray active_idx flip


def test_lorien_revealed_cast_draws_three():
    """Lórien Revealed {3}{U}{U}: cast draws three cards."""
    state = GameState(on_the_play=True)
    lorien = _lorien_def()
    state.hand = [lorien]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("d1", "d2", "d3", "d4")]
    cast_lorien_revealed(state, lorien)
    assert len(state.hand) == 3  # drew 3


def test_lorien_revealed_islandcycling_searches_island_subtype():
    """Islandcycling {1} searches for an Island-SUBTYPE card (basic Island /
    Contaminated Aquifer / Ice Tunnel), not a Mountain."""
    state = GameState(on_the_play=True)
    lorien = _lorien_def()
    state.hand = [lorien]
    state.library = [
        CardDef("Ice Tunnel", CardType.LAND, None, EffectId.ICE_TUNNEL, subtypes=("Island", "Swamp")),
        CardDef("Mountain", CardType.LAND, None, EffectId.MOUNTAIN, basic=True, subtypes=("Mountain",)),
        CardDef("Island", CardType.LAND, None, EffectId.ISLAND, basic=True, subtypes=("Island",)),
    ]
    islandcycle_lorien_revealed(state, lorien)
    assert state.pending_resolution["kind"] == "search_fetch"
    assert sorted(search_fetch_options(state)) == ["Ice Tunnel", "Island"]  # Mountain excluded (no Island subtype)
    execute_search_fetch_option(state, "Ice Tunnel")
    assert any(c.name == "Ice Tunnel" for c in state.hand)
    assert [c.name for c in state.graveyard] == ["Lórien Revealed"]  # discarded itself


def test_brainstorm():
    """Brainstorm {U}: draw 3, then put 2 hand cards on top in a chosen order
    (first placed ends on top)."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Brainstorm", CardType.INSTANT, {"U": 1}, EffectId.BRAINSTORM)]
    state.library = [_card("L1"), _card("L2"), _card("L3"), _card("L4")]
    cast_brainstorm(state, state.hand[0])
    assert state.pending_resolution["kind"] == "put_on_top"  # after drawing 3
    assert len(state.hand) == 3  # L1, L2, L3 (Brainstorm went to gy)
    execute_put_on_top_option(state, "L3")  # L3 on top
    execute_put_on_top_option(state, "L1")  # then L1
    assert state.pending_resolution is None
    assert [c.name for c in state.library[:2]] == ["L3", "L1"]  # first-placed on top
    assert len(state.hand) == 1 and state.hand[0].name == "L2"


def test_brainstorm_excludes_reserved_stack_card():
    """Brainstorm can NOT put a spell that's currently on the stack (reserved,
    still physically in the hand list) back on top of the library -- it's on
    the stack, not in hand (real Magic). Regression for the reserved-hand-
    card bug (Gurmag Angler mid-cast + Brainstorm + Mental Note crash)."""
    state = GameState(on_the_play=True)
    reserved_spell = CardDef("Gurmag Angler", CardType.CREATURE, {"generic": 6, "B": 1}, EffectId.FILLER, power=5, toughness=5)
    state.hand = [CardDef("Brainstorm", CardType.INSTANT, {"U": 1}, EffectId.BRAINSTORM), reserved_spell]
    push_to_stack(state, reserved_spell, lambda s, c: None)  # a spell cast in response, reserved on the stack
    state.library = [_card(n) for n in ("BL1", "BL2", "BL3", "BL4", "BL5")]
    cast_brainstorm(state, state.hand[0])
    assert state.pending_resolution["kind"] == "put_on_top"
    opts = put_on_top_options(state)
    assert "Gurmag Angler" not in opts, opts  # reserved on the stack -> not a legal put-on-top pick
    assert {"BL1", "BL2", "BL3"} == set(opts)  # the three genuinely-in-hand drawn cards are


def test_ponder_order():
    """Ponder {U}: look at the top three, put them back in a chosen order,
    then draw the new top."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER)]
    state.library = [_card("P1"), _card("P2"), _card("P3"), _card("P4")]
    cast_ponder(state, state.hand[0])
    assert state.pending_resolution["kind"] == "ponder"
    execute_ponder_option(state, "P3")  # P3 to top (will be drawn)
    execute_ponder_option(state, "P1")
    execute_ponder_option(state, "P2")
    assert state.pending_resolution is None
    assert state.hand[0].name == "P3"  # drew the card placed on top
    assert [c.name for c in state.library[:2]] == ["P1", "P2"]  # rest below in chosen order


def test_ponder_shuffle():
    """Ponder {U}: OR shuffle the top three instead of ordering, then draw."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER)]
    state.library = [_card(f"S{i}") for i in range(6)]
    cast_ponder(state, state.hand[0])
    execute_ponder_shuffle(state)  # shuffle instead of ordering
    assert state.pending_resolution is None and len(state.hand) == 1  # shuffled + drew 1


def test_deep_analysis_cast():
    """Deep Analysis {3}{U}: target player draws two."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)]
    state.library = [_card(f"D{i}") for i in range(4)]
    cast_deep_analysis(state, state.hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)
    assert len(state.hand) == 2


def test_deep_analysis_flashback():
    """Flashback -- {1}{U}, Pay 3 life. The {1}{U} half is paid by the generic
    mana-flashback path; the 3-life additional cost is paid here and the
    effect goes on the stack.

    The graveyard is built from a real CardInstance -- what an actual game
    puts there (plans/object-identity-zone-model.md) -- and the instance is
    what gets passed in, matching the contract drl_env._actions._flashback_
    execute now honors. Building this fixture from a raw CardDef instead is
    what let a real bug (state.graveyard.remove on the interned CardDef,
    which can never match an instance) pass this check and crash only in
    live self-play."""
    state = GameState(on_the_play=True)
    da = CardInstance(CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS))
    state.graveyard = [da]
    state.library = [_card(f"F{i}") for i in range(4)]
    state.players[0].life_total = 10
    flashback_deep_analysis(state, da)  # mana already paid upstream in real play
    assert state.graveyard == [] and state.life_total == 7 and len(state.stack) == 1
    assert state.stack[0]["card_def"] is da, "the stack entry must carry the exact graveyard instance"
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)
    assert len(state.hand) == 2


def test_deep_analysis_flashback_picks_exact_instance():
    """Two same-named graveyard copies: the instance handed in is the one that
    leaves, and the OTHER copy stays put -- the identity property the old
    by-name lookup could not guarantee."""
    state = GameState(on_the_play=True)
    da_def = CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)
    copy_a, copy_b = CardInstance(da_def), CardInstance(da_def)
    state.graveyard = [copy_a, copy_b]
    state.library = [_card(f"G{i}") for i in range(4)]
    state.players[0].life_total = 10
    flashback_deep_analysis(state, copy_b)  # explicitly the SECOND copy
    assert state.graveyard == [copy_a], "exactly the passed instance must leave; the other copy stays"


def test_counterspell():
    """Counterspell counters any spell -> the countered spell to its
    controller's graveyard."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Some Instant", controller=1)
    state.active_idx = 0
    cs = CardDef("Counterspell", CardType.INSTANT, {"U": 2}, EffectId.COUNTERSPELL)
    state.players[0].hand = [cs]  # the counter spell is in hand while on the stack (reserved), removed on resolve
    cast_counterspell(state, cs)
    assert state.pending_resolution["kind"] == "choose_stack_target"
    execute_choose_stack_target_option(state, _stack_target_named(state, "Some Instant"))
    resolve_top_of_stack(state)
    assert all(e["card_def"].name != "Some Instant" for e in state.stack)  # countered off the stack
    assert any(c.name == tgt.name for c in state.players[1].graveyard)


def test_dispel_legality_instant_only():
    """Dispel: an instant is a legal target; a sorcery is not."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    _stack_spell(state, "Ponder", controller=1)  # sorcery
    state.active_idx = 0
    assert not _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT)
    _stack_spell(state, "An Instant", controller=1)
    assert _has_stack_spell(state, lambda e: e["card_def"].card_type == CardType.INSTANT)


def test_spell_pierce_decline_counters():
    """Spell Pierce: controller declines to pay {2} -> countered."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Ponder", controller=1)  # noncreature
    state.active_idx = 0
    sp = CardDef("Spell Pierce", CardType.INSTANT, {"U": 1}, EffectId.SPELL_PIERCE)
    state.players[0].hand = [sp]
    cast_spell_pierce(state, sp)
    execute_choose_stack_target_option(state, _stack_target_named(state, "Ponder"))
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1  # payer's decision
    pay_unless_decline(state)
    assert any(c.name == tgt.name for c in state.players[1].graveyard) and state.active_idx == 0  # not paid -> countered, active_idx restored


def test_spell_pierce_pay_survives():
    """Spell Pierce: controller pays {2} -> spell survives."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "Ponder", controller=1)
    state.players[1].mana_pool = {"U": 2}  # can afford the {2}
    state.active_idx = 0
    sp2 = CardDef("Spell Pierce", CardType.INSTANT, {"U": 1}, EffectId.SPELL_PIERCE)
    state.players[0].hand = [sp2]
    cast_spell_pierce(state, sp2)
    execute_choose_stack_target_option(state, _stack_target_named(state, "Ponder"))
    resolve_top_of_stack(state)
    pay_unless_pay(state)
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert tgt not in state.players[1].graveyard  # paid -> NOT countered
    assert any(e["card_def"].name == "Ponder" for e in state.stack) and state.active_idx == 0


def test_abandon_attachments():
    """Abandon Attachments: may discard a card; if you do, draw two."""
    state = GameState(on_the_play=True)
    aa = CardDef("Abandon Attachments", CardType.INSTANT, {"generic": 1, "U": 1}, EffectId.ABANDON_ATTACHMENTS)
    state.hand = [aa, CardDef("Spare", CardType.LAND, None, EffectId.ISLAND)]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("x", "y", "z")]
    cast_abandon_attachments(state, aa)
    assert state.pending_resolution["kind"] == "discard"
    execute_discard_option(state, "Spare")
    assert len(state.hand) == 2  # discarded Spare, drew 2


def test_sleep_of_the_dead_tap_and_skip_next_untap():
    """Sleep of the Dead main cast: tap target + it skips its next untap."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    sd = CardDef("Sleep of the Dead", CardType.SORCERY, {"U": 1}, EffectId.SLEEP_OF_THE_DEAD)
    state.players[0].hand = [sd]
    state.active_idx = 0
    cast_sleep_of_the_dead(state, sd)
    execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.tapped and victim.flags.get("skip_next_untap")
    state.active_idx = 1
    state.turn_number = 2
    untap_step(state)  # its controller's untap -- stays tapped, skip consumed
    assert victim.tapped and not victim.flags.get("skip_next_untap")


def test_sleep_of_the_dead_escape():
    """Sleep of the Dead Escape: exile 3 other graveyard cards + tap; the card
    escapes the graveyard."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    # Graveyard holds real CardInstances, as a live game does -- and `sd` (the
    # instance) is what escape_sleep_of_the_dead now receives.
    sd = CardInstance(registry.CARD_DEFS["Sleep of the Dead"])
    state.players[0].graveyard = [sd, CardInstance(CardDef("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
                                  CardInstance(CardDef("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
                                  CardInstance(CardDef("g3", CardType.INSTANT, {"U": 1}, EffectId.FILLER))]
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.active_idx = 0
    assert _sleep_escape_legal(state)
    escape_sleep_of_the_dead(state, sd)
    for nm in ("g1", "g2", "g3"):
        assert state.pending_resolution["kind"] == "choose_graveyard_card"
        execute_choose_graveyard_card_option(state, next(c for c in state.graveyard if c.name == nm))
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Victim", 1)
    resolve_top_of_stack(state)
    assert victim.tapped and victim.flags.get("skip_next_untap")
    assert sd not in state.players[0].graveyard and state.players[0].graveyard == []  # 3 exiled + escaped


def test_murmuring_mystic_bird_illusion_on_cast_not_land():
    """Murmuring Mystic -- casting an instant/sorcery makes a 1/1 flying
    Bird Illusion; a land cast does not. "Whenever you cast an instant or
    sorcery spell, create a 1/1 blue Bird Illusion creature token with
    flying." Same on_cast chokepoint as Guttersnipe (fires for every cast
    path, faithful timing)."""
    state = GameState(on_the_play=True)
    mystic = Permanent(registry.CARD_DEFS["Murmuring Mystic"])
    mystic.slot = 1
    state.battlefield = [mystic]
    on_cast_trigger(state, CardDef("Some Instant", CardType.INSTANT, {"U": 1}, EffectId.FILLER))
    promote_triggers_to_stack(state)
    while state.stack:
        resolve_top_of_stack(state)
    birds = [p for p in state.battlefield if p.card_def.name == "Bird Illusion"]
    assert len(birds) == 1 and has_keyword(state, birds[0], "flying")
    on_cast_trigger(state, CardDef("Some Land", CardType.LAND, None, EffectId.FILLER))
    assert not state.trigger_queue  # a land cast doesn't trigger it


def test_sewer_veillance_cam_toggle_and_sac_draw_two():
    """Sewer-veillance Cam -- ETB may tap/untap (toggle) target creature;
    {3}{U},Sac -> draw 2."""
    def _drive(s):
        promote_triggers_to_stack(s)
        while s.stack:
            resolve_top_of_stack(s)

    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp = Permanent(CardDef("Opp", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    opp.slot = 1
    opp.tapped = False
    state.players[1].battlefield = [opp]
    state.players[0].hand = [registry.CARD_DEFS["Sewer-veillance Cam"]]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Sewer-veillance Cam"])
    _drive(state)  # ETB opens the optional tap/untap target choice
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Opp", 1)
    _drive(state)
    assert opp.tapped  # toggled untapped -> tapped
    cam = next(p for p in state.players[0].battlefield if p.card_def.name == "Sewer-veillance Cam")
    state.players[0].library = [CardDef(f"d{i}", CardType.LAND, None, EffectId.ISLAND, basic=True) for i in range(3)]
    sewer_cam_sac(state, cam)
    _drive(state)
    if state.pending_resolution is not None and state.pending_resolution["kind"] == "choose_any_target":
        execute_choose_any_target_decline(state)  # decline the LTB toggle
        _drive(state)
    assert len(state.players[0].hand) == 2  # drew 2


def test_tolarian_terror_cost_reduction_caster_graveyard_only():
    """Cost reduction is per the CASTER only. Tolarian Terror {6}{U}: -1 per
    instant/sorcery in the caster's OWN graveyard; the opponent's graveyard
    is never counted."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].graveyard = [CardDef("A", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("B", CardType.SORCERY, {"U": 1}, EffectId.FILLER),
                                  CardDef("L", CardType.LAND, None, EffectId.ISLAND)]  # 2 I/S + 1 land
    state.players[1].graveyard = [CardDef(f"opp{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER) for i in range(5)]  # must NOT count
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Tolarian Terror"])
    assert eff["generic"] == 4, eff  # 6 - 2 (own I/S only), opponent's 5 ignored


def test_deem_inferior_tuck_bottom():
    """Deem Inferior: the target's OWNER tucks it 2nd-from-top or bottom."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    victim = Permanent(CardDef("Threat", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[1].library = [CardDef(f"lib{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    di = registry.CARD_DEFS["Deem Inferior"]
    state.players[0].hand = [di]
    cast_deem_inferior(state, di)
    execute_choose_any_target_creature(state, 1, "Threat", 1)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "tuck_position" and state.active_idx == 1  # owner picks
    execute_tuck_position(state, "bottom")
    assert victim not in state.players[1].battlefield
    assert state.players[1].library[-1].name == "Threat" and state.active_idx == 0  # bottom, active restored


def test_deem_inferior_any_nonland_permanent_artifact_vs_land():
    """Deem Inferior targets ANY nonland permanent -- here an opponent's ARTIFACT
    (not a creature) -- and a land is NOT a legal target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    art = Permanent(CardDef("Gadget", CardType.ARTIFACT, None, EffectId.FILLER))
    art.slot = 1
    a_land = Permanent(CardDef("A Land", CardType.LAND, None, EffectId.ISLAND, basic=True))
    a_land.slot = 1
    state.players[1].battlefield = [art, a_land]
    state.players[1].library = [CardDef(f"lib{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    di2 = registry.CARD_DEFS["Deem Inferior"]
    state.players[0].hand = [di2]
    cast_deem_inferior(state, di2)
    opts = choose_any_target_creature_options(state)  # misnamed: really "any matching permanent"
    assert (1, "Gadget", 1) in opts and (1, "A Land", 1) not in opts  # artifact targetable, land not
    execute_choose_any_target_creature(state, 1, "Gadget", 1)
    resolve_top_of_stack(state)
    assert state.pending_resolution["kind"] == "tuck_position"
    execute_tuck_position(state, "top2")
    assert art not in state.players[1].battlefield
    assert state.players[1].library[1].name == "Gadget"  # second from top


def test_delver_transform_to_flyer_and_dies_as_front_face():
    """G10 Delver of Secrets: upkeep look -> may transform -> 3/2 flyer."""
    state = GameState(on_the_play=True)
    state.event_log = []  # capture events to check the reveal + transform logging
    state.active_idx = 0
    delver = Permanent(registry.CARD_DEFS["Delver of Secrets"])
    delver.slot = 0
    state.players[0].battlefield = [delver]
    state.players[0].library = [CardDef("Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER),
                                CardDef("x", CardType.LAND, None, EffectId.ISLAND)]
    upkeep_step(state)  # queues the "at the beginning of your upkeep" trigger
    assert state.trigger_queue and state.trigger_queue[0]["type"] == "upkeep"
    promote_triggers_to_stack(state)
    assert len(state.stack) == 1
    resolve_top_of_stack(state)  # top card is an instant -> transform offered
    assert state.pending_resolution["kind"] == "may_transform"
    assert permanent_power(state, delver) == 1 and permanent_toughness(state, delver) == 1  # front face
    execute_may_transform(state, True)
    assert delver.flags.get("transformed")
    # game state IS the back face now: card_def swapped (identity/name), front kept for revert.
    assert delver.card_def.name == "Insectile Aberration"
    assert delver.flags["front_card_def"].name == "Delver of Secrets"
    assert permanent_power(state, delver) == 3 and permanent_toughness(state, delver) == 2  # Insectile Aberration
    assert "flying" in creature_keywords(state, delver)
    reveal = next(e for e in state.event_log if e["kind"] == "reveal")
    assert reveal["card"] == "Bolt"  # the revealed top-of-library instant
    xf = next(e for e in state.event_log if e["kind"] == "transform")
    assert xf["permanent"] == ["Delver of Secrets", 0] and xf["to_card"] == "Insectile Aberration"
    assert xf["power"] == 3 and xf["toughness"] == 2
    # A DFC reverts to its FRONT face when it leaves the battlefield: dying puts
    # "Delver of Secrets" in the graveyard (not the back face, and not ceased as a
    # would-be token from the back name not being in CARD_DEFS).
    delver.damage_marked = 2
    check_state_based_actions(state)
    assert delver not in state.players[0].battlefield
    assert [c.name for c in state.players[0].graveyard] == ["Delver of Secrets"]
    death = next(e for e in state.event_log if e["kind"] == "state_based_death")
    assert death["permanent"] == ["Insectile Aberration", 0] and death["to_zone"] == "graveyard"


def test_delver_non_instant_sorcery_no_transform():
    """Non-instant/sorcery on top: look, but no transform choice ever opens."""
    state = GameState(on_the_play=True)
    state.active_idx = 0
    d2 = Permanent(registry.CARD_DEFS["Delver of Secrets"])
    d2.slot = 0
    state.players[0].battlefield = [d2]
    state.players[0].library = [CardDef("x", CardType.LAND, None, EffectId.ISLAND)]
    upkeep_step(state)
    promote_triggers_to_stack(state)
    resolve_top_of_stack(state)
    assert state.pending_resolution is None and not d2.flags.get("transformed")


def test_ward_decline_counters_spell():
    """Decline -> Zap countered, Tolarian Terror survives."""
    state, tt, zap = _cast_zap_at_tt(pay=False)
    pay_unless_decline(state)
    assert tt in state.players[0].battlefield  # not destroyed -- Zap was countered
    assert state.stack == [] and any(c.name == zap.name for c in state.players[1].graveyard)


def test_ward_pay_spell_resolves():
    """Pay {2} -> Zap survives, resolves, Tolarian Terror is destroyed."""
    state, tt, zap = _cast_zap_at_tt(pay=True)
    pay_unless_pay(state)
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 10
        execute_pool_spend(state, pool_spend_options(state)[0])
    assert len(state.stack) == 1  # Zap NOT countered
    resolve_top_of_stack(state)  # Zap resolves -> destroy Tolarian Terror
    assert tt not in state.players[0].battlefield  # destroyed
