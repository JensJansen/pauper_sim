"""Tests for game.catalog.blue_cards."""

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
    cast_dispel,
    cast_lorien_revealed,
    cast_mental_note,
    cast_ponder,
    cast_sleep_of_the_dead,
    cast_spell_pierce,
    cast_thought_scour,
    cast_thoughtcast,
    escape_sleep_of_the_dead,
    flashback_deep_analysis,
    islandcycle_lorien_revealed,
    sewer_cam_sac,
    sewer_cam_tap_or_untap,
)
from game.effects.casting import cast_permanent_from_hand, cast_targeting_creature
from game.effects.stack import on_cast_trigger, push_to_stack, resolve_top_of_stack
from game.effects.state_based import check_state_based_actions, destroy_permanent, sacrifice_to_graveyard
from game.effects.stats import creature_keywords, has_keyword, permanent_power, permanent_toughness
from game.effects.triggers import promote_triggers_to_stack
from game.mana import activate_mana_source, execute_pool_spend, float_mana, pool_spend_options
from game.resolution import (
    choose_any_target_creature_options,
    choose_stack_target_options,
    execute_choose_any_target_creature,
    execute_choose_any_target_decline,
    execute_choose_graveyard_card_option,
    execute_choose_stack_target_option,
    execute_choose_target_player_option,
    execute_discard_decline,
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
from game.turn import Phase, untap_step, upkeep_step


def _card(name, ct=CardType.LAND, eid=EffectId.ISLAND):
    return CardDef(name, ct, None, eid)


def _lorien_def():
    return CardDef(
        "Lórien Revealed", CardType.SORCERY, {"generic": 3, "U": 2}, EffectId.LORIEN_REVEALED,
        islandcycling_cost={"generic": 1},
    )


def _stack_target_named(state, name):
    """Find a stack entry by card name (choose_stack_target is pointer-addressed)."""
    return next(e for e in choose_stack_target_options(state) if e["card_def"].name == name)


def _stack_spell(st, name, controller=1):
    """Put a spell on the stack as if `controller` cast it from hand."""
    cd = CardDef(name, CardType.INSTANT if name != "Ponder" else CardType.SORCERY, {"U": 1}, EffectId.FILLER)
    st.players[controller].hand.append(cd)
    saved = st.active_idx
    st.active_idx = controller
    push_to_stack(st, cd, lambda s, c: None, reserves_hand_card=True, is_spell=True)
    st.active_idx = saved
    return cd


def _drive(s):
    """Promote queued triggers to the stack and resolve it down to empty."""
    promote_triggers_to_stack(s)
    while s.stack:
        resolve_top_of_stack(s)


def _pay_cost_fully(state):
    """Drain a pending "pay_cost" resolution by repeatedly taking the first pool-spend option."""
    guard = 0
    while state.pending_resolution is not None and state.pending_resolution["kind"] == "pay_cost":
        guard += 1
        assert guard < 30
        execute_pool_spend(state, pool_spend_options(state)[0])


def _cast_zap_at_tt(pay):
    """Cast a targeted removal spell at Tolarian Terror, driving the game up
    to its Ward {2} pay-or-counter decision."""
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
    assert len(state.stack) == 1 and any(e["type"] == "ward" for e in state.players[0].trigger_queue)
    promote_triggers_to_stack(state)  # Ward goes ON TOP of Zap
    assert len(state.stack) == 2 and state.stack[-1]["card_def"].name == "Tolarian Terror"
    resolve_top_of_stack(state)  # Ward resolves -> pay_unless for the caster (idx 1)
    assert state.pending_resolution["kind"] == "pay_unless" and state.active_idx == 1
    return state, tt, zap


def test_mental_note():
    """Mental Note: mill 2, draw 1."""
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
    """Thought Scour: target player mills 2, caster draws 1."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].hand = [CardDef("Thought Scour", CardType.INSTANT, {"U": 1}, EffectId.THOUGHT_SCOUR)]
    state.players[0].library = [CardDef("MyCard", CardType.LAND, None, EffectId.ISLAND)]
    state.players[1].library = [
        CardDef("Opp1", CardType.LAND, None, EffectId.ISLAND), CardDef("Opp2", CardType.LAND, None, EffectId.ISLAND),
        CardDef("Opp3", CardType.LAND, None, EffectId.ISLAND),
    ]
    cast_thought_scour(state, state.players[0].hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 1)  # target the opponent -- locked at cast, effect now on the stack
    resolve_top_of_stack(state)
    assert [c.name for c in state.players[1].graveyard] == ["Opp1", "Opp2"]  # THEIR top 2 milled
    assert [c.name for c in state.players[0].hand] == ["MyCard"]  # the caster drew, not the opponent
    assert state.active_idx == 0  # no stray active_idx flip


def test_lorien_revealed_cast_draws_three():
    """Lórien Revealed cast draws three cards."""
    state = GameState(on_the_play=True)
    lorien = _lorien_def()
    state.hand = [lorien]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("d1", "d2", "d3", "d4")]
    cast_lorien_revealed(state, lorien)
    assert len(state.hand) == 3  # drew 3


def test_lorien_revealed_islandcycling_searches_island_subtype():
    """Islandcycling searches for a card with the Island subtype, not just any land."""
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
    """Brainstorm: draw 3, then put 2 hand cards back on top in a chosen order."""
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
    """Brainstorm can't put a card back that's actually on the stack, even
    though the reserved card is still physically in the hand list."""
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
    """Ponder: reorder the top three, then draw the new top card."""
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
    """Ponder can shuffle the top three instead of ordering, then draw."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Ponder", CardType.SORCERY, {"U": 1}, EffectId.PONDER)]
    state.library = [_card(f"S{i}") for i in range(6)]
    cast_ponder(state, state.hand[0])
    execute_ponder_shuffle(state)  # shuffle instead of ordering
    assert state.pending_resolution is None and len(state.hand) == 1  # shuffled + drew 1


def test_deep_analysis_cast():
    """Deep Analysis: target player draws two."""
    state = GameState(on_the_play=True)
    state.hand = [CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)]
    state.library = [_card(f"D{i}") for i in range(4)]
    cast_deep_analysis(state, state.hand[0])
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)  # locked at cast, effect now on the stack
    resolve_top_of_stack(state)
    assert len(state.hand) == 2


def test_deep_analysis_flashback():
    """Flashback: pay 3 life plus mana, then cast from the graveyard."""
    state = GameState(on_the_play=True)
    da = CardInstance(CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS))
    state.graveyard = [da]
    state.library = [_card(f"F{i}") for i in range(4)]
    state.players[0].life_total = 10
    flashback_deep_analysis(state, da)  # mana already paid upstream in real play
    assert state.graveyard == [] and state.life_total == 7 and state.stack == []
    assert state.pending_resolution["kind"] == "choose_target_player"
    execute_choose_target_player_option(state, 0)  # locked; the effect now goes on the stack
    assert len(state.stack) == 1 and state.stack[0]["card_def"] is da, \
        "the stack entry must carry the exact graveyard instance"
    resolve_top_of_stack(state)
    assert len(state.hand) == 2


def test_deep_analysis_flashback_picks_exact_instance():
    """With two same-named graveyard copies, only the exact instance passed in leaves."""
    state = GameState(on_the_play=True)
    da_def = CardDef("Deep Analysis", CardType.SORCERY, {"generic": 3, "U": 1}, EffectId.DEEP_ANALYSIS)
    copy_a, copy_b = CardInstance(da_def), CardInstance(da_def)
    state.graveyard = [copy_a, copy_b]
    state.library = [_card(f"G{i}") for i in range(4)]
    state.players[0].life_total = 10
    flashback_deep_analysis(state, copy_b)  # explicitly the SECOND copy
    assert state.graveyard == [copy_a], "exactly the passed instance must leave; the other copy stays"


def test_counterspell():
    """Counterspell counters any spell, sending it to its controller's graveyard."""
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
    """Spell Pierce: controller declines to pay {2}, so the spell is countered."""
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
    """Spell Pierce: controller pays {2}, so the spell survives."""
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
    _pay_cost_fully(state)
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
    """Sleep of the Dead: taps its target, which then skips its next untap.
    Also checks the tap and skipped-untap events log the target's own
    controller as owner, not the caster."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)], event_log=[])
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
    tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(tap_events) == 1
    assert tap_events[0]["permanent"] == ["Victim", 1]
    assert tap_events[0]["now_tapped"] is True
    assert tap_events[0]["owner_idx"] == 1  # the victim's controller, NOT the caster (0)

    state.active_idx = 1
    state.turn_number = 2
    state.event_log = []
    untap_step(state)  # its controller's untap -- stays tapped, skip consumed
    assert victim.tapped and not victim.flags.get("skip_next_untap")
    untap_step_events = [e for e in state.event_log if e["kind"] == "untap_step"]
    assert not untap_step_events or ["Victim", 1] not in untap_step_events[0]["untapped"]
    re_tap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap"]
    assert len(re_tap_events) == 1
    assert re_tap_events[0]["permanent"] == ["Victim", 1]
    assert re_tap_events[0]["now_tapped"] is True
    assert re_tap_events[0]["owner_idx"] == 1
    assert re_tap_events[0]["reason"] == "skip_next_untap"


def test_sleep_of_the_dead_escape():
    """Sleep of the Dead Escape: exile 3 other graveyard cards, then cast
    from the graveyard to tap a target."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
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
    """Murmuring Mystic: casting an instant/sorcery makes a 1/1 flying Bird
    Illusion token; casting a land does not."""
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
    """Sewer-veillance Cam: ETB may toggle tap state of a target creature;
    {3}{U}, Sacrifice draws 2."""
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


def test_sewer_cam_untap_direction_discounts_mana_dork_tag():
    """Toggling an already-tapped mana dork untapped discounts its tagged
    mana pip, since tapping it for mana first cost nothing real. The
    tap/untap event also logs an explicit owner_idx."""
    state = GameState(on_the_play=True, event_log=[])
    cam = Permanent(registry.CARD_DEFS["Sewer-veillance Cam"])
    elves = Permanent(registry.CARD_DEFS["Llanowar Elves"])
    elves.slot = 1
    elves.summoning_sick = False
    state.battlefield = [cam, elves]
    activate_mana_source(state, elves)  # float+tag {G} before it gets untapped
    assert state.mana_pool_single_pip == {"G": 1}

    sewer_cam_tap_or_untap(state, cam.card_def)
    execute_choose_any_target_creature(state, 0, "Llanowar Elves", 1)
    resolve_top_of_stack(state)
    assert not elves.tapped  # toggled tapped -> untapped
    untap_events = [e for e in state.event_log if e["kind"] == "tap_or_untap" and e["reason"] == "sewer_cam"]
    assert len(untap_events) == 1
    assert untap_events[0]["permanent"] == ["Llanowar Elves", 1]
    assert untap_events[0]["now_tapped"] is False
    assert untap_events[0]["owner_idx"] == state.active_idx
    assert state.mana_pool_single_pip == {}  # the tag is excused
    assert state.mana_pool == {"G": 1}  # the floated mana itself is untouched


def test_sewer_cam_tap_direction_never_discounts():
    """Toggling untapped to tapped never discounts mana; an unrelated
    already-tagged pip stays tagged."""
    state = GameState(on_the_play=True)
    cam = Permanent(registry.CARD_DEFS["Sewer-veillance Cam"])
    elves = Permanent(registry.CARD_DEFS["Llanowar Elves"])
    elves.slot = 1
    elves.summoning_sick = False
    state.battlefield = [cam, elves]
    float_mana(state, ["G"])  # tagged G from an unrelated source
    assert state.mana_pool_single_pip == {"G": 1}

    sewer_cam_tap_or_untap(state, cam.card_def)
    execute_choose_any_target_creature(state, 0, "Llanowar Elves", 1)
    resolve_top_of_stack(state)
    assert elves.tapped  # toggled untapped -> tapped
    assert state.mana_pool_single_pip == {"G": 1}  # untouched


def test_sewer_veillance_cam_etb_target_chosen_at_promotion_not_resolution():
    """The ETB's target is chosen as the triggered ability is put on the
    stack, before it resolves."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp = Permanent(CardDef("Opp", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    opp.slot = 1
    state.players[1].battlefield = [opp]
    state.players[0].hand = [registry.CARD_DEFS["Sewer-veillance Cam"]]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Sewer-veillance Cam"])
    promote_triggers_to_stack(state)
    assert state.stack == []  # not yet on the stack -- the hook ran directly at promotion
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 1, "Opp", 1)
    assert len(state.stack) == 1  # target locked; the effect now waits on the stack
    resolve_top_of_stack(state)
    assert opp.tapped


def test_sewer_veillance_cam_ltb_target_chosen_at_promotion():
    """Same target-at-promotion timing applies to the leaves-the-battlefield
    trigger half."""
    state = GameState(on_the_play=True)
    cam = Permanent(registry.CARD_DEFS["Sewer-veillance Cam"])
    bear = Permanent(CardDef("Bear", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    bear.slot = 1
    state.battlefield = [cam, bear]
    sacrifice_to_graveyard(state, cam)
    promote_triggers_to_stack(state)
    assert state.stack == []
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_creature(state, 0, "Bear", 1)
    resolve_top_of_stack(state)
    assert bear.tapped


def test_tolarian_terror_cost_reduction_caster_graveyard_only():
    """Tolarian Terror's cost reduces by 1 per instant/sorcery in the
    caster's own graveyard; the opponent's graveyard doesn't count."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].graveyard = [CardDef("A", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("B", CardType.SORCERY, {"U": 1}, EffectId.FILLER),
                                  CardDef("L", CardType.LAND, None, EffectId.ISLAND)]  # 2 I/S + 1 land
    state.players[1].graveyard = [CardDef(f"opp{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER) for i in range(5)]  # must NOT count
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Tolarian Terror"])
    assert eff["generic"] == 4, eff  # 6 - 2 (own I/S only), opponent's 5 ignored


def test_deem_inferior_tuck_bottom():
    """Deem Inferior: the target's owner tucks it, choosing bottom."""
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
    """Deem Inferior can target any nonland permanent, such as an artifact;
    a land is not a legal target."""
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
    """Delver of Secrets: upkeep look may transform it into a 3/2 flyer."""
    state = GameState(on_the_play=True)
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
    assert delver.card_def.name == "Insectile Aberration"
    assert delver.flags["front_card_def"].name == "Delver of Secrets"
    assert permanent_power(state, delver) == 3 and permanent_toughness(state, delver) == 2  # Insectile Aberration
    assert "flying" in creature_keywords(state, delver)
    # a DFC reverts to its front face when it leaves the battlefield
    delver.damage_marked = 2
    check_state_based_actions(state)
    assert delver not in state.players[0].battlefield
    assert [c.name for c in state.players[0].graveyard] == ["Delver of Secrets"]


def test_delver_non_instant_sorcery_no_transform():
    """A non-instant/sorcery on top: no transform choice opens."""
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
    _pay_cost_fully(state)
    assert len(state.stack) == 1  # Zap NOT countered
    resolve_top_of_stack(state)  # Zap resolves -> destroy Tolarian Terror
    assert tt not in state.players[0].battlefield  # destroyed


def test_island_taps_for_blue_mana():
    """A basic Island, tapped for mana."""
    state = GameState(on_the_play=True)
    island = Permanent(registry.CARD_DEFS["Island"])
    state.battlefield = [island]
    activate_mana_source(state, island)
    assert island.tapped and state.mana_pool.get("U") == 1
    assert state.mana_pool_single_pip == {"U": 1}  # a 1-symbol event -- tagged single-pip


def test_seat_of_the_synod_taps_for_blue_mana():
    """Seat of the Synod: {T}: Add {U}."""
    state = GameState(on_the_play=True)
    seat = Permanent(registry.CARD_DEFS["Seat of the Synod"])
    state.battlefield = [seat]
    activate_mana_source(state, seat)
    assert seat.tapped and state.mana_pool.get("U") == 1
    assert state.mana_pool_single_pip == {"U": 1}  # a 1-symbol event -- tagged single-pip


def test_seat_of_the_synod_counts_for_affinity_as_an_artifact():
    """Seat of the Synod is an artifact land, so it counts toward Affinity
    for artifacts (here, Thoughtcast's cost reduction)."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Seat of the Synod"])]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Thoughtcast"])
    assert eff["generic"] == 3, eff  # 4 - 1 (Seat counts as an artifact)


def test_thoughtcast_affinity_reduces_cost_and_draws_two():
    """Thoughtcast's Affinity for artifacts reduces its cost by artifacts
    controlled, and resolving it draws two cards."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]) for _ in range(3)]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Thoughtcast"])
    assert eff["generic"] == 1, eff  # 4 - 3 artifacts
    state.hand = [registry.CARD_DEFS["Thoughtcast"]]
    state.library = [CardDef(f"d{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    cast_thoughtcast(state, state.hand[0])
    assert len(state.hand) == 2  # drew 2


def test_cryptic_serpent_cost_reduction_caster_graveyard_only():
    """Cryptic Serpent's cost reduces by 1 per instant/sorcery in the
    caster's own graveyard; the opponent's graveyard doesn't count.

    NOTE (correctness, not a test gap): the real Cryptic Serpent also has
    Ward {2}, but EffectId.CRYPTIC_SERPENT's registry entry has no "ward"
    key, so Ward never fires for this card here. Flagged, not fixed."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    state.players[0].graveyard = [CardDef("A", CardType.INSTANT, {"U": 1}, EffectId.FILLER),
                                  CardDef("B", CardType.SORCERY, {"U": 1}, EffectId.FILLER),
                                  CardDef("C", CardType.INSTANT, {"U": 1}, EffectId.FILLER)]  # 3 I/S
    state.players[1].graveyard = [CardDef(f"opp{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER) for i in range(5)]  # must NOT count
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Cryptic Serpent"])
    assert eff["generic"] == 2, eff  # 5 - 3 (own I/S only), opponent's 5 ignored


def test_dispel_counters_targeted_instant():
    """Dispel: cast for real and counter a targeted instant on the stack."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    tgt = _stack_spell(state, "An Instant", controller=1)
    state.active_idx = 0
    dispel = CardDef("Dispel", CardType.INSTANT, {"U": 1}, EffectId.DISPEL)
    state.players[0].hand = [dispel]
    cast_dispel(state, dispel)
    assert state.pending_resolution["kind"] == "choose_stack_target"
    execute_choose_stack_target_option(state, _stack_target_named(state, "An Instant"))
    resolve_top_of_stack(state)
    assert all(e["card_def"].name != "An Instant" for e in state.stack)  # countered off the stack
    assert any(c.name == tgt.name for c in state.players[1].graveyard)


def test_abandon_attachments_decline_no_draw():
    """Abandon Attachments: declining the optional discard means no draw."""
    state = GameState(on_the_play=True)
    aa = CardDef("Abandon Attachments", CardType.INSTANT, {"generic": 1, "U": 1}, EffectId.ABANDON_ATTACHMENTS)
    state.hand = [aa, CardDef("Spare", CardType.LAND, None, EffectId.ISLAND)]
    state.library = [CardDef(n, CardType.LAND, None, EffectId.ISLAND) for n in ("x", "y", "z")]
    cast_abandon_attachments(state, aa)
    assert state.pending_resolution["kind"] == "discard"
    execute_discard_decline(state)
    assert [c.name for c in state.hand] == ["Spare"]  # untouched, no draw


def test_murmuring_mystic_cast_from_hand_via_action_table():
    """Murmuring Mystic's cast, driven through the action table's
    cast/pay-cost/push-to-stack/resolve pipeline."""
    dl = [("Murmuring Mystic", 4), ("Island", 8)]
    byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Murmuring Mystic"]]
    islands = [Permanent(registry.CARD_DEFS["Island"]) for _ in range(4)]
    state.battlefield = islands
    for isl in islands:
        activate_mana_source(state, isl)
    assert state.mana_pool.get("U", 0) == 4
    cast_legal, cast_execute = byname["Cast Murmuring Mystic"]
    assert cast_legal(state)
    cast_execute(state)
    _pay_cost_fully(state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Murmuring Mystic" for p in state.battlefield)


def test_delver_cast_from_hand_via_action_table():
    """Delver of Secrets's cast, driven through the action table's cast pipeline."""
    dl = [("Delver of Secrets", 4), ("Island", 8)]
    byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    state.hand = [registry.CARD_DEFS["Delver of Secrets"]]
    island = Permanent(registry.CARD_DEFS["Island"])
    state.battlefield = [island]
    activate_mana_source(state, island)
    cast_legal, cast_execute = byname["Cast Delver of Secrets"]
    assert cast_legal(state)
    cast_execute(state)
    _pay_cost_fully(state)
    resolve_top_of_stack(state)
    assert any(p.card_def.name == "Delver of Secrets" for p in state.battlefield)


def test_sewer_veillance_cam_sac_ability_via_action_table():
    """Sewer-veillance Cam's {3}{U}, Sacrifice: draw 2 ability, driven
    through the action table's activated-ability legality/payment path."""
    dl = [("Sewer-veillance Cam", 4), ("Island", 8)]
    byname = {a[0]: (a[1], a[2]) for a in drl_env.build_action_table(dl, registry.EFFECT_REGISTRY)}
    state = GameState(on_the_play=True)
    state.phase = Phase.MAIN1
    state.turn_player_idx = 0
    state.active_idx = 0
    cam = Permanent(registry.CARD_DEFS["Sewer-veillance Cam"])
    islands = [Permanent(registry.CARD_DEFS["Island"]) for _ in range(4)]
    state.battlefield = [cam] + islands
    for isl in islands:
        activate_mana_source(state, isl)
    state.library = [CardDef(f"d{i}", CardType.LAND, None, EffectId.ISLAND, basic=True) for i in range(3)]
    legal, execute = byname["Activate Sewer-veillance Cam (draw)"]
    assert legal(state)
    execute(state)
    _pay_cost_fully(state)
    assert cam not in state.battlefield  # sacrificed once the {3}{U} was actually paid
    _drive(state)
    if state.pending_resolution is not None and state.pending_resolution["kind"] == "choose_any_target":
        execute_choose_any_target_decline(state)  # decline the LTB toggle
        _drive(state)
    assert len(state.hand) == 2


def test_sewer_veillance_cam_etb_decline_no_toggle():
    """Declining the ETB's optional toggle still resolves the ability doing
    nothing, rather than skipping the stack."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    opp = Permanent(CardDef("Opp", CardType.CREATURE, None, EffectId.FILLER, power=2, toughness=2))
    opp.slot = 1
    opp.tapped = False
    state.players[1].battlefield = [opp]
    state.players[0].hand = [registry.CARD_DEFS["Sewer-veillance Cam"]]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Sewer-veillance Cam"])
    promote_triggers_to_stack(state)
    assert state.pending_resolution["kind"] == "choose_any_target"
    execute_choose_any_target_decline(state)
    assert len(state.stack) == 1  # the ability still resolves -- just doing nothing
    resolve_top_of_stack(state)
    assert not opp.tapped  # untouched -- declined


def test_utrom_monitor_affinity_reduces_cost_and_resolves_flying():
    """Utrom Monitor's Affinity for artifacts reduces its cost, and it
    resolves as a flying 3/3."""
    state = GameState(on_the_play=True)
    state.battlefield = [Permanent(registry.CARD_DEFS["Great Furnace"]) for _ in range(3)]
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Utrom Monitor"])
    assert eff["generic"] == 1, eff  # 4 - 3 artifacts
    state.hand = [registry.CARD_DEFS["Utrom Monitor"]]
    cast_permanent_from_hand(state, registry.CARD_DEFS["Utrom Monitor"])
    monitor = next(p for p in state.battlefield if p.card_def.name == "Utrom Monitor")
    assert permanent_power(state, monitor) == 3 and has_keyword(state, monitor, "flying")


def test_deem_inferior_cost_reduction_by_cards_drawn_this_turn():
    """Deem Inferior's cost reduces by 1 per card drawn this turn."""
    state = GameState(on_the_play=True)
    state.cards_drawn_this_turn = 2
    eff = drl_env._effective_cast_cost(state, registry.CARD_DEFS["Deem Inferior"])
    assert eff["generic"] == 1, eff  # 3 - 2 cards drawn this turn
    state.cards_drawn_this_turn = 0
    assert drl_env._effective_cast_cost(state, registry.CARD_DEFS["Deem Inferior"])["generic"] == 3  # no draws -> full cost


def test_sleep_escape_illegal_no_sleep_in_graveyard():
    """_sleep_escape_legal is false with no Sleep of the Dead in the graveyard."""
    state = GameState(on_the_play=True)
    state.graveyard = [CardInstance(CardDef(f"g{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER)) for i in range(4)]
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.battlefield = [victim]
    assert not _sleep_escape_legal(state)


def test_sleep_escape_illegal_fewer_than_three_other_graveyard_cards():
    """_sleep_escape_legal is false with fewer than 3 other graveyard cards
    to exile for Escape's additional cost."""
    state = GameState(on_the_play=True)
    sd = CardInstance(registry.CARD_DEFS["Sleep of the Dead"])
    state.graveyard = [sd, CardInstance(CardDef("g1", CardType.INSTANT, {"U": 1}, EffectId.FILLER)),
                       CardInstance(CardDef("g2", CardType.INSTANT, {"U": 1}, EffectId.FILLER))]  # only 2 others
    victim = Permanent(CardDef("Victim", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1))
    victim.slot = 1
    state.battlefield = [victim]
    assert not _sleep_escape_legal(state)


def test_sleep_escape_illegal_no_creature_target():
    """_sleep_escape_legal is false with zero legal creature targets."""
    state = GameState(on_the_play=True)
    sd = CardInstance(registry.CARD_DEFS["Sleep of the Dead"])
    state.graveyard = [sd] + [CardInstance(CardDef(f"g{i}", CardType.INSTANT, {"U": 1}, EffectId.FILLER)) for i in range(3)]
    assert not _sleep_escape_legal(state)  # no creature anywhere


def test_deem_inferior_tuck_removes_an_attacker_from_combat():
    """Tucking an attacking creature into the library removes it from combat."""
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 0
    victim = Permanent(CardDef("Threat", CardType.CREATURE, None, EffectId.FILLER, power=3, toughness=3))
    victim.slot = 1
    state.players[1].battlefield = [victim]
    state.players[1].attackers = [victim]  # player 1 is attacking; player 0 answers during blocks
    state.players[1].library = [CardDef(f"lib{i}", CardType.LAND, None, EffectId.ISLAND) for i in range(3)]
    di = registry.CARD_DEFS["Deem Inferior"]
    state.players[0].hand = [di]

    cast_deem_inferior(state, di)
    execute_choose_any_target_creature(state, 1, "Threat", 1)
    resolve_top_of_stack(state)
    execute_tuck_position(state, "bottom")

    assert victim not in state.players[1].battlefield
    assert victim not in state.players[1].attackers, "506.4: a tucked attacker leaves combat"
