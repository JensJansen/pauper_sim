"""Tests for game.resolution's library/graveyard/hand handlers
(handlers_library.py): discard (including Madness routing), sacrifice,
explore, and up-to-N graveyard selection. Exercises these primitives
directly against hand-built states, bypassing drl_env entirely (no card
wires into every one of these yet)."""

import pytest

from game import registry
from game.cards import CardDef, CardType, EffectId
from game.resolution import (
    begin_choose_up_to_graveyard,
    begin_discard,
    begin_madness_decision,
    begin_sacrifice,
    choose_graveyard_card_options,
    choose_permanent_options,
    discard_options,
    execute_choose_graveyard_card_decline,
    execute_choose_graveyard_card_option,
    execute_choose_permanent_option,
    execute_discard_decline,
    execute_discard_option,
    execute_madness_decline,
    execute_scry_surveil_option,
    explore,
    madness_decision_options,
)
from game.state import GameState, Permanent, PlayerState


def _card(name):
    return CardDef(name, CardType.SORCERY, {"generic": 1}, None)


def _permanent(name, card_type):
    return Permanent(CardDef(name, card_type, None, None))


def test_discard_mandatory_fewer_than_n_available():
    # Mandatory discard of fewer cards than n asks for: never crashes,
    # stops once hand is exhausted instead of running remaining negative.
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 2, optional=False, on_complete=lambda s, cards: completed.append(cards))
    assert discard_options(state) == ["A"]
    execute_discard_option(state, "A")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A"]
    assert state.hand == []
    assert [c.name for c in state.graveyard] == ["A"]


def test_discard_mandatory_exactly_n():
    # Mandatory discard of exactly n, from a larger hand.
    state = GameState(on_the_play=True)
    state.hand = [_card("A"), _card("B"), _card("C")]
    completed = []
    begin_discard(state, 2, optional=False, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_option(state, "A")
    assert completed == []  # one more still required
    execute_discard_option(state, "B")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A", "B"]
    assert [c.name for c in state.hand] == ["C"]
    assert sorted(c.name for c in state.graveyard) == ["A", "B"]


@pytest.mark.parametrize("take", [False, True], ids=["declined", "taken"])
def test_discard_optional(take):
    # Optional discard, both branches: declined leaves hand/graveyard
    # untouched but still completes with an empty discarded_cards list
    # (Highway Robbery/Melded Moxite's own "if you do" check reads
    # bool(discarded_cards) for exactly this); taken moves the card for real.
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 1, optional=True, on_complete=lambda s, cards: completed.append(cards))
    if take:
        execute_discard_option(state, "A")
        assert len(completed) == 1 and [c.name for c in completed[0]] == ["A"]
        assert state.hand == []
        assert [c.name for c in state.graveyard] == ["A"]
    else:
        execute_discard_decline(state)
        assert completed == [[]]
        assert [c.name for c in state.hand] == ["A"]
        assert state.graveyard == []


def test_discard_madness_routes_to_exile_and_queues_decision():
    # Madness routing: a discarded card whose EffectId has a "madness"
    # registry spec goes to exile + the trigger queue, not the graveyard.
    # No real madness card exists yet (deck assembly is out of scope), so
    # this borrows EffectId.FILLER for the duration of the check, saving
    # and restoring its real (empty) registry entry around it.
    filler_entry_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"madness": {"cost": {"R": 1}, "resolve": lambda s, c: None}}
    try:
        madness_card = CardDef("Fake Madness Card", CardType.INSTANT, {"generic": 1, "R": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.hand = [madness_card]
        completed = []
        begin_discard(state, 1, optional=False, on_complete=lambda s, cards: completed.append(cards))
        execute_discard_option(state, "Fake Madness Card")
        assert len(completed) == 1 and completed[0] == [madness_card]
        assert state.hand == [] and state.graveyard == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Madness Card"]
        assert state.trigger_queue == [{"type": "decision", "kind": "madness", "card_def": madness_card}]

        # Promoting the queue (game.effects.triggers.promote_triggers_to_
        # stack's job in real play) and declining: back out of exile, into
        # the graveyard.
        state.trigger_queue.clear()
        drain_completed = []
        begin_madness_decision(state, madness_card, on_complete=lambda s: drain_completed.append(True))
        assert madness_decision_options(state) == ["cast", "decline"]
        execute_madness_decline(state)
        assert drain_completed == [True]
        assert state.exile == []
        assert [c.name for c in state.graveyard] == ["Fake Madness Card"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = filler_entry_backup


def test_sacrifice_creature_predicate():
    # begin_sacrifice: predicate-based, not hardcoded to creatures --
    # exercise a creature predicate (Dread Return's own shape) against the
    # primitive. Each pick is now a real choose_permanent sub-decision
    # (exact name+slot), chained one at a time until n are sacrificed --
    # never an arbitrary first-same-name match (see test below for why that
    # distinction is load-bearing).
    state = GameState(on_the_play=True)
    bear, wolf = _permanent("Bear", CardType.CREATURE), _permanent("Wolf", CardType.CREATURE)
    state.battlefield = [bear, wolf, _permanent("Mountain", CardType.LAND)]
    # sacrifice_to_graveyard (the real removal path this now goes through)
    # treats an unregistered name as a TOKEN that ceases to exist (real
    # Magic 111.7) rather than hitting the graveyard -- register these two
    # fake names for real so this test's graveyard assertion below actually
    # exercises the ordinary "real card" path, not the token one.
    registry.CARD_DEFS["Bear"], registry.CARD_DEFS["Wolf"] = bear.card_def, wolf.card_def
    try:
        completed = []
        begin_sacrifice(state, lambda p: p.card_def.card_type == CardType.CREATURE, 2, lambda s, ok: completed.append(ok))
        assert state.pending_resolution["kind"] == "choose_permanent"
        assert choose_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]  # the Mountain never qualifies
        execute_choose_permanent_option(state, "Bear", 1)
        assert completed == []
        assert state.pending_resolution["kind"] == "choose_permanent"  # re-opened for the 2nd pick
        assert choose_permanent_options(state) == [("Wolf", 1)]
        execute_choose_permanent_option(state, "Wolf", 1)
        assert completed == [True]
        assert sorted(p.card_def.name for p in state.battlefield) == ["Mountain"]
        assert sorted(c.name for c in state.graveyard) == ["Bear", "Wolf"]
    finally:
        del registry.CARD_DEFS["Bear"]
        del registry.CARD_DEFS["Wolf"]


def test_sacrifice_land_predicate():
    # ...and a land predicate (Fireblast's alt-cost, Lava Dart's Flashback)
    # against the same primitive.
    state = GameState(on_the_play=True)
    state.battlefield = [_permanent("Mountain", CardType.LAND), _permanent("Bear", CardType.CREATURE)]
    completed = []
    begin_sacrifice(state, lambda p: p.card_def.name == "Mountain", 1, lambda s, ok: completed.append(ok))
    assert choose_permanent_options(state) == [("Mountain", 1)]  # the Bear never qualifies, even though it's a permanent
    execute_choose_permanent_option(state, "Mountain", 1)
    assert completed == [True]
    assert [p.card_def.name for p in state.battlefield] == ["Bear"]


def test_sacrifice_lets_agent_choose_which_same_named_copy():
    # The whole point of routing sacrifice through choose_permanent instead
    # of a plain by-name pick: two same-named permanents are NOT
    # interchangeable once one differs from the other (an attached Aura, a
    # counter, ...) -- the agent must be able to choose exactly which one
    # to sacrifice, not have the engine silently take "the first one found
    # in battlefield order."
    state = GameState(on_the_play=True)
    tagged = _permanent("Bear", CardType.CREATURE)
    tagged.slot = 1
    tagged.flags["enchanted"] = True  # stand-in for "this copy carries something the other doesn't"
    plain = _permanent("Bear", CardType.CREATURE)
    plain.slot = 2
    state.battlefield = [tagged, plain]  # the "first found" copy is the TAGGED one
    registry.CARD_DEFS["Bear"] = tagged.card_def  # a real (non-token) name -- see the test above for why
    try:
        completed = []
        begin_sacrifice(state, lambda p: p.card_def.card_type == CardType.CREATURE, 1, lambda s, ok: completed.append(ok))
        assert choose_permanent_options(state) == [("Bear", 1), ("Bear", 2)]
        execute_choose_permanent_option(state, "Bear", 2)  # deliberately choose the PLAIN copy
        assert completed == [True]
        assert [p.card_def.name for p in state.battlefield] == ["Bear"]
        assert state.battlefield[0] is tagged and state.battlefield[0].flags.get("enchanted") is True
    finally:
        del registry.CARD_DEFS["Bear"]


def test_explore_land_goes_to_hand():
    # explore (Map token / Fanatical Offering): a land on top goes to hand.
    state = GameState(on_the_play=True)
    creature = Permanent(CardDef("Explorer", CardType.CREATURE, None, None, power=1, toughness=1))
    state.battlefield = [creature]
    state.library = [CardDef("A Land", CardType.LAND, None, None), CardDef("A Spell", CardType.INSTANT, {"U": 1}, None)]
    explore(state, creature)
    assert [c.name for c in state.hand] == ["A Land"]  # land -> hand
    assert creature.counters.get("+1/+1", 0) == 0  # no counter for a land


def test_explore_nonland_adds_counter_then_surveils():
    # a nonland puts a +1/+1 counter on the exploring creature, then
    # surveil 1 (keep on top or bin) on that same card.
    state = GameState(on_the_play=True)
    creature = Permanent(CardDef("Explorer", CardType.CREATURE, None, None, power=1, toughness=1))
    state.battlefield = [creature]
    state.library = [CardDef("A Spell", CardType.INSTANT, {"U": 1}, None), CardDef("Next", CardType.LAND, None, None)]
    explore(state, creature)
    assert creature.counters["+1/+1"] == 1  # nonland -> +1/+1
    assert state.pending_resolution["kind"] == "surveil"  # then "may put it in graveyard"
    execute_scry_surveil_option(state, "dispose")
    assert [c.name for c in state.graveyard] == ["A Spell"]


def test_choose_up_to_graveyard_identity_exclusion():
    # begin_choose_up_to_graveyard: N-ary "up to X" multi-target selection
    # with BY-IDENTITY exclusion (two same-named copies both reachable) and
    # optional decline (the "up to" slack). graveyard up-to-2: two
    # same-named "Bolt" instances -- both offered, and picking one excludes
    # only THAT instance (its twin stays choosable).
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.active_idx = 0
    g1 = st.new_instance(CardDef("Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER))
    g2 = st.new_instance(CardDef("Bolt", CardType.INSTANT, {"R": 1}, EffectId.FILLER))
    st.players[0].graveyard = [g1, g2]
    picked = []
    begin_choose_up_to_graveyard(st, lambda c: True, 2, lambda s, chosen: picked.extend(chosen))
    assert sorted(o.name for o in choose_graveyard_card_options(st)) == ["Bolt", "Bolt"]
    execute_choose_graveyard_card_option(st, g1)
    assert g1 not in choose_graveyard_card_options(st) and g2 in choose_graveyard_card_options(st)  # identity exclusion
    execute_choose_graveyard_card_option(st, g2)  # 2nd pick -> max reached
    assert st.pending_resolution is None and picked == [g1, g2]


def test_choose_up_to_graveyard_decline_stops_early():
    # graveyard decline after one (the "up to" slack)
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.active_idx = 0
    h1 = st.new_instance(CardDef("A", CardType.INSTANT, {"R": 1}, EffectId.FILLER))
    h2 = st.new_instance(CardDef("B", CardType.INSTANT, {"R": 1}, EffectId.FILLER))
    st.players[0].graveyard = [h1, h2]
    picked = []
    begin_choose_up_to_graveyard(st, lambda c: True, 2, lambda s, chosen: picked.extend(chosen))
    execute_choose_graveyard_card_option(st, h1)
    assert st.pending_resolution["kind"] == "choose_graveyard_card"
    execute_choose_graveyard_card_decline(st)
    assert st.pending_resolution is None and picked == [h1]
