"""Tests for game.resolution.handlers: the deck-agnostic pending-resolution
handlers (discard, mulligan, madness routing, sacrifice, cross-player/any-
target choosing, blocking, trigger ordering, explore, up-to-N multi-target
selection). Exercises these primitives directly against hand-built states,
bypassing drl_env entirely (no card wires into every one of these yet)."""

from game import registry
from game.cards import CardDef, CardType, EffectId
from game.resolution.handlers import (
    begin_choose_any_target,
    begin_choose_opponent_permanent,
    begin_choose_permanent,
    begin_choose_up_to_any_target,
    begin_choose_up_to_graveyard,
    begin_declare_blockers,
    begin_discard,
    begin_madness_decision,
    begin_mulligan,
    begin_order_triggers,
    begin_sacrifice,
    begin_search_fetch,
    bottom_options,
    choose_any_target_creature_options,
    choose_any_target_options,
    choose_graveyard_card_options,
    choose_opponent_permanent_options,
    choose_permanent_options,
    complete_resolution,
    declare_blocker_assignment,
    discard_options,
    execute_bottom_option,
    execute_choose_any_target_creature,
    execute_choose_any_target_player,
    execute_choose_graveyard_card_decline,
    execute_choose_graveyard_card_option,
    execute_choose_opponent_permanent_option,
    execute_discard_decline,
    execute_discard_option,
    execute_madness_decline,
    execute_mulligan_keep,
    execute_mulligan_take,
    execute_order_triggers_option,
    execute_sacrifice_option,
    execute_scry_surveil_option,
    explore,
    madness_decision_options,
    mulligan_decision_options,
    order_triggers_options,
    refizzle_if_now_targetless,
    sacrifice_options,
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


def test_discard_optional_declined():
    # Optional discard, declined: hand/graveyard untouched, still completes
    # with an empty discarded_cards list (Highway Robbery/Melded Moxite's
    # own "if you do" check reads bool(discarded_cards) for exactly this).
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 1, optional=True, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_decline(state)
    assert completed == [[]]
    assert [c.name for c in state.hand] == ["A"]
    assert state.graveyard == []


def test_discard_optional_taken():
    # Optional discard, taken.
    state = GameState(on_the_play=True)
    state.hand = [_card("A")]
    completed = []
    begin_discard(state, 1, optional=True, on_complete=lambda s, cards: completed.append(cards))
    execute_discard_option(state, "A")
    assert len(completed) == 1 and [c.name for c in completed[0]] == ["A"]
    assert state.hand == []
    assert [c.name for c in state.graveyard] == ["A"]


def test_mulligan_london_style_bottoms_and_logs_draws():
    # Mulligan (London style): begin_mulligan/execute_mulligan_take loop
    # twice (redraw to 7 each time, mulligans_taken incrementing), then
    # execute_mulligan_keep bottoms exactly mulligans_taken (2) cards before
    # completing.
    events = []
    state = GameState(on_the_play=True, event_log=events)
    state.library = [_card(f"L{i}") for i in range(20)]
    state.rng.shuffle(state.library)
    state.draw(7)  # new_multiplayer_game_state's own eager opening draw -- begin_mulligan's own precondition
    completed = []
    begin_mulligan(state, on_complete=lambda s: completed.append(True))
    assert mulligan_decision_options(state) == ["keep", "mulligan"]
    assert state.pending_resolution["kind"] == "mulligan_decision"
    assert len(state.hand) == 7

    execute_mulligan_take(state)
    assert state.mulligans_taken == 1
    assert len(state.hand) == 7  # redrawn fresh, not bottomed yet
    assert state.pending_resolution["kind"] == "mulligan_decision"

    execute_mulligan_take(state)
    assert state.mulligans_taken == 2
    assert len(state.hand) == 7
    assert completed == []  # still deciding -- on_complete hasn't fired

    execute_mulligan_keep(state)
    assert completed == []  # not yet -- 2 cards still need to be bottomed
    assert state.pending_resolution["kind"] == "mulligan_bottom"
    bottomed = []
    while state.pending_resolution is not None:
        name = bottom_options(state)[0]
        bottomed.append(name)
        execute_bottom_option(state, name)
    assert completed == [True]
    assert len(state.hand) == 5  # 7 - 2 bottomed
    assert [c.name for c in state.library[-2:]] == bottomed  # bottomed, in the order chosen

    # Every hand SEEN is a library->hand "draw" zone_move (GameState.draw,
    # the single generic hook): three here -- the opener, then the two
    # redraws -- 7 cards each, in order.
    draws = [e["cards"] for e in events if e.get("reason") == "draw"]
    assert len(draws) == 3 and all(len(d) == 7 for d in draws)
    # Each thrown-back hand (mulligan_take) is exactly the hand drawn just
    # before it, so draws[0] and draws[1] are the two mulliganed hands.
    takes = [e["cards"] for e in events if e.get("reason") == "mulligan_take"]
    assert takes == draws[:2]  # seen == thrown back
    assert [e["card"] for e in events if e.get("reason") == "mulligan_bottom"] == bottomed


def test_mulligan_keep_with_zero_mulligans_skips_bottom():
    # Keeping with 0 mulligans taken never opens a mulligan_bottom at all.
    state = GameState(on_the_play=True)
    state.library = [_card(f"L{i}") for i in range(20)]
    state.draw(7)
    completed = []
    begin_mulligan(state, on_complete=lambda s: completed.append(True))
    execute_mulligan_keep(state)
    assert completed == [True]
    assert state.pending_resolution is None
    assert len(state.hand) == 7


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
    # exercise a creature predicate (Dread Return's own shape) against
    # the primitive.
    state = GameState(on_the_play=True)
    state.battlefield = [
        _permanent("Bear", CardType.CREATURE),
        _permanent("Wolf", CardType.CREATURE),
        _permanent("Mountain", CardType.LAND),
    ]
    completed = []
    begin_sacrifice(state, lambda p: p.card_def.card_type == CardType.CREATURE, 2, lambda s, ok: completed.append(ok))
    assert sacrifice_options(state) == ["Bear", "Wolf"]  # the Mountain never qualifies
    execute_sacrifice_option(state, "Bear")
    assert completed == []
    execute_sacrifice_option(state, "Wolf")
    assert completed == [True]
    assert sorted(p.card_def.name for p in state.battlefield) == ["Mountain"]
    assert sorted(c.name for c in state.graveyard) == ["Bear", "Wolf"]


def test_sacrifice_land_predicate():
    # ...and a land predicate (Fireblast/Lava Dart's shape, Highway
    # Robbery's discard-or-sac choice) against the same primitive.
    state = GameState(on_the_play=True)
    state.battlefield = [_permanent("Mountain", CardType.LAND), _permanent("Bear", CardType.CREATURE)]
    completed = []
    begin_sacrifice(state, lambda p: p.card_def.name == "Mountain", 1, lambda s, ok: completed.append(ok))
    assert sacrifice_options(state) == ["Mountain"]  # the Bear never qualifies, even though it's a permanent
    execute_sacrifice_option(state, "Mountain")
    assert completed == [True]
    assert [p.card_def.name for p in state.battlefield] == ["Bear"]


def test_choose_opponent_permanent_targets_specific_slot():
    # Cross-player targeting: begin_choose_opponent_permanent
    # targets state.opponent's battlefield, addressed by (name, slot) --
    # not name alone, since two same-named OPPOSING permanents aren't
    # necessarily interchangeable. Only correct once the referencing player
    # is already the active one (blocking's own defender-decision channel
    # flips active_idx before ever calling this) -- simulated here by
    # setting active_idx directly to "the defender," same as that channel
    # would.
    attacker_bogle_1 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2 = _permanent("Slippery Bogle", CardType.CREATURE)
    attacker_bogle_2.slot = 2
    attacker_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2, attacker_land]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective

    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert choose_opponent_permanent_options(state) == [("Slippery Bogle", 1), ("Slippery Bogle", 2)]  # the Forest never qualifies
    execute_choose_opponent_permanent_option(state, "Slippery Bogle", 2)
    assert completed == [("Slippery Bogle", 2)]  # the SPECIFIC slot chosen, not an arbitrary same-named match


def test_choose_opponent_permanent_empty_options_fizzles():
    # Empty-options safety net: no eligible opposing permanent -> fizzles
    # immediately with None, same convention as begin_choose_permanent.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == [None]


def test_refizzle_if_now_targetless_fizzles_choose_opponent_permanent():
    # begin_choose_opponent_permanent only validates non-empty options ONCE,
    # at open time. If a state_based_actions pass removes the only legal
    # target before the next decision point, the pending resolution would
    # otherwise sit there with an all-False mask and no recovery --
    # game.turn._run_priority_round_gen's own refizzle_if_now_targetless call
    # is what catches that. Simulates the gap directly: open with one legal
    # target, remove it (standing in for an SBA), then confirm the re-check
    # fizzles cleanly with None instead of leaving a dead resolution.
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == []  # still open -- one legal target existed at open time
    state.players[0].battlefield = []  # the SBA's own effect: the only target just died
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_fizzles_choose_permanent():
    # Same gap, own-battlefield half (begin_choose_permanent).
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    completed = []
    begin_choose_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert completed == []
    state.players[0].battlefield = []
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_fizzles_choose_any_target_creature_only():
    # Same gap, choose_any_target's creature-only mode (allow_players=False,
    # optional=False -- the one configuration that can ever go all-False; with
    # allow_players=True a player is always legal, and optional=True always
    # offers a decline, so neither ever needs this re-check).
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    completed = []
    begin_choose_any_target(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
        allow_players=False, optional=False,
    )
    assert completed == []
    state.players[0].battlefield = []
    assert refizzle_if_now_targetless(state) is True
    assert completed == [None]
    assert state.pending_resolution is None


def test_refizzle_if_now_targetless_leaves_still_legal_resolution_alone():
    # No-op when the pending resolution's options are still non-empty -- the
    # common case, every priority-loop iteration.
    target = _permanent("Slippery Bogle", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [target]
    state.active_idx = 1
    completed = []
    begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert refizzle_if_now_targetless(state) is False
    assert completed == []  # untouched -- the target is still there
    assert state.pending_resolution is not None


def test_refizzle_if_now_targetless_ignores_unrelated_kinds():
    # search_fetch reads the library, which state_based_actions never mutates --
    # not in refizzle_if_now_targetless's covered set, so even an empty-options
    # search_fetch (its own open-time safety net already fizzled it) is simply
    # not its concern. No pending resolution at all is the other no-op case.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.library = []
    completed = []
    begin_search_fetch(state, lambda c: True, lambda s, name: completed.append(name))
    assert completed == [None]  # already fizzled by its OWN open-time check
    assert state.pending_resolution is None
    assert refizzle_if_now_targetless(state) is False  # nothing pending -- no-op


def test_choose_any_target_creature_and_player():
    # "Any target" (begin_choose_any_target): a single target spanning BOTH
    # battlefields' creatures plus either player -- real Magic's "any
    # target" (Lightning Bolt). Creatures addressed by (side, name, slot)
    # so a same-named creature on each side stays distinguishable; players
    # by index.
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    assert choose_any_target_creature_options(state) == [(0, "Grizzly Bears", 1), (1, "Grizzly Bears", 1)]  # both sides, Forest excluded
    assert ("player", 0) in choose_any_target_options(state) and ("player", 1) in choose_any_target_options(state)
    execute_choose_any_target_creature(state, 1, "Grizzly Bears", 1)  # the OPPONENT's copy specifically
    assert completed == [("creature", 1, "Grizzly Bears", 1)]

    completed = []
    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t))
    execute_choose_any_target_player(state, 0)  # legal to target yourself (real Magic)
    assert completed == [("player", 0)]


def test_choose_any_target_no_players_no_creatures_fizzles():
    # allow_players=False + no creature anywhere -> immediate None (fizzle/
    # can't-target), same empty-options net as the other primitives.
    empty = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    empty.players[0].battlefield = [_permanent("Forest", CardType.LAND)]
    completed = []
    begin_choose_any_target(empty, lambda p: p.card_type == CardType.CREATURE, lambda s, t: completed.append(t), allow_players=False)
    assert completed == [None]


def test_choose_any_target_no_players_with_creature_offers_creatures_only():
    # allow_players=False WITH a creature -> creatures only, no player option offered
    mine = _permanent("Grizzly Bears", CardType.CREATURE)
    theirs = _permanent("Grizzly Bears", CardType.CREATURE)  # same name, opposite side
    my_land = _permanent("Forest", CardType.LAND)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [mine, my_land]
    state.players[1].battlefield = [theirs]

    begin_choose_any_target(state, lambda p: p.card_type == CardType.CREATURE, lambda s, t: None, allow_players=False)
    assert all(o[0] == "creature" for o in choose_any_target_options(state))


def test_declare_blockers_gang_blocking_and_done():
    # Blocking: begin_declare_blockers/
    # declare_blocker_assignment, driven directly against a hand-built
    # state (bypassing game.turn._declare_blockers_gen's active_idx-flip --
    # simulated here the same way the cross-player check above does, by
    # setting active_idx to "the defender" up front). Also bypasses
    # drl_env's own _assign_blocker_legal eligibility gate -- this
    # exercises the resolution primitives directly, so a "re-open
    # begin_declare_blockers after each assignment" step is done by hand
    # here rather than relying on drl_env._assign_blocker_execute's own
    # nested on_complete to do it.
    bear = _permanent("Bear", CardType.CREATURE)
    wolf = _permanent("Wolf", CardType.CREATURE)
    grizzly = _permanent("Grizzly Bears", CardType.CREATURE)
    panther = _permanent("Panther", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [bear, wolf]
    state.players[0].attackers = [bear, wolf]
    state.players[1].battlefield = [grizzly, panther]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []  # real attackers declared -- does not auto-complete
    assert state.pending_resolution["kind"] == "declare_blockers"

    # Assign Grizzly Bears to block Bear specifically (not Wolf) -- the
    # nested choose_opponent_permanent offers both.
    step1_done = []
    declare_blocker_assignment(state, grizzly, on_complete=lambda s: step1_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step1_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly]}  # attacker -> LIST of blockers (gang-blocking)

    # GANG-BLOCKING: re-open the consult and assign Panther to the SAME
    # attacker (Bear). An already-blocked attacker is STILL offered --
    # multiple blockers may pile onto one attacker.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    step2_done = []
    declare_blocker_assignment(state, panther, on_complete=lambda s: step2_done.append(True))
    assert choose_opponent_permanent_options(state) == [("Bear", 1), ("Wolf", 1)]  # Bear STILL offered -- gang-block
    execute_choose_opponent_permanent_option(state, "Bear", 1)
    assert step2_done == [True]
    assert state.players[0].blocked_by == {bear: [grizzly, panther]}  # two blockers on one attacker

    # "Done blocking" (drl_env's action): closes a still-open
    # declare_blockers resolution outright, no assignment required.
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    complete_resolution(state)
    assert completed == [True]


def test_declare_blockers_no_attackers_auto_completes():
    # No attackers at all: auto-completes immediately, same empty-options
    # precedent as begin_choose_permanent/begin_search_fetch.
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.active_idx = 1
    completed = []
    begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == [True]


def test_declare_blocker_assignment_extra_predicate():
    # declare_blocker_assignment's extra_predicate: this module stays
    # effect-agnostic (see its own module docstring) and doesn't import
    # game.effects.stats itself, so the actual restriction is supplied by
    # the CALLER (drl_env._assign_blocker_execute, using game.has_keyword)
    # -- this proves the parameter itself is correctly applied on top of
    # the usual "unblocked attacker" filter, using a plain stand-in
    # predicate rather than a real keyword lookup.
    flyer = _permanent("Flyer", CardType.CREATURE)
    grounded = _permanent("Grounded", CardType.CREATURE)
    non_flying_blocker = _permanent("Non-Flying Blocker", CardType.CREATURE)
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [flyer, grounded]
    state.players[0].attackers = [flyer, grounded]
    state.players[1].battlefield = [non_flying_blocker]
    state.active_idx = 1

    completed = []
    declare_blocker_assignment(
        state, non_flying_blocker, on_complete=lambda s: completed.append(True),
        extra_predicate=lambda p: p is not flyer,  # stand-in: "flyer needs a flying blocker, and this one isn't"
    )
    assert choose_opponent_permanent_options(state) == [("Grounded", 1)]  # Flyer excluded by extra_predicate
    execute_choose_opponent_permanent_option(state, "Grounded", 1)
    assert completed == [True]
    assert state.players[0].blocked_by == {grounded: [non_flying_blocker]}


def test_order_triggers_placement_order_is_lifo_at_resolution():
    # begin_order_triggers: 2+ simultaneous
    # triggers get a real placement-order choice -- PLACEMENT order, not
    # resolution order (the stack is LIFO). Driven directly against a
    # hand-built state, bypassing game.effects.triggers.promote_triggers_
    # to_stack entirely (this module doesn't import game.effects.triggers
    # -- see its own docstring), using plain no-op resolve functions since
    # only the ordering mechanism itself is under test here.
    resolved_order = []
    entry_a = {"card_def": CardDef("Trigger A", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    entry_b = {"card_def": CardDef("Trigger B", CardType.CREATURE, None, None), "resolve": lambda s, cd: resolved_order.append(cd.name)}
    state = GameState(on_the_play=True)
    completed = []
    begin_order_triggers(state, [entry_a, entry_b], on_complete=lambda s: completed.append(True))
    assert order_triggers_options(state) == ["Trigger A", "Trigger B"]

    execute_order_triggers_option(state, "Trigger A")  # placed FIRST -- resolves LAST
    assert completed == []  # one more still to place
    assert state.stack == [entry_a]
    assert order_triggers_options(state) == ["Trigger B"]  # already-placed one no longer offered

    execute_order_triggers_option(state, "Trigger B")  # placed LAST -- resolves FIRST
    assert completed == [True]
    assert state.stack == [entry_a, entry_b]  # placement order: A then B
    assert state.pending_resolution is None

    while state.stack:  # LIFO: B (placed last) actually resolves first
        entry = state.stack.pop()
        entry["resolve"](state, entry["card_def"])
    assert resolved_order == ["Trigger B", "Trigger A"]


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
    # begin_choose_up_to_graveyard / begin_choose_up_to_any_target: N-ary
    # "up to X" multi-target selection with BY-IDENTITY exclusion (two
    # same-named copies both reachable) and optional decline (the "up to"
    # slack). graveyard up-to-2: two same-named "Bolt" instances -- both
    # offered, and picking one excludes only THAT instance (its twin stays
    # choosable).
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


def test_choose_up_to_any_target_board_identity_exclusion():
    # board up-to-2: two same-named "Bear" (distinct slots) -- both
    # reachable, picking slot 1 excludes only it (slot 2 stays choosable).
    st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    st.active_idx = 0
    bear1 = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    bear1.slot = 1
    bear2 = Permanent(CardDef("Bear", CardType.CREATURE, {"G": 1}, EffectId.FILLER, power=2, toughness=2))
    bear2.slot = 2
    st.players[0].battlefield = [bear1, bear2]
    descs = []
    begin_choose_up_to_any_target(st, lambda p: p.card_type == CardType.CREATURE, 2, lambda s, d: descs.extend(d))
    opts = choose_any_target_creature_options(st)
    assert (0, "Bear", 1) in opts and (0, "Bear", 2) in opts
    execute_choose_any_target_creature(st, 0, "Bear", 1)
    opts = choose_any_target_creature_options(st)
    assert (0, "Bear", 1) not in opts and (0, "Bear", 2) in opts  # identity (slot) exclusion
    execute_choose_any_target_creature(st, 0, "Bear", 2)
    assert st.pending_resolution is None and descs == [("creature", 0, "Bear", 1), ("creature", 0, "Bear", 2)]
