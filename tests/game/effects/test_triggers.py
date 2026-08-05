"""The Madness "decision" branch is exercised together with
madness_and_plot.execute_madness_cast in
tests/game/effects/test_integration_check.py instead, since that chain
needs both modules working together."""
from game import registry, resolution
from game.cards import CardDef, CardType, EffectId
from game.effects import state_based
from game.effects.stack import push_to_stack, resolve_top_of_stack
from game.effects.triggers import promote_triggers_to_stack
from game.state import GameState, Permanent, PlayerState


def test_draw_counter_and_automatic_return():
    _filler_backup = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"on_draw_count": {"count": 3}}
    try:
        snacker = CardDef("Fake Snacker", CardType.CREATURE, {"generic": 1}, EffectId.FILLER)
        state = GameState(on_the_play=True)
        state.library = [CardDef(f"Filler {i}", CardType.SORCERY, {}, None) for i in range(5)]
        state.graveyard = [snacker, snacker]  # two physical copies

        state.draw(1)
        assert state.cards_drawn_this_turn == 1 and state.trigger_queue == []
        state.draw(1)
        assert state.cards_drawn_this_turn == 2 and state.trigger_queue == []
        state.draw(1)  # the third card this turn -- both copies trigger
        assert state.cards_drawn_this_turn == 3
        assert len(state.trigger_queue) == 2
        assert all(e == {"type": "automatic", "kind": "on_draw_count", "card_def": snacker} for e in state.trigger_queue)

        state.draw(1)  # a 4th draw must NOT re-trigger (exactly == 3, not >= 3)
        assert len(state.trigger_queue) == 2

        # 2 simultaneous triggers -- a real placement-order choice, not fixed
        # queue order.
        promote_triggers_to_stack(state)
        assert state.players[0].triggers_fired_this_phase is True  # dense mana-burn-penalty exemption signal
        assert state.pending_resolution["kind"] == "order_triggers"
        assert resolution.order_triggers_options(state) == ["Fake Snacker"]
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place
        resolution.execute_order_triggers_option(state, "Fake Snacker")
        assert state.pending_resolution is None
        assert len(state.stack) == 2
        assert state.trigger_queue == []

        # No decision at any point once each stack entry resolves -- both
        # copies return to the battlefield tapped.
        while state.stack:
            resolve_top_of_stack(state)
        assert state.pending_resolution is None
        assert state.graveyard == []
        assert len(state.battlefield) == 2
        assert all(p.card_def is snacker and p.tapped for p in state.battlefield)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup


def test_two_simultaneous_targeting_etbs():
    # 2 simultaneous targeting ETBs: both go through ONE begin_order_triggers
    # ordering choice, each PLACED via its own etb_trigger hook (not a plain
    # resolve). Placement order is the active player's choice; LIFO means
    # placed-LAST resolves-FIRST.
    etb_calls = []

    def _fake_targeting_etb(tag):
        def hook(state, permanent):
            etb_calls.append(f"{tag}-placed")
            push_to_stack(
                state, permanent.card_def,
                lambda s, cd: etb_calls.append(f"{tag}-resolved"),
                reserves_hand_card=False, is_spell=False,
            )
        return hook

    _filler_backup2 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"etb_trigger": _fake_targeting_etb("A"), "etb_targets": True}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"etb_trigger": _fake_targeting_etb("B"), "etb_targets": True}
    try:
        card_a = CardDef("Targeting A", CardType.CREATURE, None, EffectId.FILLER)
        card_b = CardDef("Targeting B", CardType.CREATURE, None, EffectId.GENEROUS_ENT)
        perm_a = Permanent(card_a)
        perm_b = Permanent(card_b)
        state = GameState(on_the_play=True)
        state.battlefield = [perm_a, perm_b]
        state.trigger_queue = [
            {"type": "etb", "card_def": card_a, "permanent": perm_a},
            {"type": "etb", "card_def": card_b, "permanent": perm_b},
        ]

        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert resolution.order_triggers_options(state) == ["Targeting A", "Targeting B"]
        assert etb_calls == []  # neither hook has run yet -- only PLACEMENT runs a targeting hook

        resolution.execute_order_triggers_option(state, "Targeting A")  # placed FIRST -- resolves LAST
        assert etb_calls == ["A-placed"]
        assert len(state.stack) == 1
        assert state.pending_resolution["kind"] == "order_triggers"  # one more still to place

        resolution.execute_order_triggers_option(state, "Targeting B")  # placed LAST -- resolves FIRST
        assert etb_calls == ["A-placed", "B-placed"]
        assert len(state.stack) == 2
        assert state.pending_resolution is None

        while state.stack:
            resolve_top_of_stack(state)
        assert etb_calls == ["A-placed", "B-placed", "B-resolved", "A-resolved"]  # LIFO: B (placed last) resolves first
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup2
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup


def test_cross_owner_ltb_trigger_apnap_ordering():
    # Cross-owner LTB trigger: a NON-active player's creature dying via
    # state-based action (combat,
    # removal -- check_state_based_actions scans BOTH battlefields every
    # round) must queue into THAT player's own trigger_queue, not the
    # active player's proxy -- and promote_triggers_to_stack must place each
    # owner's group under ITS OWN active_idx (APNAP, 603.3b): the active
    # player's single trigger placed first (resolves last), the opponent's
    # single trigger placed second (resolves first), each stack entry
    # controller-stamped to its TRUE owner, active_idx restored to the real
    # active player once both groups are placed.
    ltb_fired = []
    _filler_backup3 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup2 = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"ltb_trigger": lambda s, p: ltb_fired.append(p.card_def.name)}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"ltb_trigger": lambda s, p: ltb_fired.append(p.card_def.name)}
    _card_defs_backup2 = dict(registry.CARD_DEFS)
    try:
        mine_def = CardDef("Mine Dies", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1)
        theirs_def = CardDef("Theirs Dies", CardType.CREATURE, None, EffectId.GENEROUS_ENT, power=1, toughness=1)
        registry.CARD_DEFS["Mine Dies"] = mine_def
        registry.CARD_DEFS["Theirs Dies"] = theirs_def
        mine = Permanent(mine_def)
        theirs = Permanent(theirs_def)
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0
        state.players[0].battlefield = [mine]
        state.players[1].battlefield = [theirs]
        mine.damage_marked = 1  # lethal for both -- simultaneous SBA deaths, cross-owner
        theirs.damage_marked = 1
        state_based.check_state_based_actions(state)
        # Each landed in its OWN owner's queue, never the active-player proxy.
        assert [e["card_def"].name for e in state.players[0].trigger_queue] == ["Mine Dies"]
        assert [e["card_def"].name for e in state.players[1].trigger_queue] == ["Theirs Dies"]

        promote_triggers_to_stack(state)
        assert state.pending_resolution is None  # 1 entry per owner -- no ordering decision needed by either
        assert len(state.stack) == 2
        assert state.stack[0]["card_def"].name == "Mine Dies" and state.stack[0]["controller"] == 0  # active player's own -- placed first (deepest)
        assert state.stack[1]["card_def"].name == "Theirs Dies" and state.stack[1]["controller"] == 1  # opponent's -- placed second (on top)
        assert state.active_idx == 0  # restored to the true active player, not left on the opponent

        while state.stack:  # LIFO: the opponent's (placed last) resolves FIRST
            resolve_top_of_stack(state)
        assert ltb_fired == ["Theirs Dies", "Mine Dies"]
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup3
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup2
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup2)


def test_opponent_owned_order_triggers_offers_only_their_own_names():
    # The OPPONENT has 2+ simultaneous triggers of their OWN -- order_triggers
    # must be answered by THAT player (active_idx
    # set to them), offering ONLY their own card names, never the active
    # player's -- this is what keeps order_triggers a safe by-name resolution
    # (drl_env._actions_resolution._CHOOSE_NAME_PENDING_KINDS's own guaranteed invariant:
    # every candidate it can ever offer is confined to the deciding player's
    # own deck).
    _filler_backup4 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup3 = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {"ltb_trigger": lambda s, p: None}
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {"ltb_trigger": lambda s, p: None}
    _card_defs_backup3 = dict(registry.CARD_DEFS)
    try:
        opp_a_def = CardDef("Opp Dies A", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=1)
        opp_b_def = CardDef("Opp Dies B", CardType.CREATURE, None, EffectId.GENEROUS_ENT, power=1, toughness=1)
        registry.CARD_DEFS["Opp Dies A"] = opp_a_def
        registry.CARD_DEFS["Opp Dies B"] = opp_b_def
        opp_a = Permanent(opp_a_def)
        opp_b = Permanent(opp_b_def)
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0
        state.players[1].battlefield = [opp_a, opp_b]
        opp_a.damage_marked = 1
        opp_b.damage_marked = 1
        state_based.check_state_based_actions(state)

        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"
        assert state.active_idx == 1  # the OPPONENT orders these -- never the active player
        assert resolution.order_triggers_options(state) == ["Opp Dies A", "Opp Dies B"]  # only THEIR own names

        resolution.execute_order_triggers_option(state, "Opp Dies A")
        assert state.active_idx == 1  # still theirs to place the second one
        resolution.execute_order_triggers_option(state, "Opp Dies B")
        assert state.pending_resolution is None
        assert state.active_idx == 0  # restored once every group (here, just the one) is placed
        assert all(e["controller"] == 1 for e in state.stack)
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup4
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup3
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup3)


def test_cross_owner_single_targeting_etb_does_not_stomp_active_idx():
    # Regression for a real crash: a NON-active player's lone queued
    # targeting ETB (a real target existed, e.g. Masked Vandal) must keep
    # active_idx pointed at its TRUE owner -- the actual decision-maker,
    # and whose "opponent" choose_opponent_permanent's own predicate reads
    # -- until that decision is genuinely answered by _place_trigger_groups'
    # single-entry targeting branch, not get stomped back to
    # original_active_idx the instant the hook opens it (begin_resolution
    # is a flat, single-slot assignment -- state.pending_resolution's own
    # docstring). Confirmed live: an all-False choose_opponent_permanent
    # mask + a RuntimeError, in real cross-deck league play, because the
    # stomp both reassigned who answers the decision AND (since state.
    # opponent re-reads state.active_idx fresh) flipped whose board was
    # being offered as "the opponent."
    resolved = []
    _filler_backup5 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {
        "etb_trigger": lambda state, permanent: resolution.begin_choose_opponent_permanent(
            state, lambda p: p.card_type == CardType.ARTIFACT, lambda s, choice: resolved.append(choice),
        ),
        "etb_targets": True,
    }
    _card_defs_backup4 = dict(registry.CARD_DEFS)
    try:
        vandal_def = CardDef("Fake Vandal", CardType.CREATURE, None, EffectId.FILLER, power=1, toughness=3)
        art_def = CardDef("Their Relic", CardType.ARTIFACT, None, EffectId.FILLER)
        registry.CARD_DEFS["Fake Vandal"] = vandal_def
        registry.CARD_DEFS["Their Relic"] = art_def
        vandal = Permanent(vandal_def)
        art = Permanent(art_def)

        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0  # player 0 holds priority; player 1 (the OTHER player) owns the queued trigger
        state.players[1].battlefield = [vandal]  # trigger owner's own board
        state.players[0].battlefield = [art]  # the target pool: player 1's real opponent
        state.players[1].trigger_queue = [{"type": "etb", "card_def": vandal_def, "permanent": vandal}]

        promote_triggers_to_stack(state)

        assert state.pending_resolution is not None and state.pending_resolution["kind"] == "choose_opponent_permanent"
        assert state.active_idx == 1, "must stay on the trigger's TRUE owner, not stomp back to original_active_idx=0"
        assert resolution.choose_opponent_permanent_options(state) == [("Their Relic", art.slot)], (
            "state.opponent must still mean player 1's real opponent (player 0) -- an active_idx stomp would flip this"
        )

        resolution.execute_choose_opponent_permanent_option(state, "Their Relic", art.slot)

        assert resolved == [("Their Relic", art.slot)]
        assert state.pending_resolution is None
        assert state.active_idx == 0  # restored to the true original active player once the decision (and the group) finishes
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup5
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup4)


def test_targeting_entry_inside_order_triggers_does_not_orphan_the_pick():
    # Same root cause as test_cross_owner_single_targeting_etb_does_not_stomp_
    # active_idx, but inside execute_order_triggers_option's own targeting
    # branch (2+ simultaneous triggers for one owner): placing a targeting
    # entry that opens a REAL nested decision replaces state.pending_
    # resolution with that decision (begin_resolution is a flat, single-slot
    # assignment), detaching the "order_triggers" dict this function's own
    # `pending` local still references. Finishing this pick immediately
    # afterward would either complete/clear the WRONG (freshly-opened)
    # resolution or, if entries remain, silently strand them with nothing in
    # state.pending_resolution ever pointing back to this order_triggers
    # pick again. This drives BOTH remaining entries through to confirm
    # neither one is lost.
    resolved = []
    _filler_backup6 = registry.EFFECT_REGISTRY[EffectId.FILLER]
    _ent_backup4 = registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]
    registry.EFFECT_REGISTRY[EffectId.FILLER] = {
        "etb_trigger": lambda state, permanent: resolution.begin_choose_opponent_permanent(
            state, lambda p: p.card_type == CardType.ARTIFACT, lambda s, choice: resolved.append(choice),
        ),
        "etb_targets": True,
    }
    registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "etb_trigger": lambda state, permanent: resolved.append("plain-placed"),
    }
    _card_defs_backup5 = dict(registry.CARD_DEFS)
    try:
        targeting_def = CardDef("Targeting Vandal", CardType.CREATURE, None, EffectId.FILLER)
        plain_def = CardDef("Plain Trigger", CardType.CREATURE, None, EffectId.GENEROUS_ENT)
        art_def = CardDef("Their Relic", CardType.ARTIFACT, None, EffectId.FILLER)
        registry.CARD_DEFS["Targeting Vandal"] = targeting_def
        registry.CARD_DEFS["Plain Trigger"] = plain_def
        registry.CARD_DEFS["Their Relic"] = art_def
        state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        state.active_idx = 0
        art = Permanent(art_def)
        state.players[1].battlefield = [art]  # the active player's real opponent
        state.trigger_queue = [
            {"type": "etb", "card_def": targeting_def, "permanent": Permanent(targeting_def)},
            {"type": "etb", "card_def": plain_def, "permanent": Permanent(plain_def)},
        ]

        promote_triggers_to_stack(state)
        assert state.pending_resolution["kind"] == "order_triggers"

        resolution.execute_order_triggers_option(state, "Targeting Vandal")  # placed FIRST -- opens its own nested decision
        assert state.pending_resolution["kind"] == "choose_opponent_permanent", (
            "the just-opened decision must be live, not immediately (and wrongly) completed by the order_triggers pick"
        )
        assert resolved == []  # the targeting hook opened its choice but hasn't fired on_complete yet

        resolution.execute_choose_opponent_permanent_option(state, "Their Relic", art.slot)
        assert resolved == [("Their Relic", art.slot)]
        assert state.pending_resolution is not None and state.pending_resolution["kind"] == "order_triggers", (
            "the order_triggers pick must be reinstalled/continued, not lost, once the nested decision completes"
        )
        assert resolution.order_triggers_options(state) == ["Plain Trigger"]  # the remaining entry is still there to place

        resolution.execute_order_triggers_option(state, "Plain Trigger")  # a plain entry is only PUSHED at placement, not run
        assert state.pending_resolution is None
        assert state.active_idx == 0
        assert len(state.stack) == 1  # the second entry made it onto the stack, not stranded
        resolve_top_of_stack(state)
        assert resolved == [("Their Relic", art.slot), "plain-placed"]  # its own hook only fires once its stack entry resolves
    finally:
        registry.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup6
        registry.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _ent_backup4
        registry.CARD_DEFS.clear()
        registry.CARD_DEFS.update(_card_defs_backup5)
