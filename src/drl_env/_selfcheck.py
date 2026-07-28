"""Assert-based self-checks for the drl_env action table -- run via
`python -m drl_env` from src/. Kept in their own module so the engine
(_actions) stays free of test code."""

import os

import game

from . import _actions
from ._actions import *
from ._seat import _for_player, _lost


def _run_self_checks():
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m drl_env` from
    # src/ (drl_env/__main__.py calls this). Exercises Plot and the on-cast
    # trigger hook (item 11) through the REAL _plot_legal/_plot_execute/
    # _cast_from_exile_legal/_cast_from_exile_execute functions -- not a
    # parallel reimplementation. No real Plot/Guttersnipe card exists yet
    # (deck assembly out of scope), so this temporarily injects into the
    # global game.CARD_DEFS/game.EFFECT_REGISTRY, saving/restoring both.
    from game.cards import CardDef, CardType, EffectId
    from game.state import GameState, Permanent, PlayerState

    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    PLOT_COST = {"generic": 1, "B": 1}  # {B}, not {R} -- EffectId.SWAMP is a real, already-correctly-wired
    # mana source (registry.py's derived views like SIMPLE_MANA_SOURCE_EFFECTS/_FIXED_SOURCE_COLOR are built
    # once at import time; injecting a fake "mana" spec onto FILLER here wouldn't be reflected in them, so the
    # legality pre-check (plan_payment) would wrongly see no valid source -- reusing a real fixed-color land
    # sidesteps that entirely rather than also having to patch the derived views to match).

    on_cast_calls = []
    plot_spell = CardDef("Fake Plot Spell", CardType.SORCERY, PLOT_COST, EffectId.FILLER)
    game.CARD_DEFS["Fake Plot Spell"] = plot_spell
    game.EFFECT_REGISTRY[EffectId.FILLER] = {
        "cast": {"resolve": lambda s, c: None},
        "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
    }
    # Guttersnipe stand-in: a permanent whose registry entry has an
    # "on_cast" trigger -- borrows EffectId.GENEROUS_ENT for the duration.
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1  # Plot Speed defaults to SORCERY (a CardType.SORCERY card, no override) -- needs a sorcery-speed phase to be legal at all now
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]

        # Plot it: pay {1}{B}, exile with this turn's stamp. Both Swamps
        # are needed (1 generic + 1 B); pay_cost is always interactive
        # regardless of what the legality pre-check found, so this taps
        # them one at a time.
        assert _plot_legal("Fake Plot Spell", PLOT_COST, game.turn.Speed.SORCERY)(state)
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        assert state.pending_resolution["kind"] == "pay_cost"
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                # Both Swamps produce only B -- the 2nd tap's B floats
                # into the pool instead of auto-filling the outstanding
                # {generic:1} pip (this engine deliberately never
                # auto-spends floated mana toward generic -- mana.py's
                # own documented design). Spend it explicitly.
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        assert state.pending_resolution is None
        assert state.hand == []
        assert [c.name for c, _stamp in state.exile] == ["Fake Plot Spell"]
        assert on_cast_calls == []  # plotting itself never fires on_cast -- it isn't casting the spell

        # Same turn: not castable yet ("on a later turn").
        assert not _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)

        # A later turn: castable for free, queues on_cast_trigger (Guttersnipe).
        state.turn_number += 1
        assert _cast_from_exile_legal("Fake Plot Spell", None, game.turn.Speed.SORCERY)(state)
        _cast_from_exile_execute("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])(state)
        assert state.exile == []
        # on_cast now QUEUES the trigger (faithful timing) -- it fires only
        # once game.turn's priority round promotes it onto the stack (above
        # the spell) and it resolves, not inline at cast.
        assert on_cast_calls == []
        assert [e["type"] for e in state.trigger_queue] == ["cast_trigger"]
        game.promote_triggers_to_stack(state)
        game.resolve_top_of_stack(state)
        assert on_cast_calls == ["Guttersnipe-ish"]

        # extra_legal gate on the cast-from-exile path (Highway Robbery's
        # own need: Plot waives the mana cost, not other costs a normal
        # cast's extra_legal already checks). Re-plot, then simulate an
        # extra_legal that's never satisfiable.
        game.EFFECT_REGISTRY[EffectId.FILLER] = {
            "cast": {"resolve": lambda s, c: None, "extra_legal": lambda s: False},
            "plot": {"cost": PLOT_COST, "resolve": lambda s, c: (s.hand.remove(c), s.exile.append((c, s.turn_number)))},
        }
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [plot_spell]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
        ]
        _plot_execute("Fake Plot Spell", PLOT_COST, game.EFFECT_REGISTRY[EffectId.FILLER]["plot"]["resolve"])(state)
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, _color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, None, is_filter)
            else:
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        state.turn_number += 1
        assert not _cast_from_exile_legal("Fake Plot Spell", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["extra_legal"], game.turn.Speed.SORCERY)(state)
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup

    print("drl_env Plot + on-cast-trigger self-check: OK")

    # Regression: on_cast_trigger (Guttersnipe) must only fire once a
    # spell's cost is actually, irreversibly paid. Casting a spell and
    # then choosing "Abandon payment" (game.abandon_pay_cost) must NOT
    # have collected the trigger for free -- and must be repeatable
    # without ever firing it, since the card never actually left hand and
    # no mana was ever spent. Before the fix, _cast_execute fired
    # on_cast_trigger BEFORE begin_pay_cost even started, so this exact
    # cast-then-abandon loop collected Guttersnipe's damage for free,
    # indefinitely.
    _card_defs_backup = dict(game.CARD_DEFS)
    _filler_backup = game.EFFECT_REGISTRY[EffectId.FILLER]
    _generous_ent_backup = game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT]

    on_cast_calls = []
    fake_bolt = CardDef("Fake Bolt", CardType.INSTANT, {"B": 1}, EffectId.FILLER)
    game.CARD_DEFS["Fake Bolt"] = fake_bolt
    game.EFFECT_REGISTRY[EffectId.FILLER] = {"cast": {"resolve": lambda s, c: None}}
    game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = {
        "on_cast": lambda s, permanent: on_cast_calls.append(permanent.card_def.name),
    }
    try:
        state = GameState(on_the_play=True)
        state.phase = game.turn.Phase.MAIN1
        state.hand = [fake_bolt]
        state.battlefield = [
            Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)),
            Permanent(CardDef("Guttersnipe-ish", CardType.CREATURE, None, EffectId.GENEROUS_ENT)),
        ]
        cast_legal = _cast_legal("Fake Bolt", None, game.turn.Speed.INSTANT)
        cast_execute = _cast_execute("Fake Bolt", game.EFFECT_REGISTRY[EffectId.FILLER]["cast"]["resolve"])

        for _ in range(5):  # "infinitely" -- a handful of reps proves the loop, not just one
            assert cast_legal(state)
            cast_execute(state)
            assert on_cast_calls == []  # not fired yet -- cost isn't paid
            assert state.pending_resolution["kind"] == "pay_cost"
            game.abandon_pay_cost(state)
            assert state.pending_resolution is None
            assert on_cast_calls == []  # declining payment must never have collected it
            assert state.hand == [fake_bolt]  # never actually cast -- still sitting in hand

        # Actually pay this time -- now, and only now, the trigger is queued
        # (once), then fires when promoted to the stack and resolved.
        assert cast_legal(state)
        cast_execute(state)
        while state.pending_resolution is not None:
            tap_opts = game.tap_cost_options(state)
            if tap_opts:
                name, color, is_filter = tap_opts[0]
                game.execute_tap_cost_option(state, name, color, is_filter)
            else:
                # A tap only floats mana into the pool -- never auto-spends
                # it toward the cost (mana.py's own design, see the Plot
                # check above) -- spend it explicitly.
                game.execute_pool_spend(state, game.pool_spend_options(state)[0])
        # Paid: the spell is on the stack and the on_cast trigger is queued
        # (not fired inline). Promote + resolve to fire it, above the spell.
        assert on_cast_calls == []
        assert [e["type"] for e in state.trigger_queue] == ["cast_trigger"]
        assert len(state.stack) == 1  # the spell itself, paid and on the stack
        game.promote_triggers_to_stack(state)
        game.resolve_top_of_stack(state)
        assert on_cast_calls == ["Guttersnipe-ish"]
        assert len(state.stack) == 1  # the spell still waiting below the resolved trigger
    finally:
        game.CARD_DEFS.clear()
        game.CARD_DEFS.update(_card_defs_backup)
        game.EFFECT_REGISTRY[EffectId.FILLER] = _filler_backup
        game.EFFECT_REGISTRY[EffectId.GENEROUS_ENT] = _generous_ent_backup

    print("drl_env abandon-payment on-cast-trigger regression: OK")

    # Tokens (item 8): build_action_table's token_card_defs param is what
    # actually makes "Activate Blood (sac)" exist as an action at all --
    # "Blood" is never a decklist name, so the plain distinct_names-driven
    # loop alone (used by every other activated ability) can't find it.
    empty_decklist = []
    no_token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY)
    assert not any("Blood" in nm for nm, _l, _e in no_token_actions)  # opt-in: omitted => absent, zero effect on existing decks

    token_actions = build_action_table(empty_decklist, game.EFFECT_REGISTRY, token_card_defs=(game.BLOOD_TOKEN_CARD_DEF,))
    activate_name, activate_legal, activate_execute = next(
        (nm, lg, ex) for nm, lg, ex in token_actions if nm == "Activate Blood (sac)"
    )

    state = GameState(on_the_play=True)
    game.create_token(state, game.BLOOD_TOKEN_CARD_DEF)
    state.battlefield.append(Permanent(CardDef("Swamp", CardType.LAND, None, EffectId.SWAMP)))
    state.hand = [CardDef("Card To Discard", CardType.SORCERY, {}, None)]
    state.library = [CardDef("Library Card", CardType.SORCERY, {}, None)]

    assert activate_legal(state)
    activate_execute(state)  # pays {1} via the real begin_pay_cost path, same as every other cost_key ability
    assert state.pending_resolution["kind"] == "pay_cost"
    tap_name, tap_color, tap_filter = game.tap_cost_options(state)[0]
    game.execute_tap_cost_option(state, tap_name, tap_color, tap_filter)
    if state.pending_resolution is not None:
        # Swamp produces B, which floats into the pool instead of
        # auto-filling the {generic:1} need -- same lesson as the Plot
        # check above. Spend it explicitly.
        game.execute_pool_spend(state, game.pool_spend_options(state)[0])

    assert state.pending_resolution["kind"] == "discard"  # Blood's discard -- a COST, paid before the effect
    game.execute_discard_option(state, "Card To Discard")
    assert state.pending_resolution is None
    assert [p.card_def.name for p in state.battlefield] == ["Swamp"]  # Blood is gone, never added to any zone
    # The DRAW is Blood's effect -- on the stack now (faithful timing), not
    # fired the instant its costs (sac + discard) were paid. Resolve it.
    assert len(state.stack) == 1 and state.hand == []
    game.resolve_top_of_stack(state)
    assert [c.name for c in state.hand] == ["Library Card"]  # discarded one, drew one

    print("drl_env tokens self-check: OK")


    # Cross-player targeting: build_action_table's
    # opponent_decklist/opponent_token_card_defs params register "Choose
    # opponent's: X (slot k)" actions from the OTHER side's own card pool
    # -- blocking's first consumer, but exercised standalone here since
    # blocking itself isn't built yet. Boggles on both sides -- what's
    # under test is MY OWN action table's opponent-facing entries, not
    # anything about my own cards, so a real decklist with real creature
    # quantities on both sides is all this needs.
    # __file__ is now src/drl_env/__init__.py (one dir deeper since drl_env
    # became a package), so data/ is two levels up, not one.
    boggles_decklist = game.parse_decklist_file(os.path.join(os.path.dirname(__file__), "..", "..", "data", "boggles.txt"))

    no_opponent_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY)
    assert not any(nm.startswith("Choose opponent's:") for nm, _l, _e in no_opponent_actions)  # 1p mode: never registered at all

    my_actions = build_action_table(boggles_decklist, game.EFFECT_REGISTRY, opponent_decklist=boggles_decklist)
    bogle_slot_actions = [nm for nm, _l, _e in my_actions if nm.startswith("Choose opponent's: Slippery Bogle")]
    assert bogle_slot_actions == [f"Choose opponent's: Slippery Bogle (slot {k})" for k in range(1, 5)]  # boggles.txt: 4 copies
    assert not any(nm.startswith("Choose opponent's: Forest") for nm, _l, _e in my_actions)  # a land, never a targetable creature

    def _midx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(my_actions) if nm == action_name)

    target_slot_2 = _midx("Choose opponent's: Slippery Bogle (slot 2)")
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")

    attacker_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacker_bogle_2.slot = 2
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacker_bogle_1, attacker_bogle_2]
    state.active_idx = 1  # simulating the defender's own already-flipped perspective (see game.begin_choose_opponent_permanent's own docstring)

    _, legal_slot_2, execute_slot_2 = my_actions[target_slot_2]
    _, legal_slot_1, _ = my_actions[target_slot_1]
    assert not legal_slot_2(state) and not legal_slot_1(state)  # nothing pending yet

    completed = []
    game.begin_choose_opponent_permanent(
        state, lambda p: p.card_def.card_type == game.CardType.CREATURE, lambda s, choice: completed.append(choice),
    )
    assert legal_slot_1(state) and legal_slot_2(state)
    execute_slot_2(state)
    assert completed == [("Slippery Bogle", 2)]  # the specific slot targeted, not an arbitrary same-named match
    assert not legal_slot_2(state)  # resolution is complete, nothing pending anymore

    print("drl_env cross-player targeting self-check: OK")

    # Turn-owner / priority-holder split:
    # _land_drop_legal (via speed_legal) and _attack_legal must both
    # refuse the non-turn player even when state.phase/state.
    # lands_played_this_turn/their own eligible creature would otherwise
    # look legal -- simulates a priority consult (active_idx flipped away
    # from turn_player_idx) without needing the full priority round built
    # yet.
    play_forest_idx = _midx("Play land: Forest")
    attack_bogle_idx = _midx("Attack: Slippery Bogle (slot 1)")
    _, play_forest_legal, _ = my_actions[play_forest_idx]
    _, attack_bogle_legal, _ = my_actions[attack_bogle_idx]

    turn_owner_state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    turn_owner_state.turn_player_idx = 0
    turn_owner_state.active_idx = 0
    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    attacking_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    attacking_bogle.summoning_sick = False
    turn_owner_state.players[0].hand = [game.CARD_DEFS["Forest"]]
    turn_owner_state.players[0].battlefield = [attacking_bogle]
    assert attack_bogle_legal(turn_owner_state)  # the turn player's own creature, their own DECLARE_ATTACKERS -- legal

    turn_owner_state.phase = game.turn.Phase.MAIN1
    assert play_forest_legal(turn_owner_state)  # the turn player's own MAIN1, land in hand, none played yet -- legal

    turn_owner_state.active_idx = 1  # simulating a priority consult of the OTHER player
    turn_owner_state.players[1].hand = [game.CARD_DEFS["Forest"]]  # even with their OWN land available
    assert not play_forest_legal(turn_owner_state)  # refused -- not their turn, regardless of their own hand/lands_played_this_turn

    turn_owner_state.phase = game.turn.Phase.DECLARE_ATTACKERS
    non_turn_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    non_turn_bogle.summoning_sick = False
    turn_owner_state.players[1].battlefield = [non_turn_bogle]  # even with their OWN eligible creature at the same (name, slot)
    assert not attack_bogle_legal(turn_owner_state)  # refused -- declaring attackers is the turn player's own special action

    print("drl_env turn-owner (land drop / declare attacker) self-check: OK")


    # Blocking: build_action_table's "Assign Blocker:
    # <name> (slot j)" / "Done blocking" entries, end to end through the
    # REAL production functions (_assign_blocker_legal/_execute,
    # _done_blocking_legal/_execute) -- not a parallel reimplementation.
    # Two attacking Slippery Bogles (real power=1 stats), one defending
    # Slippery Bogle blocks only ONE of them.
    boggles_pending_kinds = game.derive_pending_kinds(boggles_decklist)
    assign_slot_1 = _midx("Assign Blocker: Slippery Bogle (slot 1)")
    done_blocking_idx = _midx("Done blocking")
    _, assign_legal, assign_execute = my_actions[assign_slot_1]
    _, done_legal, done_execute = my_actions[done_blocking_idx]

    atk_bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    atk_bogle_2.slot = 2
    defender_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])  # slot 1 by default
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [atk_bogle_1, atk_bogle_2]
    state.players[0].attackers = [atk_bogle_1, atk_bogle_2]
    atk_bogle_1.tapped = True
    atk_bogle_2.tapped = True  # declare_attacker's own effect -- simulated directly, attacking itself isn't under test here
    state.players[1].battlefield = [defender_bogle]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    assert not assign_legal(state) and not done_legal(state)  # nothing pending yet

    completed = []
    game.begin_declare_blockers(state, on_complete=lambda s: completed.append(True))
    assert completed == []
    assert assign_legal(state) and done_legal(state)

    assign_execute(state)  # "Assign Blocker: Slippery Bogle (slot 1)" -- parks defender_bogle as a blocker
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    target_slot_1 = _midx("Choose opponent's: Slippery Bogle (slot 1)")
    _, target_legal, target_execute = my_actions[target_slot_1]
    assert target_legal(state)
    target_execute(state)  # assigns it to block atk_bogle_1 specifically, not atk_bogle_2

    # Re-opened (drl_env._assign_blocker_execute's own nested on_complete):
    # defender_bogle is now spoken for, so it's no longer offered again --
    # confirms creature_block_eligible actually gates the SAME action a
    # second time, not just the first.
    assert state.pending_resolution["kind"] == "declare_blockers"
    assert not assign_legal(state)
    assert done_legal(state)
    done_execute(state)  # "Done blocking" -- atk_bogle_2 goes unblocked
    assert completed == [True]
    assert state.pending_resolution is None
    assert state.players[0].blocked_by == {atk_bogle_1: [defender_bogle]}


    print("drl_env blocking self-check: OK")

    # Flying: _assign_blocker_execute's own
    # extra_predicate (game.has_keyword), end to end through the REAL
    # action table -- Silhana Ledgewalker (real "can't be blocked except
    # by creatures with flying," modeled as the "flying" keyword) can only
    # be blocked by a creature that itself has flying (Kitchen Imp, real
    # flying) -- a plain Slippery Bogle is otherwise a perfectly legal
    # (untapped) blocker, but can never be assigned to THIS specific
    # attacker. Mixes cards from different real color catalogs
    # (green/multicolor + black) purely to exercise the engine mechanism
    # -- not a claim either card is ever actually run together in a real
    # deck.
    flying_decklist = [("Silhana Ledgewalker", 2), ("Slippery Bogle", 2), ("Kitchen Imp", 2)]
    flying_actions = build_action_table(flying_decklist, game.EFFECT_REGISTRY, opponent_decklist=flying_decklist)

    def _fidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(flying_actions) if nm == action_name)

    _, bogle_legal, bogle_execute = flying_actions[_fidx("Assign Blocker: Slippery Bogle (slot 1)")]
    _, imp_legal, imp_execute = flying_actions[_fidx("Assign Blocker: Kitchen Imp (slot 1)")]

    attacking_ledgewalker = Permanent(game.CARD_DEFS["Silhana Ledgewalker"])
    attacking_ledgewalker.tapped = True  # already attacked
    defending_bogle = Permanent(game.CARD_DEFS["Slippery Bogle"])
    defending_imp = Permanent(game.CARD_DEFS["Kitchen Imp"])
    state = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state.players[0].battlefield = [attacking_ledgewalker]
    state.players[0].attackers = [attacking_ledgewalker]
    state.players[1].battlefield = [defending_bogle, defending_imp]
    state.active_idx = 1  # simulating _declare_blockers_gen's own flip to the defender

    game.begin_declare_blockers(state, on_complete=lambda s: None)
    # GANG-BLOCKING eligibility fix: Slippery Bogle (no flying) is NOT even
    # offered here -- the only attacker (Silhana Ledgewalker, modeled with
    # flying) can only be blocked by a flyer, so Bogle has no legal target
    # and "Assign Blocker: Slippery Bogle" is illegal (it used to be an
    # offered no-op that fizzled -- exactly what this fix removes at the
    # source). Kitchen Imp (flying) IS a legal blocker.
    assert not bogle_legal(state)
    assert imp_legal(state)

    imp_execute(state)  # Kitchen Imp HAS flying -- opens a real nested choice
    assert state.pending_resolution["kind"] == "choose_opponent_permanent"
    assert game.choose_opponent_permanent_options(state) == [("Silhana Ledgewalker", 1)]
    game.execute_choose_opponent_permanent_option(state, "Silhana Ledgewalker", 1)
    assert state.players[0].blocked_by == {attacking_ledgewalker: [defending_imp]}

    print("drl_env flying self-check: OK")

    # Targeting (real MTG rule, per drl_env._precast_choice_execute /
    # game.effects.casting.cast_aura's own docstrings): a target is chosen once,
    # at cast time, exact (name, slot) addressed -- not just by name -- and
    # re-validated by identity only once the spell resolves off the stack.
    # End to end through the REAL action table: "Cast Rancor" pays its {G}
    # cost, then (precast_choice, not deferred) immediately opens
    # choose_permanent with BOTH Slippery Bogles offered by their own
    # distinct slot; "Choose target: Slippery Bogle (slot 2)" picks the
    # specific one, which pushes to the stack (not yet attached, still in
    # hand); resolving the stack attaches it to exactly the one chosen, not
    # an arbitrary same-named match.
    targeting_decklist = [("Slippery Bogle", 2), ("Rancor", 2), ("Forest", 10)]
    targeting_actions = build_action_table(targeting_decklist, game.EFFECT_REGISTRY)

    def _gidx(action_name):
        return next(i for i, (nm, _l, _e) in enumerate(targeting_actions) if nm == action_name)

    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    bogle_2.slot = 2
    forest = Permanent(game.CARD_DEFS["Forest"])
    rancor_card = game.CARD_DEFS["Rancor"]
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1  # sorcery-speed cast requires this -- GameState defaults phase=None
    state.battlefield = [bogle_1, bogle_2, forest]
    state.hand = [rancor_card]

    _, cast_rancor_legal, cast_rancor_execute = targeting_actions[_gidx("Cast Rancor")]
    assert cast_rancor_legal(state)
    cast_rancor_execute(state)
    assert state.pending_resolution["kind"] == "pay_cost"
    targeting_actions[_gidx("Choose: Forest")][2](state)  # tap the Forest -- floats {G}
    targeting_actions[_gidx("Spend G from pool")][2](state)  # pays Rancor's {G}

    # Cost fully paid -- precast_choice means cast_aura runs its target
    # choice IMMEDIATELY here, NOT deferred to when this eventually pops
    # off the stack (that's the whole point of this redesign).
    # Aura now targets any creature on EITHER battlefield (real "Enchant
    # creature"), hexproof-aware -- the creature half rides the identity
    # pointer scheme (rl.action_bridge), so we drive it via the same
    # execute the pointer path calls, not a "Choose target:" fixed action.
    assert state.pending_resolution["kind"] == "choose_any_target"
    assert set(game.choose_any_target_creature_options(state)) == {(0, "Slippery Bogle", 1), (0, "Slippery Bogle", 2)}

    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 2)  # the SPECIFIC slot-2 bogle
    # Target chosen -- pushed to the stack, not yet attached. The card LEFT
    # hand at cast (push_to_stack), so it can't be re-cast off the stack;
    # resolve_top_of_stack restores it transiently for the attach resolve.
    assert state.pending_resolution is None
    assert state.hand == [] and len(state.stack) == 1

    game.resolve_top_of_stack(state)
    assert state.hand == []
    rancor_permanent = next(p for p in state.battlefield if p.card_def.name == "Rancor")
    assert rancor_permanent.flags["enchanting"] is bogle_2  # the SPECIFIC one chosen -- not bogle_1, despite the identical name

    print("drl_env Aura targeting (exact slot addressing) self-check: OK")

    # Fizzle, same end-to-end path: the exact chosen permanent (bogle_1
    # this time) is gone by the time the cast resolves -- the whole spell
    # fails, no effect, straight to the graveyard, never attaches.
    state = GameState(on_the_play=True)
    state.phase = game.turn.Phase.MAIN1
    bogle_1 = Permanent(game.CARD_DEFS["Slippery Bogle"])
    forest = Permanent(game.CARD_DEFS["Forest"])
    state.battlefield = [bogle_1, forest]
    state.hand = [rancor_card]

    targeting_actions[_gidx("Cast Rancor")][2](state)
    targeting_actions[_gidx("Choose: Forest")][2](state)
    targeting_actions[_gidx("Spend G from pool")][2](state)
    game.execute_choose_any_target_creature(state, 0, "Slippery Bogle", 1)  # Aura targets via the any-target pointer path now
    assert len(state.stack) == 1
    state.battlefield.remove(bogle_1)  # dies before the cast resolves

    game.resolve_top_of_stack(state)
    assert state.hand == []
    assert rancor_card in state.graveyard
    assert not any(p.card_def.name == "Rancor" for p in state.battlefield)

    print("drl_env Aura target-fizzle (end to end) self-check: OK")

    # _lost: true once someone has won and it wasn't seat_idx -- still used
    # directly by rl.train's own reward attribution.
    assert _lost(type("S", (), {"winner": 1})(), 0) is True
    assert _lost(type("S", (), {"winner": 0})(), 0) is False
    assert _lost(type("S", (), {"winner": None})(), 0) is False
    print("drl_env _lost self-check: OK")

    # tap_cost_options memoization never returns a stale answer (docs/
    #): build a pay_cost resolution with exactly 1
    # untapped Mountain, sweep the mask (populating the cache -- "Choose:
    # Mountain" legal), tap it (a real mutation -- zero untapped sources
    # left, so tap_cost_options itself now returns empty), then sweep
    # again -- the second sweep must see the mutation, not the first
    # sweep's cached answer, proving the cache doesn't leak across
    # separate legal_action_mask calls.
    perf_decklist = [("Mountain", 10), ("Lightning Bolt", 5)]
    perf_pending = game.derive_pending_kinds(perf_decklist)
    perf_actions = build_action_table(perf_decklist, game.EFFECT_REGISTRY, pending_kinds=perf_pending)
    perf_choose_mountain = next(i for i, (nm, _l, _e) in enumerate(perf_actions) if nm == "Choose: Mountain")
    perf_state = GameState(on_the_play=True, players=[PlayerState(True)])
    perf_state.hand = [game.CARD_DEFS["Lightning Bolt"]]
    perf_state.battlefield = [Permanent(game.CARD_DEFS["Mountain"])]
    game.begin_pay_cost(perf_state, {"R": 1}, on_complete=lambda s: None)
    assert legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]
    assert _actions._tap_cost_options_cache is None  # cleared again once the sweep itself returns
    game.execute_tap_cost_option(perf_state, "Mountain", None, False)  # taps the only Mountain -- 0 untapped sources left
    assert game.tap_cost_options(perf_state) == []  # ground truth: nothing left to tap
    assert not legal_action_mask(perf_state, perf_actions)[perf_choose_mountain]  # would be wrongly True if the first sweep's stale cache leaked through

    print("drl_env tap_cost_options cache self-check: OK")
