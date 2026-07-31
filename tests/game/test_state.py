"""Tests for game.state's event_log plumbing: GameState.log_event's no-op
when logging is off, its envelope shape when on, and the whole mechanism
exercised end to end through a real 2-player game (turn/stack/mana events).
NOT a re-test of engine correctness -- every game/*.py module already tests
its own instrumented behavior; this only confirms logging actually happens,
in the right shape, in the right order, and costs nothing when off."""

import random

from game import mana, registry, resolution
from game.effects.casting import play_land_from_hand
from game.effects.stack import push_to_stack
from game.effects.win_check import deal_damage_to_opponent
from game.state import GameState
from game.turn import run_multiplayer_game


def test_event_log_none_is_noop():
    # event_log=None (the default) is a true no-op -- log_event never
    # builds a dict, self.event_log stays None.
    off_state = GameState(on_the_play=True)
    assert off_state.event_log is None
    off_state.log_event("whatever", foo="bar")
    assert off_state.event_log is None


def test_log_event_envelope():
    # Direct log_event call: every event carries the same envelope
    # (turn/phase/active_idx/turn_player_idx) automatically, on top of
    # whatever fields the call site passed.
    log = []
    on_state = GameState(on_the_play=True, event_log=log)
    on_state.turn_number = 3
    on_state.log_event("tap", permanent=("Mountain", 1))
    assert len(log) == 1
    event = log[0]
    # _safe() normalizes tuples -> lists (JSON has no tuple type), so the
    # (name, slot) comes back as a list, not the tuple that was passed in.
    assert event["kind"] == "tap" and event["permanent"] == ["Mountain", 1]
    assert event["turn"] == 3 and event["active_idx"] == 0 and event["turn_player_idx"] == 0


def test_real_game_event_log():
    # End to end through a REAL 2-player game (run_multiplayer_game -> every
    # instrumented mutation site across mana.py/turn.py/resolution/*.py/
    # game/effects/*.py) -- player 0 taps a Mountain, casts Lightning Bolt at
    # player 1, and lets it resolve; player 1 only ever passes. Confirms that
    # state changes happening as an automatic side effect of "Pass" (the
    # stack push/resolve, the phase-boundary mana-pool clear) are still
    # captured -- not just changes triggered by an explicit model decision.
    bolt_def = registry.CARD_DEFS["Lightning Bolt"]

    # Lightning Bolt's production "cast" resolve is a precast "any target"
    # choice (game.catalog.red_cards) that would open a choose_any_target
    # pending this toy logging policy doesn't model -- same reason turn.py's
    # own real-game self-check uses a local direct-to-opponent resolve. This
    # test is about the event_log plumbing, not targeting, so decouple it the
    # same way.
    def bolt_resolve(s, cd):
        if cd in s.hand:
            s.hand.remove(cd)
        s.move_card(cd, s.graveyard)
        deal_damage_to_opponent(s, 3)

    def _burn_policy(state):
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision":
            return lambda: resolution.execute_mulligan_keep(state)  # both players always keep
        if state.pending_resolution is not None and state.pending_resolution["kind"] == "discard":
            name = resolution.discard_options(state)[0]  # cleanup hand-size discard, either player
            return lambda: resolution.execute_discard_option(state, name)
        if state.active_idx != 0:
            return None  # player 1 never acts otherwise
        if state.pending_resolution is not None:  # paying the Bolt's {R} -- spend floated pool mana
            # Float-first: a tap floats mana into the pool via a top-level mana
            # ability BEFORE the cast (below); during payment there is only pool
            # mana to spend, never a source to tap.
            color = mana.pool_spend_options(state)[0]
            return lambda: mana.execute_pool_spend(state, color)
        if state.lands_played_this_turn == 0 and any(c.name == "Mountain" for c in state.hand):
            return lambda: play_land_from_hand(state, registry.CARD_DEFS["Mountain"])
        # hand_count - stacked_count, same accounting drl_env._hand_count_available
        # and turn.py's own self-check use: a copy already paid-for-but-
        # unresolved on the stack is still physically in hand (push_to_stack
        # only removes it once it actually resolves) but isn't really available.
        hand_count = sum(1 for c in state.hand if c.name == "Lightning Bolt")
        stacked_count = sum(1 for e in state.stack if e["card_def"].name == "Lightning Bolt")
        if hand_count > stacked_count:
            if mana.plan_payment(state, bolt_def.cast_cost) is not None:
                def _cast_bolt():
                    mana.begin_pay_cost(state, bolt_def.cast_cost, on_complete=lambda s: push_to_stack(s, bolt_def, bolt_resolve))
                return _cast_bolt
            # Can't pay yet -- float {R} from an untapped Mountain first (float-first).
            mtn = next((p for p in state.battlefield if p.card_def.name == "Mountain" and not p.tapped), None)
            if mtn is not None:
                return lambda: mana.activate_mana_source(state, mtn)
        return None  # Pass -- resolves the stack if non-empty, else advances the phase

    events = []
    state = run_multiplayer_game(
        decklists=[[("Mountain", 10), ("Lightning Bolt", 5)], [("Mountain", 20)]],
        rng=random.Random(0), starting_player_idx=0,
        choose_action=_burn_policy, horizon=12, event_log=events,
    )
    assert state.players[1].life_total < 20  # at least one Bolt landed on the opponent
    kinds = [e["kind"] for e in events]
    assert "turn_start" in kinds and "phase_change" in kinds
    assert "zone_move" in kinds  # land entering the battlefield, and the Bolt hitting the stack then leaving it
    assert "mana_tap" in kinds and "mana_spend" in kinds
    assert "pass" in kinds  # Pass itself is a real, loggable event, not silently skipped
    assert "life_change" in kinds  # the Bolt's own damage to the opponent's life_total
    # Every draw -- both opening hands AND every in-game draw -- is recorded as
    # one library->hand zone_move (reason="draw") naming the cards, via the
    # single generic hook; each opening 7 IS a draw event at turn 0.
    draw_moves = [e for e in events if e.get("reason") == "draw"]
    assert draw_moves, "expected draws to be logged"
    assert all(e["from_zone"] == "library" and e["to_zone"] == "hand" and e["cards"] for e in draw_moves)
    assert any(e["turn"] == 0 and len(e["cards"]) == 7 for e in draw_moves)  # an opening hand
    assert any(e["turn"] > 0 for e in draw_moves)  # at least one in-game draw (a turn's draw_step)
    # Every event's envelope is well-formed regardless of kind.
    for e in events:
        assert set(e) >= {"kind", "turn", "phase", "active_idx", "turn_player_idx"}
    # Order is causally sensible: a Mountain tap (mana_tap) happens before the
    # spend (mana_spend) that actually pays for the Bolt.
    assert kinds.index("mana_tap") < kinds.index("mana_spend")
