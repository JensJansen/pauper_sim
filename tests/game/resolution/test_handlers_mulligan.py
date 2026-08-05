"""Tests for game.resolution's mulligan handlers (handlers_mulligan.py):
keep-or-mulligan (London Mulligan) and the resulting bottom-N-cards
placement. Exercises these primitives directly against hand-built states,
bypassing drl_env entirely (no card wires into every one of these yet)."""

from game.cards import CardDef, CardType
from game.resolution import (
    begin_mulligan,
    bottom_options,
    execute_bottom_option,
    execute_mulligan_keep,
    execute_mulligan_take,
    mulligan_decision_options,
)
from game.state import GameState


def _card(name):
    return CardDef(name, CardType.SORCERY, {"generic": 1}, None)


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
