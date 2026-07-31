"""Tests for rl.rewards's win/loss reward functions."""
import pytest

from game.state import GameState, PlayerState
from rl.rewards import action_count_win_reward, deploy_reward


@pytest.mark.slow
def test_action_count_win_reward():
    # action_count_win_reward: per-seat (state.players[winner].actions_taken),
    # not turn_number -- a real 2-player state (state.winner needs a second
    # seat to mean anything).

    # Default instance (0.25 floor) -- built locally just for this check; the
    # only pre-baked module-level instance the pipeline ships is the 0.2-floor one.
    rf = action_count_win_reward()

    state2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    state2.turn_won = None
    assert rf(state2, done=True, horizon=120) == 0.0  # no winner -> 0
    assert rf(state2, done=False, horizon=120) == 0.0  # not done -> 0

    state2.turn_won = 5
    state2.winner = 0

    # Plateau: anything at or under plateau_actions (80) scores the full
    # 1.0, no reward at all for going even faster -- "sufficiently fast",
    # per request.
    state2.players[1].actions_taken = 999  # the LOSER's own count must never matter
    state2.players[0].actions_taken = 1
    assert rf(state2, done=True, horizon=120) == 1.0
    state2.players[0].actions_taken = 80
    assert rf(state2, done=True, horizon=120) == 1.0

    # Linear ramp from (80, 1.0) to (200, 0.25) -- midpoint (140) should
    # land exactly halfway between.
    state2.players[0].actions_taken = 140
    assert abs(rf(state2, done=True, horizon=120) - 0.625) < 1e-9

    # Floor: exactly 0.25 at max_actions (200), and bottoms out there --
    # never continues down toward 0 for a wildly long game past the cap.
    state2.players[0].actions_taken = 200
    win_at_cap = rf(state2, done=True, horizon=120)
    assert abs(win_at_cap - 0.25) < 1e-9
    state2.players[0].actions_taken = 5000
    win_past_cap = rf(state2, done=True, horizon=120)
    assert win_at_cap == win_past_cap  # bottomed out -- doesn't keep decaying below this

    # This legacy reward self-contains the loser gate: a non-winning seat
    # (state.winner != state.active_idx) scores 0 on its own.
    state2.active_idx = 1  # score seat 1; winner is 0 -> a loss for this seat
    assert rf(state2, done=True, horizon=120) == 0.0
    state2.active_idx = 0


@pytest.mark.slow
def test_deploy_reward():
    # --- deploy_reward: two-band terminal reward, scored per seat ---
    dr = deploy_reward()  # defaults: plateau 80, max 200, win_floor 0.5, discard_base 0.02
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0  # score seat 0 throughout (rl.train._reward_for flips this in real use)

    assert dr(s, done=False, horizon=120) == 0.0  # terminal only

    # WIN band (seat 0 is the winner): 0.5 + 0.5*efficiency, efficiency on
    # GAMEPLAY actions (actions_taken - pregame_actions).
    s.winner = 0
    s.players[1].actions_taken = 999  # loser's count must never matter
    s.players[0].pregame_actions = 5  # 5 mulligan/keep/bottom picks -- excluded below
    s.players[0].actions_taken = 5 + 10   # 10 gameplay actions -> under plateau
    assert dr(s, done=True, horizon=120) == 1.0
    s.players[0].actions_taken = 5 + 80   # exactly plateau gameplay actions
    assert dr(s, done=True, horizon=120) == 1.0
    s.players[0].actions_taken = 5 + 140  # 140 gameplay -> midpoint of [80,200]
    assert abs(dr(s, done=True, horizon=120) - 0.75) < 1e-9  # 0.5 + 0.5*0.5
    s.players[0].actions_taken = 5 + 200  # >= max gameplay -> win floor
    assert abs(dr(s, done=True, horizon=120) - 0.5) < 1e-9
    s.players[0].actions_taken = 5 + 5000
    assert abs(dr(s, done=True, horizon=120) - 0.5) < 1e-9  # floored, never below win_floor

    # LOSS band (seat 0 is NOT the winner): exactly 0.0 with no cleanup
    # discards, else -(discard_base ** cleanup_discard_turns) -- SHRINKS toward
    # 0 as discard-turns pile up (discard_base=0.02 < 1), so the FIRST one is
    # the loudest signal, not the worst-case cumulative one. Mulligans are not
    # scored here -- the mulligan model owns that.
    s.winner = 1
    s.players[0].cleanup_discard_turns = 0
    s.players[0].mulligans_taken = 7   # mulligans don't affect the loss band
    assert dr(s, done=True, horizon=120) == 0.0  # played its hand, lost (mulligans ignored)
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - (-0.02)) < 1e-9  # -(0.02**1) -- the loudest single penalty
    s.players[0].cleanup_discard_turns = 3
    assert abs(dr(s, done=True, horizon=120) - (-0.000008)) < 1e-12  # -(0.02**3) -- already much smaller
    s.players[0].cleanup_discard_turns = 6
    val_6 = dr(s, done=True, horizon=120)
    assert -1e-9 < val_6 < 0  # keeps shrinking toward (never reaching) 0 -- never grows past the n=1 case

    # No-winner timeout (winner None) uses the loss band for whichever seat.
    s.winner = None
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - (-0.02)) < 1e-9


@pytest.mark.slow
def test_deploy_reward_v2_flat_win():
    # --- deploy_reward_v2 == deploy_reward(win_floor=1.0): the win band flattens to
    # 1.0 regardless of game length (the loss band is deploy_reward's, checked above) ---
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    v2 = deploy_reward(win_floor=1.0)
    s.winner = 0  # reuse the fixture shape from test_deploy_reward (pregame_actions=5)
    s.players[0].pregame_actions = 5
    s.players[0].actions_taken = 5 + 5000   # a long, grindy win
    assert v2(s, done=True, horizon=120) == 1.0  # flat -- no efficiency dock
    s.players[0].actions_taken = 5 + 1          # a fast win scores identically
    assert v2(s, done=True, horizon=120) == 1.0
