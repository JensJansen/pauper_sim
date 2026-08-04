"""Tests for rl.rewards's win/loss reward functions."""
import pytest

from game.state import GameState, PlayerState
from rl.rewards import action_count_win_reward, deploy_reward, with_mana_mistake_penalty


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
    # defaults: plateau 80, max 200, win_floor 0.5, discard_c/p=4/2
    dr = deploy_reward()
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0  # score seat 0 throughout (rl.train._reward_for flips this in real use)

    assert dr(s, done=False, horizon=120) == 0.0  # terminal only

    # WIN band (seat 0 is the winner): 0.5 + 0.5*efficiency - q, efficiency on
    # GAMEPLAY actions (actions_taken - pregame_actions). q == 0 throughout this
    # block (cleanup_discard_turns defaults to 0), so these match the pure
    # efficiency curve -- q's own effect is checked below.
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

    # LOSS band (seat 0 is NOT the winner): exactly -q. q = Hill(discard_turns;
    # c=4, p=2) -- 0 when cleanup_discard_turns is 0, saturating toward (never
    # reaching) 1 as it grows. Mana burn no longer feeds q at all (see
    # with_mana_mistake_penalty for its dense, per-transition replacement).
    # Mulligans are not scored here -- the mulligan model owns that.
    s.winner = 1
    s.players[0].cleanup_discard_turns = 0
    s.players[0].mulligans_taken = 7   # mulligans don't affect the loss band
    assert dr(s, done=True, horizon=120) == 0.0  # played its hand, lost (mulligans ignored)

    # Discard-only: Hill(n; c=4, p=2) -- convex ramp, e.g. n=1 -> 1/17, n=3 -> 9/25.
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - (-1 / 17)) < 1e-9
    s.players[0].cleanup_discard_turns = 3
    assert abs(dr(s, done=True, horizon=120) - (-9 / 25)) < 1e-9
    s.players[0].cleanup_discard_turns = 6
    assert abs(dr(s, done=True, horizon=120) - (-36 / 52)) < 1e-9
    s.players[0].cleanup_discard_turns = 0

    # Mana burn no longer moves this terminal reward at all, however large.
    s.players[0].mana_burnt_total = 100_000
    assert dr(s, done=True, horizon=120) == 0.0
    s.players[0].mana_burnt_total = 0

    # No-winner timeout (winner None) uses the loss band for whichever seat.
    s.winner = None
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - (-1 / 17)) < 1e-9


@pytest.mark.slow
def test_deploy_reward_v2_flat_win():
    # --- deploy_reward(win_floor=1.0) (the base deploy_reward_v2 wraps): the
    # win band's EFFICIENCY term flattens to 1.0 regardless of game length,
    # but q (cleanup discards) still docks it -- only a perfectly clean win
    # scores exactly 1.0. Loss band is deploy_reward's, checked above. ---
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    v2 = deploy_reward(win_floor=1.0)
    s.winner = 0  # reuse the fixture shape from test_deploy_reward (pregame_actions=5)
    s.players[0].pregame_actions = 5
    s.players[0].actions_taken = 5 + 5000   # a long, grindy win
    assert v2(s, done=True, horizon=120) == 1.0  # flat -- no efficiency dock, q still 0
    s.players[0].actions_taken = 5 + 1          # a fast win scores identically
    assert v2(s, done=True, horizon=120) == 1.0

    # A sloppy win (lots of hoarded/discarded cards) scores strictly less than
    # a clean one, but -- because q < 1 always -- still strictly above every
    # possible loss (<= 0). Mana burn no longer factors in here at all.
    s.players[0].cleanup_discard_turns = 6
    sloppy_win = v2(s, done=True, horizon=120)
    assert 0.0 < sloppy_win < 1.0
    assert abs(sloppy_win - (1 - 36 / 52)) < 1e-9
    s.players[0].cleanup_discard_turns = 0
    s.players[0].mana_burnt_total = 100_000
    assert v2(s, done=True, horizon=120) == 1.0  # unaffected


@pytest.mark.slow
def test_with_mana_mistake_penalty():
    # --- with_mana_mistake_penalty: dense, per-transition wrapper. Drains
    # (reads then zeroes) PlayerState.mana_mistake_burn on every call, adds
    # -min(penalty_per_pip * mistake, per_event_cap) on top of whatever the
    # wrapped base reward_fn already returns. ---
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base_calls = []
    base = lambda state, done, horizon: base_calls.append((done, horizon)) or 0.25
    wrapped = with_mana_mistake_penalty(base, penalty_per_pip=0.01, per_event_cap=0.05)

    # No mistake pending -> base's own value passes through unchanged, on
    # both a non-terminal and a terminal call.
    assert wrapped(s, done=False, horizon=99) == 0.25
    assert wrapped(s, done=True, horizon=99) == 0.25
    assert base_calls == [(False, 99), (True, 99)]  # base_reward_fn really was called, with the same args

    # A pending mistake is drained into a linear penalty and the mailbox
    # resets to 0 -- a second call right after sees nothing left to pay.
    s.players[0].mana_mistake_burn = 2
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - 0.02)) < 1e-9
    assert s.players[0].mana_mistake_burn == 0
    assert wrapped(s, done=False, horizon=99) == 0.25  # already drained -- no double charge

    # per_event_cap bounds a single large drain regardless of penalty_per_pip.
    s.players[0].mana_mistake_burn = 50  # 50 * 0.01 = 0.5, well past the 0.05 cap
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - 0.05)) < 1e-9

    # Scored seat-relative: reads state.players[state.active_idx], same
    # convention every other reward_fn in this module follows.
    s.active_idx = 1
    s.players[1].mana_mistake_burn = 1
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - 0.01)) < 1e-9
    assert s.players[0].mana_mistake_burn == 0  # untouched -- seat 0 wasn't scored this call
