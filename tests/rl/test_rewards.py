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
    # defaults: plateau 80, max 200, win_floor 0.5, mana_burn_c/p=5/3, discard_c/p=4/2
    dr = deploy_reward()
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0  # score seat 0 throughout (rl.train._reward_for flips this in real use)

    assert dr(s, done=False, horizon=120) == 0.0  # terminal only

    # WIN band (seat 0 is the winner): 0.5 + 0.5*efficiency - q, efficiency on
    # GAMEPLAY actions (actions_taken - pregame_actions). q == 0 throughout this
    # block (mana_burnt_total/cleanup_discard_turns both default to 0), so these
    # match the pure efficiency curve -- q's own effect is checked below.
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

    # LOSS band (seat 0 is NOT the winner): exactly -q. q = _badness(mana_burnt,
    # discard_turns) -- a noisy-or of two Hill curves, 0 only when BOTH inputs
    # are 0, saturating toward (never reaching) 1 as either grows. Mulligans are
    # not scored here -- the mulligan model owns that.
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

    # Mana-burn-only: Hill(b; c=5, p=3) -- b=1 barely punished, b=5 == half-bad
    # (the curve's own c), b=10 already severe but still short of -1.
    s.players[0].mana_burnt_total = 1
    assert abs(dr(s, done=True, horizon=120) - (-1 / 126)) < 1e-9
    s.players[0].mana_burnt_total = 5
    assert abs(dr(s, done=True, horizon=120) - (-0.5)) < 1e-9
    s.players[0].mana_burnt_total = 10
    val_10 = dr(s, done=True, horizon=120)
    assert abs(val_10 - (-8 / 9)) < 1e-9
    assert -1.0 < val_10  # asymptotes toward, never reaches, -1
    s.players[0].mana_burnt_total = 100_000  # absurdly large -> still short of -1
    assert -1.0 < dr(s, done=True, horizon=120) < -0.999

    # Combined (noisy-or, not a plain sum): worse than either alone, but the two
    # half-bad (0.5) factors combine to LESS than 1.0, not exactly 1.0 or more.
    s.players[0].mana_burnt_total = 5     # h_burn = 0.5
    s.players[0].cleanup_discard_turns = 1  # h_discard = 1/17
    combined = dr(s, done=True, horizon=120)
    assert combined < -0.5  # worse than mana-burn alone
    assert combined < -1 / 17  # worse than discard alone
    assert abs(combined - (-(1 - 0.5 * (1 - 1 / 17)))) < 1e-9
    s.players[0].mana_burnt_total = 0
    s.players[0].cleanup_discard_turns = 0

    # No-winner timeout (winner None) uses the loss band for whichever seat.
    s.winner = None
    s.players[0].cleanup_discard_turns = 1
    assert abs(dr(s, done=True, horizon=120) - (-1 / 17)) < 1e-9


@pytest.mark.slow
def test_deploy_reward_v2_flat_win():
    # --- deploy_reward_v2 == deploy_reward(win_floor=1.0): the win band's
    # EFFICIENCY term flattens to 1.0 regardless of game length, but q (mana
    # burn / cleanup discards) still docks it -- only a perfectly clean win
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

    # A sloppy win (lots of burnt mana) scores strictly less than a clean one,
    # but -- because q < 1 always -- still strictly above every possible loss (<= 0).
    s.players[0].mana_burnt_total = 10
    sloppy_win = v2(s, done=True, horizon=120)
    assert 0.0 < sloppy_win < 1.0
    assert abs(sloppy_win - (1 - 8 / 9)) < 1e-9
