"""Tests for rl.rewards's win/loss reward functions."""
import pytest

from game.state import GameState, PlayerState
from rl.rewards import (
    action_count_win_reward, deploy_reward, deploy_reward_v2,
    with_dense_mana_burn_penalty, with_mana_mistake_penalty,
)


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


def _hill(x, c, p):
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


@pytest.mark.slow
def test_with_dense_mana_burn_penalty_telescopes():
    # with_dense_mana_burn_penalty (2026-08, wired into deploy_reward_v2 in
    # place of with_mana_mistake_penalty; briefly reverted the same month
    # over an unconditional-input archetype bias, then re-enabled reading
    # the metered/unmetered-filtered mana_burnt_this_turn_metered instead --
    # see rewards.py's own comment on deploy_reward_v2 for the full history;
    # tested standalone here for reference/regression coverage, not as the
    # active league reward): reads PlayerState.mana_burnt_this_turn_metered
    # (NEVER drained by this wrapper -- only reset by game.turn._run_turn_gen
    # at turn boundaries) and charges only the MARGINAL increase in
    # _hill(mana_burnt_this_turn_metered, c, p) since the last call, tracked
    # via PlayerState.mana_burn_penalty_credited.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base_calls = []
    base = lambda state, done, horizon: base_calls.append((done, horizon)) or 0.25
    wrapped = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0)

    # Nothing burnt yet -> base's own value passes through unchanged.
    assert wrapped(s, done=False, horizon=99) == 0.25
    assert base_calls == [(False, 99)]  # base_reward_fn really was called, with the same args

    # First pip burnt this turn: charged the FULL hill(1) (baseline was 0),
    # and mana_burnt_this_turn_metered itself is untouched (unlike
    # mana_mistake_burn, this wrapper never drains its input -- only the
    # engine resets it, at the next turn boundary).
    s.players[0].mana_burnt_this_turn_metered = 1
    h1 = _hill(1, 3.3, 4.0)
    assert h1 == pytest.approx(0.0084, abs=1e-3)  # matches the owner-specified "~0.008 on the first pip" anchor
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - h1)) < 1e-9
    assert s.players[0].mana_burnt_this_turn_metered == 1  # NOT drained
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)

    # Same call again with NO new burn -> baseline already matches current
    # cumulative, so the marginal charge is exactly 0 (no double-charging a
    # pip that's already been paid for).
    assert wrapped(s, done=False, horizon=99) == 0.25

    # More mana burns later in the SAME turn: charged only the DELTA, not
    # hill() of the new total from scratch -- this is what makes it
    # genuinely dense (each transition pays for what IT added) while still
    # summing to the same total a one-shot terminal charge would give.
    s.players[0].mana_burnt_this_turn_metered = 6
    h6 = _hill(6, 3.3, 4.0)
    assert h6 == pytest.approx(0.916, abs=1e-3)  # matches the owner-specified "~close to 1 by 6/7" anchor
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - (h6 - h1))) < 1e-9
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h6)

    # Telescoping identity: the SUM of every marginal charge this turn must
    # equal hill(final total) exactly, whether charged in one jump (as
    # above) or many small steps -- verify the many-small-steps path lands
    # on the identical total.
    s2 = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s2.active_idx = 0
    wrapped2 = with_dense_mana_burn_penalty(lambda state, done, horizon: 0.0, mana_burn_c=3.3, mana_burn_p=4.0)
    total_charged = 0.0
    for cum in range(1, 8):  # 7 separate single-pip burns, one reward call after each
        s2.players[0].mana_burnt_this_turn_metered = cum
        total_charged += -wrapped2(s2, done=False, horizon=99)
    assert total_charged == pytest.approx(_hill(7, 3.3, 4.0))

    # Reset (simulating a new turn, same as game.turn._run_turn_gen does)
    # zeroes the baseline too -- the next turn's first burn is cheap again,
    # not charged against the PREVIOUS turn's already-elevated baseline.
    s.players[0].mana_burnt_this_turn_metered = 0
    s.players[0].mana_burn_penalty_credited = 0.0
    assert abs(wrapped(s, done=False, horizon=99) - 0.25) < 1e-9  # nothing burnt yet this (new) turn
    s.players[0].mana_burnt_this_turn_metered = 1
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - h1)) < 1e-9  # cheap again, not h1-relative-to-6

    # Scored seat-relative, same convention as with_mana_mistake_penalty.
    s.active_idx = 1
    s.players[1].mana_burnt_this_turn_metered = 1
    assert abs(wrapped(s, done=False, horizon=99) - (0.25 - h1)) < 1e-9
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)  # untouched -- seat 0 wasn't scored this call


@pytest.mark.slow
def test_with_dense_mana_burn_penalty_game_cap():
    # game_penalty_cap: clamps PlayerState.mana_burn_penalty_charged_total
    # (never reset -- whole GAME, unlike mana_burn_penalty_credited, which
    # resets every turn) so many separate bad turns can't sum to a penalty
    # that dwarfs the terminal win/loss signal. A small cap (0.05) makes this
    # easy to actually hit within a test instead of needing dozens of turns.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base = lambda state, done, horizon: 0.0
    wrapped = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=0.05)

    # Turn 1: burn 6 (hill(6) ~= 0.916, far past the 0.05 cap on its own) --
    # charged only up to the cap, not the full curve value.
    s.players[0].mana_burnt_this_turn_metered = 6
    assert abs(wrapped(s, done=False, horizon=99) - (-0.05)) < 1e-9
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)

    # Simulate the turn boundary (game.turn._run_turn_gen resets the PER-TURN
    # fields; mana_burn_penalty_charged_total is NOT one of them).
    s.players[0].mana_burnt_this_turn_metered = 0
    s.players[0].mana_burn_penalty_credited = 0.0

    # Turn 2: even a single burnt pip would normally cost hill(1) > 0, but
    # the whole-game budget is already fully spent -- charged exactly 0, not
    # a bonus, not a further debt.
    s.players[0].mana_burnt_this_turn_metered = 1
    assert wrapped(s, done=False, horizon=99) == 0.0
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)  # unchanged -- nothing left to spend

    # A DIFFERENT player's own budget is untouched by the first player's cap.
    s.active_idx = 1
    s.players[1].mana_burnt_this_turn_metered = 6
    assert abs(wrapped(s, done=False, horizon=99) - (-0.05)) < 1e-9
    assert s.players[1].mana_burn_penalty_charged_total == pytest.approx(0.05)


@pytest.mark.slow
def test_deploy_reward_v2_applies_dense_mana_burn_penalty():
    # deploy_reward_v2 (2026-08 re-wire): now with_dense_mana_burn_penalty(
    # deploy_reward(win_floor=1.0)), not the bare deploy_reward it was
    # between the with_mana_mistake_penalty revert and this change.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    s.winner = 0
    s.players[0].mana_burnt_this_turn_metered = 6  # a real (metered) mistake
    unpenalized = deploy_reward(win_floor=1.0)(s, done=True, horizon=99)
    penalized = deploy_reward_v2(s, done=True, horizon=99)
    assert penalized < unpenalized  # the wrap actually subtracts something
