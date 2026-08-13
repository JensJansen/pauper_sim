"""Tests for rl.rewards's win/loss reward functions."""
import pytest

from game.state import GameState, PlayerState
from rl.rewards import (
    action_count_win_reward, deploy_reward, deploy_reward_v2, deploy_reward_v3,
    deploy_reward_v4, deploy_reward_v5, deploy_reward_v6, flat_win_loss_reward,
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
    # over an unconditional-input archetype bias, then re-enabled reading a
    # per-pip single-pip-tagged mana_burnt_this_turn_single_pip instead --
    # see rewards.py's own comment on deploy_reward_v2 for the full history;
    # tested standalone here for reference/regression coverage, not as the
    # active league reward).
    #
    # SECOND iteration (2026-08-10): reward_fn is now a plain passthrough of
    # base_reward_fn -- the actual Hill-curve charge lives in the returned
    # closure's own charge_single_pip_burn(player) attribute instead, called
    # by rl.train's on_single_pip_burn hook at the moment of an actual burn
    # rather than at an arbitrary later reward_fn call (see this module's
    # own docstring on with_dense_mana_burn_penalty for why). Reads
    # PlayerState.mana_burnt_this_turn_single_pip (NEVER drained by this --
    # only reset by game.turn._run_turn_gen at turn boundaries) and charges
    # only the MARGINAL increase in _hill(mana_burnt_this_turn_single_pip,
    # c, p) since the last call, tracked via PlayerState.
    # mana_burn_penalty_credited.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base_calls = []
    base = lambda state, done, horizon: base_calls.append((done, horizon)) or 0.25
    wrapped = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0)

    # reward_fn itself is a pure passthrough now, regardless of burn state --
    # the dense charge is applied elsewhere (rl.train.collect_rollout's
    # buffer patch), never through this return value.
    assert wrapped(s, done=False, horizon=99) == 0.25
    assert base_calls == [(False, 99)]  # base_reward_fn really was called, with the same args
    s.players[0].mana_burnt_this_turn_single_pip = 6
    assert wrapped(s, done=False, horizon=99) == 0.25  # still unaffected, even with real burn on the books
    s.players[0].mana_burnt_this_turn_single_pip = 0  # reset for the charge_single_pip_burn checks below

    # First pip burnt this turn: charge_single_pip_burn charges the FULL
    # hill(1) (baseline was 0), and mana_burnt_this_turn_single_pip itself
    # is untouched (unlike mana_mistake_burn, this never drains its input --
    # only the engine resets it, at the next turn boundary).
    s.players[0].mana_burnt_this_turn_single_pip = 1
    h1 = _hill(1, 3.3, 4.0)
    assert h1 == pytest.approx(0.0084, abs=1e-3)  # matches the owner-specified "~0.008 on the first pip" anchor
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h1)
    assert s.players[0].mana_burnt_this_turn_single_pip == 1  # NOT drained
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)

    # Same call again with NO new burn -> baseline already matches current
    # cumulative, so the marginal charge is exactly 0 (no double-charging a
    # pip that's already been paid for).
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)

    # More mana burns later in the SAME turn: charged only the DELTA, not
    # hill() of the new total from scratch -- this is what makes it
    # genuinely dense (each burn pays for what IT added) while still
    # summing to the same total a one-shot terminal charge would give.
    s.players[0].mana_burnt_this_turn_single_pip = 6
    h6 = _hill(6, 3.3, 4.0)
    assert h6 == pytest.approx(0.916, abs=1e-3)  # matches the owner-specified "~close to 1 by 6/7" anchor
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h6 - h1)
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h6)

    # Telescoping identity: the SUM of every marginal charge this turn must
    # equal hill(final total) exactly, whether charged in one jump (as
    # above) or many small steps -- verify the many-small-steps path lands
    # on the identical total.
    p2 = PlayerState(True)
    wrapped2 = with_dense_mana_burn_penalty(lambda state, done, horizon: 0.0, mana_burn_c=3.3, mana_burn_p=4.0)
    total_charged = 0.0
    for cum in range(1, 8):  # 7 separate single-pip burns, one charge call after each
        p2.mana_burnt_this_turn_single_pip = cum
        total_charged += wrapped2.charge_single_pip_burn(p2)
    assert total_charged == pytest.approx(_hill(7, 3.3, 4.0))

    # Reset (simulating a new turn, same as game.turn._run_turn_gen does)
    # zeroes the baseline too -- the next turn's first burn is cheap again,
    # not charged against the PREVIOUS turn's already-elevated baseline.
    s.players[0].mana_burnt_this_turn_single_pip = 0
    s.players[0].mana_burn_penalty_credited = 0.0
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)  # nothing burnt yet this (new) turn
    s.players[0].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h1)  # cheap again, not h1-relative-to-6

    # charge_single_pip_burn takes the player directly -- a DIFFERENT
    # player's own credited/charged_total is untouched by another's charge.
    s.players[1].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[1]) == pytest.approx(h1)
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)  # untouched -- player 0 wasn't charged this call


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
    s.players[0].mana_burnt_this_turn_single_pip = 6
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.05)
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)

    # Simulate the turn boundary (game.turn._run_turn_gen resets the PER-TURN
    # fields; mana_burn_penalty_charged_total is NOT one of them).
    s.players[0].mana_burnt_this_turn_single_pip = 0
    s.players[0].mana_burn_penalty_credited = 0.0

    # Turn 2: even a single burnt pip would normally cost hill(1) > 0, but
    # the whole-game budget is already fully spent -- charged exactly 0, not
    # a bonus, not a further debt.
    s.players[0].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)  # unchanged -- nothing left to spend

    # A DIFFERENT player's own budget is untouched by the first player's cap.
    s.players[1].mana_burnt_this_turn_single_pip = 6
    assert wrapped.charge_single_pip_burn(s.players[1]) == pytest.approx(0.05)
    assert s.players[1].mana_burn_penalty_charged_total == pytest.approx(0.05)


@pytest.mark.slow
def test_deploy_reward_v2_applies_dense_mana_burn_penalty():
    # deploy_reward_v2 (2026-08 re-wire): now with_dense_mana_burn_penalty(
    # deploy_reward(win_floor=1.0)), not the bare deploy_reward it was
    # between the with_mana_mistake_penalty revert and this change.
    #
    # SECOND iteration (2026-08-10): deploy_reward_v2 itself (the reward_fn
    # callable) is a pure passthrough now -- calling it directly is
    # byte-identical to the unwrapped base regardless of burn state. The
    # actual penalty lives in its own charge_single_pip_burn(player)
    # attribute, applied by rl.train's on_single_pip_burn hook instead (see
    # with_dense_mana_burn_penalty's own docstring for why).
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    s.winner = 0
    s.players[0].mana_burnt_this_turn_single_pip = 6  # a real (single-pip) mistake
    unpenalized = deploy_reward(win_floor=1.0)(s, done=True, horizon=99)
    passthrough = deploy_reward_v2(s, done=True, horizon=99)
    assert passthrough == pytest.approx(unpenalized)  # reward_fn itself no longer subtracts anything
    assert deploy_reward_v2.charge_single_pip_burn(s.players[0]) > 0  # the real charge lives here now


def test_deploy_reward_discard_weight_scales_q_asymptote():
    # discard_weight (new 2026-08-10 param, default 1.0 -- v1/v2 byte-
    # identical to before, since neither passes it) scales q's own
    # asymptote down WITHOUT reshaping its curve: q approaches
    # discard_weight, never 1.0, as cleanup_discard_turns grows -- lets
    # deploy_reward_v3 (below) size discard's own share of a combined
    # badness budget shared with the dense mana-burn wrap.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    s.winner = 0
    s.players[0].cleanup_discard_turns = 1000  # deep into the asymptote either way
    q_default = 1.0 - deploy_reward(win_floor=1.0)(s, done=True, horizon=99)
    q_weighted = 1.0 - deploy_reward(win_floor=1.0, discard_weight=0.4)(s, done=True, horizon=99)
    assert q_default == pytest.approx(1.0, abs=1e-3)
    assert q_weighted == pytest.approx(0.4, abs=1e-3)


@pytest.mark.slow
def test_deploy_reward_v3_curve_considerably_steeper_than_v2():
    # Owner spec (2026-08-10): "up the cost of burning the first mana pip
    # considerably... more aggressive punishment for burning tagged mana
    # sources." Verifies the new mana_burn_c=2.0/mana_burn_p=2.5 curve
    # against the exact numbers documented on deploy_reward_v3 itself, and
    # that the first pip really is considerably (not just nominally)
    # costlier than v2's own curve (mana_burn_c=3.3/mana_burn_p=4.0).
    for x, expected in [(1, 0.1502), (2, 0.5), (3, 0.7337), (4, 0.8498), (6, 0.9397)]:
        assert _hill(x, 2.0, 2.5) == pytest.approx(expected, abs=1e-3)
    ratio = _hill(1, 2.0, 2.5) / _hill(1, 3.3, 4.0)
    assert ratio == pytest.approx(17.97, abs=0.1)  # ~18x costlier first pip than v2


def _effective_episode_score(reward_fn, s):
    """reward_fn's own terminal return MINUS whatever its charge_
    single_pip_burn (if any) would take for the active seat's CURRENT
    mana_burnt_this_turn_single_pip -- reconstructs the single EFFECTIVE
    number a real episode's return would sum to now that with_dense_mana_
    burn_penalty applies its charge via rl.train's on_single_pip_burn hook
    (a buffer patch on earlier transitions) rather than through reward_fn's
    own return value. deploy_reward_v1/action_count_win_reward etc. have no
    charge_single_pip_burn attribute at all -- getattr defaults to a no-op
    so this helper works uniformly across every reward_fn in this module."""
    terminal = reward_fn(s, done=True, horizon=99)
    charge_fn = getattr(reward_fn, "charge_single_pip_burn", None)
    charge = charge_fn(s.players[s.active_idx]) if charge_fn is not None else 0.0
    return terminal - charge


@pytest.mark.slow
def test_deploy_reward_v3_best_win_worst_loss_spread_is_two():
    # Owner spec (2026-08-10): "normalize... so the best wins are ~2 > the
    # worst losses (IE games without discarding, without mana burn score
    # approximately 2 greater than a game which discards and burns mana
    # frivolously)." discard_weight=0.4 + game_penalty_cap=0.6 sum to
    # exactly win_floor (1.0) by construction, verified here numerically at
    # the actual asymptote/cap rather than trusted as arithmetic. Uses
    # _effective_episode_score (terminal reward_fn return minus the SEPARATE
    # charge_single_pip_burn charge) since that's how the two now compose
    # within a real episode -- see with_dense_mana_burn_penalty's own
    # docstring for why the dense charge moved out of reward_fn's own
    # return value.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    # Best win: no discarding, no mana burn -> exactly 1.0.
    s.winner = 0
    assert _effective_episode_score(deploy_reward_v3, s) == pytest.approx(1.0)

    # Worst loss: heavy discarding (q approaching its 0.4 asymptote) AND
    # mana burn saturating its 0.6 whole-game cap -> approaches -1.0. The
    # first score above was a no-op on the mana-burn wrapper's own
    # credited/charged_total state (nothing had burnt yet), so reusing the
    # same player here is still a clean read.
    s.winner = 1  # this seat (0) loses
    s.players[0].cleanup_discard_turns = 100  # q -> ~0.3994, close to its 0.4 asymptote
    s.players[0].mana_burnt_this_turn_single_pip = 50  # raw hill far past the 0.6 cap on its own
    worst_loss = _effective_episode_score(deploy_reward_v3, s)
    assert worst_loss == pytest.approx(-1.0, abs=1e-2)

    spread = 1.0 - worst_loss
    assert spread == pytest.approx(2.0, abs=1e-2)


@pytest.mark.slow
def test_deploy_reward_v3_restores_every_win_beats_every_loss():
    # v2's own comment explicitly ABANDONED "every win beats every loss" as
    # an accepted tradeoff once its mana-burn wrap was re-enabled
    # (game_penalty_cap=2.0 stacked on top of q's own up-to-1.0 asymptote
    # can in principle push a sloppy win below a clean loss). v3's
    # discard_weight=0.4 + game_penalty_cap=0.6 summing to exactly
    # win_floor (1.0) restores this as an actual guarantee -- proven here
    # at the worst case for each reward (via _effective_episode_score, see
    # its own docstring), using INDEPENDENT GameStates per call so the
    # mana-burn wrapper's own per-player credited/charged_total bookkeeping
    # can't leak between the v3 and v2 comparison calls below.
    def _worst_case_win(reward_fn):
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.active_idx = 0
        s.winner = 0
        s.players[0].cleanup_discard_turns = 100
        s.players[0].mana_burnt_this_turn_single_pip = 50
        return _effective_episode_score(reward_fn, s)

    def _clean_loss(reward_fn):
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.active_idx = 0
        s.winner = 1
        return reward_fn(s, done=True, horizon=99)

    clean_loss_v3 = _clean_loss(deploy_reward_v3)
    assert clean_loss_v3 == pytest.approx(0.0)
    assert _worst_case_win(deploy_reward_v3) > clean_loss_v3  # v3: restored, holds even at the worst case

    clean_loss_v2 = _clean_loss(deploy_reward_v2)
    assert clean_loss_v2 == pytest.approx(0.0)
    assert _worst_case_win(deploy_reward_v2) < clean_loss_v2  # v2: the accepted tradeoff actually firing, same input


@pytest.mark.slow
def test_deploy_reward_v4_curve_reverts_to_v2_shape():
    # deploy_reward_v4 (2026-08-10, second revision) SUPERSEDES
    # deploy_reward_v3: a real training run under v3 drove elves/
    # rakdos_madness into a zero-mana-development collapse once the same-day
    # on_single_pip_burn credit-assignment fix turned v3's steepened curve
    # into a double-count (see rewards.py's own SUPERSEDED comment on
    # deploy_reward_v3). v4 keeps v3's discard_weight=0.4/game_penalty_cap=
    # 0.6 split but reverts mana_burn_c/mana_burn_p to v2's original 3.3/4.0
    # (from v3's 2.0/2.5) -- verified here against the wired-in
    # charge_single_pip_burn value itself, not just the source constants, so
    # a future edit to deploy_reward_v4's call or with_dense_mana_burn_
    # penalty's defaults would actually break this test.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    s.players[0].mana_burnt_this_turn_single_pip = 1
    h1_v2_shape = _hill(1, 3.3, 4.0)
    assert h1_v2_shape == pytest.approx(0.0084, abs=1e-3)  # v2's "~0.008 on the first pip" anchor, NOT v3's ~0.150
    assert deploy_reward_v4.charge_single_pip_burn(s.players[0]) == pytest.approx(h1_v2_shape)


@pytest.mark.slow
def test_deploy_reward_v4_best_win_worst_loss_spread_is_two():
    # Same guarantee as test_deploy_reward_v3_best_win_worst_loss_spread_is_
    # two above, re-verified for v4: discard_weight=0.4 + game_penalty_cap=
    # 0.6 (unchanged from v3, unrelated to the curve's own steepness) still
    # sum to exactly win_floor (1.0), so the best-win/worst-loss spread
    # holds regardless of the mana_burn_c/p revert.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    # Best win: no discarding, no mana burn -> exactly 1.0.
    s.winner = 0
    assert _effective_episode_score(deploy_reward_v4, s) == pytest.approx(1.0)

    # Worst loss: heavy discarding (q approaching its 0.4 asymptote) AND
    # mana burn saturating its 0.6 whole-game cap -> approaches -1.0, same
    # as v3 -- the milder v4 curve still saturates well past the cap at 50
    # burnt pips, so the cap (not the curve) determines the worst case.
    s.winner = 1  # this seat (0) loses
    s.players[0].cleanup_discard_turns = 100  # q -> ~0.3994, close to its 0.4 asymptote
    s.players[0].mana_burnt_this_turn_single_pip = 50  # raw hill far past the 0.6 cap on its own
    worst_loss = _effective_episode_score(deploy_reward_v4, s)
    assert worst_loss == pytest.approx(-1.0, abs=1e-2)

    spread = 1.0 - worst_loss
    assert spread == pytest.approx(2.0, abs=1e-2)


@pytest.mark.slow
def test_deploy_reward_v4_restores_every_win_beats_every_loss():
    # Same "every win beats every loss" guarantee test as
    # test_deploy_reward_v3_restores_every_win_beats_every_loss above,
    # re-verified for v4 -- the guarantee comes from discard_weight/
    # game_penalty_cap summing to win_floor, which v4 keeps unchanged from
    # v3, so it must hold identically despite the curve revert.
    def _worst_case_win(reward_fn):
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.active_idx = 0
        s.winner = 0
        s.players[0].cleanup_discard_turns = 100
        s.players[0].mana_burnt_this_turn_single_pip = 50
        return _effective_episode_score(reward_fn, s)

    def _clean_loss(reward_fn):
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.active_idx = 0
        s.winner = 1
        return reward_fn(s, done=True, horizon=99)

    clean_loss_v4 = _clean_loss(deploy_reward_v4)
    assert clean_loss_v4 == pytest.approx(0.0)
    assert _worst_case_win(deploy_reward_v4) > clean_loss_v4  # v4: guarantee holds, same as v3


@pytest.mark.slow
def test_flat_win_loss_reward_bands():
    # flat_win_loss_reward (2026-08-11): deploy_reward_v5's base. Flat +1/-1,
    # NO discard penalty q on either band, terminal only. Scored per seat off
    # state.active_idx exactly like deploy_reward (rl.train._reward_for flips
    # it), so "did this seat win" is state.winner == state.active_idx.
    rf = flat_win_loss_reward()
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    assert rf(s, done=False, horizon=99) == 0.0  # terminal only

    s.winner = 0
    assert rf(s, done=True, horizon=99) == pytest.approx(1.0)

    s.winner = 1  # this seat lost
    assert rf(s, done=True, horizon=99) == pytest.approx(-1.0)

    s.winner = None  # no-winner horizon timeout scores as a loss, not 0
    assert rf(s, done=True, horizon=99) == pytest.approx(-1.0)

    # q is GONE: hoarding must not move either band (the whole point -- see
    # deploy_reward_v5's comment on why q was dropped entirely).
    s.players[0].cleanup_discard_turns = 100
    s.winner = 0
    assert rf(s, done=True, horizon=99) == pytest.approx(1.0)
    s.winner = 1
    assert rf(s, done=True, horizon=99) == pytest.approx(-1.0)


@pytest.mark.slow
def test_deploy_reward_v5_curve_matches_owner_spec():
    # v5's curve (2026-08-11, owner spec "cap at 5 mana burnt and a score of
    # -1.5"): mana_burn_weight=1.5 SCALES the Hill curve (new param -- _hill
    # itself asymptotes at 1.0), mana_burn_c=2.9/p=4.0 place that asymptote
    # near 5 burnt pips, game_penalty_cap=1.5 bounds the whole-GAME total.
    # Locks the documented per-turn numbers so a future edit to v5's call, or
    # to _charge_single_pip_burn's weight handling, can't silently reshape it.
    def _charge_for(pips):
        s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        s.players[0].mana_burnt_this_turn_single_pip = pips
        return deploy_reward_v5.charge_single_pip_burn(s.players[0])

    assert _charge_for(1) == pytest.approx(1.5 * _hill(1, 2.9, 4.0), abs=1e-4)  # ~0.021, first pip nearly free
    assert _charge_for(2) == pytest.approx(0.277, abs=1e-3)
    assert _charge_for(3) == pytest.approx(0.801, abs=1e-3)
    assert _charge_for(4) == pytest.approx(1.175, abs=1e-3)
    assert _charge_for(5) == pytest.approx(1.348, abs=1e-3)  # ~90% of the 1.5 weight, per spec
    assert _charge_for(50) < 1.5  # asymptotic: approaches the weight, never reaches it

    # Whole-game cap: repeated bad turns accumulate but never exceed 1.5.
    # (mana_burn_penalty_credited resets per turn -- game.turn._run_turn_gen --
    # so each turn re-charges from 0; only charged_total persists.)
    p = PlayerState(True)
    p.mana_burnt_this_turn_single_pip = 5
    total = 0.0
    for _turn in range(10):
        p.mana_burn_penalty_credited = 0.0
        total += deploy_reward_v5.charge_single_pip_burn(p)
    assert total == pytest.approx(1.5)


@pytest.mark.slow
def test_deploy_reward_v5_is_winner_only_and_v4_is_not():
    # The opt-in flag rl.train._winner_only_burn_for reads. v5 sets it
    # (refund_on_loss=True); every earlier version must NOT, since that flag
    # is what makes collect_rollout defer charges instead of applying them --
    # a regression here would silently change v2/v3/v4's semantics.
    assert getattr(deploy_reward_v5, "mana_burn_winner_only", False) is True
    assert getattr(deploy_reward_v6, "mana_burn_winner_only", False) is True
    for older in (deploy_reward_v2, deploy_reward_v3, deploy_reward_v4):
        assert getattr(older, "mana_burn_winner_only", False) is False


@pytest.mark.slow
def test_deploy_reward_v5_guarantee_and_no_discard_dependence():
    # v5's guarantee is re-derived, NOT inherited: it no longer depends on
    # v3/v4's discard_weight + game_penalty_cap == win_floor equation (there
    # is no discard_weight anymore). Worst win = 1.0 - cap = -0.5; EVERY loss
    # = -1.0 exactly, with no range at all.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    s.winner = 0
    assert _effective_episode_score(deploy_reward_v5, s) == pytest.approx(1.0)  # clean win

    # Worst-case win: burn saturating the whole-game cap -> 1.0 - 1.5 = -0.5.
    # A sloppy win CAN score negative under v5 (v4's floor was +0.4) -- fine,
    # since the only ordering that matters is win-vs-loss.
    s.players[0].mana_burnt_this_turn_single_pip = 50
    worst_win = _effective_episode_score(deploy_reward_v5, s)
    assert worst_win == pytest.approx(-0.5, abs=1e-2)
    assert worst_win > -1.0  # still strictly beats every loss

    # THE FIX ITSELF: a loss scores exactly -1.0 no matter how it was played.
    # Passive (never burnt, never hoarded) and active-and-sloppy (heavy burn,
    # heavy hoarding) must be INDISTINGUISHABLE -- v4 scored the passive loss
    # ~2x better, which is what taught dmir_terror to stop playing.
    def _loss_score(discard_turns, burnt):
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        st.active_idx = 0
        st.winner = 1
        st.players[0].cleanup_discard_turns = discard_turns
        st.players[0].mana_burnt_this_turn_single_pip = burnt
        # Terminal return only: on a LOSS rl.train never applies the deferred
        # burn charges at all (see collect_rollout's deferred_charges), so the
        # effective episode score IS the terminal number.
        return deploy_reward_v5(st, done=True, horizon=99)

    assert _loss_score(0, 0) == pytest.approx(-1.0)      # passive loss
    assert _loss_score(100, 50) == pytest.approx(-1.0)   # active, sloppy loss
    assert _loss_score(0, 0) == _loss_score(100, 50)     # no gradation to exploit


def _charge_for(reward_fn, pips):
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.players[0].mana_burnt_this_turn_single_pip = pips
    return reward_fn.charge_single_pip_burn(s.players[0])


def _turns_to_saturate(reward_fn, pips_per_turn, cap=1.5):
    """How many consecutive `pips_per_turn` turns it takes to fully exhaust
    the whole-game cap, after which every further burnt pip is charged 0.0."""
    p = PlayerState(True)
    p.mana_burnt_this_turn_single_pip = pips_per_turn
    for turn in range(1, 100):
        p.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        reward_fn.charge_single_pip_burn(p)
        if p.mana_burn_penalty_charged_total >= cap - 1e-9:
            return turn
    return None


@pytest.mark.slow
def test_deploy_reward_v6_curve_is_v5_shape_scaled_to_weight_half():
    # v6 (2026-08-11) moves exactly ONE constant from v5: mana_burn_weight
    # 1.5 -> 0.5. Same c=2.9/p=4.0 shape, same cap. Locks the documented
    # per-turn numbers, and pins them to v5's curve scaled by 1/3 so a future
    # edit can't reshape one without the other.
    assert _charge_for(deploy_reward_v6, 1) == pytest.approx(0.007, abs=1e-3)  # first pip still nearly free
    assert _charge_for(deploy_reward_v6, 2) == pytest.approx(0.092, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 3) == pytest.approx(0.267, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 4) == pytest.approx(0.392, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 5) == pytest.approx(0.449, abs=1e-3)  # ~90% of the 0.5 weight
    assert _charge_for(deploy_reward_v6, 50) < 0.5  # asymptotic: approaches the weight, never reaches it
    for pips in (1, 2, 3, 4, 5, 6):
        assert _charge_for(deploy_reward_v6, pips) == pytest.approx(
            _charge_for(deploy_reward_v5, pips) / 3.0, abs=1e-6)


@pytest.mark.slow
def test_deploy_reward_v6_one_bad_turn_no_longer_eats_the_whole_budget():
    # THE POINT OF v6. Under v5 a single 5-pip turn charged 1.348 of a 1.5
    # cap -- 90% of the whole-game budget -- so two such turns saturated it
    # and every later burnt pip that game was free. Measured over v5's
    # 10,003-games/deck logs, dmir_terror/elves hit that in 64%/69% of games,
    # ~2/3 of the way through; a maxed-out penalty is a flat toll, not a
    # gradient. v6 keeps the cap (the guarantee depends on it) and shrinks
    # the curve under it instead, so the charge stays proportional to actual
    # waste for far longer.
    cap = 1.5
    assert _charge_for(deploy_reward_v5, 5) / cap > 0.85   # v5: one turn ~= the entire budget
    assert _charge_for(deploy_reward_v6, 5) / cap < 0.35   # v6: one turn is a real fraction of it
    assert _turns_to_saturate(deploy_reward_v5, 5) == 2
    assert _turns_to_saturate(deploy_reward_v6, 5) == 4
    # Ordinary (not disastrous) turns: v5 saturated in a handful, v6 does not
    # saturate on anything like a realistic count. 2.8 pips was the measured
    # mean over turns that burnt anything at all, so 3 is the typical case.
    assert _turns_to_saturate(deploy_reward_v5, 3) == 2
    assert _turns_to_saturate(deploy_reward_v6, 3) == 6


@pytest.mark.slow
def test_deploy_reward_v6_guarantee_is_unchanged_from_v5():
    # game_penalty_cap and the base band are byte-identical to v5, so the
    # guarantee needs no re-derivation -- but assert it directly rather than
    # by inspection, since v6 is the version the league will actually run.
    # Worst win = 1.0 - cap = -0.5; every loss = -1.0 exactly.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    s.winner = 0
    assert _effective_episode_score(deploy_reward_v6, s) == pytest.approx(1.0)  # clean win

    # Worst case needs SEVERAL bad turns to reach, not one -- _effective_
    # episode_score charges once, which was enough to saturate v5's cap but
    # deliberately isn't under v6. That gap IS the fix, so the worst case is
    # summed over turns here instead of taken from a single charge.
    p = s.players[0]
    p.mana_burnt_this_turn_single_pip = 50
    charged = 0.0
    for _turn in range(10):
        p.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        charged += deploy_reward_v6.charge_single_pip_burn(p)
    assert charged == pytest.approx(1.5)  # the cap is still reachable, just not in one turn
    worst_win = deploy_reward_v6(s, done=True, horizon=99) - charged
    assert worst_win == pytest.approx(-0.5, abs=1e-2)
    assert worst_win > -1.0  # margin of 0.5 preserved -- v6 deliberately did NOT spend it on a bigger cap

    # Losses stay flat and ungraded, same as v5 (this is what fixed the
    # passive-loss asymmetry; v6 must not reintroduce a gradient here).
    def _loss_score(discard_turns, burnt):
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        st.active_idx = 0
        st.winner = 1
        st.players[0].cleanup_discard_turns = discard_turns
        st.players[0].mana_burnt_this_turn_single_pip = burnt
        return deploy_reward_v6(st, done=True, horizon=99)

    assert _loss_score(0, 0) == pytest.approx(-1.0)
    assert _loss_score(0, 0) == _loss_score(100, 50)
