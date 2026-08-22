"""Tests for rl.rewards's win/loss reward functions."""
import pytest

from game.state import GameState, PlayerState
from rl.rewards import deploy_reward_v6, flat_win_loss_reward, with_dense_mana_burn_penalty


def _hill(x, c, p):
    if x <= 0:
        return 0.0
    xp = x ** p
    return xp / (xp + c ** p)


@pytest.mark.slow
def test_with_dense_mana_burn_penalty_telescopes():
    # reward_fn is a plain passthrough; the actual Hill-curve charge lives
    # in charge_single_pip_burn(player), called by the on_single_pip_burn
    # hook at the moment of a burn. Reads
    # PlayerState.mana_burnt_this_turn_single_pip (never drained, only reset
    # at turn boundaries) and charges only the MARGINAL increase in
    # _hill(mana_burnt_this_turn_single_pip, c, p) since the last call.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base_calls = []
    base = lambda state, done, horizon: base_calls.append((done, horizon)) or 0.25
    wrapped = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0)

    # reward_fn is a pure passthrough regardless of burn state -- the dense
    # charge is applied elsewhere (collect_rollout's buffer patch).
    assert wrapped(s, done=False, horizon=99) == 0.25
    assert base_calls == [(False, 99)]  # base_reward_fn really was called, with the same args
    s.players[0].mana_burnt_this_turn_single_pip = 6
    assert wrapped(s, done=False, horizon=99) == 0.25  # still unaffected, even with real burn on the books
    s.players[0].mana_burnt_this_turn_single_pip = 0  # reset for the charge_single_pip_burn checks below

    # First pip burnt this turn: charges the full hill(1) (baseline was 0);
    # mana_burnt_this_turn_single_pip itself is not drained by this call.
    s.players[0].mana_burnt_this_turn_single_pip = 1
    h1 = _hill(1, 3.3, 4.0)
    assert h1 == pytest.approx(0.0084, abs=1e-3)
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h1)
    assert s.players[0].mana_burnt_this_turn_single_pip == 1  # NOT drained
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)

    # Same call again with no new burn -> marginal charge is exactly 0.
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)

    # More burns later in the SAME turn: charged only the DELTA, not
    # hill() of the new total from scratch.
    s.players[0].mana_burnt_this_turn_single_pip = 6
    h6 = _hill(6, 3.3, 4.0)
    assert h6 == pytest.approx(0.916, abs=1e-3)
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h6 - h1)
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h6)

    # Telescoping identity: the sum of every marginal charge this turn must
    # equal hill(final total) exactly, whether charged in one jump or many
    # small steps.
    p2 = PlayerState(True)
    wrapped2 = with_dense_mana_burn_penalty(lambda state, done, horizon: 0.0, mana_burn_c=3.3, mana_burn_p=4.0)
    total_charged = 0.0
    for cum in range(1, 8):  # 7 separate single-pip burns, one charge call after each
        p2.mana_burnt_this_turn_single_pip = cum
        total_charged += wrapped2.charge_single_pip_burn(p2)
    assert total_charged == pytest.approx(_hill(7, 3.3, 4.0))

    # Reset (a new turn) zeroes the baseline too -- the next turn's first
    # burn is cheap again, not charged against the previous elevated baseline.
    s.players[0].mana_burnt_this_turn_single_pip = 0
    s.players[0].mana_burn_penalty_credited = 0.0
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)  # nothing burnt yet this (new) turn
    s.players[0].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(h1)  # cheap again, not h1-relative-to-6

    # charge_single_pip_burn takes the player directly -- a different
    # player's credited/charged_total is untouched by another's charge.
    s.players[1].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[1]) == pytest.approx(h1)
    assert s.players[0].mana_burn_penalty_credited == pytest.approx(h1)  # untouched -- player 0 wasn't charged this call


@pytest.mark.slow
def test_with_dense_mana_burn_penalty_game_cap():
    # game_penalty_cap clamps PlayerState.mana_burn_penalty_charged_total
    # (whole-game, never reset). A small cap (0.05) makes it easy to hit
    # within a test instead of needing dozens of turns.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0
    base = lambda state, done, horizon: 0.0
    wrapped = with_dense_mana_burn_penalty(base, mana_burn_c=3.3, mana_burn_p=4.0, game_penalty_cap=0.05)

    # Turn 1: burn 6 (hill(6) ~= 0.916, far past the 0.05 cap) -- charged
    # only up to the cap.
    s.players[0].mana_burnt_this_turn_single_pip = 6
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.05)
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)

    # Simulate the turn boundary (per-turn fields reset;
    # mana_burn_penalty_charged_total is not one of them).
    s.players[0].mana_burnt_this_turn_single_pip = 0
    s.players[0].mana_burn_penalty_credited = 0.0

    # Turn 2: budget is already fully spent -- charged exactly 0.
    s.players[0].mana_burnt_this_turn_single_pip = 1
    assert wrapped.charge_single_pip_burn(s.players[0]) == pytest.approx(0.0)
    assert s.players[0].mana_burn_penalty_charged_total == pytest.approx(0.05)  # unchanged -- nothing left to spend

    # A different player's budget is untouched by the first player's cap.
    s.players[1].mana_burnt_this_turn_single_pip = 6
    assert wrapped.charge_single_pip_burn(s.players[1]) == pytest.approx(0.05)
    assert s.players[1].mana_burn_penalty_charged_total == pytest.approx(0.05)


@pytest.mark.slow
def test_flat_win_loss_reward_bands():
    # flat +1/-1, no discard penalty on either band, terminal only. Scored
    # per seat off state.active_idx (state.winner == state.active_idx).
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

    # Hoarding must not move either band -- no cleanup-discard term exists
    # in this reward.
    s.players[0].cleanup_discard_turns = 100
    s.winner = 0
    assert rf(s, done=True, horizon=99) == pytest.approx(1.0)
    s.winner = 1
    assert rf(s, done=True, horizon=99) == pytest.approx(-1.0)


def _effective_episode_score(reward_fn, s):
    """reward_fn's terminal return minus whatever charge_single_pip_burn (if
    any) would take for the active seat's current
    mana_burnt_this_turn_single_pip -- reconstructs the effective number a
    real episode would sum to, since with_dense_mana_burn_penalty applies
    its charge via a buffer patch rather than through reward_fn's return
    value."""
    terminal = reward_fn(s, done=True, horizon=99)
    charge_fn = getattr(reward_fn, "charge_single_pip_burn", None)
    charge = charge_fn(s.players[s.active_idx]) if charge_fn is not None else 0.0
    return terminal - charge


def _charge_for(reward_fn, pips):
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.players[0].mana_burnt_this_turn_single_pip = pips
    return reward_fn.charge_single_pip_burn(s.players[0])


def _turns_to_saturate(reward_fn, pips_per_turn, cap=1.5):
    """Consecutive `pips_per_turn` turns needed to fully exhaust the
    whole-game cap, after which every further burnt pip is charged 0.0."""
    p = PlayerState(True)
    p.mana_burnt_this_turn_single_pip = pips_per_turn
    for turn in range(1, 100):
        p.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        reward_fn.charge_single_pip_burn(p)
        if p.mana_burn_penalty_charged_total >= cap - 1e-9:
            return turn
    return None


def test_deploy_reward_v6_is_winner_only():
    # deploy_reward_v6 is built with refund_on_loss=True, so a losing seat's
    # dense mana-burn charges must never be applied.
    assert getattr(deploy_reward_v6, "mana_burn_winner_only", False) is True


@pytest.mark.slow
def test_deploy_reward_v6_curve_matches_owner_spec():
    # Locks deploy_reward_v6's per-turn charge curve so a future edit can't
    # silently reshape it.
    assert _charge_for(deploy_reward_v6, 1) == pytest.approx(0.007, abs=1e-3)  # first pip still nearly free
    assert _charge_for(deploy_reward_v6, 2) == pytest.approx(0.092, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 3) == pytest.approx(0.267, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 4) == pytest.approx(0.392, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 5) == pytest.approx(0.449, abs=1e-3)  # ~90% of the 0.5 weight
    assert _charge_for(deploy_reward_v6, 50) < 0.5  # asymptotic: approaches the weight, never reaches it


@pytest.mark.slow
def test_deploy_reward_v6_one_bad_turn_does_not_eat_the_whole_budget():
    # A maxed-out penalty is a flat toll, not a gradient. This curve/weight
    # keeps the charge proportional to actual waste: one 5-pip turn is a
    # real fraction of the whole-game budget, and it takes several such
    # turns to exhaust it.
    cap = 1.5
    assert _charge_for(deploy_reward_v6, 5) / cap < 0.35
    assert _turns_to_saturate(deploy_reward_v6, 5) == 4
    assert _turns_to_saturate(deploy_reward_v6, 3) == 6


@pytest.mark.slow
def test_deploy_reward_v6_guarantee():
    # Worst win = 1.0 - game_penalty_cap = -0.5; every loss = -1.0 exactly.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    s.winner = 0
    assert _effective_episode_score(deploy_reward_v6, s) == pytest.approx(1.0)  # clean win

    # Worst case needs several bad turns to reach, not one (see the test
    # above), so it's summed over turns here.
    p = s.players[0]
    p.mana_burnt_this_turn_single_pip = 50
    charged = 0.0
    for _turn in range(10):
        p.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        charged += deploy_reward_v6.charge_single_pip_burn(p)
    assert charged == pytest.approx(1.5)  # the cap is still reachable, just not in one turn
    worst_win = deploy_reward_v6(s, done=True, horizon=99) - charged
    assert worst_win == pytest.approx(-0.5, abs=1e-2)
    assert worst_win > -1.0  # margin of 0.5 preserved

    # Losses stay flat and ungraded.
    def _loss_score(discard_turns, burnt):
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        st.active_idx = 0
        st.winner = 1
        st.players[0].cleanup_discard_turns = discard_turns
        st.players[0].mana_burnt_this_turn_single_pip = burnt
        return deploy_reward_v6(st, done=True, horizon=99)

    assert _loss_score(0, 0) == pytest.approx(-1.0)
    assert _loss_score(0, 0) == _loss_score(100, 50)
