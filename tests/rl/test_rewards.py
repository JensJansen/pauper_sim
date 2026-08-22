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
    # with_dense_mana_burn_penalty: reward_fn is a plain passthrough of
    # base_reward_fn -- the actual Hill-curve charge lives in the returned
    # closure's own charge_single_pip_burn(player) attribute instead, called
    # by rl.training.train's on_single_pip_burn hook at the moment of an actual burn
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
    # the dense charge is applied elsewhere (rl.training.train.collect_rollout's
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
def test_flat_win_loss_reward_bands():
    # flat_win_loss_reward: flat +1/-1, NO discard penalty of any kind on
    # either band, terminal only. Scored per seat off state.active_idx (this
    # module's own contract: rl.training.train._reward_for flips it), so "did this
    # seat win" is state.winner == state.active_idx.
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

    # Hoarding must not move either band -- there is no cleanup-discard term
    # at all in this reward (see deploy_reward_v6's own comment for why).
    s.players[0].cleanup_discard_turns = 100
    s.winner = 0
    assert rf(s, done=True, horizon=99) == pytest.approx(1.0)
    s.winner = 1
    assert rf(s, done=True, horizon=99) == pytest.approx(-1.0)


def _effective_episode_score(reward_fn, s):
    """reward_fn's own terminal return MINUS whatever its charge_
    single_pip_burn (if any) would take for the active seat's CURRENT
    mana_burnt_this_turn_single_pip -- reconstructs the single EFFECTIVE
    number a real episode's return would sum to now that with_dense_mana_
    burn_penalty applies its charge via rl.training.train's on_single_pip_burn hook
    (a buffer patch on earlier transitions) rather than through reward_fn's
    own return value."""
    terminal = reward_fn(s, done=True, horizon=99)
    charge_fn = getattr(reward_fn, "charge_single_pip_burn", None)
    charge = charge_fn(s.players[s.active_idx]) if charge_fn is not None else 0.0
    return terminal - charge


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


def test_deploy_reward_v6_is_winner_only():
    # The opt-in flag rl.training.train._winner_only_burn_for reads -- deploy_reward_v6
    # is built with refund_on_loss=True, so a losing seat's dense mana-burn
    # charges must never be applied (see rl.training.train's deferred_charges).
    assert getattr(deploy_reward_v6, "mana_burn_winner_only", False) is True


@pytest.mark.slow
def test_deploy_reward_v6_curve_matches_owner_spec():
    # deploy_reward_v6's curve (mana_burn_c=2.9/p=4.0, mana_burn_weight=0.5):
    # locks the documented per-turn numbers so a future edit to its call, or
    # to _charge_single_pip_burn's weight handling, can't silently reshape it.
    assert _charge_for(deploy_reward_v6, 1) == pytest.approx(0.007, abs=1e-3)  # first pip still nearly free
    assert _charge_for(deploy_reward_v6, 2) == pytest.approx(0.092, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 3) == pytest.approx(0.267, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 4) == pytest.approx(0.392, abs=1e-3)
    assert _charge_for(deploy_reward_v6, 5) == pytest.approx(0.449, abs=1e-3)  # ~90% of the 0.5 weight
    assert _charge_for(deploy_reward_v6, 50) < 0.5  # asymptotic: approaches the weight, never reaches it


@pytest.mark.slow
def test_deploy_reward_v6_one_bad_turn_does_not_eat_the_whole_budget():
    # THE POINT OF this curve/weight combination: measured under an earlier,
    # 3x-steeper weight, a single 5-pip turn charged 90% of the whole-game
    # cap, so two such turns saturated it and every later burnt pip that game
    # was free -- observed in 64%/69% of games for two archetypes at a
    # 20,065-games/deck checkpoint. A maxed-out penalty is a flat toll, not a
    # gradient. This weight keeps the cap (the guarantee depends on it) and
    # keeps the charge proportional to actual waste for far longer: one
    # 5-pip turn is a real fraction of the budget, and it takes several such
    # turns (not two) to exhaust it.
    cap = 1.5
    assert _charge_for(deploy_reward_v6, 5) / cap < 0.35   # one turn is a real fraction of the whole-game budget
    assert _turns_to_saturate(deploy_reward_v6, 5) == 4    # not 2, as an earlier 3x-steeper weight measured
    assert _turns_to_saturate(deploy_reward_v6, 3) == 6    # 2.8 pips was the measured mean burnt-turn -- 3 is typical


@pytest.mark.slow
def test_deploy_reward_v6_guarantee():
    # Worst win = 1.0 - game_penalty_cap = -0.5; every loss = -1.0 exactly.
    s = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
    s.active_idx = 0

    s.winner = 0
    assert _effective_episode_score(deploy_reward_v6, s) == pytest.approx(1.0)  # clean win

    # Worst case needs SEVERAL bad turns to reach, not one -- one bad turn is
    # deliberately not enough to saturate the cap (see the test above), so the
    # worst case is summed over turns here instead of taken from a single
    # charge.
    p = s.players[0]
    p.mana_burnt_this_turn_single_pip = 50
    charged = 0.0
    for _turn in range(10):
        p.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        charged += deploy_reward_v6.charge_single_pip_burn(p)
    assert charged == pytest.approx(1.5)  # the cap is still reachable, just not in one turn
    worst_win = deploy_reward_v6(s, done=True, horizon=99) - charged
    assert worst_win == pytest.approx(-0.5, abs=1e-2)
    assert worst_win > -1.0  # margin of 0.5 preserved -- a bigger cap was deliberately not spent on more margin

    # Losses stay flat and ungraded -- no gradient a policy could optimize
    # independently of actually winning.
    def _loss_score(discard_turns, burnt):
        st = GameState(on_the_play=True, players=[PlayerState(True), PlayerState(False)])
        st.active_idx = 0
        st.winner = 1
        st.players[0].cleanup_discard_turns = discard_turns
        st.players[0].mana_burnt_this_turn_single_pip = burnt
        return deploy_reward_v6(st, done=True, horizon=99)

    assert _loss_score(0, 0) == pytest.approx(-1.0)
    assert _loss_score(0, 0) == _loss_score(100, 50)
