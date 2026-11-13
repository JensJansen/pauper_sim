"""Schedules and progress bookkeeping as properties (plan P5): they are keyed on CUMULATIVE games, so a run split into many
sessions behaves as one run. test_run_league.py (integration) covers the end-to-end cases; these are the invariants."""
import json
import random

import pytest

from rl.league import league_runner as R
from rl.training.train import batch_size_for_iteration, ent_coef_schedule


# ---- snapshot cadence ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_snapshots_fire_once_per_cadence_multiple_however_the_run_is_split_into_sessions(seed):
    rng = random.Random(seed)
    g = rng.choice([1, 2, 5, 6, 10])
    every = rng.choice([1, 2, 3, 7, 20])
    total_iterations = rng.randint(30, 200)
    fired, games = 0, 0
    remaining = total_iterations
    while remaining:  # sessions of random length; `iteration` restarts at 0 each time, cumulative games do not
        session = min(remaining, rng.randint(1, 25))
        for _it in range(session):
            fired += R.should_snapshot(games, g, every)
            games += g
        remaining -= session
    assert fired == games // (every * g), f"g={g} every={every} iterations={total_iterations}"


def test_a_session_shorter_than_the_cadence_still_snapshots_when_the_cumulative_count_crosses():
    assert R.should_snapshot(games_before=95, games_per_iteration=10, snapshot_every=10) is True
    assert R.should_snapshot(games_before=85, games_per_iteration=10, snapshot_every=10) is False
    assert R.should_snapshot(games_before=0, games_per_iteration=10, snapshot_every=10) is False


def test_no_double_snapshot_exactly_on_a_session_boundary():
    # session 1 ends with games == 100 (a multiple); session 2's first iteration starts there and must not re-fire it
    assert R.should_snapshot(90, 10, 10) is True   # the iteration that reaches 100
    assert R.should_snapshot(100, 10, 10) is False  # the next one begins at 100


def _ctx(**kw):
    base = {"train_mulligan": True, "cumulative_games": 0, "games_per_iteration": 10, "matchup": None}
    base.update(kw)
    return base


def test_snapshot_due_uses_cumulative_games_and_never_fires_for_a_matchup():
    assert R._snapshot_due(_ctx(cumulative_games=90), 0, 10) is True
    assert R._snapshot_due(_ctx(cumulative_games=0), 9, 10) is True  # iteration 9 takes 90 -> 100
    assert R._snapshot_due(_ctx(cumulative_games=0), 8, 10) is False
    assert R._snapshot_due(_ctx(cumulative_games=90, matchup=("a", "b")), 0, 10) is False


# ---- mulligan flush cadence ------------------------------------------------------------------------------------------------

def test_mulligan_flush_fires_every_k_iterations_and_on_the_last_one():
    k = R.MULLIGAN_UPDATE_EVERY
    n = 3 * k + 3
    due = [i for i in range(n) if R._mulligan_flush_due(_ctx(), i, n)]
    assert due == [k - 1, 2 * k - 1, 3 * k - 1, n - 1]


def test_mulligan_flush_never_fires_when_the_mulligan_is_not_training():
    assert not any(R._mulligan_flush_due(_ctx(train_mulligan=False), i, 20) for i in range(20))


def test_a_one_iteration_session_still_flushes_its_mulligan_transitions():
    assert R._mulligan_flush_due(_ctx(), 0, 1) is True


# ---- the PPO schedules -----------------------------------------------------------------------------------------------------

def test_entropy_anneals_linearly_from_start_to_floor_and_then_holds():
    s, f, h = 0.02, 0.005, 50_000
    assert ent_coef_schedule(0, s, f, h) == pytest.approx(s)
    assert ent_coef_schedule(h // 2, s, f, h) == pytest.approx((s + f) / 2)
    assert ent_coef_schedule(h, s, f, h) == pytest.approx(f)
    assert ent_coef_schedule(10 * h, s, f, h) == pytest.approx(f)
    values = [ent_coef_schedule(g, s, f, h) for g in range(0, 2 * h, 997)]
    assert values == sorted(values, reverse=True)


def test_entropy_schedule_with_no_horizon_is_the_floor():
    assert ent_coef_schedule(0, horizon_games=0) == 0.005 == ent_coef_schedule(123, horizon_games=-5)


def test_batch_size_ramp_doubles_in_even_steps_to_the_cap_and_never_shrinks():
    start, cap, steps, h = 4, 16, 6, 60_000
    sizes = [batch_size_for_iteration(g, h, start, cap, steps) for g in range(0, 2 * h, 500)]
    assert sizes[0] == start and sizes[-1] == cap
    assert sizes == sorted(sizes) and set(sizes) <= {4, 8, 16}
    assert batch_size_for_iteration(h, h, start, cap, steps) == cap
    assert batch_size_for_iteration(0, 0, start, cap, steps) == cap, "no horizon: straight to the cap"


def test_a_resumed_session_continues_the_ramp_instead_of_restarting_it():
    """Both schedules read cumulative games/deck, so session 2 starting at 40,000 is where an uninterrupted run would be."""
    assert ent_coef_schedule(40_000) < ent_coef_schedule(10_000) < ent_coef_schedule(0)
    assert batch_size_for_iteration(40_000) > batch_size_for_iteration(0)


# ---- progress.json -----------------------------------------------------------------------------------------------------------

def test_checkpoint_progress_is_absolute_idempotent_and_leaves_the_batch_size_alone(tmp_path):
    d = str(tmp_path)
    R._save_progress(d, 7, 100)
    for _ in range(3):
        R.checkpoint_progress(d, 250)
    assert R._load_progress(d) == {"last_batch_size": 7, "cumulative_games_per_deck": 250}


def test_a_final_advance_after_mid_session_checkpoints_does_not_double_count(tmp_path):
    d = str(tmp_path)
    R._save_progress(d, 0, 1000)
    R.checkpoint_progress(d, 1060)  # a snapshot mid-session
    R.checkpoint_progress(d, 1120)
    out = R.advance_progress(d, n_iterations=10, games_per_iteration=15, auto_sizing=False, session_start_games=1000)
    assert out["cumulative_games_per_deck"] == 1150 == R._load_progress(d)["cumulative_games_per_deck"]


def test_a_forced_size_never_perturbs_the_auto_sizing_ladder(tmp_path):
    d = str(tmp_path)
    R._save_progress(d, 8, 100)
    out = R.advance_progress(d, n_iterations=3, games_per_iteration=5, auto_sizing=False)
    assert out == {"last_batch_size": 8, "cumulative_games_per_deck": 115}
    out = R.advance_progress(d, n_iterations=3, games_per_iteration=5, auto_sizing=True)
    assert out["last_batch_size"] == 15


def test_a_new_league_reads_as_nothing_run_yet(tmp_path):
    assert R._load_progress(str(tmp_path / "fresh")) == {"last_batch_size": 0, "cumulative_games_per_deck": 0}


def test_progress_file_is_plain_json_with_exactly_the_two_documented_keys(tmp_path):
    R._save_progress(str(tmp_path), 3, 9)
    assert json.loads((tmp_path / "progress.json").read_text()) == {"last_batch_size": 3, "cumulative_games_per_deck": 9}
