"""Self-check for run_league.py's progress/schedule plumbing.

RESTORED 2026-08-13. A file of this name existed and was deleted at some point;
only its stale .pyc survived in tests/__pycache__/. In its absence BUG 1 went
undetected across all 40,104 PPO iterations of a 60,001-games/deck run: both
2026-08-06 anti-plateau schedules (the minibatch ramp 32->2048 and the ent_coef
anneal 0.02->0.005) silently never executed, because the horizon they ramp
against never advanced.

The chain, verified on the real run's own metrics.jsonl (batch_size was 32 on
every single iteration; ent_coef stayed in [0.0191, 0.0200]):

    the /train skill always passes --n-iterations
      -> run_league.py leaves auto_sizing False
      -> _save_progress is never called
      -> progress.json is never created
      -> _load_progress returns cumulative_games_per_deck: 0
      -> both schedules restart at their origin EVERY session

Every hyperparameter conclusion drawn between 2026-08-06 and 2026-08-13 was
therefore measured against a configuration that was not the one the code and
README described. These tests pin the pieces of that chain that can be tested
without spawning a training subprocess.
"""
import json

import pytest

from rl.league_runner import _load_progress, _next_batch_games, _save_progress, advance_progress
from rl.train import batch_size_for_iteration, ent_coef_schedule


def test_absent_progress_json_reads_as_zero_not_as_an_error(tmp_path):
    """The silent part of BUG 1: a missing progress.json is indistinguishable
    from a genuinely fresh league, so nothing ever complained."""
    assert _load_progress(str(tmp_path)) == {"last_batch_size": 0, "cumulative_games_per_deck": 0}


def test_progress_round_trips(tmp_path):
    _save_progress(str(tmp_path), last_batch_size=64, cumulative_games_per_deck=12_800)
    assert _load_progress(str(tmp_path)) == {"last_batch_size": 64, "cumulative_games_per_deck": 12_800}
    assert json.loads((tmp_path / "progress.json").read_text())["cumulative_games_per_deck"] == 12_800


def test_both_schedules_are_flat_at_a_horizon_that_never_advances():
    """The CONSEQUENCE of BUG 1, stated as an invariant.

    With cumulative_games stuck near 0, both schedules return their origin
    value no matter how long the run continues -- which is exactly what the
    real run recorded for 40,104 iterations. A single session tops out around
    3,000 games/deck, i.e. 6% of the 50,000-game horizon, and
    int(0.06 * 6) == 0 doublings."""
    assert batch_size_for_iteration(0) == batch_size_for_iteration(3000) == 32
    frozen_start, frozen_end = ent_coef_schedule(0), ent_coef_schedule(3000)
    assert abs(frozen_end - frozen_start) < 0.001, (
        f"a session-local horizon leaves ent_coef effectively fixed ({frozen_start} -> {frozen_end})")

    # ...and both DO move once the horizon actually advances, so the schedules
    # themselves were never broken -- only the number fed to them.
    assert batch_size_for_iteration(50_000) > 32
    assert ent_coef_schedule(50_000) < frozen_start


def test_next_batch_games_doubles_and_never_overshoots(tmp_path):
    league = str(tmp_path)
    assert _next_batch_games(league, total_games=100) == 1  # fresh league
    _save_progress(league, last_batch_size=1, cumulative_games_per_deck=1)
    assert _next_batch_games(league, total_games=100) == 2
    _save_progress(league, last_batch_size=32, cumulative_games_per_deck=95)
    assert _next_batch_games(league, total_games=100) == 5  # clamped to what remains
    _save_progress(league, last_batch_size=5, cumulative_games_per_deck=100)
    assert _next_batch_games(league, total_games=100) is None  # target met


def test_forced_n_iterations_still_advances_the_cumulative_counter(tmp_path):
    """BUG 1's fix, and the single most important assertion in this file.

    A --n-iterations run (auto_sizing False -- what the /train skill and the
    webapp escalation loop ALWAYS produce) must still advance the league's
    cumulative game count, because that count is the horizon both PPO
    schedules ramp against. Gating it on auto-sizing is what silently disabled
    them for 40,104 iterations."""
    league = str(tmp_path)
    _save_progress(league, last_batch_size=64, cumulative_games_per_deck=10_000)
    after = advance_progress(league, n_iterations=100, games_per_iteration=6, auto_sizing=False)

    assert after["cumulative_games_per_deck"] == 10_600, "cumulative games must advance regardless of auto-sizing"
    assert after["last_batch_size"] == 64, (
        "a FORCED size must not perturb the doubling ladder -- that half of the old gate was correct")
    assert _load_progress(league) == after, "and it must be persisted, not just returned"


def test_auto_sized_batch_feeds_the_ladder_and_advances_the_counter(tmp_path):
    league = str(tmp_path)
    _save_progress(league, last_batch_size=64, cumulative_games_per_deck=10_000)
    after = advance_progress(league, n_iterations=20, games_per_iteration=6, auto_sizing=True)
    assert after == {"last_batch_size": 120, "cumulative_games_per_deck": 10_120}


def test_the_schedules_now_actually_move_across_sessions(tmp_path):
    """End-to-end on the invariant that matters: replay several --n-iterations
    sessions and confirm the ent_coef the trainer would use actually changes.
    Under BUG 1 every one of these returned the identical origin value."""
    league = str(tmp_path)
    seen = []
    for _session in range(12):
        p = advance_progress(league, n_iterations=500, games_per_iteration=6, auto_sizing=False)
        seen.append(ent_coef_schedule(p["cumulative_games_per_deck"]))
    assert len(set(seen)) > 1, f"ent_coef must move across sessions, got {set(seen)}"
    assert seen[-1] < seen[0], "and it must anneal downward, per its own schedule"


def test_the_minibatch_ramp_is_pinned_off_by_default():
    """Fixing the counter switches the ramp ON for the first time as a side
    effect, and that is a 24x cut in Adam steps (ppo.py slices
    range(0, total, batch_size), so batch_size >= buffer collapses the update
    to one full-batch minibatch). It is pinned at start == cap == 32 until it
    is run deliberately as its own experiment."""
    from rl.league_runner import PPO_DEFAULTS
    assert PPO_DEFAULTS["batch_size_start"] == PPO_DEFAULTS["batch_size_cap"] == 32
    pinned = PPO_DEFAULTS["batch_size_cap"]
    assert batch_size_for_iteration(0, start=32, cap=pinned) == 32
    assert batch_size_for_iteration(50_000, start=32, cap=pinned) == 32, "the ramp must not fire"
