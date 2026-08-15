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

from rl.league_runner import _load_progress, _next_batch_games, _save_progress, advance_progress, should_snapshot
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


def test_snapshot_cadence_survives_short_sessions():
    """BUG 1's sibling, found 2026-08-13 on the live restart: the snapshot gate
    read `(iteration + 1) % snapshot_every == 0` off the SESSION-LOCAL
    iteration index. The escalation ladder opens with sessions of 1, 2 and 4
    iterations against snapshot_every=8, so the counter reset three times and
    no session ever reached it -- the league passed 168 games/deck with an
    empty opponent pool, which also makes checkpoint_opponent_rate a silent
    no-op (nothing to sample). Replays that exact ladder."""
    gpi, snapshot_every = 24, 8  # run_default.json: 200 // 24 -> every 192 games/deck
    cumulative, snapshots = 0, 0
    for n_iterations in (1, 2, 4, 8, 16, 32):
        for iteration in range(n_iterations):
            if should_snapshot(cumulative + iteration * gpi, gpi, snapshot_every):
                snapshots += 1
        cumulative += n_iterations * gpi
    assert cumulative == 1512
    assert snapshots == cumulative // (snapshot_every * gpi) == 7, (
        f"expected one snapshot per 192 games/deck, got {snapshots}"
    )


def test_snapshot_fires_once_per_interval_never_twice():
    """The gate must be edge-triggered on the crossing, not `games % N == 0`:
    with a games_per_iteration that does not divide the interval, a modulo
    test silently skips intervals it steps over."""
    gpi, snapshot_every = 5, 4  # every 20 games, and 20 % 5 == 0 is NOT guaranteed in general
    fired = [g for g in range(0, 200, gpi) if should_snapshot(g, gpi, snapshot_every)]
    assert fired == [15, 35, 55, 75, 95, 115, 135, 155, 175, 195]
    # A gpi that strides OVER the boundary must still fire exactly once per interval.
    gpi = 7
    fired = [g for g in range(0, 210, gpi) if should_snapshot(g, gpi, 3)]  # every 21 games
    assert len(fired) == 210 // 21 == 10, fired


def test_ent_coef_override_is_wired_and_off_by_default():
    """Wave 2b's knob. PPO_DEFAULTS["ent_coef"]=None must reproduce the anneal
    exactly (this is the baseline 20,016-game run's behavior and changing it
    silently would invalidate every comparison against that run), while a float
    pins a constant and bypasses the schedule.

    Worth a test rather than trusting the plumbing: a config key that is
    accepted but never read would leave the A/B looking like it ran while
    actually re-running the baseline. The `unknown` assert in _run_session
    catches a MISSPELLED key; nothing else catches a correctly-spelled but
    unconsumed one."""
    from rl.league_runner import PPO_DEFAULTS
    from rl.train import ent_coef_schedule
    assert PPO_DEFAULTS["ent_coef"] is None, "the anneal must stay the default"

    # The exact expression _run_session evaluates, both branches.
    def resolve(hp, cumulative):
        return hp["ent_coef"] if hp["ent_coef"] is not None else ent_coef_schedule(cumulative)

    anneal = {**PPO_DEFAULTS}
    assert resolve(anneal, 0) == ent_coef_schedule(0) == 0.02
    assert resolve(anneal, 20_016) == ent_coef_schedule(20_016)
    assert resolve(anneal, 0) > resolve(anneal, 20_016), "default must still anneal DOWNWARD"

    pinned = {**PPO_DEFAULTS, "ent_coef": 0.05}
    assert resolve(pinned, 0) == 0.05
    assert resolve(pinned, 20_016) == 0.05, "a pinned constant must not decay"


@pytest.mark.parametrize("ab_config,expected_ppo", [
    ("run_entcoef_ab.json", {"ent_coef": 0.05}),
    ("run_lr_ab.json", {"lr": 0.00015}),
    ("run_lr2e4_ab.json", {"lr": 0.0002}),
])
def test_ab_config_differs_from_baseline_in_exactly_one_knob(ab_config, expected_ppo):
    """Each A/B is only interpretable if ONE variable moved. Pins that, so a
    later well-meaning edit to any of these configs is caught here instead of
    silently confounding the experiment.

    Parametrized rather than duplicated: every future single-variable A/B
    against run_default.json should be added to the list above, and gets the
    same guard for free."""
    from repo_paths import REPO_ROOT
    cfgs = REPO_ROOT / "training_configs"
    base = json.loads((cfgs / "run_default.json").read_text())
    ab = json.loads((cfgs / ab_config).read_text())

    shared = ["snapshot_every_games", "n_workers", "games_per_iteration",
              "pfsp_power", "checkpoint_opponent_rate", "pfsp", "roster", "heuristic_decks"]
    for k in shared:
        assert base[k] == ab[k], f"{k} must match the baseline; got {base[k]!r} vs {ab[k]!r}"

    assert ab["ppo"] == expected_ppo, "exactly one PPO knob may move"
    assert "ppo" not in base, "baseline must use PPO_DEFAULTS untouched"
    assert ab["league_name"] != base["league_name"], "must not train into the baseline's checkpoints"
    assert ab["gauntlet_league_name"] == base["league_name"], "gauntlet should point at the baseline"
    # Every A/B must stop at the same budget, or the runs are not comparable
    # to each other -- only to the baseline. 10,000 is the Wave 2b window.
    assert ab["total_games"] == 10_000, "all A/Bs share one budget so they compare to each other too"
