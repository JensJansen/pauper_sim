"""Self-check for run_league.py's progress/schedule plumbing: pins the chain
by which `--n-iterations` runs still advance progress.json's
cumulative_games_per_deck, the horizon both the PPO minibatch ramp and the
ent_coef anneal schedule ramp against. Covers the pieces of that chain that
can be tested without spawning a training subprocess.
"""
import json

import pytest

from rl.league.league_runner import (_load_progress, _next_batch_games, _save_progress, advance_progress,
                              checkpoint_progress, should_snapshot)
from rl.training.train import batch_size_for_iteration, ent_coef_schedule

# Training-mechanics keys a league config should inherit from baseline_settings.json
# via its own top-level "extends" rather than retype -- retyping them is
# exactly the duplicate-source-of-truth that silently drifted once already
# (see test_the_training_league_inherits_mechanics_and_points_back_at_the_main_league).
MECHANICS = ["snapshot_every_games", "n_workers", "games_per_iteration",
             "pfsp_power", "checkpoint_opponent_rate", "pfsp", "device"]


def _assert_extends_not_duplicates(raw_cfg, cfg_name, base_name):
    assert raw_cfg.get("extends") == base_name, f"{cfg_name} should extend {base_name}"
    dup = set(MECHANICS) & raw_cfg.keys()
    assert not dup, f"{cfg_name} duplicates {dup} instead of inheriting them via extends"


def test_absent_progress_json_reads_as_zero_not_as_an_error(tmp_path):
    """A missing progress.json reads the same as a genuinely fresh league."""
    assert _load_progress(str(tmp_path)) == {"last_batch_size": 0, "cumulative_games_per_deck": 0}


def test_progress_round_trips(tmp_path):
    _save_progress(str(tmp_path), last_batch_size=64, cumulative_games_per_deck=12_800)
    assert _load_progress(str(tmp_path)) == {"last_batch_size": 64, "cumulative_games_per_deck": 12_800}
    assert json.loads((tmp_path / "progress.json").read_text())["cumulative_games_per_deck"] == 12_800


def test_both_schedules_are_flat_at_a_horizon_that_never_advances():
    """With cumulative_games stuck near 0, both schedules return their
    origin value no matter how long the run continues."""
    assert batch_size_for_iteration(0) == batch_size_for_iteration(3000) == 4
    frozen_start, frozen_end = ent_coef_schedule(0), ent_coef_schedule(3000)
    assert abs(frozen_end - frozen_start) < 0.001, (
        f"a session-local horizon leaves ent_coef effectively fixed ({frozen_start} -> {frozen_end})")

    # Both move once the horizon actually advances.
    assert batch_size_for_iteration(50_000) > 4
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
    """A --n-iterations run (auto_sizing False -- a forced/debug batch size)
    must still advance the league's cumulative game count, since that count
    is the horizon both PPO schedules ramp against."""
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


def test_mid_session_progress_checkpoints_survive_a_crash_and_never_double_count(tmp_path):
    """Two properties, the second of which makes the first safe:
      1. a crash after some snapshots leaves the counter at those snapshots;
      2. a session that completes lands on exactly start + played, no matter
         how many mid-session writes preceded it."""
    league = str(tmp_path)
    _save_progress(league, last_batch_size=64, cumulative_games_per_deck=10_000)

    # Session of 100 iterations x 6 games; snapshots land every 200 games.
    for games_so_far in (200, 400, 600):
        checkpoint_progress(league, 10_000 + games_so_far)
    crashed = _load_progress(league)
    assert crashed["cumulative_games_per_deck"] == 10_600, "a crash keeps the snapshots it wrote"
    assert crashed["last_batch_size"] == 64, "mid-session writes must not touch the doubling ladder"

    # Session finishes: 100 * 6 = 600 games -> 10,600, not 10,600 + 600 (the
    # three writes above are already part of that total).
    after = advance_progress(league, n_iterations=100, games_per_iteration=6, auto_sizing=False,
                             session_start_games=10_000)
    assert after["cumulative_games_per_deck"] == 10_600, "session end must be absolute, not additive"
    assert _load_progress(league) == after


def test_the_schedules_now_actually_move_across_sessions(tmp_path):
    """Replays several --n-iterations sessions and confirms the ent_coef the
    trainer would use actually changes."""
    league = str(tmp_path)
    seen = []
    for _session in range(12):
        p = advance_progress(league, n_iterations=500, games_per_iteration=6, auto_sizing=False)
        seen.append(ent_coef_schedule(p["cumulative_games_per_deck"]))
    assert len(set(seen)) > 1, f"ent_coef must move across sessions, got {set(seen)}"
    assert seen[-1] < seen[0], "and it must anneal downward, per its own schedule"


def test_the_minibatch_ramp_is_measured_in_episodes_and_does_fire():
    """The ramp counts episodes, not transitions -- ppo_update samples whole
    trajectories, so a transition-level minibatch has no meaning. The old
    transition-level keys (batch_size_start/cap) must stay absent from
    PPO_DEFAULTS: PPO_DEFAULTS' unknown-key assert only catches a stale
    config carrying them while they're gone."""
    from rl.league.league_runner import PPO_DEFAULTS
    assert "batch_size_start" not in PPO_DEFAULTS and "batch_size_cap" not in PPO_DEFAULTS, (
        "the old transition-level keys must stay gone so a stale config fails loudly"
    )
    assert PPO_DEFAULTS["seq_batch_size_start"] == 4 and PPO_DEFAULTS["seq_batch_size_cap"] == 16
    start, cap = PPO_DEFAULTS["seq_batch_size_start"], PPO_DEFAULTS["seq_batch_size_cap"]
    assert batch_size_for_iteration(0, start=start, cap=cap) == start
    assert batch_size_for_iteration(50_000, start=start, cap=cap) == cap, "the ramp must reach its cap"


def test_snapshot_cadence_survives_short_sessions():
    """The snapshot gate must fire on cumulative games, not a session-local
    iteration index -- otherwise an escalation ladder of short sessions
    (1, 2, 4, ... iterations) never reaches snapshot_every within any one of
    them. Replays that ladder."""
    gpi, snapshot_every = 24, 8  # baseline_settings.json: 200 // 24 -> every 192 games/deck
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


def test_ent_coef_defaults_to_the_anneal():
    """PPO_DEFAULTS["ent_coef"]=None must reproduce the anneal; a float pins
    a constant and bypasses the schedule instead. The schedule actually
    moving across sessions is covered by
    test_the_schedules_now_actually_move_across_sessions; the resolution of
    hp["ent_coef"] itself lives in _run_session (rl.league.league_runner)."""
    from rl.league.league_runner import PPO_DEFAULTS
    assert PPO_DEFAULTS["ent_coef"] is None, "the anneal must stay the default"


def test_the_adopted_lr_matches_the_arm_that_justified_it():
    """PPO_DEFAULTS["lr"] is set from a controlled experiment, not a guess,
    so changing it back is a deliberate act with a visible test to update.
    The arm's config was deleted along with the other concluded A/Bs, so the
    value is pinned as a literal here instead of read back from it."""
    from rl.league.league_runner import PPO_DEFAULTS
    assert PPO_DEFAULTS["lr"] == 0.0002


# If a new A/B config is started, reinstate a test pinning that its arm moves
# exactly one knob off baseline_settings.json (no accidental confounds).


def test_trunk_width_is_read_off_an_existing_checkpoint_not_the_config():
    """trunk_hidden is per-league configurable, which creates a hazard: a
    config edit could silently shape-mismatch a league already on disk, or a
    cross-league load could build the wrong shape and load garbage. An
    existing checkpoint is the sole authority on its own width; the config
    applies only to a deck with no live.pt yet. Pins that the inference
    actually reads the tensors, using a real baseline checkpoint as fixture."""
    from rl.checkpoint import trunk_hidden_from_deck_checkpoint
    from rl.league.league_runner import TRUNK_HIDDEN
    from repo_paths import CHECKPOINTS_DIR

    baseline = CHECKPOINTS_DIR / "4_deck_subleague_test" / "elves" / "live.pt"
    if not baseline.exists():
        pytest.skip("baseline checkpoints not present")
    assert trunk_hidden_from_deck_checkpoint(str(baseline)) == (128, 128)
    assert TRUNK_HIDDEN == (128, 128), "the default must stay the width every existing league was trained at"

    # A path that does not exist reads as None, so a fresh deck falls back
    # to the configured width.
    assert trunk_hidden_from_deck_checkpoint(str(CHECKPOINTS_DIR / "nope" / "live.pt")) is None


def test_main_league_inherits_mechanics_instead_of_duplicating_them():
    """main_league.json used to retype every baseline_settings.json mechanics
    field (games_per_iteration, checkpoint_opponent_rate, pfsp_power, ...);
    it now extends the file instead, so the two structurally cannot drift
    apart the way a manual copy could."""
    from repo_paths import REPO_ROOT
    from config_loader import load_config
    cfgs = REPO_ROOT / "training_configs"
    raw_main = json.loads((cfgs / "main_league.json").read_text())
    _assert_extends_not_duplicates(raw_main, "main_league.json", "baseline_settings.json")

    resolved = load_config(str(cfgs / "main_league.json"))
    default = json.loads((cfgs / "baseline_settings.json").read_text())
    for k in MECHANICS:
        assert resolved[k] == default[k]

    # The full manifest, not a subset -- this league's whole point is the 11-deck meta.
    manifest = json.loads((REPO_ROOT / "data" / "league_decks.json").read_text())
    assert set(resolved["roster"]) == set(manifest), "main league must carry the full manifest roster"


def test_the_training_league_inherits_mechanics_and_points_back_at_the_main_league():
    """A training-league population exists to differ from the main (primary)
    league in nothing but its nondeterministic training trajectory.
    Extending small_league.json instead of retyping its fields
    (mechanics AND roster -- it plays the same decks) makes a mechanical
    difference structurally impossible rather than something a test has to
    keep re-catching. Only league_name and training_league_name are set
    locally, and must point at each other, which is what makes each side
    report validation.round_robin_training against the other."""
    from repo_paths import REPO_ROOT
    from config_loader import load_config
    cfgs = REPO_ROOT / "training_configs"
    raw_twin = json.loads((cfgs / "small_league_training_league.json").read_text())
    _assert_extends_not_duplicates(raw_twin, "small_league_training_league.json", "small_league.json")
    assert "roster" not in raw_twin, "roster should be inherited via extends, not retyped"

    default = load_config(str(cfgs / "small_league.json"))
    twin = load_config(str(cfgs / "small_league_training_league.json"))

    assert twin["roster"] == default["roster"], "a training league must play the same decks"
    assert twin["league_name"] != default["league_name"], "the training league must be a SEPARATE population"
    assert twin["training_league_name"] == default["league_name"]
    assert default["training_league_name"] == twin["league_name"]


def test_bench_config_nulls_out_inherited_training_league_fields():
    """benchmarking_league.json extends small_league.json, which sets training_league_name
    -- explicitly nulled out in benchmarking_league.json so a benchmark timing run
    doesn't inherit round_robin_training's wall-time, which it exists
    specifically to exclude. Pins that an extending file's explicit null
    actually overrides the base's value rather than the merge treating an
    absent-vs-null key the same way."""
    from repo_paths import REPO_ROOT
    from config_loader import load_config
    cfgs = REPO_ROOT / "training_configs"
    resolved = load_config(str(cfgs / "benchmarking_league.json"))
    assert resolved["training_league_name"] is None
