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

from rl.league_runner import (_load_progress, _next_batch_games, _save_progress, advance_progress,
                              checkpoint_progress, should_snapshot)
from rl.train import batch_size_for_iteration, ent_coef_schedule

# The training-mechanics keys that any config running the same league (main or
# its gauntlet twin) must reproduce exactly from run_default.json -- deck
# identity (roster, league_name, total_games, gauntlet) is deliberately
# exempt. Shared by test_main_league_mechanics_match_the_validated_config and
# test_the_gauntlet_twin_is_mechanically_identical_to_the_league_it_measures.
MECHANICS = ["snapshot_every_games", "n_workers", "games_per_iteration",
             "pfsp_power", "checkpoint_opponent_rate", "pfsp"]


def _assert_mechanics_match(cfg, cfg_name, default):
    for k in MECHANICS:
        assert k in cfg, f"{cfg_name} is missing {k}; it would silently fall back to a default"
        assert cfg[k] == default[k], (
            f"{cfg_name} {k}={cfg[k]!r} but the validated config uses {default[k]!r}")


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
    assert batch_size_for_iteration(0) == batch_size_for_iteration(3000) == 4
    frozen_start, frozen_end = ent_coef_schedule(0), ent_coef_schedule(3000)
    assert abs(frozen_end - frozen_start) < 0.001, (
        f"a session-local horizon leaves ent_coef effectively fixed ({frozen_start} -> {frozen_end})")

    # ...and both DO move once the horizon actually advances, so the schedules
    # themselves were never broken -- only the number fed to them.
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


def test_mid_session_progress_checkpoints_survive_a_crash_and_never_double_count(tmp_path):
    """BUG 3's fix. _save_live_checkpoints has always written the live nets at
    every snapshot point so a mid-session crash keeps the training it did;
    progress.json was written only after _run_session returned, so a crash left
    the counter BEHIND the weights and it had to be hand-corrected three times.

    Two properties, and the second is the one that makes the first safe:
      1. a crash after some snapshots leaves the counter at those snapshots;
      2. a session that completes lands on exactly start + played, no matter
         how many mid-session writes preceded it -- otherwise the fix trades a
         counter that lagged for one that runs ahead."""
    league = str(tmp_path)
    _save_progress(league, last_batch_size=64, cumulative_games_per_deck=10_000)

    # Session of 100 iterations x 6 games; snapshots land every 200 games.
    for games_so_far in (200, 400, 600):
        checkpoint_progress(league, 10_000 + games_so_far)
    crashed = _load_progress(league)
    assert crashed["cumulative_games_per_deck"] == 10_600, "a crash keeps the snapshots it wrote"
    assert crashed["last_batch_size"] == 64, "mid-session writes must not touch the doubling ladder"

    # Now let the same session finish. 100 * 6 = 600 games -> 10,600, NOT
    # 10,600 + 600: the three writes above are already part of that total.
    after = advance_progress(league, n_iterations=100, games_per_iteration=6, auto_sizing=False,
                             session_start_games=10_000)
    assert after["cumulative_games_per_deck"] == 10_600, "session end must be absolute, not additive"
    assert _load_progress(league) == after


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


def test_the_minibatch_ramp_is_measured_in_episodes_and_does_fire():
    """The ramp counts EPISODES since the recurrent policy landed (2026-08-17)
    -- ppo_update samples whole trajectories, so a transition-level minibatch
    has no meaning. It was pinned OFF at 32/32 transitions before that; it is
    now live at 4 -> 16 episodes.

    The renamed keys are the point: a config still carrying batch_size_start
    would be read as 32 EPISODES, which at ~25-50 episodes per buffer collapses
    every update to one full-batch minibatch. PPO_DEFAULTS' unknown-key assert
    catches that instead, which only works while the OLD names are absent."""
    from rl.league_runner import PPO_DEFAULTS
    assert "batch_size_start" not in PPO_DEFAULTS and "batch_size_cap" not in PPO_DEFAULTS, (
        "the old transition-level keys must stay gone so a stale config fails loudly"
    )
    assert PPO_DEFAULTS["seq_batch_size_start"] == 4 and PPO_DEFAULTS["seq_batch_size_cap"] == 16
    start, cap = PPO_DEFAULTS["seq_batch_size_start"], PPO_DEFAULTS["seq_batch_size_cap"]
    assert batch_size_for_iteration(0, start=start, cap=cap) == start
    assert batch_size_for_iteration(50_000, start=start, cap=cap) == cap, "the ramp must reach its cap"


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


def test_ent_coef_defaults_to_the_anneal():
    """Wave 2b's knob. PPO_DEFAULTS["ent_coef"]=None must reproduce the anneal
    (this is the baseline 20,016-game run's behavior and changing it silently
    would invalidate every comparison against that run); a float pins a
    constant and bypasses the schedule instead.

    Only this default value is pinned here -- the schedule actually moving
    across sessions is covered end-to-end by
    test_the_schedules_now_actually_move_across_sessions, and the resolution
    of hp["ent_coef"] itself lives in _run_session (rl.league_runner), not
    reproduced here."""
    from rl.league_runner import PPO_DEFAULTS
    assert PPO_DEFAULTS["ent_coef"] is None, "the anneal must stay the default"


def test_the_adopted_lr_matches_the_arm_that_justified_it():
    """lr 3e-4 -> 2e-4 (2026-08-15) is the first PPO_DEFAULTS value changed on a
    controlled experiment rather than a guess, so changing it back is a
    deliberate act with a visible test to update.

    Evidence, three arms at 10,000 games/deck scored against the identical
    20,016-game baseline: 2e-4 reached the plateau band on roughly half the
    baseline's budget, and was the only value at which no deck materially
    regressed. It buys convergence SPEED, not a better final policy -- the
    stronger claim was measured and withdrawn.

    The arm's config (run_lr2e4_ab.json) was deleted in the 2026-08-17 cleanup
    along with every other concluded A/B, so the value is pinned as a literal
    here instead of read back from it. The companion assertion that
    run_pretrain.py kept its own untested 3e-4 went with that script when
    per-deck encoders removed the pretrain phase entirely."""
    from rl.league_runner import PPO_DEFAULTS
    assert PPO_DEFAULTS["lr"] == 0.0002


# test_ab_config_differs_from_baseline_in_exactly_one_knob lived here. It pinned
# that each concluded A/B arm (run_entcoef_ab / run_lr_ab / run_trunk512_ab)
# moved exactly ONE knob off run_default.json, so an experiment could not be
# silently confounded by a well-meaning config edit. All three arms were null
# results and their configs were deleted in the 2026-08-17 cleanup, leaving the
# test with nothing to parametrize over. Reinstate it if a new A/B is started --
# the single-variable property is what made those runs interpretable.


def test_trunk_width_is_read_off_an_existing_checkpoint_not_the_config():
    """trunk_hidden became per-league configurable (2026-08-15) to test whether
    the plateau is a capacity ceiling. The hazard that creates: a config edit
    silently shape-mismatching a league already on disk, or -- worse -- a
    cross-league load (_run_eval_vs_gauntlet pulling ANOTHER population's
    live.pt) building the wrong shape and either erroring or, if the shapes
    happened to line up, loading garbage.

    The rule is that an existing checkpoint is the sole authority on its own
    width and the config applies only to a deck with no live.pt yet. This pins
    that the inference actually reads the tensors, using the real 20,016-game
    baseline as the fixture."""
    from rl.checkpoint import trunk_hidden_from_deck_checkpoint
    from rl.league_runner import TRUNK_HIDDEN
    from repo_paths import CHECKPOINTS_DIR

    baseline = CHECKPOINTS_DIR / "4_deck_subleague_test" / "elves" / "live.pt"
    if not baseline.exists():
        pytest.skip("baseline checkpoints not present")
    assert trunk_hidden_from_deck_checkpoint(str(baseline)) == (128, 128)
    assert TRUNK_HIDDEN == (128, 128), "the default must stay the width every existing league was trained at"

    # A path that does not exist reads as None so callers can fall back to the
    # configured width -- that is exactly how a FRESH deck picks up a new value.
    assert trunk_hidden_from_deck_checkpoint(str(CHECKPOINTS_DIR / "nope" / "live.pt")) is None


def test_main_league_mechanics_match_the_validated_config():
    """league_main.json had silently drifted: it predated all of Wave 2a and
    carried no games_per_iteration (so run_league would have derived
    max(1, n_workers)=6 instead of the validated 24), checkpoint_opponent_rate
    0.0 instead of 0.15, and no pfsp_power. Running the main league on it would
    have trained under mechanics nothing had ever validated, while looking like
    a normal config-driven run -- the stale-config trap the /train skill warns
    about, and the same class as BUG 1 and BUG 4: a setting that is wrong in a
    way nothing errors on.

    Run MECHANICS must match run_default.json. Deck identity (roster,
    league_name, total_games, gauntlet) is deliberately different and exempt."""
    from repo_paths import REPO_ROOT
    cfgs = REPO_ROOT / "training_configs"
    default = json.loads((cfgs / "run_default.json").read_text())
    main = json.loads((cfgs / "league_main.json").read_text())

    _assert_mechanics_match(main, "league_main.json", default)

    # The full manifest, not a subset -- this league's whole point is the 11-deck meta.
    manifest = json.loads((REPO_ROOT / "data" / "league_decks.json").read_text())
    assert set(main["roster"]) == set(manifest), "main league must carry the full manifest roster"


def test_the_gauntlet_twin_is_mechanically_identical_to_the_league_it_measures():
    """A twin population exists to differ from the main league in NOTHING but
    its nondeterministic training trajectory. Any mechanical difference makes
    the comparison measure the config instead of the training -- and it had
    already drifted exactly that way: run_gauntlet_twin.json carried no
    games_per_iteration and no pfsp_power, so it would have trained at 6
    games/iteration against the main league's 24 while looking like a valid
    twin (found 2026-08-17, before its first session).

    Same class of failure as league_main.json's own drift above, but worse in
    consequence: that one trains a league badly, this one silently invalidates
    every number the twin is built to produce.

    Roster must match too. Only league_name and gauntlet_league_name may
    differ -- and they must point AT each other, which is what makes each side
    report vs_gauntlet against the other."""
    from repo_paths import REPO_ROOT
    cfgs = REPO_ROOT / "training_configs"
    default = json.loads((cfgs / "run_default.json").read_text())
    twin = json.loads((cfgs / "run_gauntlet_twin.json").read_text())

    _assert_mechanics_match(twin, "run_gauntlet_twin.json", default)

    assert set(twin["roster"]) == set(default["roster"]), "a twin must play the same decks"
    assert twin["league_name"] != default["league_name"], "the twin must be a SEPARATE population"
    assert twin["gauntlet_league_name"] == default["league_name"]
    assert default["gauntlet_league_name"] == twin["league_name"]


# test_pretrain_cross_deck_pairing_is_opt_in_and_actually_varies lived here. It
# pinned run_pretrain.py's --cross-deck opponent sampling: the shared encoder
# had only ever seen MIRROR board states while ~10/11 of real league games are
# cross-deck, so its attention was asked at league time to encode card
# co-occurrences it never saw. Per-deck encoders (2026-08-17) deleted the whole
# pretrain phase -- every encoder now learns from exactly the league games its
# own deck plays, so the mismatch it addressed cannot arise.
