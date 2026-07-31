"""Self-check for the training-ops web UI's run management (webapp/runs.py).

Mostly pure argv-construction / log-parsing logic; Flask routes and real
run_league.py/run_pretrain.py invocations are exercised by hand (run the
app, click Start) rather than through pytest. RunManager.stop() gets a real
subprocess (a plain `sleep`, not a training script) since "did it actually
verify the kill" is exactly the kind of thing that's cheap to check for
real rather than trust a mock to get right.
"""
import subprocess
import sys

import pytest

from webapp import runs as runs_module
from webapp.runs import (
    LEAGUE_GLOBAL, LEAGUE_MODES, RunManager, _batch_healthy, argspec_from_parser, build_argv,
    is_auto_sizing_league_run, parse_last_batch_timing, _league_parser,
)


def _flag_value(argv, flag):
    """None if flag absent, True if it's a bare store_true flag, else its value."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
        return True
    return argv[i + 1]


def test_build_argv_pretrain_defaults_and_freeze():
    assert build_argv("pretrain", {}) == ["1", "2"]
    assert build_argv("pretrain", {"n_iterations": "5", "games_per_iteration": "3"}) == ["5", "3"]
    assert build_argv("pretrain", {"n_iterations": "5", "games_per_iteration": "3", "freeze": True}) == \
        ["5", "3", "--freeze"]
    assert build_argv("pretrain", {"freeze": False}) == ["1", "2"]


@pytest.mark.slow
def test_argspec_from_parser_excludes_config_paths_and_help():
    spec = argspec_from_parser(_league_parser())
    dests = {f["dest"] for f in spec}
    assert "run_config" not in dests
    assert "league_config" not in dests
    assert "help" not in dests
    assert "n_workers" in dests
    matchup = next(f for f in spec if f["dest"] == "matchup")
    assert matchup["nargs"] == 2
    eval_flag = next(f for f in spec if f["dest"] == "eval")
    assert eval_flag["type"] == "store_true"


@pytest.mark.slow
def test_build_argv_league_omits_blank_fields():
    argv = build_argv("league", {"n_workers": "6", "snapshot_every": "", "eval": False})
    assert _flag_value(argv, "--n-workers") == "6"
    assert "--snapshot-every" not in argv  # blank -> omitted, script's own default applies
    assert "--eval" not in argv  # falsy store_true -> omitted


@pytest.mark.slow
def test_build_argv_league_never_emits_removed_flags():
    # games_per_iteration / batch_size_start,cap,steps / max_batch_size are no
    # longer real CLI flags (removed 2026-07-31 -- always derived/hardcoded
    # instead) -- a stray key in the submitted values must be silently
    # ignored, never turned into a flag that would make argparse error out.
    argv = build_argv("league", {
        "n_workers": "6", "games_per_iteration": "12", "max_batch_size": "4000",
        "batch_size_start": "32", "batch_size_cap": "2048", "batch_size_steps": "6",
    })
    for dead_flag in ("--games-per-iteration", "--max-batch-size",
                      "--batch-size-start", "--batch-size-cap", "--batch-size-steps"):
        assert dead_flag not in argv


@pytest.mark.slow
def test_build_argv_league_matchup_and_flags():
    argv = build_argv("league", {"matchup": ["elves", "boggles"], "eval": True, "greedy": True, "games": "25"})
    assert _flag_value(argv, "--matchup") == "elves"
    idx = argv.index("--matchup")
    assert argv[idx + 1:idx + 3] == ["elves", "boggles"]
    assert "--eval" in argv
    assert "--greedy" in argv
    assert _flag_value(argv, "--games") == "25"


@pytest.mark.slow
def test_league_mode_groups_cover_every_real_flag_exactly_once_or_global():
    """LEAGUE_GLOBAL/LEAGUE_MODES is hand-authored domain knowledge (argparse
    can't tell us which flags matter for which mode) -- this is the guard
    against it silently drifting from the real parser: a future flag that's
    added to run_league.py but never added to a group would just vanish from
    the UI, and a typo'd dest here would reference a flag that doesn't exist."""
    real_dests = {f["dest"] for f in argspec_from_parser(_league_parser())}
    covered = set(LEAGUE_GLOBAL)
    for mode in LEAGUE_MODES:
        covered |= set(mode["dests"])
        covered |= set(mode.get("implies", {}))  # e.g. eval mode implies eval=True, never rendered as a field

    assert real_dests <= covered, f"flags missing from every group (invisible in the UI): {real_dests - covered}"
    assert covered <= real_dests, f"grouped dest doesn't exist on the real parser: {covered - real_dests}"
    for mode in LEAGUE_MODES:
        overlap = set(LEAGUE_GLOBAL) & set(mode["dests"])
        assert not overlap, f"mode {mode['key']!r} duplicates global field(s): {overlap}"


def test_is_auto_sizing_league_run():
    # the case that actually confused a real run: matchup/eval both stay off,
    # n_iterations blank, total_games given (max_batch_size removed 2026-07-31 --
    # no longer a field at all, see run_league.py's _next_batch_games)
    assert is_auto_sizing_league_run({"total_games": "50000"}) is True
    # an explicit --n-iterations is a forced one-off size -- never escalate it
    assert not is_auto_sizing_league_run(
        {"total_games": "50000", "n_iterations": "2"})
    # matchup/eval runs aren't auto-sized at all
    assert not is_auto_sizing_league_run(
        {"total_games": "50000", "matchup": ["elves", "boggles"]})
    assert not is_auto_sizing_league_run(
        {"total_games": "50000", "eval": True})
    # missing total_games -> can't auto-size
    assert not is_auto_sizing_league_run({})
    assert not is_auto_sizing_league_run({})


def test_batch_healthy():
    # the real case this exists for: a parallel (--n-workers > 1) run exits 1
    # on Windows from ProcessPoolExecutor teardown even on full success -- so
    # this checks log CONTENT, never an exit code.
    assert _batch_healthy("League session 3: ...\n  iter 0 ...\nsession 3 done in 12.0s ...\n")
    assert not _batch_healthy("League session 3: ...\n")  # no "session N done" -- never finished
    assert not _batch_healthy("Traceback (most recent call last):\n  File ...\nsession 3 done in 1.0s")


def test_parse_last_batch_timing_none_before_any_batch_finishes():
    assert parse_last_batch_timing("") is None
    assert parse_last_batch_timing("League session 0: n_iterations=1 ...\n  iter 0 [elves]: games=2 ...\n") is None


def test_parse_last_batch_timing_league_format():
    # real run_league.py line shape, incl. the trailing collect/update breakdown
    log = "session 4 done in 25.4s (0.53s/game across 48 games) -- collect=11.5s (45%), update=13.9s (55%)\n"
    assert parse_last_batch_timing(log) == {"total_seconds": 25.4, "avg_seconds_per_game": 0.53, "games": 48}


def test_parse_last_batch_timing_pretrain_format_has_no_games_count():
    # run_pretrain.py's summary omits "across N games" entirely
    log = "session 2 done in 12.0s (2.40s/game)\n"
    assert parse_last_batch_timing(log) == {"total_seconds": 12.0, "avg_seconds_per_game": 2.4, "games": None}


def test_parse_last_batch_timing_eval_format_computes_the_average_itself():
    # eval mode never prints a per-game rate -- only total games + total seconds
    log = "eval done: 4 games in 15.2s\n"
    assert parse_last_batch_timing(log) == {"total_seconds": 15.2, "avg_seconds_per_game": 15.2 / 4, "games": 4}


def test_parse_last_batch_timing_picks_the_latest_batch_in_an_escalating_log():
    log = (
        "=== batch 1 ===\nsession 0 done in 10.0s (5.00s/game across 2 games)\n"
        "=== batch 2 ===\nsession 1 done in 12.0s (3.00s/game across 4 games)\n"
    )
    assert parse_last_batch_timing(log) == {"total_seconds": 12.0, "avg_seconds_per_game": 3.0, "games": 4}


@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    """A RunManager pointed at a throwaway registry/log dir -- never touches
    the real logs/webapp_runs/ used by an actual running server."""
    monkeypatch.setattr(runs_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(runs_module, "REGISTRY_PATH", tmp_path / "registry.json")
    return RunManager()


def test_stop_verifies_the_process_actually_died(isolated_manager, tmp_path):
    # The bug this guards against: stop() used to fire taskkill and unconditionally
    # report success, even if the kill silently failed -- the UI would then claim
    # "stopped" while the process kept running. Uses a real subprocess so the
    # verification (proc.poll() after the kill) is checked for real, not mocked.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    run_id = "test-stop-real-process"
    isolated_manager._procs[run_id] = proc
    isolated_manager._registry[run_id] = {
        "id": run_id, "status": "running", "mode": "single", "log_path": str(tmp_path / "x.log"),
    }
    try:
        assert isolated_manager.stop(run_id) is True
        assert proc.poll() is not None, "process must actually be dead, not just assumed dead"
        assert isolated_manager._registry[run_id]["status"] == "stopped"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_stop_reports_failure_when_this_process_never_held_the_handle(isolated_manager):
    # Simulates the most common real-world case: the run was started by an
    # EARLIER server process that's since been restarted, so the current
    # process's self._procs has no Popen for it -- stop() must say so
    # (False), not silently claim success while the run keeps training.
    run_id = "orphaned-from-a-restart"
    isolated_manager._registry[run_id] = {"id": run_id, "status": "running", "mode": "single"}
    assert isolated_manager.stop(run_id) is False


def test_stop_on_an_already_finished_process_reports_true_without_mislabeling_it(isolated_manager, tmp_path):
    # If it finished on its own (a race between the click and the check), stop()
    # shouldn't force-label a natural finish as a manual "stopped" -- the real
    # exit-code-based status (here: "finished", exit 0) should win instead.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    run_id = "already-done"
    isolated_manager._procs[run_id] = proc
    isolated_manager._registry[run_id] = {
        "id": run_id, "status": "running", "mode": "single", "log_path": str(tmp_path / "x.log"),
    }
    assert isolated_manager.stop(run_id) is True
    assert isolated_manager._registry[run_id]["status"] == "finished"
    assert isolated_manager._registry[run_id]["exit_code"] == 0
