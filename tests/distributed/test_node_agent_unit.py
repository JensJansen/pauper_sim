"""Fast, no-torch, no-network unit tests for node_agent.py's pure disk/file
helpers -- the loop's own orchestration is covered by
test_node_agent_loopback.py (real, slow); these pin the individual pieces
in isolation."""
import json
import os
import signal

import pytest

from distributed import node_agent


def test_stop_flag_starts_unrequested():
    flag = node_agent.StopFlag()
    assert flag.requested is False


def test_sigterm_handler_sets_the_flag_when_invoked():
    # Invokes the installed handler directly rather than os.kill(pid, SIGTERM)
    # -- on Windows (this dev machine), os.kill with SIGTERM terminates the
    # process via TerminateProcess instead of invoking a registered Python
    # handler, unlike the standard POSIX delivery on the Linux nodes this
    # code actually targets. This tests the handler's own logic, which is
    # platform-independent; real end-to-end signal delivery only needs
    # verifying once against a real Linux node (Phase 3).
    flag = node_agent.StopFlag()
    flag.install_sigterm_handler()
    handler = signal.getsignal(signal.SIGTERM)
    handler(signal.SIGTERM, None)
    assert flag.requested is True


def test_collect_deck_files_reads_known_files_only(tmp_path):
    deck_dir = tmp_path / "elves"
    deck_dir.mkdir()
    (deck_dir / "live.pt").write_bytes(b"live")
    (deck_dir / "snapshot_3.pt").write_bytes(b"snap")
    (deck_dir / "checks").mkdir()
    (deck_dir / "checks" / "x.json").write_text("{}")

    files = node_agent._collect_deck_files(str(tmp_path), "elves")
    assert set(files) == {"live.pt", "snapshot_3.pt"}


def test_collect_deck_files_empty_for_a_deck_with_no_directory_yet(tmp_path):
    assert node_agent._collect_deck_files(str(tmp_path), "elves") == {}


def test_read_new_metrics_lines_only_returns_lines_since_the_given_offset(tmp_path):
    league_dir = tmp_path
    metrics_path = league_dir / "metrics.jsonl"
    metrics_path.write_text('{"a": 1}\n')
    lines, offset1 = node_agent._read_new_metrics_lines(str(league_dir), 0)
    assert lines == ['{"a": 1}']

    with open(metrics_path, "a") as f:
        f.write('{"a": 2}\n')
    lines2, offset2 = node_agent._read_new_metrics_lines(str(league_dir), offset1)
    assert lines2 == ['{"a": 2}']
    assert offset2 > offset1


def test_read_new_metrics_lines_missing_file_returns_empty(tmp_path):
    lines, offset = node_agent._read_new_metrics_lines(str(tmp_path), 0)
    assert lines == [] and offset == 0


def test_write_roster_to_disk_excludes_named_decks(tmp_path):
    roster = {"elves": {"live.pt": b"elves-weights"}, "boggles": {"live.pt": b"boggles-weights"}}
    node_agent._write_roster_to_disk(roster, str(tmp_path), exclude={"elves"})
    assert not (tmp_path / "elves").exists()
    assert (tmp_path / "boggles" / "live.pt").read_bytes() == b"boggles-weights"


def test_write_roster_to_disk_writes_everything_when_exclude_is_empty(tmp_path):
    roster = {"elves": {"live.pt": b"elves-weights"}}
    node_agent._write_roster_to_disk(roster, str(tmp_path))
    assert (tmp_path / "elves" / "live.pt").read_bytes() == b"elves-weights"


def test_seed_local_progress_writes_the_hub_value_on_a_fresh_node(tmp_path):
    node_agent._seed_local_progress(str(tmp_path), cumulative_games=5000)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["cumulative_games_per_deck"] == 5000


def test_seed_local_progress_takes_the_hubs_value_even_when_local_is_ahead(tmp_path):
    # local ahead = a crashed, never-pushed call; bootstrap also rolls the weights back to the hub's
    from rl.league.league_runner import _save_progress
    _save_progress(str(tmp_path), last_batch_size=0, cumulative_games_per_deck=9000)
    node_agent._seed_local_progress(str(tmp_path), cumulative_games=100)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["cumulative_games_per_deck"] == 100


def test_node_id_is_created_once_and_reused(tmp_path):
    path = str(tmp_path / "sub" / "node_id")
    first = node_agent.load_or_create_node_id(path)
    assert first and node_agent.load_or_create_node_id(path) == first


def test_register_with_backoff_retries_only_409(monkeypatch):
    calls, sleeps = [], []

    def fake_register(hub_url, credential, node_id=None, **kw):
        calls.append(node_id)
        if len(calls) < 3:
            raise node_agent.node_sync.SyncError("409: full", status_code=409)
        return "boggles", "tok", 0, 0
    monkeypatch.setattr(node_agent.node_sync, "register", fake_register)
    out = node_agent.register_with_backoff("u", "c", "n1", node_agent.StopFlag(), sleep_fn=sleeps.append)
    assert out == ("boggles", "tok", 0, 0) and sleeps == [5, 10]

    def bad_cred(*a, **k):
        raise node_agent.node_sync.SyncError("401: no", status_code=401)
    monkeypatch.setattr(node_agent.node_sync, "register", bad_cred)
    with pytest.raises(node_agent.node_sync.SyncError):
        node_agent.register_with_backoff("u", "c", "n1", node_agent.StopFlag(), sleep_fn=sleeps.append)


# --- run(): hub-failure handling, with training and the hub both faked ---

push_kwargs = []  # kwargs of each push in the last _drive_run


session_iters = []  # n_iterations of each _run_session call in the last _drive_run


def _drive_run(monkeypatch, tmp_path, push, register, poll=lambda *a: "continue", max_calls=8,
               metrics_acked=0, total_games=10**9, start_games=0, **kw):
    """Runs node_agent.run with _run_session & friends stubbed out. Returns the
    list of (token, metrics_lines) each push attempt carried."""
    opts = {"device": "cpu", "n_workers": 1, "games_per_iteration": 1, "snapshot_every": 1, "seed": 0,
            "pfsp": True, "checkpoint_rate": 0.0, "ppo_hparams": {}, "pfsp_power": 0.5, "trunk_hidden": 8,
            "stratify_0land_pct": 0.0, "league_name": "t", "total_games": 10**9}
    league = tmp_path / "league"
    league.mkdir()
    (league / "metrics.jsonl").write_text('{"kind": "ppo"}\n')
    cred = tmp_path / "cred.json"
    cred.write_text(json.dumps({"registration_credential": "c"}))
    attempts = []
    push_kwargs.clear()
    monkeypatch.setattr(node_agent, "load_config", lambda p: {})
    monkeypatch.setattr(node_agent, "_resolve_options", lambda c: opts)
    monkeypatch.setattr(node_agent, "build_pool", lambda: (None, None, None, None))
    monkeypatch.setattr(node_agent, "_reported_version", lambda vocab: {})
    monkeypatch.setattr(node_agent, "_sync_vocab", lambda url, cred: None)
    # progress advances by n_iterations * games_per_iteration (1) per call, like advance_progress
    games = {"n": start_games}
    session_iters.clear()
    monkeypatch.setattr(node_agent, "_run_session", lambda n, *a, **k: session_iters.append(n))
    monkeypatch.setattr(node_agent, "advance_progress",
                        lambda d, n, gpi, **k: games.__setitem__("n", games["n"] + n * gpi))
    monkeypatch.setattr(node_agent, "_load_progress",
                        lambda d: {"cumulative_games_per_deck": games["n"], "last_batch_size": 1})
    opts["total_games"] = total_games
    monkeypatch.setattr(node_agent, "_collect_deck_files", lambda d, deck: {})
    monkeypatch.setattr(node_agent.node_sync, "register", lambda url, c, node_id=None, **kw: register() + (metrics_acked,))
    monkeypatch.setattr(node_agent.node_sync, "pull", lambda *a: {})
    monkeypatch.setattr(node_agent.node_sync, "poll", lambda *a: poll(*a))

    def fake_push(url, token, deck, files, lines, games, **kw):
        attempts.append((token, list(lines)))
        push_kwargs.append(kw)
        return push(len(attempts), token)
    monkeypatch.setattr(node_agent.node_sync, "push", fake_push)
    node_agent.run("cfg", "http://hub", str(cred), max_calls=max_calls, league_dir=str(league),
                   sleep_fn=lambda s: None, stop_flag=node_agent.StopFlag(), **kw)
    return attempts


def _conn_err():
    return node_agent.node_sync.SyncError("unreachable")


def _rejected():
    return node_agent.node_sync.SyncError("403: superseded", status_code=403)


def test_run_keeps_training_through_failed_syncs_then_waits_for_hub_at_the_cap(monkeypatch, tmp_path):
    polls = []

    def push(n, token):
        if n <= 5:
            raise _conn_err()
        return "continue"

    def poll(*a):
        polls.append(1)
        if len(polls) == 1:
            raise _conn_err()
        return "continue"
    attempts = _drive_run(monkeypatch, tmp_path, push, lambda: ("elves", "t1", 0), poll=poll, max_calls=6)
    assert len(attempts) == 6          # 5 failures + 1 success: no crash
    assert len(polls) == 2             # cap hit on the 5th failure -> waited (one poll failed, one succeeded)
    # the metrics line was re-sent every attempt (offset only advances on success)
    assert all(lines == ['{"kind": "ppo"}'] for _t, lines in attempts)


def test_run_reregisters_on_403_and_continues_with_the_new_token(monkeypatch, tmp_path):
    regs = iter([("elves", "t1", 0), ("elves", "t2", 0)])

    def push(n, token):
        if token == "t1":
            raise _rejected()
        return "continue"
    attempts = _drive_run(monkeypatch, tmp_path, push, lambda: next(regs), max_calls=3)
    assert [t for t, _ in attempts] == ["t1", "t2", "t2"]


def test_run_exits_when_reregistration_returns_a_different_deck(monkeypatch, tmp_path):
    regs = iter([("elves", "t1", 0), ("boggles", "t2", 0)])

    def push(n, token):
        raise _rejected()
    with pytest.raises(SystemExit):
        _drive_run(monkeypatch, tmp_path, push, lambda: next(regs))


def test_run_exits_on_a_second_403_with_no_successful_sync_between(monkeypatch, tmp_path):
    regs = iter([("elves", "t1", 0), ("elves", "t2", 0)])

    def push(n, token):
        raise _rejected()
    with pytest.raises(SystemExit):
        _drive_run(monkeypatch, tmp_path, push, lambda: next(regs))


# --- metrics line tracking ---

def test_read_new_metrics_lines_leaves_a_partial_trailing_line_for_later(tmp_path):
    (tmp_path / "metrics.jsonl").write_bytes(b'{"a": 1}\n{"b": 2}\n{"c"')
    lines, off = node_agent._read_new_metrics_lines(str(tmp_path), 0)
    assert lines == ['{"a": 1}', '{"b": 2}'] and off == len(b'{"a": 1}\n{"b": 2}\n')
    (tmp_path / "metrics.jsonl").write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": 3}\n')
    lines, _ = node_agent._read_new_metrics_lines(str(tmp_path), off)
    assert lines == ['{"c": 3}']


def test_resume_metrics_position(tmp_path):
    (tmp_path / "metrics.jsonl").write_bytes(b"a\nb\nc\n")
    assert node_agent._resume_metrics_position(str(tmp_path), 0) == (0, 0, False)
    assert node_agent._resume_metrics_position(str(tmp_path), 2) == (4, 2, False)
    # hub holds more lines than the local file has: start at the end and rebase
    assert node_agent._resume_metrics_position(str(tmp_path), 5) == (6, 3, True)
    assert node_agent._resume_metrics_position(str(tmp_path / "nope"), 0) == (0, 0, False)


def test_run_after_restart_resends_only_lines_the_hub_lacks(monkeypatch, tmp_path):
    # local file has 1 line and the hub already holds it -> first push carries none, starts at line 1
    attempts = _drive_run(monkeypatch, tmp_path, lambda n, t: "continue", lambda: ("elves", "t1", 0),
                          max_calls=1, metrics_acked=1)
    assert attempts[0][1] == [] and push_kwargs[0]["metrics_start_line"] == 1
    assert push_kwargs[0]["metrics_rebase"] is False


def test_run_against_a_fresh_hub_sends_the_whole_local_file_from_line_zero(monkeypatch, tmp_path):
    attempts = _drive_run(monkeypatch, tmp_path, lambda n, t: "continue", lambda: ("elves", "t1", 0),
                          max_calls=1, metrics_acked=0)
    assert attempts[0][1] == ['{"kind": "ppo"}'] and push_kwargs[0]["metrics_start_line"] == 0


# --- total_games: the node ends its own run ---

def test_node_exits_and_releases_at_total_games_with_a_clipped_final_call(monkeypatch, tmp_path):
    released = []
    monkeypatch.setattr(node_agent.node_sync, "release", lambda url, token, deck: released.append(deck))
    attempts = _drive_run(monkeypatch, tmp_path, lambda n, t: "continue", lambda: ("elves", "t1", 0),
                          max_calls=None, total_games=7, call_granularity=3)
    assert session_iters == [3, 3, 1]      # last call clipped to the 1 iteration left, not a full 3
    assert len(attempts) == 3 and released == ["elves"]


def test_node_never_exits_at_target_with_its_final_state_unpushed(monkeypatch, tmp_path):
    released = []
    monkeypatch.setattr(node_agent.node_sync, "release", lambda url, token, deck: released.append(deck))

    def push(n, token):
        if n == 3:                          # the push carrying the target-reaching state fails
            raise _conn_err()
        return "continue"
    attempts = _drive_run(monkeypatch, tmp_path, push, lambda: ("elves", "t1", 0),
                          max_calls=None, total_games=7, call_granularity=3)
    assert session_iters == [3, 3, 1]      # re-pushed without training further
    assert len(attempts) == 4 and released == ["elves"]


def test_node_already_past_target_does_not_train(monkeypatch, tmp_path):
    monkeypatch.setattr(node_agent.node_sync, "release", lambda *a: None)
    _drive_run(monkeypatch, tmp_path, lambda n, t: "continue", lambda: ("elves", "t1", 0),
               max_calls=None, total_games=5, start_games=9, call_granularity=3)
    assert session_iters == []
