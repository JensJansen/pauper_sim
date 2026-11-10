"""webapp_sync: retention (plain filesystem) and the commit/push/bump flow, the
latter against REAL throwaway git repos (a bare remote, a parent repo with a
submodule) -- never the real src/webapp (conftest stubs sync_async suite-wide)."""
import json
import subprocess
import shutil
import threading
import time

import pytest
from pathlib import Path

from webapp_publish import sync as ws


# ---- retention ----

def _touch(path, text="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_prune_replays_keeps_the_newest_ten_by_game_count_per_league(tmp_path):
    for n in range(1, 13):                       # 1000..12000: sorting by name would put 9000 above 12000
        _touch(tmp_path / "main-league" / f"round_robin_review_{n * 1000}games.json")
    for n in (5000, 6000):
        _touch(tmp_path / "other-league" / f"round_robin_review_{n}games.json")
    _touch(tmp_path / "main-league" / "notes.json")          # not a generated review: never touched
    _touch(tmp_path / "cross_league_50k.json")                # hand-made top-level file: never touched

    removed = ws.prune_replays(tmp_path)

    left = sorted(int(p.name.split("_")[-1][:-10]) for p in (tmp_path / "main-league").glob("round_robin_*"))
    assert left == [n * 1000 for n in range(3, 13)]
    assert len(removed) == 2 and len(list((tmp_path / "other-league").iterdir())) == 2
    assert (tmp_path / "main-league" / "notes.json").exists() and (tmp_path / "cross_league_50k.json").exists()


def test_migrate_flat_replays_moves_legacy_names_into_league_folders(tmp_path):
    _touch(tmp_path / "main-league_round_robin_review_15780games.json", '{"a": 1}')
    _touch(tmp_path / "cross_league_50k.json")
    assert ws.migrate_flat_replays(tmp_path) == 1
    assert (tmp_path / "main-league" / "round_robin_review_15780games.json").read_text() == '{"a": 1}'
    assert (tmp_path / "cross_league_50k.json").exists() and ws.migrate_flat_replays(tmp_path) == 0


def _write_jsonl(path, ns, dup_last=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"cumulative_games": n, "v": "first"}) for n in ns]
    if dup_last:
        lines.append(json.dumps({"cumulative_games": ns[-1], "v": "rewrite"}))   # crash + resume replay
    path.write_text("\n".join(lines) + "\n")


def test_prune_checks_keeps_newest_ten_ticks_last_line_wins_and_skips_other_files(tmp_path):
    league = tmp_path / "main-league"
    _write_jsonl(league / "checks" / "strength_history.jsonl", list(range(1, 15)), dup_last=True)
    _write_jsonl(league / "elves" / "checks" / "mulligan_audit.jsonl", list(range(1, 4)))
    (league / "metrics.jsonl").write_text('{"kind": "ppo", "cumulative_games": 1}\n' * 50)

    assert ws.prune_checks(league) == 1                      # only the 14-tick file needed a rewrite

    rows = [json.loads(line) for line in (league / "checks" / "strength_history.jsonl").read_text().splitlines()]
    assert [r["cumulative_games"] for r in rows] == list(range(5, 15))
    assert rows[-1]["v"] == "rewrite"                         # the replayed point's last write survives
    assert len((league / "elves" / "checks" / "mulligan_audit.jsonl").read_text().splitlines()) == 3
    assert len((league / "metrics.jsonl").read_text().splitlines()) == 50   # metrics are never pruned


# ---- git flow, against real throwaway repos ----

def git(cwd, *args):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"git {args} failed: {r.stderr}"
    return r.stdout.strip()


@pytest.fixture
def repos(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t", "GIT_ALLOW_PROTOCOL": "file"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ws, "GIT_TIMEOUT_SECONDS", 60)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(remote), str(seed)], check=True, capture_output=True)
    git(seed, "checkout", "-q", "-b", "main")
    _touch(seed / "app.py", "viewer v1")
    _touch(seed / "logs" / "validation" / "README.md", "docs v1")
    _touch(seed / "logs" / "replays" / ".gitkeep", "")
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "seed")
    git(seed, "push", "-q", "origin", "main")

    parent = tmp_path / "parent"
    subprocess.run(["git", "init", "-q", "-b", "main", str(parent)], check=True)
    _touch(parent / "README.md", "parent")
    git(parent, "add", "-A")
    git(parent, "commit", "-q", "-m", "parent init")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(remote), "src/webapp")
    git(parent, "commit", "-q", "-m", "add submodule")
    webapp = parent / "src" / "webapp"
    git(webapp, "checkout", "-q", "main")
    return {"remote": remote, "parent": parent, "webapp": webapp, "tmp": tmp_path}


def _clone_remote(repos, name="check"):
    dest = repos["tmp"] / name
    subprocess.run(["git", "clone", "-q", str(repos["remote"]), str(dest)], check=True, capture_output=True)
    return dest


def _fill_data(webapp, ticks=12):
    for n in range(1, ticks + 1):
        _touch(webapp / "logs" / "replays" / "main-league" / f"round_robin_review_{n * 1000}games.json")
    _write_jsonl(webapp / "logs" / "validation" / "main-league" / "checks" / "strength_history.jsonl",
                 [n * 1000 for n in range(1, ticks + 1)])
    (webapp / "logs" / "validation" / "main-league" / "metrics.jsonl").write_text(
        '{"kind": "ppo", "cumulative_games": 1}\n' * 20)


def test_sync_commits_pushes_only_data_and_bumps_the_parent_pointer(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    _fill_data(webapp)
    (webapp / "app.py").write_text("viewer v2 -- half-finished edit")            # viewer code: must NOT be committed
    (webapp / "logs" / "validation" / "README.md").write_text("docs v2")         # docs: must NOT be committed
    parent_commits_before = git(parent, "rev-list", "--count", "HEAD")

    result = ws.sync("main-league through 12000 games/deck", webapp_dir=webapp, repo_root=parent)

    assert result["git"]["error"] is None and result["git"]["committed"] and result["git"]["pushed"]
    remote = _clone_remote(repos)
    assert len(list((remote / "logs" / "replays" / "main-league").glob("round_robin_review_*"))) == 10
    checks = (remote / "logs" / "validation" / "main-league" / "checks" / "strength_history.jsonl").read_text().splitlines()
    assert len(checks) == 10
    assert len((remote / "logs" / "validation" / "main-league" / "metrics.jsonl").read_text().splitlines()) == 20
    assert (remote / "app.py").read_text() == "viewer v1"                           # code edit not pushed
    assert (remote / "logs" / "validation" / "README.md").read_text() == "docs v1"
    assert (webapp / "app.py").read_text() == "viewer v2 -- half-finished edit"      # ...and still there locally
    assert git(webapp, "rev-parse", "HEAD") == git(remote, "rev-parse", "HEAD")      # local is at the pushed head
    # parent: one local pointer-bump commit, only touching the submodule, not pushed anywhere
    assert result["parent_bumped"] and git(parent, "rev-list", "--count", "HEAD") == str(int(parent_commits_before) + 1)
    assert git(parent, "show", "--name-only", "--format=", "HEAD") == "src/webapp"
    assert git(parent, "ls-tree", "HEAD", "src/webapp").split()[2] == git(webapp, "rev-parse", "HEAD")


def test_sync_with_nothing_new_makes_no_commits(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    _fill_data(webapp, ticks=3)
    ws.sync("first", webapp_dir=webapp, repo_root=parent)
    head, parent_head = git(webapp, "rev-parse", "HEAD"), git(parent, "rev-parse", "HEAD")
    result = ws.sync("second", webapp_dir=webapp, repo_root=parent)
    assert result["git"]["error"] is None and not result["git"]["committed"] and not result["parent_bumped"]
    assert git(webapp, "rev-parse", "HEAD") == head and git(parent, "rev-parse", "HEAD") == parent_head


def test_sync_checks_out_main_when_the_submodule_is_detached(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    git(webapp, "checkout", "-q", "--detach")
    _fill_data(webapp, ticks=2)
    result = ws.sync("detached", webapp_dir=webapp, repo_root=parent)
    assert result["git"]["error"] is None
    assert git(webapp, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_sync_rebases_onto_a_remote_that_moved_ahead(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    other = _clone_remote(repos, "other")
    _touch(other / "logs" / "replays" / "elsewhere.json", "{}")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "someone else pushed")
    git(other, "push", "-q", "origin", "main")

    _fill_data(webapp, ticks=2)
    result = ws.sync("behind", webapp_dir=webapp, repo_root=parent)

    assert result["git"]["error"] is None and result["git"]["pushed"]
    remote = _clone_remote(repos, "final")
    assert (remote / "logs" / "replays" / "elsewhere.json").exists()
    assert (remote / "logs" / "replays" / "main-league" / "round_robin_review_2000games.json").exists()
    assert git(webapp, "rev-parse", "HEAD") == git(remote, "rev-parse", "HEAD")


def test_sync_reports_a_rebase_conflict_without_forcing_or_leaving_a_mess(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    other = _clone_remote(repos, "other")
    _touch(other / "logs" / "validation" / "main-league" / "metrics.jsonl", "remote version\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "remote metrics")
    git(other, "push", "-q", "origin", "main")

    _touch(webapp / "logs" / "validation" / "main-league" / "metrics.jsonl", "local version\n")
    result = ws.sync("conflict", webapp_dir=webapp, repo_root=parent)

    assert result["git"]["error"] and "rebase" in result["git"]["error"] and not result["git"]["pushed"]
    assert not (webapp.parent.parent / ".git" / "modules" / "src" / "webapp" / "rebase-merge").exists()
    assert git(webapp, "status", "--porcelain") == ""                           # no conflict markers left behind
    assert not result["parent_bumped"]
    remote = _clone_remote(repos, "final")
    assert (remote / "logs" / "validation" / "main-league" / "metrics.jsonl").read_text() == "remote version\n"


def test_sync_never_raises_and_skips_when_the_submodule_isnt_checked_out(monkeypatch):
    monkeypatch.setattr(ws.webapp_mirror, "webapp_ready", lambda: False)
    assert ws.sync("x") is None
    monkeypatch.setattr(ws.webapp_mirror, "webapp_ready", lambda: True)
    monkeypatch.setattr(ws, "prune_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ws.sync("x") is None                                               # swallowed, logged


def test_sync_async_coalesces_overlapping_requests(monkeypatch):
    calls, gate = [], threading.Event()

    def slow_sync(label, **kw):
        calls.append(label)
        gate.wait(5)
    monkeypatch.setattr(ws, "sync", slow_sync)
    t = ws.sync_async("first")
    time.sleep(0.2)
    assert ws.sync_async("second") is None and ws.sync_async("third") is None   # coalesced into one pending run
    gate.set()
    t.join(10)
    assert calls == ["first", "third"]                                         # latest label wins; "second" dropped
    assert ws._running is False


# ---- per-game index sidecars (webapp/replay_index.py, loaded by path) ----

REAL_INDEXER = Path(__file__).resolve().parent.parent / "src" / "webapp" / "replay_index.py"


def _write_replay(path, n_games=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    games = [{"game_index": i, "deck_a": "elves", "deck_b": "boggles", "events": [{"kind": "turn_start"}] * 3}
             for i in range(n_games)]
    path.write_text(json.dumps({"meta": {}, "games": games}))


def test_prune_replays_deletes_a_replays_sidecar_with_it(tmp_path):
    for n in range(1, 13):
        _write_replay(tmp_path / "lg" / f"round_robin_review_{n * 1000}games.json")
        (tmp_path / "lg" / f"round_robin_review_{n * 1000}games.json.gameindex").write_text("{}")
    ws.prune_replays(tmp_path)
    left = sorted(p.name for p in (tmp_path / "lg").iterdir())
    assert len([n for n in left if n.endswith(".json")]) == 10 and len([n for n in left if n.endswith(".gameindex")]) == 10
    assert "round_robin_review_1000games.json.gameindex" not in left


def test_build_sidecars_skips_a_checkout_without_the_indexer(tmp_path):
    _write_replay(tmp_path / "logs" / "replays" / "lg" / "round_robin_review_1000games.json")
    assert ws.build_sidecars(tmp_path) == (0, 0)


def test_sync_commits_sidecars_with_their_replays_and_drops_pruned_ones(repos):
    webapp, parent = repos["webapp"], repos["parent"]
    shutil.copy(REAL_INDEXER, webapp / "replay_index.py")                    # untracked code: never committed
    for n in range(1, 13):
        _write_replay(webapp / "logs" / "replays" / "main-league" / f"round_robin_review_{n * 1000}games.json")

    result = ws.sync("sidecars", webapp_dir=webapp, repo_root=parent)

    assert result["git"]["error"] is None and result["sidecars"][0] == 10         # built for the 10 survivors only
    remote = _clone_remote(repos)
    names = sorted(p.name for p in (remote / "logs" / "replays" / "main-league").iterdir())
    assert len([n for n in names if n.endswith(".json")]) == 10
    assert len([n for n in names if n.endswith(".gameindex")]) == 10
    assert not (remote / "replay_index.py").exists()
    index = json.loads((remote / "logs" / "replays" / "main-league"
                        / "round_robin_review_12000games.json.gameindex").read_text())
    assert [g["game_index"] for g in index["games"]] == [0, 1]

    # a second tick with an unchanged replay set rebuilds nothing and makes no new commit
    again = ws.sync("sidecars again", webapp_dir=webapp, repo_root=parent)
    assert again["sidecars"] == (0, 0) and not again["git"]["committed"]
