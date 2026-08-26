"""Self-check for webapp_mirror.py's two guards (_real_league_name,
webapp_ready) and the three mirror_* writers -- covers both "does the write
land" and "does it silently skip" without ever touching the real
checkpoints/ or src/webapp/."""
import json

import webapp_mirror as wm


def _make_webapp(tmp_path, with_git=True, with_logs=True):
    """A fake src/webapp/ tree with just enough on disk to pass/fail
    webapp_ready()'s two checks independently."""
    webapp_dir = tmp_path / "webapp"
    webapp_dir.mkdir(parents=True)
    if with_git:
        (webapp_dir / ".git").write_text("gitdir: ../../.git/modules/src/webapp")
    if with_logs:
        (webapp_dir / "logs").mkdir()
    return webapp_dir


def test_real_league_name_accepts_a_direct_child_of_checkpoints_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(wm, "CHECKPOINTS_DIR", tmp_path)
    assert wm._real_league_name(str(tmp_path / "main-league")) == "main-league"


def test_real_league_name_rejects_a_nested_or_unrelated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(wm, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    (tmp_path / "checkpoints").mkdir()
    # nested two levels down (e.g. a deck subdir, never a league itself)
    assert wm._real_league_name(str(tmp_path / "checkpoints" / "main-league" / "elves")) is None
    # a benchmark harness's throwaway dir entirely outside CHECKPOINTS_DIR
    assert wm._real_league_name(str(tmp_path / "some_benchmark_scratch_dir")) is None


def test_webapp_ready_requires_both_git_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(wm, "WEBAPP_DIR", _make_webapp(tmp_path, with_git=True, with_logs=True))
    assert wm.webapp_ready() is True

    monkeypatch.setattr(wm, "WEBAPP_DIR", _make_webapp(tmp_path / "no_git", with_git=False, with_logs=True))
    assert wm.webapp_ready() is False

    monkeypatch.setattr(wm, "WEBAPP_DIR", _make_webapp(tmp_path / "no_logs", with_git=True, with_logs=False))
    assert wm.webapp_ready() is False


def _ready(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    webapp_dir = _make_webapp(tmp_path)
    monkeypatch.setattr(wm, "CHECKPOINTS_DIR", checkpoints_dir)
    monkeypatch.setattr(wm, "WEBAPP_DIR", webapp_dir)
    monkeypatch.setattr(wm, "WEBAPP_VALIDATION_DIR", webapp_dir / "logs" / "validation")
    return checkpoints_dir


def test_mirror_json_writes_under_the_mirrored_league_when_ready(tmp_path, monkeypatch):
    checkpoints_dir = _ready(tmp_path, monkeypatch)
    league_dir = str(checkpoints_dir / "main-league")
    wm.mirror_json(league_dir, "checks/mulligan_audit_100games.json", {"a": 1})

    dest = tmp_path / "webapp" / "logs" / "validation" / "main-league" / "checks" / "mulligan_audit_100games.json"
    assert json.loads(dest.read_text()) == {"a": 1}


def test_mirror_json_is_a_noop_when_webapp_not_checked_out(tmp_path, monkeypatch):
    checkpoints_dir = _ready(tmp_path, monkeypatch)
    monkeypatch.setattr(wm, "WEBAPP_DIR", tmp_path / "webapp_never_initialized")  # no .git, no logs/
    league_dir = str(checkpoints_dir / "main-league")

    wm.mirror_json(league_dir, "checks/x_100games.json", {"a": 1})

    assert not (tmp_path / "webapp" / "logs" / "validation").exists()


def test_mirror_json_is_a_noop_for_a_throwaway_league_dir(tmp_path, monkeypatch):
    _ready(tmp_path, monkeypatch)
    wm.mirror_json(str(tmp_path / "some_benchmark_scratch_dir"), "checks/x_100games.json", {"a": 1})

    assert not (tmp_path / "webapp" / "logs" / "validation").exists()


def test_mirror_metrics_line_appends_jsonl(tmp_path, monkeypatch):
    checkpoints_dir = _ready(tmp_path, monkeypatch)
    league_dir = str(checkpoints_dir / "main-league")

    wm.mirror_metrics_line(league_dir, {"kind": "ppo", "deck": "elves"})
    wm.mirror_metrics_line(league_dir, {"kind": "ppo", "deck": "boggles"})

    dest = tmp_path / "webapp" / "logs" / "validation" / "main-league" / "metrics.jsonl"
    lines = dest.read_text().splitlines()
    assert [json.loads(l)["deck"] for l in lines] == ["elves", "boggles"]


def test_mirror_progress_overwrites_not_appends(tmp_path, monkeypatch):
    checkpoints_dir = _ready(tmp_path, monkeypatch)
    league_dir = str(checkpoints_dir / "main-league")

    wm.mirror_progress(league_dir, {"cumulative_games_per_deck": 100})
    wm.mirror_progress(league_dir, {"cumulative_games_per_deck": 200})

    dest = tmp_path / "webapp" / "logs" / "validation" / "main-league" / "progress.json"
    assert json.loads(dest.read_text()) == {"cumulative_games_per_deck": 200}
