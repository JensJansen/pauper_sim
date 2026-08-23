"""Self-check for validation/_common.py's path construction and JSON
writing -- the plumbing every check module shares. Doesn't touch torch/game
nets (load_deck_net/load_agent aren't exercised here; they're covered by a
real smoke run instead, same as the checks that call them)."""
import json

from validation import _common


def _ctx(**overrides):
    fields = dict(primary_league_name="my_league", train_decks=["elves", "dmir_terror"],
                 decklists={}, vocab=None, deck_ctxs={}, fixed_tables={}, games_per_check=50,
                 seed=None, cumulative_games=2400)
    fields.update(overrides)
    return _common.ValidationContext(**fields)


def test_primary_league_dir_and_training_league_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(training_league_name="the_twin")
    assert ctx.primary_league_dir == str(tmp_path / "my_league")
    assert ctx.training_league_dir == str(tmp_path / "the_twin")


def test_training_league_dir_is_none_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx()  # training_league_name defaults to None
    assert ctx.training_league_dir is None


def test_write_league_json_stamps_the_games_count_and_writes_under_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(cumulative_games=4800)
    path = _common.write_league_json(ctx, "some_check", {"a": 1})
    assert path == f"{ctx.primary_league_dir}/checks/some_check_4800games.json"
    assert json.loads(open(path).read()) == {"a": 1}


def test_write_deck_json_writes_under_the_decks_own_subfolder(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(cumulative_games=4800)
    path = _common.write_deck_json(ctx, "elves", "mulligan_audit", {"b": 2})
    assert path == f"{ctx.primary_league_dir}/elves/checks/mulligan_audit_4800games.json"
    assert json.loads(open(path).read()) == {"b": 2}


def test_append_metric_writes_one_jsonl_line_with_cumulative_games_tagged(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(cumulative_games=4800)
    _common.append_metric(ctx, kind="round_robin_primary", deck="elves", win_rate=0.6)

    lines = (tmp_path / "my_league" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"kind": "round_robin_primary", "deck": "elves", "win_rate": 0.6, "cumulative_games": 4800}


def test_collected_game_logs_and_deck_league_start_empty_and_are_independent_per_context():
    """Each ValidationContext gets its own fresh accumulator lists -- a
    dataclass field with a mutable default must use default_factory, not a
    shared list every instance would silently alias."""
    a, b = _ctx(), _ctx()
    a.collected_game_logs.append("game")
    assert b.collected_game_logs == []
