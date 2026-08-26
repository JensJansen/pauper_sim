"""Self-check for validation/_common.py's path construction and JSON
writing -- the plumbing every check module shares. Doesn't touch torch/game
nets (load_deck_net/load_agent aren't exercised here; they're covered by a
real smoke run instead, same as the checks that call them)."""
import json

import webapp_mirror as wm
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


def test_write_league_json_also_mirrors_into_the_webapp_submodule(tmp_path, monkeypatch, fake_webapp):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp
    ctx = _ctx(cumulative_games=4800)

    _common.write_league_json(ctx, "some_check", {"a": 1})

    mirrored = validation_dir / "my_league" / "checks" / "some_check.jsonl"
    assert json.loads(mirrored.read_text().strip()) == {"a": 1}


def test_write_league_json_accumulates_across_cadence_points_in_the_mirror(tmp_path, monkeypatch, fake_webapp):
    """Two cadence points for the same check mirror as two lines in one
    growing file, not two separate files -- see webapp_mirror.mirror_json."""
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp

    _common.write_league_json(_ctx(cumulative_games=4800), "some_check", {"cumulative_games": 4800})
    _common.write_league_json(_ctx(cumulative_games=9600), "some_check", {"cumulative_games": 9600})

    mirrored = validation_dir / "my_league" / "checks" / "some_check.jsonl"
    lines = [json.loads(l) for l in mirrored.read_text().splitlines()]
    assert lines == [{"cumulative_games": 4800}, {"cumulative_games": 9600}]


def test_write_deck_json_also_mirrors_into_the_webapp_submodule(tmp_path, monkeypatch, fake_webapp):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp
    ctx = _ctx(cumulative_games=4800)

    _common.write_deck_json(ctx, "elves", "mulligan_audit", {"b": 2})

    mirrored = validation_dir / "my_league" / "elves" / "checks" / "mulligan_audit.jsonl"
    assert json.loads(mirrored.read_text().strip()) == {"b": 2}


def test_append_metric_also_mirrors_into_the_webapp_submodule(tmp_path, monkeypatch, fake_webapp):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp
    ctx = _ctx(cumulative_games=4800)

    _common.append_metric(ctx, kind="round_robin_primary", deck="elves", win_rate=0.6)

    mirrored = validation_dir / "my_league" / "metrics.jsonl"
    record = json.loads(mirrored.read_text().splitlines()[0])
    assert record == {"kind": "round_robin_primary", "deck": "elves", "win_rate": 0.6, "cumulative_games": 4800}


def test_write_league_json_mirror_is_a_noop_when_webapp_not_checked_out(tmp_path, monkeypatch):
    """No fake src/webapp/ set up here -- the real webapp_mirror.WEBAPP_DIR
    stays whatever this machine's actual checkout is, so this only proves
    the mirror call doesn't crash the primary write when the mirror itself
    fails its own guard (or the real submodule happens to be uninitialized)."""
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    monkeypatch.setattr(wm, "CHECKPOINTS_DIR", tmp_path)  # real league_dir, but...
    monkeypatch.setattr(wm, "WEBAPP_DIR", tmp_path / "no_such_webapp_dir")  # ...never checked out
    ctx = _ctx(cumulative_games=4800)

    path = _common.write_league_json(ctx, "some_check", {"a": 1})

    assert json.loads(open(path).read()) == {"a": 1}  # primary write still succeeded


def test_write_league_json_embeds_games_in_the_local_file_untouched(tmp_path, monkeypatch):
    """A check (round_robin_primary) that embeds a "games" key in its own
    payload -- so the SAME checks/<check>_<N>games.json file is directly
    openable in the webapp's replay viewer, no separate file -- must get
    that key back in full when read from the real, local checkpoints/
    copy."""
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(cumulative_games=4800)
    games = [{"game_index": 0, "deck_a": "elves", "deck_b": "dmir_terror", "events": [{"kind": "game_over"}]}]

    path = _common.write_league_json(ctx, "primary_vs_primary_round_robin", {"a": 1, "games": games})

    doc = json.loads(open(path).read())
    assert doc == {"a": 1, "games": games}


def test_write_league_json_strips_games_from_the_mirrored_copy(tmp_path, monkeypatch, fake_webapp):
    """The webapp submodule's small, git-committed logs/validation/ copy
    must never receive the embedded "games" key -- a full round robin's
    games can be hundreds of MB, and /stats never reads that key anyway."""
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp
    ctx = _ctx(cumulative_games=4800)
    games = [{"game_index": 0, "deck_a": "elves", "deck_b": "dmir_terror", "events": [{"kind": "game_over"}]}]

    _common.write_league_json(ctx, "primary_vs_primary_round_robin", {"a": 1, "games": games})

    mirrored = validation_dir / "my_league" / "checks" / "primary_vs_primary_round_robin.jsonl"
    assert json.loads(mirrored.read_text().strip()) == {"a": 1}, "the mirrored copy must have \"games\" stripped"


def test_write_deck_json_also_strips_games_from_the_mirrored_copy(tmp_path, monkeypatch, fake_webapp):
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    validation_dir = fake_webapp
    ctx = _ctx(cumulative_games=4800)

    _common.write_deck_json(ctx, "elves", "some_check", {"b": 2, "games": [{"game_index": 0}]})

    local = json.loads(open(f"{ctx.primary_league_dir}/elves/checks/some_check_4800games.json").read())
    assert local == {"b": 2, "games": [{"game_index": 0}]}, "the local file must keep \"games\""
    mirrored = validation_dir / "my_league" / "elves" / "checks" / "some_check.jsonl"
    assert json.loads(mirrored.read_text().strip()) == {"b": 2}, "the mirrored copy must have \"games\" stripped"


def test_write_json_skips_indent_when_a_games_key_is_present(tmp_path, monkeypatch):
    """Pretty-printing a full embedded event log substantially bloats it and
    slows the write down for no benefit -- same reasoning
    rl.league.league_runner._write_event_log already applies. A payload
    with no "games" key still gets the readable indent=2 formatting."""
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    ctx = _ctx(cumulative_games=4800)

    with_games = _common.write_league_json(ctx, "check_a", {"games": [{"game_index": 0}]})
    without_games = _common.write_league_json(ctx, "check_b", {"a": 1})

    assert "\n" not in open(with_games).read(), "a payload with games must be written compact, no indent"
    assert "\n" in open(without_games).read(), "a payload without games keeps the readable indent=2 formatting"


def test_collected_game_logs_and_deck_league_start_empty_and_are_independent_per_context():
    """Each ValidationContext gets its own fresh accumulator lists -- a
    dataclass field with a mutable default must use default_factory, not a
    shared list every instance would silently alias."""
    a, b = _ctx(), _ctx()
    a.collected_game_logs.append("game")
    assert b.collected_game_logs == []
