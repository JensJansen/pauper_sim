"""Self-check for run_training_pipeline.py's config resolution and cadence
arithmetic (_resolve_options) and its --fresh wipe (_wipe) -- the pieces
testable without a real training run (build_pool()/_run_session/torch).
"""
import run_training_pipeline as pipeline


def test_resolve_options_requires_league_name_and_total_games():
    try:
        pipeline._resolve_options({"total_games": 100})
        assert False, "missing league_name should raise"
    except AssertionError as e:
        assert "league_name" in str(e)

    try:
        pipeline._resolve_options({"league_name": "x"})
        assert False, "missing total_games should raise"
    except AssertionError as e:
        assert "total_games" in str(e)


def test_resolve_options_defaults():
    opts = pipeline._resolve_options({"league_name": "x", "total_games": 1000})
    assert opts["checks_cadence_pct"] == 5
    assert opts["checks_games"] == 50
    assert opts["n_workers"] == 6
    assert opts["games_per_iteration"] == 6  # max(1, n_workers) when not overridden
    assert opts["snapshot_every"] == 33  # max(1, 200 // 6)
    assert opts["checkpoint_rate"] == 0.0
    assert opts["pfsp"] is True
    assert opts["device"] == "cpu"
    assert opts["roster"] is None
    assert opts["training_league_name"] is None
    assert opts["seed"] is None


def test_resolve_options_reads_config_overrides():
    cfg = {"league_name": "x", "total_games": 1000, "n_workers": 4, "games_per_iteration": 10,
           "snapshot_every_games": 100, "checkpoint_opponent_rate": 0.2, "pfsp": False,
           "device": "cuda", "training_league_name": "y",
           "roster": ["a", "b", "c"], "checks_cadence_pct": 10, "checks_games": 25, "seed": 7}
    opts = pipeline._resolve_options(cfg)
    assert opts["n_workers"] == 4
    assert opts["games_per_iteration"] == 10
    assert opts["snapshot_every"] == 10  # max(1, 100 // 10)
    assert opts["checkpoint_rate"] == 0.2
    assert opts["pfsp"] is False
    assert opts["device"] == "cuda"
    assert opts["training_league_name"] == "y"
    assert opts["roster"] == ["a", "b", "c"]
    assert opts["checks_games"] == 25
    assert opts["seed"] == 7


def test_cadence_games_is_a_percent_of_total_games_rounded_and_at_least_one():
    assert pipeline._resolve_options({"league_name": "x", "total_games": 1000,
                                      "checks_cadence_pct": 5})["cadence_games"] == 50
    assert pipeline._resolve_options({"league_name": "x", "total_games": 1000,
                                      "checks_cadence_pct": 33})["cadence_games"] == 330
    # Small total_games + small cadence_pct must never round down to zero
    # (a zero-size chunk would spin the training loop forever).
    assert pipeline._resolve_options({"league_name": "x", "total_games": 10,
                                      "checks_cadence_pct": 1})["cadence_games"] == 1


def test_wipe_removes_the_league_dir_and_shared_vocab(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CHECKPOINTS_DIR", tmp_path)
    league_dir = tmp_path / "some_league"
    (league_dir / "elves").mkdir(parents=True)
    (league_dir / "elves" / "live.pt").write_text("x")
    (league_dir / "progress.json").write_text("{}")
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("{}")

    pipeline._wipe(league_dir)

    assert not league_dir.exists()
    assert not vocab_path.exists()


def test_wipe_is_a_noop_on_a_league_that_was_never_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CHECKPOINTS_DIR", tmp_path)
    never_trained = tmp_path / "brand_new_league"
    pipeline._wipe(never_trained)  # must not raise even though the dir never existed
    assert not never_trained.exists()
