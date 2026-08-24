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
    assert opts["stratify_0land_pct"] == 0.0
    assert opts["pfsp"] is True
    assert opts["device"] == "cpu"
    assert opts["roster"] is None
    assert opts["create_training_league"] is False
    assert opts["training_league_name"] is None
    assert opts["seed"] is None


def test_resolve_options_reads_config_overrides():
    cfg = {"league_name": "x", "total_games": 1000, "n_workers": 4, "games_per_iteration": 10,
           "snapshot_every_games": 100, "checkpoint_opponent_rate": 0.2, "stratify_0land_pct": 0.33, "pfsp": False,
           "device": "cuda", "create_training_league": True,
           "roster": ["a", "b", "c"], "checks_cadence_pct": 10, "checks_games": 25, "seed": 7}
    opts = pipeline._resolve_options(cfg)
    assert opts["n_workers"] == 4
    assert opts["games_per_iteration"] == 10
    assert opts["snapshot_every"] == 10  # max(1, 100 // 10)
    assert opts["checkpoint_rate"] == 0.2
    assert opts["stratify_0land_pct"] == 0.33
    assert opts["pfsp"] is False
    assert opts["device"] == "cuda"
    assert opts["create_training_league"] is True
    assert opts["training_league_name"] == "x-training"  # derived from league_name, not a free string
    assert opts["roster"] == ["a", "b", "c"]
    assert opts["checks_games"] == 25
    assert opts["seed"] == 7


def test_resolve_options_training_league_name_is_none_when_toggle_is_off():
    opts = pipeline._resolve_options({"league_name": "x", "total_games": 1000, "create_training_league": False})
    assert opts["training_league_name"] is None


def _fake_train_to_opts():
    return {"cadence_games": 40, "games_per_iteration": 10, "snapshot_every": 1, "n_workers": 1,
            "seed": None, "roster": None, "pfsp": True, "checkpoint_rate": 0.0, "ppo_hparams": None,
            "pfsp_power": 0.5, "trunk_hidden": (1,), "stratify_0land_pct": 0.0}


def test_train_to_chunks_until_target_and_invokes_on_chunk_per_chunk(monkeypatch):
    state = {"cumulative_games_per_deck": 0, "last_batch_size": 0}
    run_calls = []

    def fake_run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers, **kw):
        run_calls.append(kw["cumulative_games"])

    def fake_advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing, session_start_games=None):
        state["cumulative_games_per_deck"] = session_start_games + n_iterations * games_per_iteration

    monkeypatch.setattr(pipeline, "_run_session", fake_run_session)
    monkeypatch.setattr(pipeline, "advance_progress", fake_advance_progress)
    monkeypatch.setattr(pipeline, "_load_progress", lambda league_dir: dict(state))

    on_chunk_calls = []
    pipeline._train_to("some/dir", 100, _fake_train_to_opts(), executor=None, device="cpu",
                       on_chunk=on_chunk_calls.append)

    assert run_calls == [0, 40, 80]  # cadence_games=40 chunks, clipped on the last one (20 left)
    assert state["cumulative_games_per_deck"] == 100
    assert [c["cumulative_games_per_deck"] for c in on_chunk_calls] == [40, 80, 100]


def test_train_to_chunk_games_override_beats_opts_cadence_games(monkeypatch):
    # Pins the fix for the twin build silently collapsing into one giant
    # session: opts["cadence_games"] is sized off the PRIMARY league's own
    # total_games and can exceed the training league's much smaller fixed
    # cap (e.g. cadence_games=30000 vs TRAINING_LEAGUE_GAMES=10000) -- an
    # explicit chunk_games must win over that oversized default so the
    # twin build still chunks (and persists progress) even then.
    state = {"cumulative_games_per_deck": 0, "last_batch_size": 0}
    run_calls = []

    def fake_run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers, **kw):
        run_calls.append(kw["cumulative_games"])

    def fake_advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing, session_start_games=None):
        state["cumulative_games_per_deck"] = session_start_games + n_iterations * games_per_iteration

    monkeypatch.setattr(pipeline, "_run_session", fake_run_session)
    monkeypatch.setattr(pipeline, "advance_progress", fake_advance_progress)
    monkeypatch.setattr(pipeline, "_load_progress", lambda league_dir: dict(state))

    opts = _fake_train_to_opts()
    opts["cadence_games"] = 30000  # sized for a much larger primary league's total_games
    pipeline._train_to("some/dir", 100, opts, executor=None, device="cpu", chunk_games=40)

    assert run_calls == [0, 40, 80]  # chunk_games=40 wins, not opts["cadence_games"]=30000
    assert state["cumulative_games_per_deck"] == 100


def test_train_to_is_a_noop_once_target_already_reached(monkeypatch):
    monkeypatch.setattr(pipeline, "_run_session", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(pipeline, "advance_progress", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not advance")))
    monkeypatch.setattr(pipeline, "_load_progress", lambda league_dir: {"cumulative_games_per_deck": 100, "last_batch_size": 0})

    on_chunk_calls = []
    result = pipeline._train_to("some/dir", 100, _fake_train_to_opts(), executor=None, device="cpu",
                                on_chunk=on_chunk_calls.append)

    assert result["cumulative_games_per_deck"] == 100
    assert on_chunk_calls == []


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
