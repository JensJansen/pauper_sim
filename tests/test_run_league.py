"""Self-check for run_league.py's auto-sizing doubling ladder
(_next_batch_games), which lost its max_batch_size cap 2026-07-31 -- see its
own docstring for why. Marked slow: importing run_league pulls in torch/rl.*.
"""
import json
from pathlib import Path

import pytest

import run_league
from rl.pool import build_pool as _real_build_pool

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.mark.slow
def test_next_batch_games_fresh_league_starts_at_one(tmp_path):
    assert run_league._next_batch_games(str(tmp_path), total_games=100) == 1


@pytest.mark.slow
def test_next_batch_games_doubles_from_the_last_real_batch(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(
        {"last_batch_size": 8, "cumulative_games_per_deck": 15}))
    assert run_league._next_batch_games(str(tmp_path), total_games=1000) == 16


@pytest.mark.slow
def test_next_batch_games_never_overshoots_the_remaining_target():
    # no separate ceiling anymore (max_batch_size removed) -- the ONLY cap left
    # is "don't play more than what's left of total_games"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/progress.json", "w") as f:
            json.dump({"last_batch_size": 512, "cumulative_games_per_deck": 900}, f)
        # doubling would want 1024, but only 100 remain (total_games=1000)
        assert run_league._next_batch_games(d, total_games=1000) == 100


@pytest.mark.slow
def test_next_batch_games_returns_none_once_target_already_met(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(
        {"last_batch_size": 500, "cumulative_games_per_deck": 1000}))
    assert run_league._next_batch_games(str(tmp_path), total_games=1000) is None


@pytest.mark.slow
def test_run_eval_labels_each_game_with_its_real_pairing(tmp_path, monkeypatch):
    """_run_eval's whole point for the round-robin case (--eval, no --matchup)
    is that game N in a many-pairing log can be ANY pairing -- confirm it
    returns a (deck_a, deck_b) per game_logs entry, in round-robin order
    (combinations_with_replacement: AA, AB, BB for a 2-deck roster), and that
    _write_event_log round-trips it into the JSON as real deck_a/deck_b
    fields instead of a bare, unlabeled game_index. fresh_stack=True + a
    tmp_path league_dir sidesteps needing a real frozen_stack/live checkpoint
    (untrained random-init nets play real, complete games -- slow because of
    that, not because anything here is flaky); a private vocab_path keeps
    this test from writing to the repo's own real checkpoints/vocab.json."""
    monkeypatch.chdir(_SRC_DIR)  # league_decks.json/data/*.txt are loaded via "../data/..." (rl.pool's own convention)
    monkeypatch.setattr(
        run_league, "build_pool",
        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")),
    )

    game_logs = []
    eval_decks, game_pairings = run_league._run_eval(
        ["rakdos_madness", "dmir_terror"], games_per_pairing=2, greedy=False, seed=0,
        game_logs=game_logs, fresh_stack=True, league_dir=str(tmp_path / "league"),
    )
    assert eval_decks == ["rakdos_madness", "dmir_terror"]
    assert len(game_logs) == 6  # 3 pairings (AA, AB, BB) x 2 games
    assert game_pairings == (
        [("rakdos_madness", "rakdos_madness")] * 2
        + [("rakdos_madness", "dmir_terror")] * 2
        + [("dmir_terror", "dmir_terror")] * 2
    )

    log_path = str(tmp_path / "eval_log.json")
    run_league._write_event_log(log_path, game_logs, {"mode": "eval"}, game_pairings=game_pairings)
    with open(log_path) as f:
        doc = json.load(f)
    assert [(g["deck_a"], g["deck_b"]) for g in doc["games"]] == game_pairings
    assert [g["game_index"] for g in doc["games"]] == list(range(6))
