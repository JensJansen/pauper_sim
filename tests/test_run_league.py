"""Self-check for run_league.py's auto-sizing doubling ladder
(_next_batch_games), which lost its max_batch_size cap 2026-07-31 -- see its
own docstring for why. Marked slow: importing run_league pulls in torch/rl.*.
"""
import json

import pytest

import run_league


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
