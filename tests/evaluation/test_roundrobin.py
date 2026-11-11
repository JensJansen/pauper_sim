"""evaluation.roundrobin.run_round_robin: the round robin run_league.py --eval and the validation checks share, built on
play_matchups (moved here from rl.league.league_runner._run_eval, whose tests these are)."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evaluation import play
from evaluation.roundrobin import run_round_robin
from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
DECKS = ["rakdos_madness", "dmir_terror"]


@pytest.fixture
def league(tmp_path, monkeypatch):
    """Fixed (saved) weights for two decks: a deterministic league, and a private vocab so the repo's real
    checkpoints/vocab.json is never written."""
    monkeypatch.chdir(_SRC_DIR)  # data paths are relative to src/
    monkeypatch.setattr(league_runner, "HORIZON", 30)
    vocab_path = str(tmp_path / "vocab.json")
    monkeypatch.setattr(play, "build_pool", lambda: _real_build_pool(vocab_path=vocab_path))
    monkeypatch.setattr(play, "_pool_cache", None)
    monkeypatch.setattr(play, "_agent_cache", {})
    _dl, vocab, _ctx, tables = _real_build_pool(vocab_path=vocab_path)
    league_dir = tmp_path / "league"
    for name in DECKS:
        net = league_runner.build_deck_net(vocab.size, len(tables[name]), trunk_hidden=(24, 24))
        ckpt_io.save_deck_checkpoint(str(league_dir / name / "live.pt"), net)
        ckpt_io.save_deck_checkpoint(str(league_dir / name / "mulligan.pt"), MulliganNet(net.encoder))
    return str(league_dir)


@pytest.mark.slow
def test_each_game_is_labelled_with_its_real_pairing(league, tmp_path):
    """The round robin is every pair with mirrors (AA, AB, BB for two decks); game N of a many-pairing log can be
    any pairing, so a (deck_a, deck_b) comes back per game_logs entry, in that order, and _write_event_log
    round-trips it into the JSON."""
    game_logs = []
    decks, game_pairings = run_round_robin(league, DECKS, 2, greedy=False, seed=0, game_logs=game_logs)
    assert decks == DECKS and len(game_logs) == 6
    assert game_pairings == ([("rakdos_madness", "rakdos_madness")] * 2 + [("rakdos_madness", "dmir_terror")] * 2
                             + [("dmir_terror", "dmir_terror")] * 2)
    log_path = str(tmp_path / "eval_log.json")
    league_runner._write_event_log(log_path, game_logs, {"mode": "eval"}, game_pairings=game_pairings)
    with open(log_path) as f:
        doc = json.load(f)
    assert [(g["deck_a"], g["deck_b"]) for g in doc["games"]] == game_pairings
    assert [g["game_index"] for g in doc["games"]] == list(range(6))


@pytest.mark.slow
def test_a_single_matchup_plays_just_that_pairing(league):
    game_logs = []
    decks, game_pairings = run_round_robin(league, None, 3, greedy=True, seed=1, game_logs=game_logs,
                                           matchup=("dmir_terror", "rakdos_madness"))
    assert decks == ["dmir_terror", "rakdos_madness"] and len(game_logs) == 3
    assert game_pairings == [("dmir_terror", "rakdos_madness")] * 3


@pytest.mark.slow
def test_parallel_covers_every_pairing_and_matches_sequential_for_a_seed(league):
    """An executor (a thread pool stands in for the process pool: it shares this test's monkeypatches) changes
    nothing about the results -- each pairing's games are seeded from (seed, deck names, chunk), not from scheduling."""
    def winners(executor=None):
        logs = []
        _d, pairings = run_round_robin(league, DECKS, 2, greedy=True, seed=7, game_logs=logs, executor=executor,
                                       n_workers=2)
        return pairings, [next(e for e in ev if e["kind"] == "game_over")["winner"] for ev in logs]

    sequential = winners()
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = winners(executor)
    assert Counter(parallel[0]) == Counter(sequential[0]) and len(parallel[1]) == 6
    assert parallel == sequential
    assert winners() == sequential  # reproducible run to run
