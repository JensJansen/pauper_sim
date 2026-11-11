"""Self-check for validation.round_robin_primary. Marked slow: real game
engine + torch, same as round_robin_training's own tests."""
import json
from pathlib import Path

import pytest

from evaluation import play
from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, round_robin_primary

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, decks):
    for name in decks:
        net = league_runner.build_deck_net(vocab.size, len(fixed_tables[name]), trunk_hidden=(24, 24))
        ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / name / "live.pt"), net)
        ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / name / "mulligan.pt"), MulliganNet(net.encoder))


def _make_ctx(tmp_path, monkeypatch, decks, games_per_check=2, seed=1, cumulative_games=100):
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(play, "build_pool", lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")))
    monkeypatch.setattr(play, "_pool_cache", None)
    monkeypatch.setattr(play, "_agent_cache", {})
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    return _common.ValidationContext(
        primary_league_name="primary", train_decks=list(decks), decklists=decklists, vocab=vocab,
        deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=games_per_check,
        seed=seed, cumulative_games=cumulative_games,
    )


@pytest.mark.slow
def test_embeds_one_games_entry_per_game_aligned_with_pairings(tmp_path, monkeypatch):
    """primary_vs_primary_round_robin's own output file embeds a "games" key
    (see the module's own docstring) so it's directly openable in the
    webapp's replay viewer -- assert that embed actually lines up game-index,
    deck_a/deck_b, and non-empty events with the pairings the check played,
    not just that a "games" key exists."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"])
    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], games_per_check=2)

    result = round_robin_primary.run(ctx)

    # rakdos x rakdos, rakdos x dmir, dmir x dmir (mirrors included), 2 games/pairing.
    assert result["pairings"] == 3
    assert result["games"] == 6

    payload_path = tmp_path / "primary" / "checks" / "primary_vs_primary_round_robin_100games.json"
    payload = json.loads(payload_path.read_text())
    games = payload["games"]

    assert len(games) == 6
    assert [g["game_index"] for g in games] == list(range(6))
    assert all(g["events"] for g in games), "every embedded game must carry a non-empty event log"
    assert {g["deck_a"] for g in games} <= {"rakdos_madness", "dmir_terror"}
    assert {g["deck_b"] for g in games} <= {"rakdos_madness", "dmir_terror"}

    # deck_a/deck_b on each embedded game must match the aggregate pairing totals' own deck names.
    embedded_pairs = {(g["deck_a"], g["deck_b"]) for g in games}
    aggregate_pairs = {(p["deck_a"], p["deck_b"]) for p in payload["pairings"]}
    assert embedded_pairs == aggregate_pairs


def test_deck_win_totals_leave_out_mirrors():
    """A mirror game always has a winner, so counting it for the deck handed every deck a guaranteed win in
    1/len(decks) of its games (pooled win rate pinned at 6/11 for 11 decks). Mirrors must not enter the totals."""
    by_pairing = {
        ("a", "a"): {"games": 10, "a_wins": 6, "b_wins": 4, "no_winner": 0},
        ("a", "b"): {"games": 10, "a_wins": 7, "b_wins": 3, "no_winner": 0},
        ("b", "b"): {"games": 10, "a_wins": 5, "b_wins": 5, "no_winner": 0},
    }
    totals = round_robin_primary.deck_win_totals(["a", "b"], by_pairing)
    assert totals == {"a": {"wins": 7, "games": 10}, "b": {"wins": 3, "games": 10}}
    # zero-sum without mirrors: the pooled win rate is exactly 50%
    assert sum(t["wins"] for t in totals.values()) * 2 == sum(t["games"] for t in totals.values())
