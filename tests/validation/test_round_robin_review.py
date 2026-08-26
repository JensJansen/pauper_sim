"""Self-check for validation.round_robin_review: writes straight into the
webapp submodule's logs/replays/ (nothing under checkpoints/, no
metrics.jsonl line), and is a clean no-op when the submodule isn't checked
out. Marked slow: real game engine + torch, same as round_robin_training's
own tests."""
import json
from pathlib import Path

import pytest

import webapp_mirror
from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, round_robin_review

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, decks):
    for name in decks:
        net = league_runner.build_deck_net(vocab.size, len(fixed_tables[name]), trunk_hidden=(24, 24))
        ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / name / "live.pt"), net)
        ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / name / "mulligan.pt"), MulliganNet(net.encoder))


def _make_ctx(tmp_path, monkeypatch, decks, cumulative_games=100):
    # round_robin_review reuses league_runner._run_eval as-is (no build_pool
    # import of its own, unlike round_robin_training) -- _run_eval calls
    # build_pool() via league_runner's own module namespace, so that's what
    # needs patching, same as test_league_runner.py's own _run_eval tests.
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(league_runner, "build_pool",
                        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")))
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    return _common.ValidationContext(
        primary_league_name="primary", train_decks=list(decks), decklists=decklists, vocab=vocab,
        deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=999,  # unused: GAMES_PER_PAIRING is fixed
        seed=1, cumulative_games=cumulative_games,
    )


def _webapp_dir(tmp_path, monkeypatch, *, ready):
    webapp_dir = tmp_path / "fake_webapp"
    if ready:
        (webapp_dir / "logs").mkdir(parents=True)
        (webapp_dir / ".git").write_text("gitdir: whatever")
    monkeypatch.setattr(webapp_mirror, "WEBAPP_DIR", webapp_dir)
    return webapp_dir


@pytest.mark.slow
def test_writes_full_round_robin_straight_into_webapp_replays(tmp_path, monkeypatch):
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"])
    webapp_dir = _webapp_dir(tmp_path, monkeypatch, ready=True)

    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], cumulative_games=4200)
    result = round_robin_review.run(ctx)

    assert result["pairings"] == 3  # rakdos x rakdos, rakdos x dmir, dmir x dmir (mirrors included)
    assert result["games"] == 3 * round_robin_review.GAMES_PER_PAIRING

    log_path = webapp_dir / "logs" / "replays" / "primary_round_robin_review_4200games.json"
    doc = json.loads(log_path.read_text())
    assert len(doc["games"]) == result["games"]
    assert doc["meta"]["games_per_pairing"] == round_robin_review.GAMES_PER_PAIRING
    assert {g["deck_a"] for g in doc["games"]} <= {"rakdos_madness", "dmir_terror"}

    # Not folded into mulligan_audit's sample -- see the module's own docstring.
    assert ctx.collected_game_logs == []

    # Nothing written to checkpoints/, no metrics.jsonl line.
    assert not (tmp_path / "primary" / "checks").exists()
    assert not (tmp_path / "primary" / "metrics.jsonl").exists()


def test_skips_cleanly_when_webapp_submodule_not_checked_out(tmp_path, monkeypatch):
    _webapp_dir(tmp_path, monkeypatch, ready=False)
    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness"])

    result = round_robin_review.run(ctx)

    assert result == {"skipped": "webapp submodule not checked out"}
