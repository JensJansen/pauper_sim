"""Self-check for validation.round_robin_training's parallel path (executor/
n_workers on ValidationContext). Marked slow: real game engine + torch, same
as rl.league.league_runner's own _run_eval tests this mirrors."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, round_robin_training

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, primary_decks, training_decks):
    """Writes ONE randomly-initialized (but then FIXED on disk) net per deck
    per league -- called exactly once per test, before any ctx is built or
    run. build_deck_net randomly initializes on every construction (torch's
    own global RNG, untouched by this repo's `seed` params); calling it
    again for a second run (e.g. to compare sequential vs parallel) would
    silently give the second run different weights and make outcomes
    incomparable regardless of seeding -- not a parallelism bug, a test-setup
    one, so this is deliberately separated from context construction."""
    for league, decks in (("primary", primary_decks), ("training", training_decks)):
        for name in decks:
            net = league_runner.build_deck_net(vocab.size, len(fixed_tables[name]), trunk_hidden=(24, 24))
            ckpt_io.save_deck_checkpoint(str(tmp_path / league / name / "live.pt"), net)
            ckpt_io.save_deck_checkpoint(str(tmp_path / league / name / "mulligan.pt"), MulliganNet(net.encoder))


def _make_ctx(tmp_path, monkeypatch, primary_decks, seed, executor=None, n_workers=1):
    """Builds a ValidationContext against whatever checkpoints already sit
    on disk under tmp_path (see _save_fixed_checkpoints) -- never writes new
    ones itself, so calling this more than once (to compare two runs)
    reuses the identical fixed weights both times. A fresh ValidationContext
    per call: collected_game_logs/collected_deck_league must not leak
    between the runs being compared."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(round_robin_training, "build_pool",
                        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")))
    monkeypatch.setattr(round_robin_training, "_worker_pool_cache", None)
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)

    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    return _common.ValidationContext(
        primary_league_name="primary", training_league_name="training",
        train_decks=list(primary_decks), decklists=decklists, vocab=vocab,
        deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=2,
        seed=seed, cumulative_games=100, executor=executor, n_workers=n_workers,
    )


@pytest.mark.slow
def test_sequential_and_parallel_agree_on_pairing_results(tmp_path, monkeypatch):
    """Same seed, same (fixed) weights: the parallel path (a ThreadPoolExecutor
    stand-in for a real process pool, sharing this test's monkeypatched
    build_pool()) must produce the SAME per-pairing win/burn accounting as
    the sequential path -- not just the same shape, the same numbers, since
    _pairing_worker is the identical function either way (see its own
    docstring)."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"], ["rakdos_madness"])

    payload_path = tmp_path / "primary" / "checks" / "primary_vs_training_round_robin_100games.json"

    ctx_seq = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=3)
    seq_result = round_robin_training.run(ctx_seq)
    seq_payload = json.loads(payload_path.read_text())  # capture before the parallel run overwrites the same path

    ctx_par = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=3)
    with ThreadPoolExecutor(max_workers=2) as executor:
        ctx_par.executor, ctx_par.n_workers = executor, 2
        par_result = round_robin_training.run(ctx_par)
    par_payload = json.loads(payload_path.read_text())

    assert seq_result["pairings"] == par_result["pairings"] == 2  # rakdos_madness x rakdos_madness, dmir_terror x rakdos_madness
    assert seq_result["games"] == par_result["games"]
    # Same seed, same weights -> same per-pairing win/burn accounting, not just the same shape.
    key = lambda p: (p["primary_deck"], p["training_deck"])
    assert sorted(seq_payload["pairings"], key=key) == sorted(par_payload["pairings"], key=key)
    assert seq_payload["primary_totals"] == par_payload["primary_totals"]
    assert seq_payload["mana_burnt_by_deck"] == par_payload["mana_burnt_by_deck"]


@pytest.mark.slow
def test_parallel_path_is_deterministic_for_a_given_seed(tmp_path, monkeypatch):
    """Seeds are pre-drawn sequentially (itertools.product order) before
    dispatch -- see round_robin_training._pairing_worker's own docstring --
    so a given ctx.seed must reproduce identical outcomes regardless of
    worker scheduling, given the SAME fixed weights both times."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"], ["rakdos_madness"])

    def _run():
        ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=11)
        with ThreadPoolExecutor(max_workers=2) as executor:
            ctx.executor, ctx.n_workers = executor, 2
            round_robin_training.run(ctx)
        return [(e["kind"], e.get("winner")) for ev in ctx.collected_game_logs for e in ev if e["kind"] == "game_over"]

    assert _run() == _run()


@pytest.mark.slow
def test_executor_none_still_runs_every_pairing_sequentially(tmp_path, monkeypatch):
    """The default (no executor passed) must keep working exactly as
    before -- every existing caller of this check never sets ctx.executor."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints(tmp_path, vocab, fixed_tables, ["rakdos_madness"], ["rakdos_madness"])

    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness"], seed=1)
    assert ctx.executor is None and ctx.n_workers == 1
    result = round_robin_training.run(ctx)
    assert result["pairings"] == 1
    assert result["games"] == 2
