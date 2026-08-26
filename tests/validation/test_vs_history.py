"""Self-check for validation.vs_history's parallel path (executor/n_workers
on ValidationContext). Marked slow: real game engine + torch, same as
rl.league.league_runner's own _run_eval_vs_history test this mirrors."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.league.league import LeaguePool
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool as _real_build_pool
from validation import _common, vs_history

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _save_fixed_checkpoints_and_snapshots(tmp_path, vocab, fixed_tables, decks):
    """Writes ONE fixed (not re-randomized per call -- see
    test_round_robin_training's own note on why) live.pt/mulligan.pt per
    deck, plus a couple of registered snapshots so _run_eval_vs_history has
    real milestones to compare against instead of returning []."""
    league_dir = str(tmp_path / "primary")
    for name in decks:
        net = league_runner.build_deck_net(vocab.size, len(fixed_tables[name]), trunk_hidden=(24, 24))
        mnet = MulliganNet(net.encoder)
        ckpt_io.save_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)
        ckpt_io.save_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        pool = LeaguePool(league_dir, [name], max_snapshots_per_deck=2)
        for _ in range(3):  # 3rd registration evicts snapshot_0 into archive/ (cap=2) -- both milestones exist
            pool.register_snapshot(name, net, mnet)


def _make_ctx(tmp_path, monkeypatch, decks, seed, executor=None, n_workers=1):
    """Builds a ValidationContext against whatever checkpoints/snapshots
    already sit on disk (see _save_fixed_checkpoints_and_snapshots) --
    never writes new ones itself, so two contexts built from this compare
    against identical fixed weights."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(vs_history, "build_pool",
                        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")))
    monkeypatch.setattr(vs_history, "_worker_pool_cache", None)
    monkeypatch.setattr(_common, "CHECKPOINTS_DIR", tmp_path)

    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    return _common.ValidationContext(
        primary_league_name="primary", train_decks=list(decks), decklists=decklists, vocab=vocab,
        deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=2,
        seed=seed, cumulative_games=100, executor=executor, n_workers=n_workers,
    )


@pytest.mark.slow
def test_sequential_and_parallel_agree_on_milestone_results(tmp_path, monkeypatch):
    """Same seed, same fixed weights/snapshots: the parallel path (a
    ThreadPoolExecutor stand-in for a real process pool, sharing this
    test's monkeypatched build_pool()) must produce the SAME per-deck
    milestone results as the sequential path -- each deck's work is fully
    independent (own net, own snapshots), so this is the simplest of the
    three checks to get right."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints_and_snapshots(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"])

    payload_path = tmp_path / "primary" / "checks" / "vs_history_100games.json"

    ctx_seq = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=5)
    seq_result = vs_history.run(ctx_seq)
    seq_payload = json.loads(payload_path.read_text())  # capture before the parallel run overwrites the same path

    ctx_par = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        ctx_par.executor, ctx_par.n_workers = executor, 2
        par_result = vs_history.run(ctx_par)
    par_payload = json.loads(payload_path.read_text())

    assert seq_result["decks"] == par_result["decks"] == 2
    assert seq_payload["decks"].keys() == par_payload["decks"].keys() == {"rakdos_madness", "dmir_terror"}
    for name in ("rakdos_madness", "dmir_terror"):
        seq_labels = {m["label"] for m in seq_payload["decks"][name]}
        assert seq_labels == {"archive_oldest", "active_oldest"}, f"{name}: expected both milestones"
        # Same seed, same weights -> same per-milestone outcome, not just the same shape.
        assert seq_payload["decks"][name] == par_payload["decks"][name]


@pytest.mark.slow
def test_parallel_path_is_deterministic_for_a_given_seed(tmp_path, monkeypatch):
    """Each deck's own random.Random(seed) inside _run_eval_vs_history is
    already independent of every other deck's -- see vs_history._deck_worker's
    own docstring -- so running every deck's identical `seed` in parallel
    must reproduce identical outcomes across two full runs."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints_and_snapshots(tmp_path, vocab, fixed_tables, ["rakdos_madness", "dmir_terror"])

    def _run():
        ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness", "dmir_terror"], seed=13)
        with ThreadPoolExecutor(max_workers=2) as executor:
            ctx.executor, ctx.n_workers = executor, 2
            return vs_history.run(ctx)

    payload_path = tmp_path / "primary" / "checks" / "vs_history_100games.json"
    _run()
    first = json.loads(payload_path.read_text())
    _run()
    second = json.loads(payload_path.read_text())
    assert first == second


@pytest.mark.slow
def test_executor_none_still_runs_every_deck_sequentially(tmp_path, monkeypatch):
    """The default (no executor passed) must keep working exactly as
    before -- every existing caller of this check never sets ctx.executor."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    _save_fixed_checkpoints_and_snapshots(tmp_path, vocab, fixed_tables, ["rakdos_madness"])

    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness"], seed=1)
    assert ctx.executor is None and ctx.n_workers == 1
    result = vs_history.run(ctx)
    assert result["decks"] == 1


@pytest.mark.slow
def test_a_deck_with_no_snapshots_yet_is_a_cheap_no_op_even_in_parallel(tmp_path, monkeypatch):
    """A deck early in training (no snapshots registered yet) must still
    resolve to an empty milestone list, not raise or hang, when dispatched
    through the parallel path."""
    monkeypatch.chdir(_SRC_DIR)
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    # live.pt/mulligan.pt exist, but no LeaguePool snapshots were ever registered.
    net = league_runner.build_deck_net(vocab.size, len(fixed_tables["rakdos_madness"]), trunk_hidden=(24, 24))
    ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / "rakdos_madness" / "live.pt"), net)
    ckpt_io.save_deck_checkpoint(str(tmp_path / "primary" / "rakdos_madness" / "mulligan.pt"), MulliganNet(net.encoder))

    ctx = _make_ctx(tmp_path, monkeypatch, ["rakdos_madness"], seed=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        ctx.executor, ctx.n_workers = executor, 2
        result = vs_history.run(ctx)
    assert result["decks"] == 1
    payload = json.loads((tmp_path / "primary" / "checks" / "vs_history_100games.json").read_text())
    assert payload["decks"]["rakdos_madness"] == []
