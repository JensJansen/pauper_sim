"""Session-level tests for rl.league.league_runner._run_session: every play/train
mode, end to end over a throwaway 2-deck league on tiny nets.

These pin the invariants a mode must keep (which decks' weights move, which
files and metric records appear, that a seeded run reproduces), not exact
weights -- exact hashes would break on any torch upgrade. The same modes were
also verified bit-for-bit across the _run_session split with a seeded
before/after golden hash; this file is the permanent, version-robust form.

The "parallel collect path" tests use an executor that runs each submitted task
in this process, so they are deterministic; the one real ProcessPoolExecutor
test checks the spawn/pickling boundary, which no stand-in can.
"""
import json
import random
import shutil
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.league import league_runner

pytestmark = pytest.mark.slow

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
ROSTER = ["elves", "boggles"]
TRUNK = (8, 8)  # tiny on purpose: these tests need real, checkpoint-loadable shapes, not trained ones


class InProcessExecutor:
    """Runs submitted tasks immediately in this process, so the parallel
    collect path (collect_rollout_league_parallel) is exercised deterministically."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


@pytest.fixture(autouse=True)
def short_games(monkeypatch):
    """Untrained nets play long games; a 30-turn horizon keeps each session in seconds. The horizon
    reaches the worker processes as an argument, so this covers the real-pool test too."""
    monkeypatch.setattr(league_runner, "HORIZON", 30)


@pytest.fixture(scope="module")
def pretrained_league(tmp_path_factory):
    """A league that has already run one session (weights, optimizer state, snapshots on disk),
    built once; tests that need 'a trained league' start from a private copy of it."""
    league_dir = tmp_path_factory.mktemp("pretrained")
    mp = pytest.MonkeyPatch()
    mp.setattr(league_runner, "HORIZON", 30)
    mp.chdir(_SRC_DIR)  # the per-test chdir in tests/rl/conftest.py doesn't cover a module-scoped fixture
    try:
        _session(league_dir, checkpoint_rate=0.3)
    finally:
        mp.undo()
    return league_dir


def _copy(src, tmp_path):
    dst = tmp_path / "league"
    shutil.copytree(src, dst)
    return dst


def _seed(seed=5):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # ppo_update shuffles with the global numpy RNG


def _session(league_dir, iters=1, gpi=2, executor=None, n_workers=1, seed=5, snapshot_every=1, **kw):
    """Runs one seeded session; returns that session's metrics.jsonl records (from its session_start on)."""
    _seed(seed)
    kw.setdefault("roster", ROSTER)
    kw.setdefault("trunk_hidden", TRUNK)
    kw.setdefault("ppo_hparams", {"n_epochs": 1})  # these tests check which things move, not how well PPO fits
    league_runner._run_session(iters, gpi, snapshot_every, executor, n_workers, league_dir=str(league_dir), seed=seed, **kw)
    with open(league_dir / "metrics.jsonl") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    last_start = max(i for i, r in enumerate(records) if r["kind"] == "session_start")
    return records[last_start:]


def _weights(league_dir, deck, filename="live.pt"):
    """The saved model weights of one deck, as {param name: tensor}."""
    return torch.load(league_dir / deck / filename, map_location="cpu", weights_only=False)["net"]


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _ppo(records):
    return [r for r in records if r["kind"] == "ppo"]


def test_sequential_session_trains_every_deck_and_writes_its_artifacts(tmp_path):
    records = _session(tmp_path, iters=2, checkpoint_rate=0.5, stratify_0land_pct=0.5)
    assert [r["kind"] for r in records][0] == "session_start"
    ppo = _ppo(records)
    assert sorted((r["iteration"], r["deck"]) for r in ppo) == [(i, d) for i in range(2) for d in sorted(ROSTER)]
    assert all(r["games"] == 2 and r["epochs_run"] >= 1 for r in ppo)
    for deck in ROSTER:
        assert (tmp_path / deck / "live.pt").exists() and (tmp_path / deck / "mulligan.pt").exists()
        assert list((tmp_path / deck).glob("snapshot_*.pt")), "snapshot_every=1 must register a snapshot"
    assert (tmp_path / "session.txt").read_text() == "0"
    assert json.loads((tmp_path / "progress.json").read_text())["cumulative_games_per_deck"] == 4


def test_a_second_session_resumes_the_saved_weights_and_numbering(pretrained_league, tmp_path):
    league = _copy(pretrained_league, tmp_path)
    first = _weights(league, "elves")
    records = _session(league, seed=6)
    assert (league / "session.txt").read_text() == "1"
    assert records[0]["kind"] == "session_start" and records[0]["session"] == 1
    assert not _same(first, _weights(league, "elves")), "the resumed session must keep training the saved net"


def test_same_seed_reproduces_the_session_exactly(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _session(a, checkpoint_rate=0.5)
    _session(b, checkpoint_rate=0.5)
    for deck in ROSTER:
        assert _same(_weights(a, deck), _weights(b, deck)), f"{deck}: same seed, different weights"
        assert _same(_weights(a, deck, "mulligan.pt"), _weights(b, deck, "mulligan.pt"))
    def ppo_rows(league_dir):
        with open(league_dir / "metrics.jsonl") as f:
            return _ppo(json.loads(ln) for ln in f if ln.strip())

    assert ppo_rows(a) == ppo_rows(b)


def test_parallel_collect_path_matches_the_session_invariants(tmp_path):
    records = _session(tmp_path, executor=InProcessExecutor(), n_workers=2, gpi=2, checkpoint_rate=0.5)
    ppo = _ppo(records)
    assert sorted(r["deck"] for r in ppo) == sorted(ROSTER) and all(r["games"] == 2 for r in ppo)
    for deck in ROSTER:
        assert list((tmp_path / deck).glob("snapshot_*.pt"))


def test_matchup_trains_both_named_decks_and_never_snapshots(tmp_path):
    records = _session(tmp_path, matchup=("elves", "boggles"))
    assert sorted(r["deck"] for r in _ppo(records)) == sorted(ROSTER)
    for deck in ROSTER:
        assert not list((tmp_path / deck).glob("snapshot_*.pt")), "a fixed matchup has no historical snapshots"
    assert not (tmp_path / "elves" / "opponent_stats.json").exists()


def test_frozen_opponent_deck_is_played_but_never_updated(pretrained_league, tmp_path):
    league = _copy(pretrained_league, tmp_path)
    boggles_before, elves_before = _weights(league, "boggles"), _weights(league, "elves")
    snapshots_before = len(list((league / "boggles").glob("snapshot_*.pt")))
    records = _session(league, train_decks=["elves"], checkpoint_rate=0.3, seed=6)
    assert {r["deck"] for r in _ppo(records)} == {"elves"}
    assert _same(boggles_before, _weights(league, "boggles")), "a frozen deck's weights must not move"
    assert not _same(elves_before, _weights(league, "elves")), "the train deck must keep learning"
    assert len(list((league / "boggles").glob("snapshot_*.pt"))) == snapshots_before, "only train decks are snapshotted"


def test_mulligan_frozen_session_logs_no_mulligan_updates(tmp_path):
    records = _session(tmp_path, train_mulligan=False)
    assert not [r for r in records if r["kind"] == "mulligan"]
    assert _ppo(records)


def test_deck_frozen_session_trains_only_the_mulligan_model(pretrained_league, tmp_path):
    """--train-mulligan-only: the PPO stats are all zero but the mulligan model still learns."""
    league = _copy(pretrained_league, tmp_path)
    deck_before = _weights(league, "elves")
    mulligan_before = _weights(league, "elves", "mulligan.pt")
    records = _session(league, train_deck=False, seed=6)
    assert _same(deck_before, _weights(league, "elves")), "the frozen policy must not move"
    assert all(r["epochs_run"] == 0 and r["explained_variance"] == 0.0 for r in _ppo(records))
    assert [r for r in records if r["kind"] == "mulligan"], "the mulligan model must still update"
    assert not _same(mulligan_before, _weights(league, "elves", "mulligan.pt"))


def test_pfsp_off_and_game_logging_run_clean(tmp_path):
    logs = []
    records = _session(tmp_path, pfsp=False, game_logs=logs)
    assert _ppo(records)
    assert len(logs) == 2 * len(ROSTER), "game_logs must receive one event log per game played"


def test_cpu_mirrors_follow_the_gpu_weights():
    """On a CUDA league the play paths read CPU copies of the nets: they must start equal, share
    the deck net's encoder with the mulligan mirror, and be re-synced after training moves the GPU
    weights -- the one thing a CPU-only run cannot exercise."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from rl.model.mulligan import MulliganNet
    from rl.roster import build_pool

    _decklists, vocab, _ctxs, fixed_tables = build_pool()
    live = {"elves": league_runner.build_deck_net(vocab.size, len(fixed_tables["elves"]), TRUNK).to("cuda")}
    mulligan = {"elves": MulliganNet(live["elves"].encoder).to("cuda")}
    cpu_nets, cpu_mulligan = league_runner._make_cpu_mirrors(live, mulligan, "cuda")
    assert cpu_nets["elves"] is not live["elves"] and next(cpu_nets["elves"].parameters()).device.type == "cpu"
    assert cpu_mulligan["elves"].encoder is cpu_nets["elves"].encoder, "mirror must share the mirrored deck net's encoder"
    ctx = {"cpu_nets": cpu_nets, "live_nets": live, "cpu_mulligan_nets": cpu_mulligan,
           "mulligan_nets": mulligan, "deck_names": ["elves"]}
    with torch.no_grad():
        for p in live["elves"].parameters():
            p.add_(1.0)
        for p in mulligan["elves"].parameters():
            p.add_(1.0)
    stale = next(cpu_nets["elves"].parameters()).clone()
    league_runner._sync_cpu_mirrors(ctx)
    synced = next(cpu_nets["elves"].parameters())
    assert not torch.equal(stale, synced) and torch.equal(synced, next(live["elves"].parameters()).cpu())
    assert all(torch.equal(m, g.cpu()) for m, g in zip(cpu_mulligan["elves"].parameters(), mulligan["elves"].parameters()))


def test_cpu_league_mirrors_are_the_live_nets_and_sync_is_a_no_op():
    live, mulligan = {"a": object()}, {"a": object()}
    assert league_runner._make_cpu_mirrors(live, mulligan, "cpu") == (live, mulligan)
    ctx = {"cpu_nets": live, "live_nets": live}
    league_runner._sync_cpu_mirrors(ctx)  # must return before touching anything else


def test_a_real_process_pool_runs_a_session(tmp_path):
    """The spawn/pickling boundary itself: weights, mulligan state dicts and buffers cross real
    process boundaries here, which an in-process executor cannot catch."""
    with ProcessPoolExecutor(max_workers=2) as executor:
        records = _session(tmp_path, iters=1, gpi=2, executor=executor, n_workers=2, checkpoint_rate=0.3)
    ppo = _ppo(records)
    assert sorted(r["deck"] for r in ppo) == sorted(ROSTER) and all(r["games"] == 2 for r in ppo)
    assert all(r["buffer_size"] > 0 for r in ppo), "workers must return real transitions"


class RecordingExecutor(InProcessExecutor):
    """An in-process executor that logs each collection submit ("C") into a shared event list, with a
    fingerprint of the first deck's weights as shipped to the workers."""

    def __init__(self, events, fingerprints):
        self.events, self.fingerprints = events, fingerprints

    def submit(self, fn, *args, **kwargs):
        self.events.append("C")
        shipped = args[1][ROSTER[0]]  # all_state_dicts[deck] of the first train deck
        self.fingerprints.append(float(sum(v.double().sum() for v in shipped.values() if v.is_floating_point())))
        return super().submit(fn, *args, **kwargs)


def _schedule(tmp_path, monkeypatch, overlap_collect, iters, snapshot_every, n_workers=1):
    """Runs a session and returns (events, shipped-weight fingerprints): "C" per collection submit
    (one per worker per round), "U" per round's update, in the order they happened."""
    events, fingerprints = [], []
    real_update = league_runner._update_round

    def update(*a, **kw):
        events.append("U")
        return real_update(*a, **kw)

    monkeypatch.setattr(league_runner, "_update_round", update)
    # one worker => one "C" per round, so the event string reads as the round schedule
    _session(tmp_path, iters=iters, snapshot_every=snapshot_every, executor=RecordingExecutor(events, fingerprints),
             n_workers=n_workers, overlap_collect=overlap_collect, checkpoint_rate=0.3)
    return "".join(events), fingerprints


def test_overlap_collect_ships_the_weights_from_before_the_previous_update(tmp_path, monkeypatch):
    """The documented one-update lag, pinned: round 2 is submitted before round 1's update, so the
    workers get round 1's weights; the sequential schedule ships the updated ones."""
    seq_events, seq_fp = _schedule(tmp_path / "s", monkeypatch, False, iters=1, snapshot_every=100)
    ovl_events, ovl_fp = _schedule(tmp_path / "o", monkeypatch, True, iters=1, snapshot_every=100)
    assert seq_events == "CUCU" and ovl_events == "CCUU"
    assert seq_fp[1] != seq_fp[0], "sequential: round 2 collects from the weights round 1's update produced"
    assert ovl_fp[1] == ovl_fp[0], "overlap: round 2 collects from the weights round 1 collected with"


def test_overlap_collect_prefetches_across_an_iteration_boundary_only_when_nothing_is_due(tmp_path, monkeypatch):
    # no snapshot, and the mulligan flush is only due on the session's last iteration: the boundary prefetches
    events, _ = _schedule(tmp_path / "a", monkeypatch, True, iters=2, snapshot_every=100)
    assert events == "CCUCUCUU"
    # a snapshot at the boundary must land before anything is collected after it: no prefetch there
    events, _ = _schedule(tmp_path / "b", monkeypatch, True, iters=2, snapshot_every=1)
    assert events == "CCUUCCUU"


def test_overlap_collect_session_trains_like_a_normal_one(tmp_path):
    records = _session(tmp_path, iters=2, executor=InProcessExecutor(), n_workers=2, gpi=2, checkpoint_rate=0.5,
                       overlap_collect=True)
    assert records[0]["kind"] == "session_start" and records[0]["overlap_collect"] is True
    ppo = _ppo(records)
    assert sorted((r["iteration"], r["deck"]) for r in ppo) == [(i, d) for i in range(2) for d in sorted(ROSTER)]
    assert all(r["games"] == 2 and r["buffer_size"] > 0 for r in ppo)


def test_overlap_collect_needs_a_worker_pool_and_no_matchup(tmp_path):
    with pytest.raises(AssertionError, match="overlap_collect"):
        _session(tmp_path, overlap_collect=True)
    with pytest.raises(AssertionError, match="overlap_collect"):
        _session(tmp_path, executor=InProcessExecutor(), n_workers=2, matchup=("elves", "boggles"), overlap_collect=True)
