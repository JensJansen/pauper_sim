"""Rollback and crash-recovery of a league directory (plan P6).

Contract: `run_rollback` puts back exactly the snapshot it was asked for, touches nothing else, and is itself reversible; a
session killed at ANY checkpoint write leaves a directory the next session resumes from, with progress.json never ahead of
the games the weights have seen. (Round-trip exactness of the checkpoint format and a resumed session keeping its saved weights
are tests/unit/rl/test_checkpoint.py and tests/integration/rl_pipeline/test_run_session_modes.py; the owner decided on
2026-11-12 that restoring the saved state exactly is the contract, not bit-identical continuation.)"""
import hashlib
import json
import random
import shutil

import numpy as np
import pytest
import torch

import run_rollback
from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.roster import build_pool
from support.paths import SRC

ROSTER = ["elves", "boggles"]
TRUNK = (8, 8)


@pytest.fixture(autouse=True)
def short_games(monkeypatch):
    monkeypatch.setattr(league_runner, "HORIZON", 30)


def _session(league_dir, iters=2, gpi=2, seed=5, **kw):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # run_league.py's job, done here the same way: a session is told how many games/deck the league has already trained
    done = league_runner._load_progress(str(league_dir))["cumulative_games_per_deck"]
    league_runner._run_session(iters, gpi, 1, None, 1, league_dir=str(league_dir), seed=seed, roster=ROSTER,
                               trunk_hidden=TRUNK, ppo_hparams={"n_epochs": 1}, checkpoint_rate=0.3,
                               cumulative_games=done, **kw)
    league_runner.advance_progress(str(league_dir), iters, gpi, auto_sizing=False, session_start_games=done)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    league = tmp_path_factory.mktemp("trained")
    mp = pytest.MonkeyPatch()
    mp.setattr(league_runner, "HORIZON", 30)
    mp.chdir(SRC)
    try:
        _session(league)
        _session(league, seed=6)  # two sessions: live weights have moved well past snapshot 0
    finally:
        mp.undo()
    return league


@pytest.fixture()
def league(trained, tmp_path):
    dst = tmp_path / "league"
    shutil.copytree(trained, dst)
    return dst


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(league):
    return {str(p.relative_to(league)).replace("\\", "/"): _digest(p) for p in sorted(league.rglob("*")) if p.is_file()}


def _ctx(deck):
    _decklists, _vocab, deck_ctxs, _tables = build_pool()
    return deck_ctxs[deck]


def _state(path, key="net"):
    return torch.load(path, map_location="cpu", weights_only=True)[key]


def _equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_rollback_restores_exactly_the_named_snapshot_and_nothing_else(league):
    snap = torch.load(league / "elves" / "snapshot_0.pt", map_location="cpu", weights_only=True)
    assert not _equal(_state(league / "elves" / "live.pt"), snap["state_dict"]), "live must differ from snapshot 0 first"
    before = _tree(league)
    msg = run_rollback.rollback_deck(str(league), "elves", 0, _ctx("elves"))
    assert "restored snapshot_0" in msg
    assert _equal(_state(league / "elves" / "live.pt"), snap["state_dict"])
    assert _equal(_state(league / "elves" / "mulligan.pt"), snap["mulligan_state_dict"])
    after = _tree(league)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed == {"elves/live.pt", "elves/mulligan.pt", "elves/live.pt.before_rollback",
                       "elves/mulligan.pt.before_rollback"}, "nothing outside the rolled-back deck's live pair may change"


def test_the_backup_is_the_pre_rollback_live_pair_so_a_rollback_is_reversible(league):
    live_before, mull_before = _digest(league / "elves" / "live.pt"), _digest(league / "elves" / "mulligan.pt")
    run_rollback.rollback_deck(str(league), "elves", 0, _ctx("elves"))
    assert _digest(league / "elves" / "live.pt.before_rollback") == live_before
    assert _digest(league / "elves" / "mulligan.pt.before_rollback") == mull_before


def test_a_second_rollback_refuses_to_destroy_the_only_backup_unless_forced(league):
    run_rollback.rollback_deck(str(league), "elves", 0, _ctx("elves"))
    before = _tree(league)
    with pytest.raises(FileExistsError):
        run_rollback.rollback_deck(str(league), "elves", 1, _ctx("elves"))
    assert _tree(league) == before, "a refused rollback must change nothing"
    run_rollback.rollback_deck(str(league), "elves", 1, _ctx("elves"), force=True)
    snap1 = torch.load(league / "elves" / "snapshot_1.pt", map_location="cpu", weights_only=True)
    assert _equal(_state(league / "elves" / "live.pt"), snap1["state_dict"])


def test_dry_run_changes_nothing_and_says_what_it_would_do(league):
    before = _tree(league)
    msg = run_rollback.rollback_deck(str(league), "elves", 0, _ctx("elves"), dry_run=True)
    assert "would restore" in msg and _tree(league) == before


def test_an_unknown_snapshot_fails_before_anything_is_touched(league):
    before = _tree(league)
    with pytest.raises(FileNotFoundError):
        run_rollback.rollback_deck(str(league), "elves", 999, _ctx("elves"))
    assert _tree(league) == before


def test_a_rolled_back_league_resumes_training_from_the_restored_weights(league):
    run_rollback.rollback_deck(str(league), "elves", 0, _ctx("elves"))
    snap = torch.load(league / "elves" / "snapshot_0.pt", map_location="cpu", weights_only=True)["state_dict"]
    games_before = json.loads((league / "progress.json").read_text())["cumulative_games_per_deck"]
    _session(league, iters=1, seed=9)
    assert not _equal(_state(league / "elves" / "live.pt"), snap), "the resumed session must train from the restored net"
    assert json.loads((league / "progress.json").read_text())["cumulative_games_per_deck"] == games_before + 2


# ---- a crash at any checkpoint write ------------------------------------------------------------------------------------------

def _total_deck_checkpoint_writes(tmp_path, monkeypatch):
    calls = []
    real = ckpt_io.save_deck_checkpoint
    monkeypatch.setattr(league_runner.ckpt_io, "save_deck_checkpoint", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    _session(tmp_path / "count")
    monkeypatch.setattr(league_runner.ckpt_io, "save_deck_checkpoint", real)
    return len(calls)


def test_a_session_killed_at_any_checkpoint_write_leaves_a_directory_the_next_session_resumes_from(
        trained, tmp_path, monkeypatch):
    total = _total_deck_checkpoint_writes(tmp_path, monkeypatch)
    assert total >= 8
    real = ckpt_io.save_deck_checkpoint
    for kill_at in sorted({1, 2, 3, 4, 5, total // 2, total - 1, total}):
        league = tmp_path / f"crash{kill_at}"
        shutil.copytree(trained, league)
        start = json.loads((league / "progress.json").read_text())["cumulative_games_per_deck"]
        seen = []

        def flaky(*a, _seen=seen, _kill=kill_at, **k):
            _seen.append(1)
            if len(_seen) == _kill:
                raise RuntimeError("simulated kill")
            return real(*a, **k)

        monkeypatch.setattr(league_runner.ckpt_io, "save_deck_checkpoint", flaky)
        with pytest.raises(RuntimeError, match="simulated kill"):
            _session(league, seed=10 + kill_at)
        monkeypatch.setattr(league_runner.ckpt_io, "save_deck_checkpoint", real)

        progress = json.loads((league / "progress.json").read_text())["cumulative_games_per_deck"]
        assert start <= progress <= start + 4, f"kill_at={kill_at}: progress.json ran ahead of the games played"
        for deck in ROSTER:  # every checkpoint file is whole: the atomic write never leaves a torn file
            assert _state(league / deck / "live.pt") and _state(league / deck / "mulligan.pt")
        _session(league, iters=1, seed=50)  # the recovery session starts and finishes
        after = json.loads((league / "progress.json").read_text())["cumulative_games_per_deck"]
        assert after == progress + 2, f"kill_at={kill_at}: recovery resumed from {progress}, ended at {after}"
