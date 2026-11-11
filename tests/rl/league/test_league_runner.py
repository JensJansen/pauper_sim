"""Self-check for rl.league.league_runner's auto-sizing doubling ladder
(_next_batch_games) and other league_runner helpers. Marked slow: importing
rl.league.league_runner pulls in torch/rl.*."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl import checkpoint as ckpt_io
from rl.league import league_runner
from rl.roster import build_pool as _real_build_pool

_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


@pytest.mark.slow
def test_next_batch_games_fresh_league_starts_at_one(tmp_path):
    assert league_runner._next_batch_games(str(tmp_path), total_games=100) == 1


@pytest.mark.slow
def test_next_batch_games_doubles_from_the_last_real_batch(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(
        {"last_batch_size": 8, "cumulative_games_per_deck": 15}))
    assert league_runner._next_batch_games(str(tmp_path), total_games=1000) == 16


@pytest.mark.slow
def test_next_batch_games_never_overshoots_the_remaining_target():
    # The only cap left is "don't play more than what's left of total_games".
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/progress.json", "w") as f:
            json.dump({"last_batch_size": 512, "cumulative_games_per_deck": 900}, f)
        # doubling would want 1024, but only 100 remain (total_games=1000)
        assert league_runner._next_batch_games(d, total_games=1000) == 100


@pytest.mark.slow
def test_next_batch_games_returns_none_once_target_already_met(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps(
        {"last_batch_size": 500, "cumulative_games_per_deck": 1000}))
    assert league_runner._next_batch_games(str(tmp_path), total_games=1000) is None


@pytest.mark.slow
def test_a_checkpoint_carries_its_own_encoder(tmp_path):
    """Each checkpoint carries its own encoder, so a loaded net can only ever
    be paired with its own perception -- cross-population comparison is
    valid by construction, not by a guard.

    Pinned by round-tripping a net whose encoder has been deliberately moved
    away from its initialization -- a loader that rebuilt a fresh encoder
    instead of restoring the saved one would come back at the random init."""
    from rl.model.deck import DeckNetwork  # noqa: F401 -- built via build_deck_net below

    net = league_runner.build_deck_net(12, 4)
    with torch.no_grad():
        net.encoder.embedding.weight.fill_(0.5)  # unmistakably not a random init
    path = str(tmp_path / "live.pt")
    ckpt_io.save_deck_checkpoint(path, net)

    reloaded = league_runner.build_deck_net(12, 4)
    assert not torch.allclose(reloaded.encoder.embedding.weight, net.encoder.embedding.weight),         "a freshly built net must NOT already match -- otherwise this test proves nothing"
    ckpt_io.load_deck_checkpoint(path, reloaded)
    assert torch.allclose(reloaded.encoder.embedding.weight, net.encoder.embedding.weight),         "the encoder must be restored FROM the checkpoint, not rebuilt fresh alongside it"


@pytest.mark.slow
def test_run_session_skips_record_outcome_for_a_stratified_game(tmp_path, monkeypatch):
    """_run_session must never call pool.record_outcome for a game whose
    training seat had its opening hand forced (stratify_0land_pct) -- see
    collect_rollout_league's own docstring / rl.league.league.LeaguePool
    .record_outcome's docstring for why: it would permanently bias PFSP's
    difficulty estimate for whichever opponent happened to be sampled,
    unrelated to real skill.

    Single-deck roster with a snapshot of itself pre-registered on disk (so
    LeaguePool.__init__'s directory scan finds it) and checkpoint_rate=1.0
    makes every game this session a frozen-snapshot-opponent game -- the
    only case with exactly one recorded seat, the one case stratify can fire
    in at all (see collect_rollout's own "len(recorded_seats) == 1" gate).
    stratify_0land_pct=1.0 then makes every one of those actually
    stratified. So record_outcome must be called ZERO times."""
    from rl.league.league import LeaguePool

    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool()
    trunk_hidden = (8, 8)  # small on purpose -- this test only needs a real, checkpoint-loadable shape, not a trained one
    seed_net = league_runner.build_deck_net(vocab.size, len(fixed_tables["elves"]), trunk_hidden)
    seed_pool = LeaguePool(str(tmp_path), ["elves"])
    seed_pool.register_snapshot("elves", seed_net)  # gives checkpoint_rate=1.0 a real snapshot to sample

    calls = []
    monkeypatch.setattr(LeaguePool, "record_outcome", lambda self, *a, **k: calls.append((a, k)))

    league_runner._run_session(1, 6, 1, None, 1, league_dir=str(tmp_path), seed=0, roster=["elves"],
                                checkpoint_rate=1.0, stratify_0land_pct=1.0, trunk_hidden=trunk_hidden)
    assert not calls, ("every game this session is a stratified frozen-snapshot game -- "
                        f"record_outcome must never fire, but was called with {calls}")


@pytest.mark.slow
def test_run_session_with_frozen_deck_logs_zeroed_ppo_stats(tmp_path):
    """--train-mulligan-only (train_deck=False): ppo_update never runs, so each
    round's metric must report 0.0 for every PPO stat -- including
    explained_variance/adv_std, which used to be assigned only inside the
    training branch and raised an UnboundLocalError on the first round."""
    import json

    league_runner._run_session(1, 2, 99, None, 1, league_dir=str(tmp_path), seed=0, roster=["elves"],
                               train_deck=False, trunk_hidden=(8, 8))
    with open(tmp_path / "metrics.jsonl") as f:
        ppo = [r for r in (json.loads(ln) for ln in f if ln.strip()) if r["kind"] == "ppo"]
    assert ppo, "the round must still log a ppo record"
    for r in ppo:
        assert r["epochs_run"] == 0 and r["explained_variance"] == 0.0 and r["adv_std"] == 0.0
        assert r["policy_loss"] == 0.0 and r["value_loss"] == 0.0


@pytest.mark.slow
def test_unknown_ppo_hyperparameter_is_a_hard_error():
    """A typo'd hyperparameter that silently does nothing must fail loudly,
    not run un-executed for thousands of iterations."""
    with pytest.raises(AssertionError, match="unknown ppo hyperparameter"):
        league_runner._run_session(1, 1, 1, None, 1, league_dir=str(tempfile.mkdtemp()),
                                    roster=["elves"], ppo_hparams={"learning_rate": 1e-4})


@pytest.mark.slow
def test_rollback_restores_a_snapshot_and_backs_up_the_replaced_live(tmp_path):
    """Rolling back must itself be reversible: the live pair it replaces is
    copied aside, and a SECOND rollback refuses to clobber that backup (which
    would destroy the only copy of the pre-rollback state -- the exact mistake
    the backup exists to prevent)."""
    import torch
    import run_rollback
    from rl.model.mulligan import MulliganNet
    from rl import checkpoint as ckpt_io

    deck_ctx = (SimpleNamespace(size=12), list(range(4)))  # rollback only reads vocab.size + len(fixed_table)
    deck_dir = tmp_path / "elves"
    (deck_dir / "archive").mkdir(parents=True)

    old = league_runner.build_deck_net(12, 4)
    ckpt_io.save_snapshot(str(deck_dir / "archive" / "snapshot_58.pt"), old, MulliganNet(old.encoder))
    current = league_runner.build_deck_net(12, 4)
    with torch.no_grad():
        current.critic_head.weight.add_(5.0)  # make "now" clearly different from the snapshot
    ckpt_io.save_deck_checkpoint(str(deck_dir / "live.pt"), current)
    ckpt_io.save_deck_checkpoint(str(deck_dir / "mulligan.pt"), MulliganNet(current.encoder))

    run_rollback.rollback_deck(str(tmp_path), "elves", 58, deck_ctx)

    restored = league_runner.build_deck_net(12, 4)
    ckpt_io.load_deck_checkpoint(str(deck_dir / "live.pt"), restored)
    assert torch.equal(restored.critic_head.weight, old.critic_head.weight), "live.pt must be the snapshot"

    backed_up = league_runner.build_deck_net(12, 4)
    ckpt_io.load_deck_checkpoint(str(deck_dir / ("live.pt" + run_rollback.BACKUP_SUFFIX)), backed_up)
    assert torch.equal(backed_up.critic_head.weight, current.critic_head.weight), \
        "the replaced live.pt must be recoverable"

    with pytest.raises(FileExistsError):
        run_rollback.rollback_deck(str(tmp_path), "elves", 58, deck_ctx)
    run_rollback.rollback_deck(str(tmp_path), "elves", 58, deck_ctx, force=True)  # opt in


@pytest.mark.slow
def test_rollback_dry_run_changes_nothing(tmp_path):
    import run_rollback
    from rl.model.mulligan import MulliganNet
    from rl import checkpoint as ckpt_io

    deck_dir = tmp_path / "elves"
    (deck_dir / "archive").mkdir(parents=True)
    ckpt_io.save_snapshot(str(deck_dir / "archive" / "snapshot_3.pt"),
                          league_runner.build_deck_net(12, 4), MulliganNet(league_runner.build_deck_net(12, 4).encoder))
    run_rollback.rollback_deck(str(tmp_path), "elves", 3, (SimpleNamespace(size=12), list(range(4))), dry_run=True)
    assert not (deck_dir / "live.pt").exists(), "a dry run must not write anything"
