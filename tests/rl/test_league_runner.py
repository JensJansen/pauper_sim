"""Self-check for rl.league_runner's auto-sizing doubling ladder
(_next_batch_games), which lost its max_batch_size cap 2026-07-31 -- see its
own docstring for why. Marked slow: importing rl.league_runner pulls in torch/rl.*.
"""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl import checkpoint as ckpt_io
from rl import league_runner
from rl.pool import build_pool as _real_build_pool

_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


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
    # no separate ceiling anymore (max_batch_size removed) -- the ONLY cap left
    # is "don't play more than what's left of total_games"
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
def test_run_eval_labels_each_game_with_its_real_pairing(tmp_path, monkeypatch):
    """_run_eval's whole point for the round-robin case (--eval, no --matchup)
    is that game N in a many-pairing log can be ANY pairing -- confirm it
    returns a (deck_a, deck_b) per game_logs entry, in round-robin order
    (combinations_with_replacement: AA, AB, BB for a 2-deck roster), and that
    _write_event_log round-trips it into the JSON as real deck_a/deck_b
    fields instead of a bare, unlabeled game_index. An empty league_dir (so
    every deck starts from a fresh, untrained net) + a
    tmp_path league_dir sidesteps needing a real frozen_stack/live checkpoint
    (untrained random-init nets play real, complete games -- slow because of
    that, not because anything here is flaky); a private vocab_path keeps
    this test from writing to the repo's own real checkpoints/vocab.json."""
    monkeypatch.chdir(_SRC_DIR)  # league_decks.json/data/*.txt are loaded via "../data/..." (rl.pool's own convention)
    monkeypatch.setattr(
        league_runner, "build_pool",
        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")),
    )

    game_logs = []
    eval_decks, game_pairings = league_runner._run_eval(
        ["rakdos_madness", "dmir_terror"], games_per_pairing=2, greedy=False, seed=0,
        game_logs=game_logs, league_dir=str(tmp_path / "league"),
    )
    assert eval_decks == ["rakdos_madness", "dmir_terror"]
    assert len(game_logs) == 6  # 3 pairings (AA, AB, BB) x 2 games
    assert game_pairings == (
        [("rakdos_madness", "rakdos_madness")] * 2
        + [("rakdos_madness", "dmir_terror")] * 2
        + [("dmir_terror", "dmir_terror")] * 2
    )

    log_path = str(tmp_path / "eval_log.json")
    league_runner._write_event_log(log_path, game_logs, {"mode": "eval"}, game_pairings=game_pairings)
    with open(log_path) as f:
        doc = json.load(f)
    assert [(g["deck_a"], g["deck_b"]) for g in doc["games"]] == game_pairings
    assert [g["game_index"] for g in doc["games"]] == list(range(6))


@pytest.mark.slow
def test_run_eval_vs_history_finds_archived_and_active_milestones(tmp_path, monkeypatch):
    """_run_eval_vs_history is the direct measurement of "does the live net
    beat its own past selves" -- confirms it (a) returns [] with no history
    yet, (b) finds the active pool's oldest snapshot once one exists, and (c)
    finds an ARCHIVED one too once LeaguePool.register_snapshot has evicted
    past the window (rl.league's own archive/, not deletion)."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(
        league_runner, "build_pool",
        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")),
    )
    from rl.league import LeaguePool
    from rl.mulligan import MulliganNet

    league_dir = str(tmp_path / "league")
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    deck_name = "rakdos_madness"
    net = league_runner.build_deck_net(vocab.size, len(fixed_tables[deck_name]))
    mnet = MulliganNet(net.encoder)

    # No snapshots at all yet -- nothing to compare against.
    assert league_runner._run_eval_vs_history(
        deck_name, net, mnet, deck_ctxs[deck_name], decklists[deck_name], league_dir, horizon=20,
    ) == []

    pool = LeaguePool(league_dir, [deck_name], max_snapshots_per_deck=2)
    for _ in range(3):  # 3rd registration evicts snapshot_0 into archive/ (cap=2)
        pool.register_snapshot(deck_name, net, mnet)

    results = league_runner._run_eval_vs_history(
        deck_name, net, mnet, deck_ctxs[deck_name], decklists[deck_name], league_dir,
        horizon=20, games_per_snapshot=2, seed=0,
    )
    labels = {r["label"] for r in results}
    assert labels == {"archive_oldest", "active_oldest"}, f"expected both milestones, got {labels}"
    for r in results:
        assert r["games"] == 2
        assert r["live_wins"] + r["snapshot_wins"] + r["no_winner"] == r["games"]


@pytest.mark.slow
def test_run_eval_vs_gauntlet_plays_the_independent_twin_and_handles_missing_deck(tmp_path, monkeypatch):
    """_run_eval_vs_gauntlet is the EXTERNAL reference check (an independently
    trained twin league, not this league's own history) -- confirms it (a)
    returns None when the gauntlet league has no checkpoint for this deck yet
    (its training hasn't reached it, or it doesn't exist), and (b) actually
    plays real games against the gauntlet's live net and tallies a real
    result once one exists."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(
        league_runner, "build_pool",
        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")),
    )
    deck_name = "rakdos_madness"
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    live_net = league_runner.build_deck_net(vocab.size, len(fixed_tables[deck_name]))
    from rl.mulligan import MulliganNet
    mulligan_net = MulliganNet(live_net.encoder)

    gauntlet_dir = str(tmp_path / "gauntlet")
    # No gauntlet checkpoint for this deck yet -- must return None, not crash or return an empty dict.
    assert league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name],
        gauntlet_dir, horizon=20,
    ) is None
    # gauntlet_league_dir=None entirely (the common case -- most leagues have no gauntlet) must also return None.
    assert league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name],
        None, horizon=20,
    ) is None

    # Write a real (untrained but structurally valid) gauntlet checkpoint for this deck.
    import os
    import torch
    deck_dir = os.path.join(gauntlet_dir, deck_name)
    os.makedirs(deck_dir, exist_ok=True)
    torch.save({"net": live_net.state_dict()}, os.path.join(deck_dir, "live.pt"))

    result = league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name],
        gauntlet_dir, horizon=20, games=2, seed=0,
    )
    assert result["games"] == 2
    assert result["live_wins"] + result["gauntlet_wins"] + result["no_winner"] == 2


@pytest.mark.slow
def test_run_eval_vs_heuristic_plays_real_games(tmp_path, monkeypatch):
    """_run_eval_vs_heuristic is the gauntlet's tier-1 (hand-authored,
    non-learned) member -- confirms it actually plays real games between a
    live net and rl.heuristic_agent.HeuristicAgent and returns a real tally, using the
    deck the owner scoped it to (mono_red_rally)."""
    monkeypatch.chdir(_SRC_DIR)
    monkeypatch.setattr(
        league_runner, "build_pool",
        lambda: _real_build_pool(vocab_path=str(tmp_path / "vocab.json")),
    )
    deck_name = "mono_red_rally"
    decklists, vocab, deck_ctxs, fixed_tables = _real_build_pool(vocab_path=str(tmp_path / "vocab.json"))
    live_net = league_runner.build_deck_net(vocab.size, len(fixed_tables[deck_name]))
    from rl.mulligan import MulliganNet
    mulligan_net = MulliganNet(live_net.encoder)

    result = league_runner._run_eval_vs_heuristic(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name],
        horizon=20, games=3, seed=0,
    )
    assert result["games"] == 3
    assert result["live_wins"] + result["heuristic_wins"] + result["no_winner"] == 3


# --- Wave 1 instrumentation (2026-08-13) ---


@pytest.mark.slow
def test_paired_eval_balances_the_play_exactly_not_just_on_average():
    """_play_paired_eval_games replays the SAME seed with the seats exchanged,
    so collect_rollout's per-game starting_idx draw hands the play to the other
    agent on identical shuffles. On-the-play is then balanced exactly rather
    than in expectation -- the variance reduction the n=20 checks needed."""
    import inspect
    src = inspect.getsource(league_runner._play_paired_eval_games)
    assert src.count("random.Random(seed)") == 2, \
        "both halves must be driven from the SAME seed, or the pairing is lost"
    # live's wins in the swapped half come from the OPPONENT column
    assert 'rev[opp_wins_key]' in src


@pytest.mark.slow
def test_a_checkpoint_carries_its_own_encoder(tmp_path):
    """The property that replaced the stack_id guard.

    Cross-population comparison used to be silently meaningless whenever the
    two sides had been trained against different frozen shared stacks -- one
    population's per-deck weights would be loaded onto the OTHER's encoder and
    the resulting win rate measured nothing (24,579 games/deck of vs_gauntlet
    were confounded exactly that way, which is what stack_id.txt existed to
    catch). Per-deck encoders make it unrepresentable instead of guarded: the
    encoder ships inside the checkpoint, so a loaded net can only ever be
    paired with its own perception.

    Pinned by round-tripping a net whose encoder has been deliberately moved
    away from its initialization -- a loader that rebuilt a fresh encoder
    instead of restoring the saved one would come back at the random init."""
    from rl.deck import DeckNetwork  # noqa: F401 -- built via build_deck_net below

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
def test_unknown_ppo_hyperparameter_is_a_hard_error():
    """A typo'd hyperparameter that silently does nothing is precisely how two
    anti-plateau schedules went un-executed for 40,104 iterations."""
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
    from rl.deck import DeckNetwork
    from rl.mulligan import MulliganNet
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
    from rl.deck import DeckNetwork
    from rl.mulligan import MulliganNet
    from rl import checkpoint as ckpt_io

    deck_dir = tmp_path / "elves"
    (deck_dir / "archive").mkdir(parents=True)
    ckpt_io.save_snapshot(str(deck_dir / "archive" / "snapshot_3.pt"),
                          league_runner.build_deck_net(12, 4), MulliganNet(league_runner.build_deck_net(12, 4).encoder))
    run_rollback.rollback_deck(str(tmp_path), "elves", 3, (SimpleNamespace(size=12), list(range(4))), dry_run=True)
    assert not (deck_dir / "live.pt").exists(), "a dry run must not write anything"
