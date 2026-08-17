"""Self-check for rl.league_runner's auto-sizing doubling ladder
(_next_batch_games), which lost its max_batch_size cap 2026-07-31 -- see its
own docstring for why. Marked slow: importing rl.league_runner pulls in torch/rl.*.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

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
    fields instead of a bare, unlabeled game_index. fresh_stack=True + a
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
        game_logs=game_logs, fresh_stack=True, league_dir=str(tmp_path / "league"),
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
    shared = league_runner.build_fresh_stack(vocab.size)
    net = league_runner.DeckNetwork(shared, film_condition_dim=league_runner.D_MODEL,
                                     non_targeting_n_actions=len(fixed_tables[deck_name]))
    mnet = MulliganNet(shared)

    # No snapshots at all yet -- nothing to compare against.
    assert league_runner._run_eval_vs_history(
        deck_name, net, mnet, deck_ctxs[deck_name], decklists[deck_name], shared, league_dir, horizon=20,
    ) == []

    pool = LeaguePool(league_dir, [deck_name], max_snapshots_per_deck=2)
    for _ in range(3):  # 3rd registration evicts snapshot_0 into archive/ (cap=2)
        pool.register_snapshot(deck_name, net, mnet)

    results = league_runner._run_eval_vs_history(
        deck_name, net, mnet, deck_ctxs[deck_name], decklists[deck_name], shared, league_dir,
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
    shared = league_runner.build_fresh_stack(vocab.size)
    live_net = league_runner.DeckNetwork(shared, film_condition_dim=league_runner.D_MODEL,
                                          non_targeting_n_actions=len(fixed_tables[deck_name]))
    from rl.mulligan import MulliganNet
    mulligan_net = MulliganNet(shared)

    gauntlet_dir = str(tmp_path / "gauntlet")
    # No gauntlet checkpoint for this deck yet -- must return None, not crash or return an empty dict.
    assert league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name], shared,
        gauntlet_dir, horizon=20,
    ) is None
    # gauntlet_league_dir=None entirely (the common case -- most leagues have no gauntlet) must also return None.
    assert league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name], shared,
        None, horizon=20,
    ) is None

    # Write a real (untrained but structurally valid) gauntlet checkpoint for this deck.
    import os
    import torch
    deck_dir = os.path.join(gauntlet_dir, deck_name)
    os.makedirs(deck_dir, exist_ok=True)
    torch.save({"net": live_net.state_dict()}, os.path.join(deck_dir, "live.pt"))

    result = league_runner._run_eval_vs_gauntlet(
        deck_name, live_net, mulligan_net, deck_ctxs[deck_name], decklists[deck_name], shared,
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
    shared = league_runner.build_fresh_stack(vocab.size)
    live_net = league_runner.DeckNetwork(shared, film_condition_dim=league_runner.D_MODEL,
                                          non_targeting_n_actions=len(fixed_tables[deck_name]))
    from rl.mulligan import MulliganNet
    mulligan_net = MulliganNet(shared)

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
def test_stack_id_detects_a_different_stack_but_tolerates_a_legacy_league(tmp_path):
    """The failure this exists for: a gauntlet league trained against a stack
    that no longer exists, compared against the current one, reporting a
    plausible number that means nothing (24,579 games/deck of vs_gauntlet were
    confounded exactly this way)."""
    a = league_runner.build_fresh_stack(12)
    b = league_runner.build_fresh_stack(12)  # same architecture, different weights
    assert league_runner.stack_id(a) != league_runner.stack_id(b)
    assert league_runner.stack_id(a) == league_runner.stack_id(a), "must be deterministic"

    league = str(tmp_path / "some_league")
    os.makedirs(league, exist_ok=True)
    # No stack_id.txt yet -> legacy league, tolerated (every league on disk
    # today predates the check; failing them all would be the guard's first act)
    assert league_runner.stack_id_matches(league, a)

    league_runner.write_stack_id(league, a)
    assert league_runner.stack_id_matches(league, a)
    assert not league_runner.stack_id_matches(league, b), "a DIFFERENT stack must be caught"


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

    shared = league_runner.build_fresh_stack(12)
    deck_ctx = (None, list(range(4)))
    deck_dir = tmp_path / "elves"
    (deck_dir / "archive").mkdir(parents=True)

    old = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4)
    ckpt_io.save_snapshot(str(deck_dir / "archive" / "snapshot_58.pt"), old, MulliganNet(shared))
    current = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4)
    with torch.no_grad():
        current.critic_head.weight.add_(5.0)  # make "now" clearly different from the snapshot
    ckpt_io.save_deck_checkpoint(str(deck_dir / "live.pt"), current)
    ckpt_io.save_deck_checkpoint(str(deck_dir / "mulligan.pt"), MulliganNet(shared))

    run_rollback.rollback_deck(str(tmp_path), "elves", 58, shared, deck_ctx)

    restored = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4)
    ckpt_io.load_deck_checkpoint(str(deck_dir / "live.pt"), restored)
    assert torch.equal(restored.critic_head.weight, old.critic_head.weight), "live.pt must be the snapshot"

    backed_up = DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4)
    ckpt_io.load_deck_checkpoint(str(deck_dir / ("live.pt" + run_rollback.BACKUP_SUFFIX)), backed_up)
    assert torch.equal(backed_up.critic_head.weight, current.critic_head.weight), \
        "the replaced live.pt must be recoverable"

    with pytest.raises(FileExistsError):
        run_rollback.rollback_deck(str(tmp_path), "elves", 58, shared, deck_ctx)
    run_rollback.rollback_deck(str(tmp_path), "elves", 58, shared, deck_ctx, force=True)  # opt in


@pytest.mark.slow
def test_rollback_dry_run_changes_nothing(tmp_path):
    import run_rollback
    from rl.deck import DeckNetwork
    from rl.mulligan import MulliganNet
    from rl import checkpoint as ckpt_io

    shared = league_runner.build_fresh_stack(12)
    deck_dir = tmp_path / "elves"
    (deck_dir / "archive").mkdir(parents=True)
    ckpt_io.save_snapshot(str(deck_dir / "archive" / "snapshot_3.pt"),
                          DeckNetwork(shared, film_condition_dim=shared.d_model, non_targeting_n_actions=4),
                          MulliganNet(shared))
    run_rollback.rollback_deck(str(tmp_path), "elves", 3, shared, (None, list(range(4))), dry_run=True)
    assert not (deck_dir / "live.pt").exists(), "a dry run must not write anything"
