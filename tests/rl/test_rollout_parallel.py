"""Tests for rl.rollout_parallel: the ProcessPoolExecutor multiprocessing
plumbing for league rollout collection (collect_rollout_league_parallel and
its worker, _league_rollout_worker)."""
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.pool import build_pool
from rl.rollout_parallel import collect_rollout_league_parallel

HORIZON = 20


@pytest.mark.slow
def test_collect_rollout_league_parallel_smoke():
    # Regression coverage for collect_rollout_league_parallel's own
    # all_trunk_hidden plumbing: rl.league_runner computes it ONCE per session
    # and threads it through every call instead of this function re-deriving it
    # from live_nets itself (see its own docstring). A ThreadPoolExecutor
    # stands in for the real ProcessPoolExecutor -- same submit()/
    # future.result() interface, no process-spawn/pickling overhead -- but
    # _league_rollout_worker's own build_pool() call still runs for real (this
    # directory's conftest.py chdir's to src/ for exactly that reason).
    #
    # Each net's ENCODER has no separate plumbing to regress: it is a
    # registered child, so it crosses the boundary inside the net's own
    # state_dict. There used to be a shared_state_dict/shared_hparams pair
    # here carrying the one frozen shared stack across separately.
    _decklists, vocab, _deck_ctxs, fixed_tables = build_pool()
    net = DeckNetwork(SetTransformer(vocab.size), film_condition_dim=SetTransformer(vocab.size).d_model,
                       non_targeting_n_actions=len(fixed_tables["mono_red_madness"]), trunk_hidden=(24, 24))
    live_nets = {"mono_red_madness": net}
    all_trunk_hidden = {"mono_red_madness": tuple(layer.out_features for layer in net.trunk_layers)}

    tmp_dir = tempfile.mkdtemp()
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            buffers_by_deck, _mull_by_deck, games_played, outcomes = collect_rollout_league_parallel(
                "mono_red_madness", live_nets, "action_count_win_reward_200_floor02", tmp_dir, HORIZON,
                n_games=1, executor=executor, n_workers=1, all_trunk_hidden=all_trunk_hidden,
            )
    finally:
        shutil.rmtree(tmp_dir)

    assert games_played == 1, "one submitted worker task playing one game must report one game played"
    assert "mono_red_madness" in buffers_by_deck and len(buffers_by_deck["mono_red_madness"]) > 0, (
        "the worker's own build_pool()-rebuilt tables and the DeckNetwork rebuilt with all_trunk_hidden's "
        "trunk widths (encoder included, from the net's own state_dict) must together produce a real rollout"
    )
    assert all(np.isfinite(v) for v in buffers_by_deck["mono_red_madness"].value)
    # 0 entries iff that single game hit a horizon timeout (no winner) -- excluded
    # entirely rather than recorded as a loss, see collect_rollout_league's own docstring.
    assert len(outcomes) <= 1 and all(o[2] in (True, False) for o in outcomes), (
        "the worker's own outcome (opponent, snapshot_id, won), when present, must cross the process boundary "
        "intact -- this is what feeds PFSP's record_outcome back in the MAIN process, not the worker's own read-only pool"
    )
    print(f"rl.rollout_parallel collect_rollout_league_parallel smoke test: OK ({time.time() - t0:.1f}s)")
