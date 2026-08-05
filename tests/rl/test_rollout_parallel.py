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
    # shared_state_dict/all_trunk_hidden plumbing: rl.league_runner now
    # computes both ONCE per session and threads them through every call
    # instead of this function re-deriving them from live_nets itself (see
    # its own docstring). A ThreadPoolExecutor stands in for the real
    # ProcessPoolExecutor -- same submit()/future.result() interface, no
    # process-spawn/pickling overhead -- but _league_rollout_worker's own
    # build_pool() call still runs for real (this directory's conftest.py
    # chdir's to src/ for exactly that reason). Confirms the frozen shared
    # stack and the per-deck trunk widths survive the boundary intact enough
    # to actually play a game and record a real, finite transition.
    _decklists, vocab, _deck_ctxs, fixed_tables = build_pool()
    shared_hparams = {"d_model": 16, "n_heads": 2, "n_layers": 1, "dim_feedforward": 32}
    shared = SetTransformer(vocab.size, **shared_hparams)
    net = DeckNetwork(shared, film_condition_dim=16,
                       non_targeting_n_actions=len(fixed_tables["mono_red_madness"]), trunk_hidden=(24, 24))
    live_nets = {"mono_red_madness": net}
    shared_state_dict = shared.state_dict()
    all_trunk_hidden = {"mono_red_madness": tuple(layer.out_features for layer in net.trunk_layers)}

    tmp_dir = tempfile.mkdtemp()
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            buffers_by_deck, _mull_by_deck, games_played, outcomes = collect_rollout_league_parallel(
                "mono_red_madness", live_nets, "action_count_win_reward_200_floor02", tmp_dir, HORIZON,
                n_games=1, executor=executor, n_workers=1, shared_hparams=shared_hparams,
                shared_state_dict=shared_state_dict, all_trunk_hidden=all_trunk_hidden,
            )
    finally:
        shutil.rmtree(tmp_dir)

    assert games_played == 1, "one submitted worker task playing one game must report one game played"
    assert "mono_red_madness" in buffers_by_deck and len(buffers_by_deck["mono_red_madness"]) > 0, (
        "the worker's own build_pool()-rebuilt tables, the shared stack loaded from shared_state_dict, and "
        "the DeckNetwork rebuilt with all_trunk_hidden's trunk widths must together produce a real rollout"
    )
    assert all(np.isfinite(v) for v in buffers_by_deck["mono_red_madness"].value)
    # 0 entries iff that single game hit a horizon timeout (no winner) -- excluded
    # entirely rather than recorded as a loss, see collect_rollout_league's own docstring.
    assert len(outcomes) <= 1 and all(o[2] in (True, False) for o in outcomes), (
        "the worker's own outcome (opponent, snapshot_id, won), when present, must cross the process boundary "
        "intact -- this is what feeds PFSP's record_outcome back in the MAIN process, not the worker's own read-only pool"
    )
    print(f"rl.rollout_parallel collect_rollout_league_parallel smoke test: OK ({time.time() - t0:.1f}s)")
