"""Tests for rl.training.rollout_parallel: the ProcessPoolExecutor multiprocessing
plumbing for league rollout collection (collect_rollout_league_parallel and
its worker, _league_rollout_worker)."""
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork
from rl.roster import build_pool
from rl.training.rollout_parallel import collect_rollout_league_parallel

HORIZON = 20


@pytest.mark.slow
def test_collect_rollout_league_parallel_smoke():
    # ThreadPoolExecutor stands in for the real ProcessPoolExecutor (same
    # submit()/future.result() interface, no spawn/pickling overhead), but
    # _league_rollout_worker's own build_pool() call still runs for real
    # (conftest.py chdir's to src/ for that reason). The encoder crosses the
    # boundary inside the net's own state_dict, as a registered child.
    _decklists, vocab, _deck_ctxs, fixed_tables = build_pool()
    net = DeckNetwork(SetTransformer(vocab.size), film_condition_dim=SetTransformer(vocab.size).d_model,
                       non_targeting_n_actions=len(fixed_tables["mono_red_madness"]), trunk_hidden=(24, 24))
    live_nets = {"mono_red_madness": net}
    all_trunk_hidden = {"mono_red_madness": tuple(layer.out_features for layer in net.trunk_layers)}

    tmp_dir = tempfile.mkdtemp()
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as executor:
            # stratify_0land_pct=1.0: a single-deck roster always resolves
            # to a self-mirror (checkpoint_rate defaults to 0.0 -- never a
            # snapshot), so this can't actually force the hand (both seats
            # get recorded, collect_rollout's own guard blocks it) -- but it
            # DOES prove the kwarg crosses the executor.submit(...) /
            # _league_rollout_worker(...) process boundary without raising
            # (a positional/keyword mismatch there would surface as a
            # TypeError from collect_rollout's own "stratify_0land_pct > 0.0"
            # comparison, unconditionally, before the guard is even reached).
            buffers_by_deck, _mull_by_deck, games_played, outcomes = collect_rollout_league_parallel(
                "mono_red_madness", live_nets, "deploy_reward_v6", tmp_dir, HORIZON,
                n_games=1, executor=executor, n_workers=1, all_trunk_hidden=all_trunk_hidden,
                stratify_0land_pct=1.0,
            )
    finally:
        shutil.rmtree(tmp_dir)

    assert games_played == 1, "one submitted worker task playing one game must report one game played"
    assert "mono_red_madness" in buffers_by_deck and len(buffers_by_deck["mono_red_madness"]) > 0, (
        "the worker's own build_pool()-rebuilt tables and the DeckNetwork rebuilt with all_trunk_hidden's "
        "trunk widths (encoder included, from the net's own state_dict) must together produce a real rollout"
    )
    assert all(np.isfinite(v) for v in buffers_by_deck["mono_red_madness"].value)
    # 0 entries iff that single game hit a horizon timeout (no winner) --
    # excluded rather than recorded as a loss.
    assert len(outcomes) <= 1 and all(o[2] in (True, False) for o in outcomes), (
        "the worker's own outcome (opponent, snapshot_id, won, was_stratified), when present, must cross the "
        "process boundary intact -- this is what feeds PFSP's record_outcome back in the MAIN process, not the "
        "worker's own read-only pool"
    )
    assert all(o[3] is False for o in outcomes), (
        "a single-deck roster is always a self-mirror (both seats recorded) -- stratify_0land_pct=1.0 above must "
        "still never fire here, proving the boundary crossing didn't accidentally also cross the seat-count guard"
    )
    print(f"rl.training.rollout_parallel collect_rollout_league_parallel smoke test: OK ({(time.time() - t0) * 1000:,.0f}ms)")
