"""A game played by a rollout worker is the game played in process (plan P3, the real-games half).

The worker rebuilds its networks from shipped state dicts, rebuilds the roster tables itself, plays with its own seed, and
returns its buffer as plain tuples. Run on the same weights and the same seed, that path must record exactly the transitions
the in-process path records -- every action, log-probability, value, reward, done flag, mask, scalar input and token (identities
excepted: they are dropped on purpose before crossing the boundary)."""
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import rl.rewards as rewards_module
from rl.league.league import LeaguePool
from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork
from rl.roster import build_pool
from rl.training import rollout_parallel as rp
from rl.training.train import collect_rollout_league

DECK = "mono_red_madness"
HORIZON = 20
SEED = 1234


@pytest.fixture()
def pool_and_net():
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    torch.manual_seed(0)
    net = DeckNetwork(SetTransformer(vocab.size), non_targeting_n_actions=len(fixed_tables[DECK]), trunk_hidden=(24, 24))
    net.eval()
    return decklists, deck_ctxs, net


def _in_process(decklists, deck_ctxs, net, n_games, tmp, seed=SEED):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # The worker builds its networks (random initial weights, then overwritten from the shipped state dict) before it plays,
    # and that draws from torch's RNG. Draw the same amount here so both paths sample their actions from the same stream.
    DeckNetwork(SetTransformer(deck_ctxs[DECK][0].size),
                non_targeting_n_actions=len(deck_ctxs[DECK][1]), trunk_hidden=(24, 24))
    pool = LeaguePool(tmp, [DECK])
    buffers, _m, played, outcomes = collect_rollout_league(
        DECK, {DECK: net}, None, deck_ctxs, decklists, pool, rewards_module.deploy_reward_v6, HORIZON, n_games,
        random.Random(seed), device="cpu")
    return rp._buffer_to_entries(buffers[DECK]), played, outcomes


def _via_worker(net, n_games, tmp):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    trunk = tuple(layer.out_features for layer in net.trunk_layers)
    entries, _m, _logs, played, outcomes = rp._league_rollout_worker(
        DECK, {DECK: sd}, {DECK: trunk}, "deploy_reward_v6", tmp, HORIZON, n_games, SEED)
    return entries[DECK], played, outcomes


def _same(a, b):
    assert len(a) == len(b) > 0
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        toks_x, toks_y = x[0], y[0]
        assert [(v, tuple(np.ravel(r))) for v, r, _ in toks_x] == [(v, tuple(np.ravel(r))) for v, r, _ in toks_y], i
        assert np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]), i
        assert x[3] == y[3] and x[4] == pytest.approx(y[4], abs=1e-6) and x[5] == pytest.approx(y[5], abs=1e-6), i
        assert x[6] == y[6] and x[7] == y[7], i


def test_the_worker_records_what_the_in_process_loop_records(pool_and_net):
    decklists, deck_ctxs, net = pool_and_net
    tmp = tempfile.mkdtemp()
    try:
        mine, played_a, out_a = _in_process(decklists, deck_ctxs, net, 2, tmp)
        theirs, played_b, out_b = _via_worker(net, 2, tmp)
    finally:
        shutil.rmtree(tmp)
    assert played_a == played_b == 2 and out_a == out_b
    _same(mine, theirs)


def test_the_same_through_a_submitted_and_gathered_round(pool_and_net):
    """submit_league_rollout draws each worker's seed from the global `random`; with it seeded, the first draw is the seed the
    in-process run is given. A thread pool stands in for the process pool (same submit/result interface)."""
    decklists, deck_ctxs, net = pool_and_net
    drawn = random.Random(0).randrange(2 ** 31)
    tmp = tempfile.mkdtemp()
    try:
        mine, played, out = _in_process(decklists, deck_ctxs, net, 2, tmp, seed=drawn)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(0)
        with ThreadPoolExecutor(max_workers=1) as ex:
            futures = rp.submit_league_rollout(DECK, {DECK: net}, "deploy_reward_v6", tmp, HORIZON, 2, ex, 1,
                                               {DECK: tuple(layer.out_features for layer in net.trunk_layers)})
            buffers, _m, played_w, outcomes = rp.gather_league_rollout(futures)
    finally:
        shutil.rmtree(tmp)
    assert played == played_w == 2 and out == outcomes
    _same(mine, rp._buffer_to_entries(buffers[DECK]))
