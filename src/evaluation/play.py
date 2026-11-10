"""Playing matchups between agents loaded from arbitrary checkpoint directories."""
import hashlib
import math
import random
from dataclasses import dataclass

import torch

from rl import checkpoint as ckpt_io
from rl.decision.agent import SeatAgent
from rl.league import league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool
from rl.training.train import _constant_pairing, collect_rollout

CHUNK_GAMES = 20  # games per work unit: small enough to balance across workers, big enough to amortize the dispatch


@dataclass(frozen=True)
class MatchupResult:
    a: object  # AgentRef
    b: object  # AgentRef
    games: int
    a_wins: int
    b_wins: int
    no_winner: int  # horizon timeouts


def matchup_seed(seed, deck_a, deck_b, chunk):
    """The RNG seed of one chunk of games. Depends only on the master seed, the two DECK names and the chunk index --
    not on which directories the agents come from, nor on the worker count. Two variants of the same deck therefore
    face the same shuffles and coin flips against the same opponent (common random numbers: a paired comparison, so
    the variants differ by their policies, not by luck of the deal), and results never depend on scheduling."""
    digest = hashlib.blake2b(f"{seed}|{deck_a}|{deck_b}|{chunk}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


_pool_cache = None  # (decklists, vocab, deck_ctxs, fixed_tables), built once per process
_agent_cache = {}   # (league_dir, deck) -> (deck net, mulligan net), loaded once per process


def _pool():
    global _pool_cache
    if _pool_cache is None:
        _pool_cache = build_pool()
    return _pool_cache


def _load(league_dir, deck):
    key = (league_dir, deck)
    if key not in _agent_cache:
        _decklists, vocab, _ctxs, fixed_tables = _pool()
        live_path = f"{league_dir}/{deck}/live.pt"
        net = league_runner.build_deck_net(vocab.size, len(fixed_tables[deck]),
                                           ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
        ckpt_io.load_deck_checkpoint(live_path, net)
        net.eval()
        mnet = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{deck}/mulligan.pt", mnet)
        mnet.eval()
        _agent_cache[key] = (net, mnet)
    return _agent_cache[key]


def _play_chunk(task):
    """Runs in a worker (or in-process). Plays n games of one matchup; returns (a_wins, b_wins, no_winner)."""
    (dir_a, deck_a), (dir_b, deck_b), n, seed, greedy, horizon = task
    torch.set_num_threads(1)
    decklists, _vocab, deck_ctxs, _tables = _pool()
    agents = []
    for league_dir, deck in ((dir_a, deck_a), (dir_b, deck_b)):
        net, mnet = _load(league_dir, deck)
        agents.append(SeatAgent(net, mnet, deck_ctxs[deck]))
    winners = []
    pairing = _constant_pairing(agents, [decklists[deck_a], decklists[deck_b]], [None, None], [None, None])
    collect_rollout(pairing, n, horizon, random.Random(seed), device="cpu", record=False, greedy=greedy,
                    on_game_end=lambda state: winners.append(state.winner))
    return winners.count(0), winners.count(1), winners.count(None)


def play_matchups(matchups, games, seed=0, greedy=True, executor=None, horizon=None):
    """Plays `games` games of every matchup and returns one MatchupResult per matchup, in order.

    greedy: argmax play (what validation/ uses); False samples from the policy like training does.
    executor: an existing ProcessPoolExecutor to spread the work over; None plays in this process.
    Results depend only on (matchups, games, seed, greedy), never on the executor or its size."""
    horizon = horizon if horizon is not None else league_runner.HORIZON
    tasks, owner = [], []
    for i, m in enumerate(matchups):
        for chunk in range(math.ceil(games / CHUNK_GAMES)):
            n = min(CHUNK_GAMES, games - chunk * CHUNK_GAMES)
            tasks.append(((m.a.league_dir, m.a.deck), (m.b.league_dir, m.b.deck), n,
                          matchup_seed(seed, m.a.deck, m.b.deck, chunk), greedy, horizon))
            owner.append(i)
    outputs = list(executor.map(_play_chunk, tasks)) if executor is not None else [_play_chunk(t) for t in tasks]
    totals = [[0, 0, 0] for _ in matchups]
    for i, out in zip(owner, outputs):
        for k in range(3):
            totals[i][k] += out[k]
    return [MatchupResult(m.a, m.b, games, *t) for m, t in zip(matchups, totals)]
