"""Playing matchups between agents loaded from arbitrary checkpoint directories."""
import hashlib
import math
import os
import random
from dataclasses import dataclass

import torch

from rl import checkpoint as ckpt_io
from rl.decision.agent import SeatAgent
from rl.league import league, league_runner
from rl.model.mulligan import MulliganNet
from rl.roster import build_pool
from rl.training.rollout_parallel import _sanitize_events
from rl.training.train import _constant_pairing, collect_rollout

from . import history

CHUNK_GAMES = 20  # games per work unit: small enough to balance across workers, big enough to amortize the dispatch


@dataclass(frozen=True)
class MatchupResult:
    a: object  # AgentRef
    b: object  # AgentRef
    games: int
    a_wins: int
    b_wins: int
    no_winner: int  # horizon timeouts
    logs: object = None  # play_matchups(collect_logs=True): one sanitized engine event log per game, in game order


def matchup_seed(seed, deck_a, deck_b, chunk):
    """The RNG seed of one chunk of games. Depends only on the master seed, the two DECK names and the chunk index --
    not on which directories the agents come from, nor on the worker count. Two variants of the same deck therefore
    face the same shuffles and coin flips against the same opponent (common random numbers: a paired comparison, so
    the variants differ by their policies, not by luck of the deal), and results never depend on scheduling."""
    digest = hashlib.blake2b(f"{seed}|{deck_a}|{deck_b}|{chunk}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


_pool_cache = None  # (decklists, vocab, deck_ctxs, fixed_tables), built once per process
_agent_cache = {}   # (league_dir, deck, snapshot, live-file mtimes) -> (deck net, mulligan net), loaded once per process
_pool_by_dir = {}   # (league_dir, deck) -> LeaguePool, only used for its snapshot loader


def _pool():
    global _pool_cache
    if _pool_cache is None:
        _pool_cache = build_pool()
    return _pool_cache


def _load(league_dir, deck, snapshot=None):
    """(deck net, mulligan net) for the deck's live weights, or for snapshot `snapshot` (an id).

    A snapshot never changes, so it is cached for the life of the process. live.pt/mulligan.pt do change (a
    training pipeline's worker processes outlive many cadence points), so their cache key carries the files' mtimes."""
    live = None if snapshot is not None else tuple(
        os.stat(f"{league_dir}/{deck}/{name}").st_mtime_ns for name in ("live.pt", "mulligan.pt"))
    key = (league_dir, deck, snapshot, live)
    if key not in _agent_cache:
        _decklists, vocab, deck_ctxs, fixed_tables = _pool()
        if snapshot is not None:
            pool = _pool_by_dir.setdefault((league_dir, deck), league.LeaguePool(league_dir, [deck]))
            agent = pool.load_snapshot_agent(history.snapshot_path(league_dir, deck, snapshot), deck_ctxs[deck])
            _agent_cache[key] = (agent.main, agent.mulligan)
            return _agent_cache[key]
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
    """Runs in a worker (or in-process). Plays n games of one matchup; returns (a_wins, b_wins, no_winner), or that
    plus the sanitized per-game event logs when the task's optional trailing collect_logs flag is true."""
    (dir_a, deck_a, *snap_a), (dir_b, deck_b, *snap_b), n, seed, greedy, horizon, *flags = task  # snap_*: [] or [id]
    logs = [] if flags and flags[0] else None
    torch.set_num_threads(1)
    decklists, _vocab, deck_ctxs, _tables = _pool()
    agents = []
    for league_dir, deck, snap in ((dir_a, deck_a, snap_a), (dir_b, deck_b, snap_b)):
        net, mnet = _load(league_dir, deck, *snap)
        agents.append(SeatAgent(net, mnet, deck_ctxs[deck]))
    winners = []
    pairing = _constant_pairing(agents, [decklists[deck_a], decklists[deck_b]], [None, None], [None, None])
    collect_rollout(pairing, n, horizon, random.Random(seed), device="cpu", record=False, greedy=greedy,
                    game_logs=logs, on_game_end=lambda state: winners.append(state.winner))
    counts = (winners.count(0), winners.count(1), winners.count(None))
    return counts if logs is None else (*counts, _sanitize_events(logs))  # a raw event can hold an unpicklable closure


def play_matchups(matchups, games, seed=0, greedy=True, executor=None, horizon=None, collect_logs=False):
    """Plays `games` games of every matchup and returns one MatchupResult per matchup, in order.

    greedy: argmax play (what validation/ uses); False samples from the policy like training does.
    executor: an existing ProcessPoolExecutor to spread the work over; None plays in this process.
    collect_logs: also return every game's full engine event log (MatchupResult.logs; what the replay viewer and
    mulligan_audit read), at the cost of holding them all in memory.
    Results depend only on (matchups, games, seed, greedy), never on the executor or its size."""
    horizon = horizon if horizon is not None else league_runner.HORIZON
    tasks, owner = [], []
    for i, m in enumerate(matchups):
        for chunk in range(math.ceil(games / CHUNK_GAMES)):
            n = min(CHUNK_GAMES, games - chunk * CHUNK_GAMES)
            tasks.append(((m.a.league_dir, m.a.deck, m.a.snapshot), (m.b.league_dir, m.b.deck, m.b.snapshot), n,
                          matchup_seed(seed, m.a.deck, m.b.deck, chunk), greedy, horizon, collect_logs))
            owner.append(i)
    outputs = list(executor.map(_play_chunk, tasks)) if executor is not None else [_play_chunk(t) for t in tasks]
    totals = [[0, 0, 0] for _ in matchups]
    logs = [[] for _ in matchups]
    for i, out in zip(owner, outputs):
        for k in range(3):
            totals[i][k] += out[k]
        if collect_logs:
            logs[i].extend(out[3])
    return [MatchupResult(m.a, m.b, games, *t, logs=lg if collect_logs else None)
            for m, t, lg in zip(matchups, totals, logs)]
