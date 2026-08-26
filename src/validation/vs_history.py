"""Check: vs_history -- each deck's current live net vs. its own frozen old
self (the oldest still-active snapshot, and once any exist, the oldest
archived one). The one check immune to the "everyone is improving together"
confound the round-robin checks both have, since the opponent here can never
change -- any win-rate movement is unambiguously this deck's own progress.
Per deck plus a league-wide rollup.

Absorbed from rl.league.league_runner._run_eval_vs_history (previously
triggered automatically inside _run_session every eval_every_sessions
sessions; now runs only at this pipeline's cadence instead)."""
import torch

from rl.league.league_runner import HORIZON, _run_eval_vs_history
from rl.roster import build_pool

from . import _common

NAME = "vs_history"

_worker_pool_cache = None  # build_pool()'s result, cached per WORKER PROCESS -- see _worker_pool


def _worker_pool():
    """Sibling of rl.league.league_runner._eval_worker_pool /
    validation.round_robin_training._worker_pool -- same one-time-per-process
    build_pool() cache, own copy since it's cheap and this is a different
    module."""
    global _worker_pool_cache
    if _worker_pool_cache is None:
        _worker_pool_cache = build_pool()
    return _worker_pool_cache


def _deck_worker(league_dir, name, decklist, games_per_snapshot, seed):
    """One deck's whole vs-history check -- fully independent of every
    other deck (own net, own snapshots/archive, no shared accumulator),
    which is what makes this the simplest of the three checks to run either
    in-process (executor=None) or in a SEPARATE PROCESS via
    ctx.executor.submit.

    Rebuilds vocab/deck_ctxs/fixed_tables via build_pool() (cached per
    process) and loads this one deck's live net fresh from league_dir,
    rather than having either shipped from the parent -- same rationale as
    rl.league.league_runner._eval_pairing_chunk_worker's own docstring
    (fixed_table closures can't pickle; live.pt is already the current
    checkpointed weights by the time any validation check runs).

    _run_eval_vs_history itself already builds a fresh random.Random(seed)
    per call (see rl.league.league_runner._play_paired_eval_games) rather
    than consuming a shared stream -- unlike the two round-robin checks,
    running every deck's identical `seed` in parallel changes nothing about
    reproducibility, since each deck's call was already independent of
    every other's even in the sequential version."""
    torch.set_num_threads(1)  # this worker IS the unit of parallelism when run via an executor
    _decklists, vocab, deck_ctxs, fixed_tables = _worker_pool()
    net, mnet = _common.load_deck_net(league_dir, name, vocab, fixed_tables[name])
    return _run_eval_vs_history(name, net, mnet, deck_ctxs[name], decklist, league_dir, HORIZON,
                               games_per_snapshot=games_per_snapshot, seed=seed)


def run(ctx):
    if ctx.executor is None or ctx.n_workers <= 1:
        per_deck = {name: _deck_worker(ctx.primary_league_dir, name, ctx.decklists[name],
                                       ctx.games_per_check, ctx.seed)
                   for name in ctx.train_decks}
    else:
        futures = {name: ctx.executor.submit(_deck_worker, ctx.primary_league_dir, name, ctx.decklists[name],
                                             ctx.games_per_check, ctx.seed)
                  for name in ctx.train_decks}
        per_deck = {name: future.result() for name, future in futures.items()}

    for name in ctx.train_decks:
        results = per_deck[name]
        payload = {"check": NAME, "primary_league": ctx.primary_league_name, "deck": name,
                  "cumulative_games": ctx.cumulative_games, "milestones": results}
        _common.write_deck_json(ctx, name, NAME, payload)

        for r in results:
            wr = r["live_wins"] / r["games"] if r["games"] else float("nan")
            _common.append_metric(ctx, kind=NAME, deck=name, label=r["label"], snapshot_id=r["snapshot_id"],
                                  games=r["games"], live_wins=r["live_wins"], win_rate=wr)

    league_payload = {"check": NAME, "primary_league": ctx.primary_league_name,
                      "cumulative_games": ctx.cumulative_games, "decks": per_deck}
    write_path = _common.write_league_json(ctx, NAME, league_payload)

    return {"decks": len(per_deck), "wrote": write_path}
