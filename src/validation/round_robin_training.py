"""Check: primary_vs_training_round_robin -- the FULL cross product between
the primary league's roster and the training league's roster (deck_a x
deck_b, mismatched names included, mirror-across-leagues where names match
is just one cell of the matrix) -- not just same-name pairs the way the old
per-session vs_gauntlet check did. ctx.games_per_check games per pairing,
greedy, on current live weights on both sides. No-op (skipped) if no
training league is configured, or it has no live.pt for any deck yet.

Absorbed from analysis/eval/run_cross_league_eval.py (now deleted); same
common-random-numbers seat-swap and mana-burn reporting, extended to feed
mulligan_audit's shared game-log pool (primary-controlled seat only).
"""
import itertools
import random
import time

import torch

from rl.league.league_runner import HORIZON, league_roster
from rl.roster import build_pool
from rl.training.rollout_parallel import _sanitize_events
from rl.training.train import _constant_pairing, collect_rollout

from . import _common

NAME = "primary_vs_training_round_robin"

_worker_pool_cache = None  # build_pool()'s result, cached per WORKER PROCESS -- see _worker_pool


def _worker_pool():
    """build_pool()'s (decklists, vocab, deck_ctxs, fixed_tables), cached per
    process -- same one-time-per-worker rationale as
    rl.league.league_runner._eval_worker_pool (a sibling cache, not shared:
    different module, and the cost is trivial either way)."""
    global _worker_pool_cache
    if _worker_pool_cache is None:
        _worker_pool_cache = build_pool()
    return _worker_pool_cache


def _half(agents, decks, n_games, seed):
    """One seat orientation: agents[0]/decks[0] at seat 0. Returns the raw
    per-game event logs (every event, not just game_over -- mulligan_audit
    needs the draw/mulligan events too) plus the count played."""
    pairing = _constant_pairing(agents, decks, [None, None], [None, None])
    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, n_games, HORIZON, random.Random(seed),
                                           device="cpu", record=False, greedy=True, game_logs=game_logs)
    return game_logs, played


def _pairing_worker(primary_league_dir, training_league_dir, a, b, games_per_check, pair_seed):
    """One (primary_deck, training_deck) pairing, both seat orientations --
    the exact per-pairing computation `run` used to do inline, extracted so
    it can run either in-process (executor=None: called directly, in a
    plain loop, same order as before) or in a SEPARATE PROCESS via
    ctx.executor.submit -- one function, one behavior, regardless of which.

    Loads both agents fresh from their own league_dir rather than having
    them shipped from the parent: by the time any validation check runs,
    _run_session has already checkpointed this chunk's live.pt/mulligan.pt
    for BOTH leagues (see rl.league.league_runner._eval_pairing_chunk_worker's
    own docstring for why), so there's nothing "live in memory only" to
    miss. decklists/vocab/deck_ctxs/fixed_tables are rebuilt too (cached per
    process via _worker_pool) -- fixed_table entries hold legal_fn/execute_fn
    closures pickle can't serialize.

    Returns (pairing_result, burn_records, sanitized_fwd_logs,
    sanitized_rev_logs) -- burn_records is [(side, name, games, mana_burnt,
    mana_burnt_single_pip), ...] for the two sides. Logs are always
    sanitized (rl.training.rollout_parallel._sanitize_events), even when
    called in-process, so there's exactly one code path regardless of
    executor -- the cost is trivial next to a pairing's own game-collection
    time."""
    torch.set_num_threads(1)  # this worker IS the unit of parallelism when run via an executor
    decklists, vocab, deck_ctxs, fixed_tables = _worker_pool()
    agent_a = _common.load_agent(primary_league_dir, a, vocab, deck_ctxs[a], fixed_tables[a])
    agent_b = _common.load_agent(training_league_dir, b, vocab, deck_ctxs[b], fixed_tables[b])

    half = games_per_check // 2
    fwd_logs, played_f = _half([agent_a, agent_b], [decklists[a], decklists[b]], half, pair_seed)
    rev_logs, played_r = _half([agent_b, agent_a], [decklists[b], decklists[a]],
                               games_per_check - half, pair_seed)
    played = played_f + played_r

    fwd_over = [e for ev in fwd_logs for e in ev if e["kind"] == "game_over"]
    rev_over = [e for ev in rev_logs for e in ev if e["kind"] == "game_over"]
    a_wins = sum(1 for e in fwd_over if e["winner"] == 0) + sum(1 for e in rev_over if e["winner"] == 1)
    b_wins = sum(1 for e in fwd_over if e["winner"] == 1) + sum(1 for e in rev_over if e["winner"] == 0)
    no_winner = played - a_wins - b_wins

    a_burnt = sum(e["mana_burnt_total"][0] for e in fwd_over) + sum(e["mana_burnt_total"][1] for e in rev_over)
    b_burnt = sum(e["mana_burnt_total"][1] for e in fwd_over) + sum(e["mana_burnt_total"][0] for e in rev_over)
    a_burnt_sp = (sum(e["mana_burnt_total_single_pip"][0] for e in fwd_over)
                 + sum(e["mana_burnt_total_single_pip"][1] for e in rev_over))
    b_burnt_sp = (sum(e["mana_burnt_total_single_pip"][1] for e in fwd_over)
                 + sum(e["mana_burnt_total_single_pip"][0] for e in rev_over))

    pairing_result = {"primary_deck": a, "training_deck": b, "games": played,
                      "primary_wins": a_wins, "training_wins": b_wins, "no_winner": no_winner,
                      "primary_mana_burnt_total": a_burnt, "training_mana_burnt_total": b_burnt}
    burn_records = [("primary", a, played, a_burnt, a_burnt_sp), ("training", b, played, b_burnt, b_burnt_sp)]
    return pairing_result, burn_records, _sanitize_events(fwd_logs), _sanitize_events(rev_logs)


def run(ctx):
    if ctx.training_league_dir is None:
        return {"skipped": "no training league configured"}
    training_roster = league_roster(ctx.training_league_dir)
    if not training_roster:
        return {"skipped": "training league has no live.pt for any deck yet"}

    primary_decks = ctx.train_decks
    # One seed per pairing, pre-drawn sequentially in itertools.product order
    # BEFORE dispatch -- same draw order as the old inline loop, so results
    # stay reproducible for a given ctx.seed regardless of worker completion
    # order (a worker must never draw its own seed).
    rng = random.Random(ctx.seed)
    pairs = [(a, b, rng.randrange(2 ** 31)) for a, b in itertools.product(primary_decks, training_roster)]

    t0 = time.time()
    if ctx.executor is None or ctx.n_workers <= 1:
        results = [_pairing_worker(ctx.primary_league_dir, ctx.training_league_dir, a, b, ctx.games_per_check, seed)
                  for a, b, seed in pairs]
    else:
        futures = [ctx.executor.submit(_pairing_worker, ctx.primary_league_dir, ctx.training_league_dir,
                                       a, b, ctx.games_per_check, seed)
                  for a, b, seed in pairs]
        results = [f.result() for f in futures]

    pairing_results = []
    primary_totals = {d: {"wins": 0, "games": 0} for d in primary_decks}
    burn_by_deck = {}  # ("primary"|"training", name) -> [games, mana_burnt_total, mana_burnt_total_single_pip]

    def _record_burn(side, name, n, total, total_single_pip):
        acc = burn_by_deck.setdefault((side, name), [0, 0, 0])
        acc[0] += n
        acc[1] += total
        acc[2] += total_single_pip

    for (a, b, _seed), (pairing_result, burn_records, fwd_logs, rev_logs) in zip(pairs, results):
        pairing_results.append(pairing_result)
        for side, name, n, total, total_sp in burn_records:
            _record_burn(side, name, n, total, total_sp)
        primary_totals[a]["wins"] += pairing_result["primary_wins"]
        primary_totals[a]["games"] += pairing_result["games"]

        # Feed mulligan_audit -- only the primary-controlled seat counts.
        for ev in fwd_logs:
            ctx.collected_game_logs.append(ev)
            ctx.collected_deck_league.append({0: ("primary", a), 1: ("training", b)})
        for ev in rev_logs:
            ctx.collected_game_logs.append(ev)
            ctx.collected_deck_league.append({0: ("training", b), 1: ("primary", a)})

    elapsed_ms = (time.time() - t0) * 1000
    burn_summary = {f"{side}/{name}": {"games": n, "mana_burnt_total_per_game": total / n,
                                       "mana_burnt_total_single_pip_per_game": total_sp / n}
                    for (side, name), (n, total, total_sp) in burn_by_deck.items()}
    payload = {"check": NAME, "primary_league": ctx.primary_league_name,
               "training_league": ctx.training_league_name,
               "primary_roster": primary_decks, "training_roster": training_roster,
               "games_per_pairing": ctx.games_per_check, "seed": ctx.seed,
               "cumulative_games": ctx.cumulative_games, "elapsed_ms": elapsed_ms,
               "pairings": pairing_results, "mana_burnt_by_deck": burn_summary,
               "primary_totals": {d: {"wins": t["wins"], "games": t["games"],
                                      "win_rate": t["wins"] / t["games"] if t["games"] else None}
                                 for d, t in primary_totals.items()}}
    write_path = _common.write_league_json(ctx, NAME, payload)

    for d in primary_decks:
        t = primary_totals[d]
        wr = t["wins"] / t["games"] if t["games"] else float("nan")
        _common.append_metric(ctx, kind=NAME, deck=d, games=t["games"], wins=t["wins"], win_rate=wr)

    return {"pairings": len(pairing_results), "games": sum(r["games"] for r in pairing_results),
            "elapsed_ms": elapsed_ms, "wrote": write_path}
