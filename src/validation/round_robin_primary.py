"""Check: primary_vs_primary_round_robin -- every training deck in the
primary league plays every other, mirrors included, ctx.games_per_check
games per pairing, greedy, on current live weights.

Reuses rl.league.league_runner._run_eval (the same round-robin-with-mirrors
core run_league.py --eval exposes standalone for a manual --matchup check)
rather than re-implementing the pairing/collection loop. The check's own
output file embeds every one of those games' full event log (payload
"games", same shape rl.league.league_runner._write_event_log/--log use)
alongside the aggregate win/loss summary, so the SAME
checks/primary_vs_primary_round_robin_<N>games.json write_league_json
already produces is directly openable in the webapp's replay viewer -- no
separate file. See write_league_json's own docstring on why that embedded
"games" key is excluded from the small, git-committed webapp mirror
(/stats never reads it, and a full round robin's games can be hundreds of
MB) while staying in the real, local checkpoints/ copy untouched.
"""
import time
from collections import defaultdict

from rl.league.league_runner import _run_eval

from . import _common

NAME = "primary_vs_primary_round_robin"


def run(ctx):
    game_logs = []
    t0 = time.time()
    decks, game_pairings = _run_eval(ctx.train_decks, ctx.games_per_check, greedy=True, seed=ctx.seed,
                                     game_logs=game_logs, league_dir=ctx.primary_league_dir,
                                     executor=ctx.executor, n_workers=ctx.n_workers)
    elapsed_ms = (time.time() - t0) * 1000

    by_pairing = defaultdict(lambda: {"games": 0, "a_wins": 0, "b_wins": 0, "no_winner": 0})
    for ev, (a, b) in zip(game_logs, game_pairings):
        outcome = next((e for e in ev if e["kind"] == "game_over"), None)
        p = by_pairing[(a, b)]
        p["games"] += 1
        if outcome is None:
            p["no_winner"] += 1
        elif outcome["winner"] == 0:  # seat 0 = deck a, seat 1 = deck b, fixed for the whole pairing
            p["a_wins"] += 1
        else:
            p["b_wins"] += 1

        # Feed mulligan_audit -- every seat here is a primary-league deck.
        ctx.collected_game_logs.append(ev)
        ctx.collected_deck_league.append({0: ("primary", a), 1: ("primary", b)})

    win_totals = {d: {"wins": 0, "games": 0} for d in decks}
    pairing_results = []
    for (a, b), p in by_pairing.items():
        pairing_results.append({"deck_a": a, "deck_b": b, **p})
        if a == b:  # mirror: both sides are the same deck, count games once
            win_totals[a]["wins"] += p["a_wins"] + p["b_wins"]
            win_totals[a]["games"] += p["games"]
        else:
            win_totals[a]["wins"] += p["a_wins"]
            win_totals[a]["games"] += p["games"]
            win_totals[b]["wins"] += p["b_wins"]
            win_totals[b]["games"] += p["games"]

    # Same shape rl.league.league_runner._write_event_log produces (and
    # replay_engine.list_games/reduce_game expect): one entry per game, its
    # own deck_a/deck_b stamped on from game_pairings (aligned 1:1 with
    # game_logs -- see the zip() above) so a many-pairing round-robin log is
    # self-describing.
    games = [{"game_index": i, "deck_a": a, "deck_b": b, "events": ev}
             for i, (ev, (a, b)) in enumerate(zip(game_logs, game_pairings))]

    payload = {"check": NAME, "primary_league": ctx.primary_league_name, "decks": decks,
               "games_per_pairing": ctx.games_per_check, "seed": ctx.seed,
               "cumulative_games": ctx.cumulative_games, "elapsed_ms": elapsed_ms,
               "pairings": pairing_results,
               "deck_totals": {d: {"wins": t["wins"], "games": t["games"],
                                   "win_rate": t["wins"] / t["games"] if t["games"] else None}
                              for d, t in win_totals.items()},
               "games": games}
    write_path = _common.write_league_json(ctx, NAME, payload)

    for d in decks:
        t = win_totals[d]
        wr = t["wins"] / t["games"] if t["games"] else float("nan")
        _common.append_metric(ctx, kind=NAME, deck=d, games=t["games"], wins=t["wins"], win_rate=wr)

    return {"pairings": len(by_pairing), "games": len(game_logs), "elapsed_ms": elapsed_ms, "wrote": write_path}
