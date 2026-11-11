"""Check: primary_vs_primary_round_robin -- every training deck in the
primary league plays every other, mirrors included, ctx.games_per_check
games per pairing, greedy, on current live weights. A deck's win-rate total
is its record against the OTHER decks (mirrors excluded, see
deck_win_totals); the mirror games are still played and still listed in
"pairings".

Reuses evaluation.roundrobin.run_round_robin (the same round-robin-with-mirrors
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

from evaluation.roundrobin import run_round_robin

from . import _common

NAME = "primary_vs_primary_round_robin"


def deck_win_totals(decks, by_pairing):
    """Each deck's record against the OTHER decks: {deck: {"wins", "games"}}.

    Mirror pairings (a == b) are left out. A mirror game has no winner from the deck's point of view -- one side of
    it always wins, so counting it for the deck made every deck's win rate carry a guaranteed win in 1 of every
    len(decks) pairings (a flat +1/len(decks) offset, and a pooled win rate that was the same constant at every
    cadence point). The mirror's own results stay in the output's "pairings"; metrics lines written before this
    fix (games = len(decks) * games_per_pairing, win_rate inflated by that offset) are not comparable with ones
    written after (games = (len(decks) - 1) * games_per_pairing)."""
    totals = {d: {"wins": 0, "games": 0} for d in decks}
    for (a, b), p in by_pairing.items():
        if a == b:
            continue
        totals[a]["wins"] += p["a_wins"]
        totals[a]["games"] += p["games"]
        totals[b]["wins"] += p["b_wins"]
        totals[b]["games"] += p["games"]
    return totals


def run(ctx):
    game_logs = []
    t0 = time.time()
    decks, game_pairings = run_round_robin(ctx.primary_league_dir, ctx.train_decks, ctx.games_per_check, greedy=True,
                                           seed=ctx.seed, game_logs=game_logs, executor=ctx.executor,
                                           n_workers=ctx.n_workers)
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

    pairing_results = [{"deck_a": a, "deck_b": b, **p} for (a, b), p in by_pairing.items()]
    win_totals = deck_win_totals(decks, by_pairing)

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
