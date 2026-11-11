"""Check: primary_vs_training_round_robin -- the FULL cross product between
the primary league's roster and the training league's roster (deck_a x
deck_b, mismatched names included, mirror-across-leagues where names match
is just one cell of the matrix) -- not just same-name pairs the way the old
per-session vs_gauntlet check did. ctx.games_per_check games per pairing,
greedy, on current live weights on both sides. No-op (skipped) if no
training league is configured, or it has no live.pt for any deck yet.

Plays through evaluation.play_matchups (one way of playing agents against
each other, executor-independent seeding); the primary deck is always seat 0
and who starts is each game's own coin flip. Mana-burn reporting and
mulligan_audit's shared game-log pool (primary-controlled seat only) are read
off the games' event logs.
"""
import itertools
import random
import time

from evaluation import AgentRef, Matchup, play_matchups
from evaluation.matchups import available_decks

from . import _common

NAME = "primary_vs_training_round_robin"


def _burn(over, seat, key):
    return sum(e[key][seat] for e in over)


def run(ctx):
    if ctx.training_league_dir is None:
        return {"skipped": "no training league configured"}
    training_roster = available_decks(ctx.training_league_dir)
    if not training_roster:
        return {"skipped": "training league has no live.pt for any deck yet"}

    primary_decks = ctx.train_decks
    pairs = list(itertools.product(primary_decks, training_roster))
    seed = random.randrange(2 ** 31) if ctx.seed is None else ctx.seed

    t0 = time.time()
    results = play_matchups(
        [Matchup(AgentRef(a, ctx.primary_league_dir), AgentRef(b, ctx.training_league_dir)) for a, b in pairs],
        ctx.games_per_check, seed=seed, greedy=True,
        executor=ctx.executor if ctx.n_workers > 1 else None, collect_logs=True)

    pairing_results = []
    primary_totals = {d: {"wins": 0, "games": 0} for d in primary_decks}
    burn_by_deck = {}  # ("primary"|"training", name) -> [games, mana_burnt_total, mana_burnt_total_single_pip]

    def _record_burn(side, name, n, total, total_single_pip):
        acc = burn_by_deck.setdefault((side, name), [0, 0, 0])
        acc[0] += n
        acc[1] += total
        acc[2] += total_single_pip

    for (a, b), r in zip(pairs, results):
        over = [e for ev in r.logs for e in ev if e["kind"] == "game_over"]
        a_burnt, b_burnt = _burn(over, 0, "mana_burnt_total"), _burn(over, 1, "mana_burnt_total")
        a_sp, b_sp = _burn(over, 0, "mana_burnt_total_single_pip"), _burn(over, 1, "mana_burnt_total_single_pip")
        pairing_results.append({"primary_deck": a, "training_deck": b, "games": r.games,
                                "primary_wins": r.a_wins, "training_wins": r.b_wins, "no_winner": r.no_winner,
                                "primary_mana_burnt_total": a_burnt, "training_mana_burnt_total": b_burnt})
        _record_burn("primary", a, r.games, a_burnt, a_sp)
        _record_burn("training", b, r.games, b_burnt, b_sp)
        primary_totals[a]["wins"] += r.a_wins
        primary_totals[a]["games"] += r.games

        # Feed mulligan_audit -- only the primary-controlled seat counts.
        for ev in r.logs:
            ctx.collected_game_logs.append(ev)
            ctx.collected_deck_league.append({0: ("primary", a), 1: ("training", b)})

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
