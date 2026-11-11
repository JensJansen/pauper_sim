"""Check: strength_history -- how strong the CURRENT agents are against their own training history, and against every
deck's history. Two views of one question ("is training still paying?"), both read off the league's archived
snapshots (evaluation/history.py), both greedy, both paired across anchors (evaluation.play.matchup_seed).

  own    each deck's current agent against ITS OWN snapshots at the anchor ages in `ladder`: the steady-improvement
         curve. Against ever-older selves the score should be well above 50 and rise with the age gap; against
         recent ones it sinks toward 50. Anchors far older than the agent saturate (near 100) and carry no
         information; the informative band is the recent ones.
  field  each deck's current agent against every OTHER deck's snapshots at the same anchor ages (A_now vs B@age). As
         B ages its curve falls to where B stopped improving, so it reads B's learning through a fixed yardstick, and
         shows whether progress generalises beyond a deck's own history.

The anchors are the ladder rungs (default 1k, 5k, 10k, 20k, 30k, 40k, 50k, 75k, 100k, 150k, ... games/deck) at
least two snapshot intervals older than the current point, mapped to the nearest snapshot. Unlike a
round-robin, the cost per cadence point grows with the number of rungs, not quadratically with the history.

ctx fields read: games_per_snapshot (required), history_ladder, history_games (games per `own` cell, default 100;
the cell progress_indicator reads, each deck's reference anchor, gets REF_GAMES_FACTOR times that), field_games (games
per `field` cell, default 40). Report only: nothing here changes training."""
from evaluation import AgentRef, Matchup, Tally, history, play_matchups, wilson_interval
from evaluation.verdicts import REF_GAMES_FACTOR, ref_games

from . import _common

NAME = "strength_history"
DEFAULT_LADDER = (1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000, 300000, 400000,
                  500000)


def _score(rows):
    t = Tally()
    for r in rows:
        t += Tally(r["wins"], r["no_winner"], r["games"])
    return t


def run(ctx):
    if ctx.games_per_snapshot is None:
        return {"skipped": "ctx.games_per_snapshot not set"}
    league, decks = ctx.primary_league_dir, list(ctx.train_decks)
    ladder = ctx.history_ladder or DEFAULT_LADDER
    points = {d: history.ladder_points(league, d, ctx.cumulative_games, ladder, ctx.games_per_snapshot) for d in decks}
    if not any(points.values()):
        return {"skipped": f"no anchor is old enough at {ctx.cumulative_games} games/deck"}

    current = {d: AgentRef(d, league, "current") for d in decks}
    own = [(Matchup(current[d], AgentRef(d, league, f"@{g}", sid)), d, g) for d in decks for g, sid in points[d]]
    field = [(Matchup(current[a], AgentRef(b, league, f"@{g}", sid)), a, b, g)
             for b in decks for g, sid in points[b] for a in decks if a != b]
    seed = 0 if ctx.seed is None else ctx.seed
    # The cell progress_indicator reads (each deck's reference anchor) gets REF_GAMES_FACTOR x the games: a label
    # taken from 4 decks x 100 games flickers on noise alone.
    refs = {d: ref_games([g for g, _ in points[d]], ctx.cumulative_games) for d in decks}
    boosted = [i for i, (_m, d, g) in enumerate(own) if g == refs[d]]
    own_results = [None] * len(own)
    for indices, games in (([i for i in range(len(own)) if i not in set(boosted)], ctx.history_games),
                           (boosted, ctx.history_games * REF_GAMES_FACTOR)):
        for i, r in zip(indices, play_matchups([own[i][0] for i in indices], games, seed=seed, executor=ctx.executor)
                        if indices else []):
            own_results[i] = r
    field_results = play_matchups([m for m, *_ in field], ctx.field_games, seed=seed + 1, executor=ctx.executor)

    def row(r, **keys):
        return {**keys, "games": r.games, "wins": r.a_wins, "no_winner": r.no_winner}
    own_rows = [row(r, deck=d, anchor_games=g) for (_m, d, g), r in zip(own, own_results)]
    field_rows = [row(r, deck_a=a, deck_b=b, anchor_games=g) for (_m, a, b, g), r in zip(field, field_results)]

    payload = {"check": NAME, "primary_league": ctx.primary_league_name, "cumulative_games": ctx.cumulative_games,
               "history_games": ctx.history_games, "field_games": ctx.field_games, "own": own_rows, "field": field_rows}
    write_path = _common.write_league_json(ctx, NAME, payload)
    for d in decks:
        for r in (r for r in own_rows if r["deck"] == d):
            _common.append_metric(ctx, kind=NAME, deck=d, anchor_games=r["anchor_games"], games=r["games"],
                                  score=_score([r]).score)
    for b in decks:  # per opponent deck, pooled over the other decks' current agents
        for g in sorted({r["anchor_games"] for r in field_rows if r["deck_b"] == b}):
            rs = [r for r in field_rows if r["deck_b"] == b and r["anchor_games"] == g]
            _common.append_metric(ctx, kind="strength_field", deck=b, anchor_games=g, games=sum(r["games"] for r in rs),
                                  score=_score(rs).score)

    ctx.extras[NAME] = {"own": own_rows, "field": field_rows}
    ages = sorted({r["anchor_games"] for r in own_rows})
    pooled = {g: _score([r for r in own_rows if r["anchor_games"] == g]) for g in ages}
    return {"anchors": len(ages),
            "own_vs_anchor": {g: f"{100 * t.score:.0f}+-{100 * (wilson_interval(t)[1] - wilson_interval(t)[0]) / 2:.0f}"
                              for g, t in pooled.items()},
            "wrote": write_path}
