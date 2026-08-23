"""Check: vs_history -- each deck's current live net vs. its own frozen old
self (the oldest still-active snapshot, and once any exist, the oldest
archived one). The one check immune to the "everyone is improving together"
confound the round-robin checks both have, since the opponent here can never
change -- any win-rate movement is unambiguously this deck's own progress.
Per deck plus a league-wide rollup.

Absorbed from rl.league.league_runner._run_eval_vs_history (previously
triggered automatically inside _run_session every eval_every_sessions
sessions; now runs only at this pipeline's cadence instead)."""
from rl.league.league_runner import HORIZON, _run_eval_vs_history

from . import _common

NAME = "vs_history"


def run(ctx):
    per_deck = {}
    for name in ctx.train_decks:
        net, mnet = _common.load_deck_net(ctx.primary_league_dir, name, ctx.vocab, ctx.fixed_tables[name])
        results = _run_eval_vs_history(name, net, mnet, ctx.deck_ctxs[name], ctx.decklists[name],
                                       ctx.primary_league_dir, HORIZON, games_per_snapshot=ctx.games_per_check,
                                       seed=ctx.seed)
        per_deck[name] = results
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
