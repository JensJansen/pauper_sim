"""Check: vs_heuristic -- each configured deck's current live net vs.
rl.decision.heuristic_agent.HeuristicAgent, a fixed, non-learned, hand-
scripted opponent. The cleanest regression alarm available: since the
opponent never changes and was never trained, any drop in this win rate is
unambiguously the policy breaking, not the opponent improving. Only runs
for ctx.heuristic_decks (most decks have no owner-authored heuristic).

Absorbed from rl.league.league_runner._run_eval_vs_heuristic (previously
triggered automatically inside _run_session; now runs only at this
pipeline's cadence instead)."""
from rl.league.league_runner import HORIZON, _run_eval_vs_heuristic

from . import _common

NAME = "vs_heuristic"


def run(ctx):
    decks = [d for d in ctx.heuristic_decks if d in ctx.train_decks]
    if not decks:
        return {"skipped": "no heuristic_decks configured (or none in this run's roster)"}

    per_deck = {}
    for name in decks:
        net, mnet = _common.load_deck_net(ctx.primary_league_dir, name, ctx.vocab, ctx.fixed_tables[name])
        r = _run_eval_vs_heuristic(name, net, mnet, ctx.deck_ctxs[name], ctx.decklists[name], HORIZON,
                                   games=ctx.games_per_check, seed=ctx.seed)
        wr = r["live_wins"] / r["games"] if r["games"] else float("nan")
        per_deck[name] = {**r, "win_rate": wr}
        payload = {"check": NAME, "primary_league": ctx.primary_league_name, "deck": name,
                  "cumulative_games": ctx.cumulative_games, **per_deck[name]}
        _common.write_deck_json(ctx, name, NAME, payload)
        _common.append_metric(ctx, kind=NAME, deck=name, games=r["games"], live_wins=r["live_wins"], win_rate=wr)

    league_payload = {"check": NAME, "primary_league": ctx.primary_league_name,
                      "cumulative_games": ctx.cumulative_games, "decks": per_deck}
    write_path = _common.write_league_json(ctx, NAME, league_payload)

    return {"decks": len(per_deck), "wrote": write_path}
