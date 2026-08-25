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

from rl.league.league_runner import HORIZON, league_roster
from rl.training.train import _constant_pairing, collect_rollout

from . import _common

NAME = "primary_vs_training_round_robin"


def _half(agents, decks, n_games, seed):
    """One seat orientation: agents[0]/decks[0] at seat 0. Returns the raw
    per-game event logs (every event, not just game_over -- mulligan_audit
    needs the draw/mulligan events too) plus the count played."""
    pairing = _constant_pairing(agents, decks, [None, None], [None, None])
    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, n_games, HORIZON, random.Random(seed),
                                           device="cpu", record=False, greedy=True, game_logs=game_logs)
    return game_logs, played


def run(ctx):
    if ctx.training_league_dir is None:
        return {"skipped": "no training league configured"}
    training_roster = league_roster(ctx.training_league_dir)
    if not training_roster:
        return {"skipped": "training league has no live.pt for any deck yet"}

    primary_decks = ctx.train_decks
    primary_agents = {name: _common.load_agent(ctx.primary_league_dir, name, ctx.vocab, ctx.deck_ctxs[name],
                                               ctx.fixed_tables[name])
                     for name in primary_decks}
    training_agents = {name: _common.load_agent(ctx.training_league_dir, name, ctx.vocab, ctx.deck_ctxs[name],
                                                ctx.fixed_tables[name])
                      for name in training_roster}

    rng = random.Random(ctx.seed)
    pairing_results = []
    primary_totals = {d: {"wins": 0, "games": 0} for d in primary_decks}
    burn_by_deck = {}  # ("primary"|"training", name) -> [games, mana_burnt_total, mana_burnt_total_single_pip]

    def _record_burn(side, name, n, total, total_single_pip):
        acc = burn_by_deck.setdefault((side, name), [0, 0, 0])
        acc[0] += n
        acc[1] += total
        acc[2] += total_single_pip

    t0 = time.time()
    for a, b in itertools.product(primary_decks, training_roster):
        agent_a, agent_b = primary_agents[a], training_agents[b]
        pair_seed = rng.randrange(2 ** 31)
        half = ctx.games_per_check // 2
        fwd_logs, played_f = _half([agent_a, agent_b], [ctx.decklists[a], ctx.decklists[b]], half, pair_seed)
        rev_logs, played_r = _half([agent_b, agent_a], [ctx.decklists[b], ctx.decklists[a]],
                                   ctx.games_per_check - half, pair_seed)
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
        _record_burn("primary", a, played, a_burnt, a_burnt_sp)
        _record_burn("training", b, played, b_burnt, b_burnt_sp)

        pairing_results.append({"primary_deck": a, "training_deck": b, "games": played,
                                "primary_wins": a_wins, "training_wins": b_wins, "no_winner": no_winner,
                                "primary_mana_burnt_total": a_burnt, "training_mana_burnt_total": b_burnt})
        primary_totals[a]["wins"] += a_wins
        primary_totals[a]["games"] += played

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
