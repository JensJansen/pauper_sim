"""A league's current agents, round-robin: the shape run_league.py --eval/--matchup and the validation round robins
use (every pair of decks, mirrors included, over the live weights in one league directory), built on play_matchups so
there is one way of playing agents against each other."""
import itertools
import random
import time

from rl.league import league_runner

from . import play
from .matchups import AgentRef, Matchup


def run_round_robin(league_dir, eval_decks, games_per_pairing, greedy=True, seed=None, game_logs=None, matchup=None,
                    executor=None, n_workers=1):
    """Plays games with no training over the current live agents (deck net + its mulligan net) in `league_dir`
    (default: the run_league.py league): a round robin with mirrors over `eval_decks` (default: every deck in the
    pool), or the single pairing `matchup` = (deck_a, deck_b). Greedy (argmax) by default; greedy=False samples.

    game_logs: a list to extend with one engine event log per game, pairing by pairing. Returns (the decks
    played, per-game (deck_a, deck_b) aligned 1:1 with game_logs -- or None when game_logs is None).

    executor / n_workers: an existing ProcessPoolExecutor with n_workers workers to spread the games over (a
    single `matchup` always runs in-process). Results depend only on (decks, games, seed, greedy), never on the
    executor: each pairing's games are seeded from (seed, the two deck names, chunk) -- see play.matchup_seed.
    seed=None draws a fresh one."""
    league_dir = league_dir or league_runner.LEAGUE_DIR
    deck_names = list(play._pool()[0])
    decks = list(matchup) if matchup else (list(eval_decks) if eval_decks else deck_names)
    assert set(decks) <= set(deck_names), f"eval decks {decks} not all in roster {deck_names}"
    pairs = [tuple(matchup)] if matchup else list(itertools.combinations_with_replacement(decks, 2))
    refs = {d: AgentRef(d, league_dir) for d in set(decks)}
    seed = random.randrange(2 ** 31) if seed is None else seed
    pool = executor if executor is not None and n_workers > 1 and matchup is None else None

    print(f"Eval{' (parallel, ' + str(n_workers) + ' workers)' if pool else ''}: {len(pairs)} pairing(s) x "
          f"{games_per_pairing} games ({'greedy' if greedy else 'sampled'}, seed={seed}) over decks={decks}")
    t0 = time.time()
    results = play.play_matchups([Matchup(refs[a], refs[b]) for a, b in pairs], games_per_pairing, seed=seed,
                                 greedy=greedy, executor=pool, collect_logs=game_logs is not None)
    game_pairings = None
    if game_logs is not None:
        game_pairings = []
        for (a, b), r in zip(pairs, results):
            game_logs.extend(r.logs)
            game_pairings.extend([(a, b)] * len(r.logs))
    print(f"eval done: {sum(r.games for r in results)} games in {(time.time() - t0) * 1000:,.0f}ms")
    return decks, game_pairings
