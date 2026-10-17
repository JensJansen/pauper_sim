"""Benchmark: DISTRIBUTED_TRAINING_PLAN.md gap #14 -- how much does
node_agent.py's own pattern (call _run_session repeatedly, in small
increments, rather than once with a large n_iterations) actually cost?

_run_session's setup section rebuilds every deck's net from disk and
rescans LeaguePool's snapshot directories fresh on EVERY call, uncached
across calls -- see league_runner.py's own _run_session docstring. This
isolates that marginal per-call cost from the actual game-collection cost:
runs the SAME total games/deck two ways -- once as `n_iterations=N` in a
single _run_session call, once as N separate calls of `n_iterations=1` each
(pointed at the SAME progressively-updated league_dir, exactly how
node_agent.py's own loop drives it) -- and reports the difference as the
real marginal cost of granularity.

    python src/benchmarking/distributed_call_granularity.py \
        [--games-per-iter N] [--n-calls N] [--roster A,B,...] [--seed 0]
"""
import argparse
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import _common as bench  # noqa: F401 -- sets sys.path + chdir for build_pool()'s relative paths
from rl.league import league_runner


def _run_as_one_call(n_calls, gpi, seed, train_decks, n_workers):
    # roster=None (the full 11-deck manifest), train_decks scoped to the
    # one deck being trained -- exactly node_agent.py's own call shape.
    # Restricting roster instead would undercount the real setup cost: a
    # real node's every _run_session call rebuilds ALL 11 decks' nets for
    # the opponent pool, not just the one it trains.
    league_dir = tempfile.mkdtemp(prefix="bench_granularity_onecall_")
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            t0 = time.perf_counter()
            league_runner._run_session(n_calls, gpi, snapshot_every=10_000, executor=ex, n_workers=n_workers,
                                       league_dir=league_dir, seed=seed, roster=None, train_decks=train_decks)
            return time.perf_counter() - t0
    finally:
        shutil.rmtree(league_dir, ignore_errors=True)


def _run_as_many_calls(n_calls, gpi, seed, train_decks, n_workers):
    league_dir = tempfile.mkdtemp(prefix="bench_granularity_manycalls_")
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            t0 = time.perf_counter()
            progress = 0
            for i in range(n_calls):
                league_runner._run_session(1, gpi, snapshot_every=10_000, executor=ex, n_workers=n_workers,
                                           league_dir=league_dir, seed=seed, roster=None, train_decks=train_decks,
                                           cumulative_games=progress)
                progress += gpi
            return time.perf_counter() - t0
    finally:
        shutil.rmtree(league_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--n-calls", type=int, default=10)
    ap.add_argument("--train-decks", default="elves", help="Deck(s) actually trained -- opponent pool is always the full manifest, matching node_agent.py.")
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_decks = args.train_decks.split(",")

    one = _run_as_one_call(args.n_calls, args.games_per_iter, args.seed, train_decks, args.n_workers)
    many = _run_as_many_calls(args.n_calls, args.games_per_iter, args.seed, train_decks, args.n_workers)

    total_games = args.n_calls * args.games_per_iter
    marginal_per_call = (many - one) / (args.n_calls - 1) if args.n_calls > 1 else float("nan")
    print(f"\n{total_games} games/deck, {args.n_calls} calls, train_decks={train_decks} (opponent pool: full manifest)")
    print(f"  one _run_session(n_iterations={args.n_calls}) call:  {one*1000:8.1f}ms total")
    print(f"  {args.n_calls} separate _run_session(n_iterations=1) calls: {many*1000:8.1f}ms total")
    print(f"  marginal per-call setup overhead: {marginal_per_call*1000:.1f}ms")
    print(f"  ({100*(many-one)/one:.0f}% slower running in {args.n_calls}x smaller increments)")


if __name__ == "__main__":
    main()
