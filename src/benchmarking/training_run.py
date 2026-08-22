"""Benchmark: the REAL league training loop under different collection configs.

Drives rl.league.league_runner._run_session directly -- the EXACT training sequence
(all decks, collect + ppo_update, the batch-size schedule, session-end
checkpointing) over a throwaway checkpoint dir, so every number here is a
true training session's per-iteration cost, not an isolated function
microbench. Nothing has to be done to get untrained weights: with per-deck
encoders a league with no live.pt on disk starts from freshly-initialized
nets, which is exactly the benchmark condition (there used to be a
fresh_stack=True flag for this, back when the alternative was loading a
pretrained frozen shared stack).

Each config runs with the SAME seed (identical fresh weights + shuffles) over
its own throwaway checkpoint dir, so they are comparable. _run_session prints
its own total time + collect/update split per config; this harness adds a
cross-config wall-clock summary.

Configs:
  seq        1 process, batch-of-1 collection (the sequential path)
  mp<N>      N worker processes (collect_rollout_league_parallel)

    python src/benchmarking/training_run.py [--iterations N] [--games-per-iter N] \
        [--configs seq,mp6] [--seed 0] [--snapshot-every N] [--roster A,B,...]
"""

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import _common as bench  # noqa: F401 -- sets sys.path + chdir so build_pool()'s relative paths resolve like the real script
from rl.league import league_runner


def _parse_config(name):
    """'seq' | 'mp<N>' -> n_worker_processes_or_None."""
    if name == "seq":
        return None
    if name.startswith("mp"):
        return int(name[2:])
    raise SystemExit(f"unknown config {name!r} (want seq | mp<N>)")


def _run_config(name, iters, gpi, snapshot_every, seed, roster):
    workers = _parse_config(name)
    import tempfile
    league_dir = tempfile.mkdtemp(prefix=f"bench_train_{name}_")
    kwargs = dict(league_dir=league_dir, seed=seed, roster=roster)
    t0 = time.perf_counter()
    try:
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                league_runner._run_session(iters, gpi, snapshot_every, ex, workers, **kwargs)
        else:
            league_runner._run_session(iters, gpi, snapshot_every, None, 1, **kwargs)
    finally:
        shutil.rmtree(league_dir, ignore_errors=True)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--configs", type=str, default="seq,mp6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=20, help="Real league default; won't fire in a short bench.")
    ap.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                     help="Restrict the benchmark to this comma-separated sub-roster (a true isolated sub-league, "
                          "same as run_league.py's --roster) instead of the full deck pool.")
    args = ap.parse_args()

    roster = args.roster.split(",") if args.roster else None
    print(f"true-training-loop benchmark: fresh (untrained) identical-config stack, seed={args.seed}, "
          f"iterations={args.iterations} games/iter={args.games_per_iter}"
          f"{' roster=' + str(roster) if roster else ''}")
    results = {}
    for name in args.configs.split(","):
        print(f"\n########## config: {name} ##########", flush=True)
        results[name] = _run_config(name, args.iterations, args.games_per_iter, args.snapshot_every,
                                    args.seed, roster)

    print("\n===== SUMMARY (total wall-clock per config, whole training session) =====")
    base = results.get("seq")
    for name, secs in results.items():
        rel = f"  ({base / secs:.2f}x vs seq)" if base and secs else ""
        print(f"  {name:>10}: {secs:7.1f}s{rel}")
    print("\nnote: MP config's per-worker sampling is not fully seeded, so its games differ slightly; "
          "raise --iterations/--games-per-iter to average out cross-config game-length noise.")


if __name__ == "__main__":
    main()
