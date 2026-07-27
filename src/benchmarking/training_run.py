"""Benchmark: the REAL league training loop under different collection configs.

Drives run_league._run_session directly -- the EXACT training sequence (all
decks, collect + ppo_update, the batch-size schedule, session-end
checkpointing) -- with the ONLY difference being a fresh, UNTRAINED shared stack
of identical config (fresh_stack=True) instead of the loaded trained one. So
every number here is a true training session's per-iteration cost, not an
isolated function microbench.

Each config runs with the SAME seed (identical fresh weights + shuffles) over
its own throwaway checkpoint dir, so they are comparable. _run_session prints
its own total time + collect/update split per config; this harness adds a
cross-config wall-clock summary.

Configs:
  seq        1 process, batch-of-1 collection (the sequential path)
  mp<N>      N worker processes (collect_rollout_league_parallel)

    python src/benchmarking/training_run.py [--iterations N] [--games-per-iter N] \
        [--configs seq,mp6] [--seed 0] [--snapshot-every N]
"""

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import _common as bench  # noqa: F401 -- sets sys.path + chdir so `import run_league` resolves like the real script
import run_league


def _parse_config(name):
    """'seq' | 'mp<N>' -> n_worker_processes_or_None."""
    if name == "seq":
        return None
    if name.startswith("mp"):
        return int(name[2:])
    raise SystemExit(f"unknown config {name!r} (want seq | mp<N>)")


def _run_config(name, iters, gpi, snapshot_every, seed):
    workers = _parse_config(name)
    import tempfile
    league_dir = tempfile.mkdtemp(prefix=f"bench_train_{name}_")
    kwargs = dict(fresh_stack=True, league_dir=league_dir, seed=seed)
    t0 = time.perf_counter()
    try:
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                run_league._run_session(iters, gpi, snapshot_every, ex, workers, **kwargs)
        else:
            run_league._run_session(iters, gpi, snapshot_every, None, 1, **kwargs)
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
    args = ap.parse_args()

    print(f"true-training-loop benchmark: fresh (untrained) identical-config stack, seed={args.seed}, "
          f"iterations={args.iterations} games/iter={args.games_per_iter}")
    results = {}
    for name in args.configs.split(","):
        print(f"\n########## config: {name} ##########", flush=True)
        results[name] = _run_config(name, args.iterations, args.games_per_iter, args.snapshot_every,
                                    args.seed)

    print("\n===== SUMMARY (total wall-clock per config, whole training session) =====")
    base = results.get("seq")
    for name, secs in results.items():
        rel = f"  ({base / secs:.2f}x vs seq)" if base and secs else ""
        print(f"  {name:>10}: {secs:7.1f}s{rel}")
    print("\nnote: MP config's per-worker sampling is not fully seeded, so its games differ slightly; "
          "raise --iterations/--games-per-iter to average out cross-config game-length noise.")


if __name__ == "__main__":
    main()
