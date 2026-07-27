"""Shared setup for the benchmarking scripts.

Imported FIRST by every benchmark (before any rl.*/game import) so it can
(1) put src/ on sys.path and (2) chdir into src/, making the `../data` and
`../checkpoints` relative paths every rl.* module already uses resolve the
same way they do for run_league.py -- no matter which directory the
benchmark was launched from. Only after that are rl.* importable, so the
train primitives are re-exported from here and scripts import ONLY this
module (no import-order fragility).

Run any benchmark from anywhere, e.g.:  python src/benchmarking/mp_scaling.py
"""

import os
import statistics
import sys
import tempfile
import time
import types

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_SRC)  # so build_pool()'s ../data and ../checkpoints resolve like run_league.py's

# Line-buffer stdout so every print() flushes at its newline even when output is
# redirected to a file (block-buffered by default off a TTY -- benchmark progress
# would otherwise stay invisible until the process exits). Covers every script
# that imports _common. hasattr guard: a wrapped/captured stdout may lack it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch  # noqa: E402

from rl.arch import SetTransformer  # noqa: E402
from rl.deck import DeckNetwork  # noqa: E402
from rl.league import LeaguePool  # noqa: E402
from rl.pool import build_pool  # noqa: E402
from rl.rewards import action_count_win_reward_200_floor02  # noqa: E402
# Re-exported so scripts import only _common (no rl.* import => no ordering trap).
from rl.train import (  # noqa: E402,F401
    collect_rollout_league, collect_rollout_league_batched, collect_rollout_league_parallel, ppo_update,
)

# Mirror run_league.py's real training config so benchmarks measure the real thing.
D_MODEL = 64
SHARED_HPARAMS = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}
HORIZON = 120
REWARD_FN = action_count_win_reward_200_floor02
REWARD_FN_NAME = "action_count_win_reward_200_floor02"


def build_setup():
    """Everything the benchmarks share: the real pool + one DeckNetwork per
    deck over ONE shared (random-init, frozen) stack, each deck's Adam, and a
    LeaguePool rooted at a throwaway temp dir. No checkpoints needed -- weights
    don't affect wall-clock, and with zero snapshots the pool samples only live
    nets/self, which is all the timing benchmarks exercise."""
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = SetTransformer(vocab.size, **SHARED_HPARAMS)
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()

    live_nets, optimizers = {}, {}
    for name in decklists:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_nets[name] = net
        optimizers[name] = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-4)

    league_dir = tempfile.mkdtemp(prefix="bench_league_")
    pool = LeaguePool(league_dir, list(decklists))
    return types.SimpleNamespace(
        decklists=decklists, vocab=vocab, deck_ctxs=deck_ctxs, fixed_tables=fixed_tables,
        shared=shared, live_nets=live_nets, optimizers=optimizers, pool=pool,
        league_dir=league_dir, deck_names=list(decklists),
    )


def median_time(fn, reps=5, cuda_sync=False):
    """Median wall-clock seconds over `reps` calls. cuda_sync: bracket each
    call with torch.cuda.synchronize() so async GPU kernels are actually
    finished before the clock stops (otherwise CUDA timings are meaningless)."""
    times = []
    for _ in range(reps):
        if cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)
