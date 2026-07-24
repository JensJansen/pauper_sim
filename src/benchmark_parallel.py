"""One-off diagnostic (not part of the training pipeline): wall-clock
comparison of collecting N real league games single-process vs across
multiple worker processes (token_train.collect_rollout_league_parallel).
Per-game cost is allowed to be higher in the parallel case (process/IPC
overhead) -- the only number that matters here is TOTAL wall-clock to
finish N games.

Must run guarded by `if __name__ == "__main__":` -- Windows multiprocessing
uses the "spawn" start method, which re-imports this module in every
worker process; without the guard, each spawned worker would recursively
try to re-run main() and spawn its own workers."""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from rewards import action_count_win_reward_200_floor02
from terminated import never_terminated
from token_arch import SetTransformer
from token_deck import DeckNetwork
from token_league import LeaguePool
from token_pool import build_pool
from token_train import collect_rollout_league, collect_rollout_league_parallel

CHECKPOINT_DIR = "../checkpoints"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
LEAGUE_DIR = f"{CHECKPOINT_DIR}/league"
D_MODEL = 64
N_GAMES = 50
TRAINING_DECK = "mono_red_madness"
REWARD_FN_NAME = "action_count_win_reward_200_floor02"
HORIZON = 120


def load_frozen_stack(vocab_size):
    ckpt = torch.load(FROZEN_STACK, weights_only=True)
    assert ckpt["vocab_size"] == vocab_size
    shared = SetTransformer(vocab_size, d_model=ckpt["d_model"], n_heads=4, n_layers=2, dim_feedforward=128)
    shared.load_state_dict(ckpt["shared"])
    for p in shared.parameters():
        p.requires_grad = False
    shared.eval()
    return shared


def build_live_nets(decklists, fixed_tables, shared):
    live_nets = {}
    for name in decklists:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        live_path = f"{LEAGUE_DIR}/{name}/live.pt"
        if os.path.exists(live_path):
            ckpt = torch.load(live_path, weights_only=True)
            net.load_state_dict(ckpt["net"])
        live_nets[name] = net
    return live_nets


def main():
    torch.set_num_threads(1)  # main process is also just doing the single-process baseline's own rollout work -- keep it apples-to-apples with each worker's own thread cap
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    live_nets = build_live_nets(decklists, fixed_tables, shared)
    shared_hparams = {"d_model": D_MODEL, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}

    print(f"=== Single-process baseline: {N_GAMES} games ===")
    rng = random.Random(0)
    pool = LeaguePool(LEAGUE_DIR, list(decklists))
    t0 = time.time()
    buf, played = collect_rollout_league(
        TRAINING_DECK, live_nets[TRAINING_DECK], deck_ctxs[TRAINING_DECK], decklists[TRAINING_DECK],
        action_count_win_reward_200_floor02, pool, decklists, deck_ctxs, live_nets,
        [never_terminated, never_terminated], HORIZON, N_GAMES, rng, device="cpu",
    )
    single_elapsed = time.time() - t0
    assert played == N_GAMES and len(buf) > 0
    print(f"  {single_elapsed:.2f}s total wall-clock, {single_elapsed / N_GAMES:.3f}s/game, buf={len(buf)} transitions")

    for n_workers in (6, 8, 10, 12):
        print(f"\n=== Parallel: {N_GAMES} games across {n_workers} worker processes ===")
        # ProcessPoolExecutor spawns workers lazily on first submit, not at
        # construction -- so the whole `with` block (not just the
        # collect_rollout_league_parallel call) is what actually pays the
        # spawn+re-import cost; timing the block as a whole is the only
        # honest way to capture that, matching what a real training run
        # would actually experience.
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            buf, played = collect_rollout_league_parallel(
                TRAINING_DECK, live_nets, REWARD_FN_NAME, LEAGUE_DIR, HORIZON, N_GAMES, executor, n_workers, shared_hparams,
            )
        parallel_elapsed = time.time() - t0
        assert played == N_GAMES and len(buf) > 0
        print(f"  {parallel_elapsed:.2f}s total wall-clock, {parallel_elapsed / N_GAMES:.3f}s/game, "
              f"buf={len(buf)} transitions")
        print(f"  speedup vs single-process: {single_elapsed / parallel_elapsed:.2f}x")


if __name__ == "__main__":
    main()
