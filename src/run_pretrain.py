"""Pretrain driver: pretrain the shared perception stack (embeddings +
SetTransformer + FiLM) with a throwaway DeckNetwork per pool deck -- real
mirror self-play games run through rl.train.train_selfplay for EVERY
deck in the roster (data/league_decks.json), each deck in turn every
iteration, so the shared stack sees all decks' worth of real board states
every round rather than one deck exhausted before the next starts.
Gradients from every throwaway net flow into the ONE shared
SetTransformer+FiLM instance -- exactly the "pretrain the shared layers
with junk [per-deck heads], then freeze" design. Generalized over an
arbitrary roster (was hardcoded to 2 decks) so the shared stack's embedding
table + attention actually learn representations for every deck's cards,
not just a hardcoded subset.

Checkpointed after every session (resumable across separate invocations)
so this can be run in small batches per the explicit instruction: start
tiny, watch for crashes/hangs/stalling, scale toward ~1hr/batch once
runs are healthy.

Usage: python run_pretrain.py [n_iterations] [games_per_iteration] [--freeze]
--freeze: after this session's training, freeze the shared stack
(requires_grad=False, eval mode) and write the frozen weights to
../checkpoints/shared_stack_frozen.pt for league training to load -- run
this only once satisfied pretraining is done, not on every intermediate batch.
"""
import os
import random
import sys
import time

import torch

from rl.rewards import action_count_win_reward_200_floor02
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.pool import build_pool
from rl.train import train_selfplay

CHECKPOINT_DIR = "../checkpoints"
CHECKPOINT = f"{CHECKPOINT_DIR}/pretrain_shared_stack.pt"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
D_MODEL = 64


def main():
    n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 1
    games_per_iteration = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 2
    do_freeze = "--freeze" in sys.argv

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    shared = SetTransformer(vocab.size, d_model=D_MODEL, n_heads=4, n_layers=2, dim_feedforward=128)

    # ONE optimizer for the shared stack, reused across EVERY deck's mirror
    # session, plus one optimizer per deck for its own throwaway head's
    # unique params. NOT one Adam per net over net.parameters() -- that
    # would give the shared stack's identical parameter tensors N
    # independent Adam instances with unsynchronized momentum/variance
    # state, stepping on them in alternation (confirmed the hard way, see
    # git history and rl.train.ppo_update's own docstring).
    opt_shared = torch.optim.Adam(shared.parameters(), lr=3e-4)
    nets, head_opts = {}, {}
    for name in deck_names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        nets[name] = net
        head_opts[name] = torch.optim.Adam(
            [p for n, p in net.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4,
        )

    session = 0
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, weights_only=True)
        assert ckpt["vocab_size"] == vocab.size, "vocab changed since last checkpoint -- would silently corrupt the embedding table"
        assert set(ckpt["nets"]) == set(deck_names), "roster changed since last checkpoint -- start a fresh pretrain"
        shared.load_state_dict(ckpt["shared"])
        opt_shared.load_state_dict(ckpt["opt_shared"])
        for name in deck_names:
            nets[name].load_state_dict(ckpt["nets"][name])
            head_opts[name].load_state_dict(ckpt["head_opts"][name])
        session = ckpt["session"] + 1
        print(f"resumed from {CHECKPOINT} (session {session}, vocab {ckpt['vocab_size']}=={vocab.size})")

    rng = random.Random()
    reward_fn = action_count_win_reward_200_floor02
    horizon = 120

    total_games = n_iterations * games_per_iteration * len(deck_names)
    print(f"pretrain session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"decks={deck_names} (={total_games} total games across all pool decks)")
    t0 = time.time()
    for i in range(n_iterations):
        for name in deck_names:
            print(f"[mirror: {name}] session {session} iteration {i}", flush=True)
            train_selfplay(nets[name], deck_ctxs[name], decklists[name], reward_fn,
                            nets[name], deck_ctxs[name], decklists[name], reward_fn,
                            [opt_shared, head_opts[name]], [opt_shared, head_opts[name]], horizon,
                            n_iterations=1, games_per_iteration=games_per_iteration, rng=rng, device="cpu")
    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "shared": shared.state_dict(), "opt_shared": opt_shared.state_dict(),
        "nets": {name: nets[name].state_dict() for name in deck_names},
        "head_opts": {name: head_opts[name].state_dict() for name in deck_names},
        "session": session, "vocab_size": vocab.size, "d_model": D_MODEL,
    }, CHECKPOINT)
    print(f"checkpoint saved to {CHECKPOINT}")

    if do_freeze:
        for p in shared.parameters():
            p.requires_grad = False
        shared.eval()
        torch.save({"shared": shared.state_dict(), "vocab_size": vocab.size, "d_model": D_MODEL}, FROZEN_STACK)
        print(f"shared stack FROZEN and saved to {FROZEN_STACK}")


if __name__ == "__main__":
    main()
