"""Phase 4 driver: pretrain the shared perception stack (embeddings +
SetTransformer + FiLM) with a throwaway DeckNetwork per pool deck
(mono_red_madness, rakdos_madness) -- real mirror self-play games run
through token_train.train_selfplay for each deck in turn, every iteration,
so the shared stack sees both decks' worth of real board states every
round rather than one deck exhausted before the other starts. Gradients
from BOTH throwaway nets flow into the ONE shared SetTransformer+FiLM
instance -- exactly the "pretrain the shared layers with junk [per-deck
heads], then freeze" design.

Checkpointed after every session (resumable across separate invocations)
so this can be run in small batches per the explicit instruction: start
tiny, watch for crashes/hangs/stalling, scale toward ~1hr/batch once
runs are healthy.

Usage: python run_pretrain.py [n_iterations] [games_per_iteration] [--freeze]
--freeze: after this session's training, freeze the shared stack
(requires_grad=False, eval mode) and write the frozen weights to
../checkpoints/shared_stack_frozen.pt for Stage 1/2 to load -- run this
only once satisfied pretraining is done, not on every intermediate batch.
"""
import os
import random
import sys
import time

import torch

from rewards import action_count_win_reward_200_floor02
from terminated import never_terminated
from token_arch import SetTransformer
from token_deck import DeckNetwork
from token_pool import build_pool
from token_train import train_selfplay

CHECKPOINT_DIR = "../checkpoints"
CHECKPOINT = f"{CHECKPOINT_DIR}/pretrain_shared_stack.pt"
FROZEN_STACK = f"{CHECKPOINT_DIR}/shared_stack_frozen.pt"
D_MODEL = 64


def main():
    n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 1
    games_per_iteration = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 2
    do_freeze = "--freeze" in sys.argv

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    decklist_a, decklist_b = decklists["mono_red_madness"], decklists["rakdos_madness"]
    ctx_a, ctx_b = deck_ctxs["mono_red_madness"], deck_ctxs["rakdos_madness"]
    fixed_a, fixed_b = fixed_tables["mono_red_madness"], fixed_tables["rakdos_madness"]
    shared = SetTransformer(vocab.size, d_model=D_MODEL, n_heads=4, n_layers=2, dim_feedforward=128)
    net_a = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_a))
    net_b = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_b))
    # ONE optimizer for the shared stack, reused across BOTH decks' mirror
    # sessions below, plus one optimizer per deck for its own throwaway
    # head's unique params. NOT one Adam per net over net.parameters() --
    # that would give the shared stack's identical parameter tensors TWO
    # independent Adam instances with unsynchronized momentum/variance
    # state, stepping on them in alternation (confirmed the hard way, see
    # git history and token_train.ppo_update's own docstring).
    opt_shared = torch.optim.Adam(shared.parameters(), lr=3e-4)
    opt_a_head = torch.optim.Adam([p for n, p in net_a.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)
    opt_b_head = torch.optim.Adam([p for n, p in net_b.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)

    session = 0
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, weights_only=True)
        shared.load_state_dict(ckpt["shared"])
        net_a.load_state_dict(ckpt["net_a"])
        net_b.load_state_dict(ckpt["net_b"])
        opt_shared.load_state_dict(ckpt["opt_shared"])
        opt_a_head.load_state_dict(ckpt["opt_a_head"])
        opt_b_head.load_state_dict(ckpt["opt_b_head"])
        session = ckpt["session"] + 1
        print(f"resumed from {CHECKPOINT} (session {session}, vocab must match: "
              f"{ckpt['vocab_size']} == {vocab.size} -> {ckpt['vocab_size'] == vocab.size})")
        assert ckpt["vocab_size"] == vocab.size, "vocab changed since last checkpoint -- would silently corrupt the embedding table"

    rng = random.Random()
    terminated_fns = [never_terminated, never_terminated]
    reward_fn = action_count_win_reward_200_floor02
    horizon = 120  # matches configs/mono_red_madness_mirror.json's own horizon

    print(f"Phase 4 pretrain session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"(={n_iterations * games_per_iteration * 2} total games across both pool decks)")
    t0 = time.time()
    for i in range(n_iterations):
        print(f"[deck A mirror: mono_red_madness] session {session} iteration {i}", flush=True)
        train_selfplay(net_a, ctx_a, decklist_a, reward_fn, net_a, ctx_a, decklist_a, reward_fn,
                        [opt_shared, opt_a_head], [opt_shared, opt_a_head], terminated_fns, horizon,
                        n_iterations=1, games_per_iteration=games_per_iteration, rng=rng, device="cpu")
        print(f"[deck B mirror: rakdos_madness] session {session} iteration {i}", flush=True)
        train_selfplay(net_b, ctx_b, decklist_b, reward_fn, net_b, ctx_b, decklist_b, reward_fn,
                        [opt_shared, opt_b_head], [opt_shared, opt_b_head], terminated_fns, horizon,
                        n_iterations=1, games_per_iteration=games_per_iteration, rng=rng, device="cpu")
    elapsed = time.time() - t0
    total_games = n_iterations * games_per_iteration * 2
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "shared": shared.state_dict(), "net_a": net_a.state_dict(), "net_b": net_b.state_dict(),
        "opt_shared": opt_shared.state_dict(), "opt_a_head": opt_a_head.state_dict(), "opt_b_head": opt_b_head.state_dict(),
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
