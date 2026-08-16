"""Pretrain driver: pretrain the shared perception stack (embeddings +
SetTransformer + FiLM) with a throwaway DeckNetwork per pool deck -- real
mirror self-play games run through rl.train.train_selfplay for EVERY
deck in the roster (data/league_decks.json), each deck in turn every
iteration, so the shared stack sees all decks' worth of real board states
every round rather than one deck exhausted before the next starts.
Gradients from every throwaway net flow into the ONE shared
SetTransformer+FiLM instance -- exactly the "pretrain the shared layers
with junk [per-deck heads], then freeze" design. Generalized over an
arbitrary roster, so the shared stack's embedding table + attention
actually learn representations for every deck's cards, not just a
hardcoded subset.

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
import random
import sys
import time

import torch

from rl.rewards import action_count_win_reward_200_floor02
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.league_runner import HORIZON
from rl.pool import build_pool
from rl.train import train_selfplay
from rl import checkpoint as ckpt_io
from repo_paths import CHECKPOINTS_DIR

CHECKPOINT = CHECKPOINTS_DIR / "pretrain_shared_stack.pt"
FROZEN_STACK = CHECKPOINTS_DIR / "shared_stack_frozen.pt"
D_MODEL = 64


def pretrain_opponent(deck_names, name, cross_deck, rng):
    """Which deck `name` is paired against for one pretrain round.

    Mirror (the historical behavior, and still the default) always returns
    `name` itself. --cross-deck samples uniformly from the WHOLE roster,
    including `name`, so roughly 1/len(roster) of rounds stay mirrors -- close
    to the mix a real league produces.

    Why this exists (2026-08-16, RL_METHODOLOGY_PLAN.md section 1A.15): the
    shared stack has only ever encoded MIRROR board states, both players on the
    same decklist. In an 11-deck league ~10/11 of games are cross-deck, so the
    SetTransformer's attention is asked at training time to encode combinations
    -- my elves creatures opposite their dmir Terror -- that it never saw once
    during pretraining. Every deck's CARDS were covered (embeddings are trained
    over the full roster); their cross-deck CO-OCCURRENCE was not. That is a
    sharper form of the frozen-encoder hypothesis than "not enough pretrain
    games", and it costs a few lines because rl.train.train_selfplay already
    takes fully independent a/b decks -- pretrain was simply passing the same
    one twice."""
    if not cross_deck:
        return name
    return rng.choice(deck_names)


def main():
    n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 1
    games_per_iteration = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 2
    do_freeze = "--freeze" in sys.argv
    cross_deck = "--cross-deck" in sys.argv

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    deck_names = list(decklists)
    shared = SetTransformer(vocab.size, d_model=D_MODEL, n_heads=4, n_layers=2, dim_feedforward=128)

    # ONE optimizer for the shared stack, reused across EVERY deck's mirror
    # session, plus one optimizer per deck for its own throwaway head's
    # unique params. NOT one Adam per net over net.parameters() -- that
    # would give the shared stack's identical parameter tensors N
    # independent Adam instances with unsynchronized momentum/variance
    # state, stepping on them in alternation (see rl.ppo.ppo_update's own
    # docstring).
    opt_shared = torch.optim.Adam(shared.parameters(), lr=3e-4)
    nets, head_opts = {}, {}
    for name in deck_names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        nets[name] = net
        head_opts[name] = torch.optim.Adam(
            [p for n, p in net.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4,
        )

    session = 0
    ckpt = ckpt_io.load_pretrain_checkpoint(CHECKPOINT)
    if ckpt is not None:
        assert ckpt["vocab_size"] == vocab.size, "vocab changed since last checkpoint -- would silently corrupt the embedding table"
        assert set(ckpt["nets"]) == set(deck_names), "roster changed since last checkpoint -- start a fresh pretrain"
        shared.load_state_dict(ckpt["shared"])
        opt_shared.load_state_dict(ckpt["opt_shared"])
        for name in deck_names:
            nets[name].load_state_dict(ckpt["nets"][name])
        if "head_opts" in ckpt:  # a migrated checkpoint drops per-deck head optimizers -> fresh Adam re-warms
            for name in deck_names:
                head_opts[name].load_state_dict(ckpt["head_opts"][name])
        session = ckpt["session"] + 1
        print(f"resumed from {CHECKPOINT} (session {session}, vocab {ckpt['vocab_size']}=={vocab.size})")

    rng = random.Random()
    reward_fn = action_count_win_reward_200_floor02
    horizon = HORIZON

    total_games = n_iterations * games_per_iteration * len(deck_names)
    print(f"pretrain session {session}: n_iterations={n_iterations} games_per_iteration={games_per_iteration} "
          f"decks={deck_names} (={total_games} total games across all pool decks)")
    t0 = time.time()
    for i in range(n_iterations):
        for name in deck_names:
            opp = pretrain_opponent(deck_names, name, cross_deck, rng)
            label = "mirror" if opp == name else "cross"
            print(f"[{label}: {name} vs {opp}] session {session} iteration {i}", flush=True)
            # Both sides' head optimizers step, and opt_shared appears in both
            # lists -- the same arrangement the mirror path always used, just
            # with a possibly-different deck on side b.
            train_selfplay(nets[name], deck_ctxs[name], decklists[name], reward_fn,
                            nets[opp], deck_ctxs[opp], decklists[opp], reward_fn,
                            [opt_shared, head_opts[name]], [opt_shared, head_opts[opp]], horizon,
                            n_iterations=1, games_per_iteration=games_per_iteration, rng=rng, device="cpu")
    elapsed = time.time() - t0
    print(f"session {session} done in {elapsed:.1f}s ({elapsed / total_games:.2f}s/game)")

    ckpt_io.save_pretrain_checkpoint(CHECKPOINT, shared, opt_shared, nets, head_opts, session, vocab.size, D_MODEL)
    print(f"checkpoint saved to {CHECKPOINT}")

    if do_freeze:
        for p in shared.parameters():
            p.requires_grad = False
        shared.eval()
        ckpt_io.save_frozen_stack(FROZEN_STACK, shared, vocab.size, D_MODEL)
        print(f"shared stack FROZEN and saved to {FROZEN_STACK}")


if __name__ == "__main__":
    main()
