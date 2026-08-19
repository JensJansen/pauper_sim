"""Reproduce (and locate) the collection hang seen at 22,848 games/deck.

Symptom, observed twice on 2026-08-19 against checkpoints/4_deck_subleague_test
copied at cum=22,848: one collect worker stops emitting, grows ~1 GB/minute
without bound, and never returns. The parent then either waits forever or dies
with MemoryError while unpickling the worker's result. Seen on the DEFAULT
--device cpu path and on two different training decks, so it is neither
GPU-related nor deck-specific.

horizon bounds TURNS, so a game that stops advancing its turn counter -- a
priority/trigger loop inside a single turn -- is not bounded by it at all, and
would grow the transition buffer without limit. That is the hypothesis this
script is built to confirm or kill.

Runs collection SEQUENTIALLY in-process (no worker pool) so a stall is
directly observable, with faulthandler set to dump every thread's stack if any
single batch exceeds --stall-seconds. The dumped traceback names the exact
engine function looping, which is the whole point -- a hang with no stack is
just a rumor.

Usage:
  python analysis/repro_hang.py --deck elves --games 24 --rounds 20
"""
import sys
import time
import random
import faulthandler
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.league import LeaguePool
from rl.mulligan import MulliganNet
from rl.rewards import deploy_reward_v6
from rl.train import collect_rollout_league
from rl.league_runner import HORIZON, build_deck_net
from rl import checkpoint as ckpt_io

ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--deck", default=None, help="Training deck (default: cycle all four).")
    p.add_argument("--games", type=int, default=24, help="Games per round (matches games_per_iteration).")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stall-seconds", type=int, default=180,
                   help="Dump all thread stacks and abort if one round exceeds this.")
    p.add_argument("--checkpoint-rate", type=float, default=0.15)
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    league_dir = str(CHECKPOINTS_DIR / args.league)

    live_nets, mulligan_nets = {}, {}
    for name in ROSTER:
        path = f"{league_dir}/{name}/live.pt"
        net = build_deck_net(vocab.size, len(fixed_tables[name]),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(path))
        ckpt_io.load_deck_checkpoint(path, net)
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        mnet.eval()
        mulligan_nets[name] = mnet
    print(f"loaded {args.league} live nets; horizon={HORIZON}", flush=True)

    pool = LeaguePool(league_dir, ROSTER)
    rng = random.Random(args.seed)
    decks = [args.deck] if args.deck else ROSTER

    # exit=True: if a round wedges, print every thread's stack and kill the
    # process rather than leaving another multi-GB zombie behind.
    faulthandler.dump_traceback_later(args.stall_seconds, exit=True)
    for rnd in range(args.rounds):
        for name in decks:
            t0 = time.time()
            buffers, _mull, played, _out = collect_rollout_league(
                name, live_nets, mulligan_nets, deck_ctxs, decklists, pool, deploy_reward_v6,
                HORIZON, args.games, rng, device="cpu",
                checkpoint_rate=args.checkpoint_rate, pfsp=True)
            n = len(buffers.get(name, ()))
            biggest = max((len(b) for b in buffers.values()), default=0)
            print(f"  round {rnd:3d} [{name:16}] {played:3d} games in {time.time() - t0:7.1f}s  "
                  f"buf={n:6d} largest_bucket={biggest:6d}", flush=True)
            # Re-arm for the next round: the timer is one-shot per call.
            faulthandler.dump_traceback_later(args.stall_seconds, exit=True)
    faulthandler.cancel_dump_traceback_later()
    print("no stall reproduced", flush=True)


if __name__ == "__main__":
    main()
