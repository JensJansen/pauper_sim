"""One-off cross-population tournament: every deck in league A's checkpoint
dir plays every deck in league B's checkpoint dir -- the full cross product
across BOTH rosters (deck_a x deck_b, including mismatched-name pairs), not
just same-name pairs the way rl.league_runner._run_eval_vs_gauntlet compares.
games_per_matchup games each, greedy (policy's actual best play), no
training/checkpointing. Written for benchmarking checkpoints/
4_deck_subleague_test (the actively-training subleague) against checkpoints/
4_deck_subleague_gauntlet (the frozen reference pod) before/after a training
batch, per the owner's own before/after comparison request.

Usage:
  python run_cross_league_eval.py LEAGUE_A LEAGUE_B [--games N] [--seed N] [--log PATH]
"""
import argparse
import itertools
import json
import os
import random
import time

from repo_paths import CHECKPOINTS_DIR
from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet
from rl.agent import SeatAgent
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl import checkpoint as ckpt_io
from rl.league_runner import load_frozen_stack, D_MODEL

DEFAULT_ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]


def _load_deck_nets(league_dir, names, shared, fixed_tables):
    live_nets, mulligan_nets = {}, {}
    for name in names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)  # optimizer=None: eval only needs weights
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(shared)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        mnet.eval()
        mulligan_nets[name] = mnet
    return live_nets, mulligan_nets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("league_a", help="checkpoints/<league_a>, e.g. 4_deck_subleague_test")
    p.add_argument("league_b", help="checkpoints/<league_b>, e.g. 4_deck_subleague_gauntlet")
    p.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                    help="Deck subset to test on both sides (default: the 4-deck subleague/gauntlet roster).")
    p.add_argument("--games", type=int, default=50, help="Games per matchup (default 50).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log", type=str, default=None, metavar="PATH", help="Write the summary JSON here.")
    args = p.parse_args()

    roster = args.roster.split(",") if args.roster else DEFAULT_ROSTER
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)

    live_a, mull_a = _load_deck_nets(CHECKPOINTS_DIR / args.league_a, roster, shared, fixed_tables)
    live_b, mull_b = _load_deck_nets(CHECKPOINTS_DIR / args.league_b, roster, shared, fixed_tables)

    rng = random.Random(args.seed)
    horizon = 120
    results = []
    t0 = time.time()
    for a, b in itertools.product(roster, roster):
        pairing = _constant_pairing(
            [SeatAgent(live_a[a], mull_a[a], deck_ctxs[a]), SeatAgent(live_b[b], mull_b[b], deck_ctxs[b])],
            [decklists[a], decklists[b]], [None, None], [None, None])
        game_logs = []
        _bufs, _mull, played = collect_rollout(pairing, args.games, horizon, rng, device="cpu",
                                                record=False, greedy=True, game_logs=game_logs)
        outcomes = [e for ev in game_logs for e in ev if e["kind"] == "game_over"]
        a_wins = sum(1 for e in outcomes if e["winner"] == 0)
        b_wins = sum(1 for e in outcomes if e["winner"] == 1)
        no_winner = played - a_wins - b_wins
        results.append({"deck_a": a, "deck_b": b, "games": played,
                         "a_wins": a_wins, "b_wins": b_wins, "no_winner": no_winner})
        print(f"  {a} ({args.league_a}) vs {b} ({args.league_b}): "
              f"{a_wins}-{b_wins} ({no_winner} no-winner) of {played}", flush=True)

    total_games = sum(r["games"] for r in results)
    total_a = sum(r["a_wins"] for r in results)
    total_b = sum(r["b_wins"] for r in results)
    print(f"cross-league eval done: {total_games} games in {time.time() - t0:.1f}s")
    print(f"OVERALL: {args.league_a} {total_a}-{total_b} {args.league_b} "
          f"({total_a / total_games:.1%} - {total_b / total_games:.1%}) over {total_games} games")

    out = {"league_a": args.league_a, "league_b": args.league_b, "roster": roster,
           "games_per_matchup": args.games, "seed": args.seed, "results": results,
           "total_games": total_games, "total_a_wins": total_a, "total_b_wins": total_b}
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote summary to {args.log}")


if __name__ == "__main__":
    main()
