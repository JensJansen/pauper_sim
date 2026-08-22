"""Scores league checkpoints against a fixed reference of known strength: a
fully untrained DeckNetwork (random encoder and heads), the zero of the scale.

Default vintages `0,live` decompose the run:

    row(0)     = what the first ~200 games/deck bought
    row(live)  = what all games so far bought
    the gap    = what the games between them bought

Runs both greedy and sampled play against the anchor by default (--mode
both) and reports them side by side: argmax over a randomly initialized head
can be a near-constant policy, which would understate the anchor's strength,
so a large greedy/sampled split flags a degenerate greedy anchor.

The anchor is a DeckNetwork choosing over the engine's own legality mask, not
a scripted rule set, and is never a training target. Evaluation-only.

Usage:
  python analysis/eval/run_anchor_eval.py [--games 100] [--vintages 0,live] [--mode both]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/
import argparse
import math
import time

import torch

from repo_paths import CHECKPOINTS_DIR
from rl.model.mulligan import MulliganNet
from rl.decision.agent import SeatAgent
from rl.league.league import LeaguePool
from rl.roster import build_pool
from analysis.eval.report_metrics import wilson as _wilson
from rl.league.league_runner import (HORIZON, build_deck_net, league_roster, load_vintage_agent,
                                     _play_paired_eval_games)





def _anchor_agent(deck_ctx, seed):
    """A fully untrained agent -- random encoder, random heads.

    Built with DeckNetwork's default trunk_hidden, matching what
    _run_eval_vs_gauntlet assumes when loading live.pt, so the anchor is
    architecture-matched to the thing it scores and the only difference is
    training.
    """
    vocab, fixed_table = deck_ctx
    torch.manual_seed(seed)  # reproducible anchor weights across runs
    net = build_deck_net(vocab.size, len(fixed_table))
    net.eval()
    mull = MulliganNet(net.encoder)
    mull.eval()
    return SeatAgent(net, mull, deck_ctx)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--vintages", default="0,live",
                   help="comma-separated snapshot ids and/or 'live'")
    p.add_argument("--games", type=int, default=100, help="games per (deck, vintage, mode) cell")
    p.add_argument("--mode", default="both", choices=["greedy", "sampled", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor-seed", type=int, default=1234,
                   help="torch seed for the random-init anchor's weights")
    args = p.parse_args()

    decklists, vocab, deck_ctxs, _fixed = build_pool()
    league_dir = str(CHECKPOINTS_DIR / args.league)
    vintages = args.vintages.split(",")
    modes = ["greedy", "sampled"] if args.mode == "both" else [args.mode]
    decks = league_roster(league_dir)
    assert decks, f"no trained decks (no live.pt) under {league_dir}"

    print(f"random-init anchor vs {args.league}, {args.games} games/cell, horizon {HORIZON}")
    print(f"anchor = fully untrained DeckNetwork, encoder included (torch seed {args.anchor_seed})\n")
    header = f"{'deck':<16}{'vintage':>9}"
    for m in modes:
        header += f"{m + ' win%':>13}{'95% CI':>16}"
    print(header)

    t0 = time.time()
    rows = {}
    for deck in decks:
        deck_ctx = deck_ctxs[deck]
        pool = LeaguePool(league_dir, [deck])  # reused so each snapshot loads off disk once
        anchor = _anchor_agent(deck_ctx, args.anchor_seed)
        for vintage in vintages:
            agent = load_vintage_agent(league_dir, deck, vintage, deck_ctx, pool=pool)
            line = f"{deck:<16}{vintage:>9}"
            for mode in modes:
                # Same base seed per cell: every (deck, vintage, mode) cell sees
                # the same shuffle sequence, so differences are policy, not deck order.
                res = _play_paired_eval_games(agent, anchor, decklists[deck], args.games, HORIZON,
                                              args.seed, "anchor_wins",
                                              greedy=(mode == "greedy"))
                pct, lo, hi = _wilson(res["live_wins"], res["games"])
                rows[(deck, vintage, mode)] = (res["live_wins"], res["games"])
                line += f"{100 * pct:>12.1f}%{f'[{100 * lo:.1f}, {100 * hi:.1f}]':>16}"
            print(line, flush=True)

    print(f"\ndone in {time.time() - t0:.0f}s")
    if len(vintages) > 1:
        first, last = vintages[0], vintages[-1]
        print(f"\ngap {first} -> {last} (what the games BETWEEN those vintages bought):")
        for mode in modes:
            for deck in decks:
                (w0, n0), (w1, n1) = rows[(deck, first, mode)], rows[(deck, last, mode)]
                p0, p1 = w0 / n0, w1 / n1
                pooled = (w0 + w1) / (n0 + n1)
                se = math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
                z = (p1 - p0) / se if se else 0.0
                print(f"  [{mode:<7}] {deck:<16}{100 * p0:>7.1f}% -> {100 * p1:>6.1f}%"
                      f"  ({100 * (p1 - p0):>+6.1f}pp, z={z:>+5.2f})")
        print("  |z| < 2 means the later vintage is NOT distinguishable from the earlier one\n"
              "  against a fixed reference -- i.e. those games bought nothing measurable.")


if __name__ == "__main__":
    main()
