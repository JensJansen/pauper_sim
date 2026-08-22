"""ABSOLUTE SCALE: what is a trained league checkpoint actually worth?

Every existing instrument -- vs_gauntlet, vs_history, vs_heuristic -- reports a
win rate against ONE opponent of unknown strength, so "51% vs the gauntlet"
cannot answer whether 60,001 games/deck bought anything at all. Nothing in the
repo has ever measured against a fixed point whose strength is known by
construction. This does: a fully UNTRAINED DeckNetwork -- random encoder and
random heads alike -- is the zero of the scale.

Default vintages `0,live` DECOMPOSE the run rather than merely scoring it.
snapshot_0 is a ~200-game policy (snapshot_every_games=200), so:

    row(0)     = what the first ~200 games/deck bought
    row(live)  = what all 60,001 bought
    the gap    = what the remaining ~59,800 bought

That decomposition is the point. vs_history already shows three of four decks
at parity with their own 200-game-old selves, so if the two rows here land on
top of each other, the plateau is not a plateau -- almost nothing was ever
learned past the first few hundred games, and the diagnosis ends there.

GREEDY vs SAMPLED: the in-training evals all use greedy=True (measure best
play, not an exploration sample), and matching them keeps these numbers
comparable. But argmax over a RANDOMLY initialized head is a near-constant
policy -- it can lock onto one action index and, say, never play a land -- which
would understate the anchor and inflate every gap measured against it. So this
runs BOTH by default (--mode both) and reports them side by side; a large split
between them means the greedy anchor degenerated and the sampled row is the
honest one.

AUTONOMY: the anchor is a DeckNetwork choosing over the engine's own legality
mask -- a policy making its own decisions, just an untrained one. It is not a
scripted rule set and is never a training target. Evaluation-only.

Usage:
  python analysis/eval/run_anchor_eval.py [--games 100] [--vintages 0,live] [--mode both]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/, for `repo_paths` / `rl.*` -- these live two levels up now that this script sits in analysis/eval/
import argparse
import math
import time

import torch

from repo_paths import CHECKPOINTS_DIR
from rl.model.mulligan import MulliganNet
from rl.decision.agent import SeatAgent
from rl.league.league import LeaguePool
from rl.roster import build_pool
from analysis.eval.report_metrics import wilson as _wilson  # same helper, one definition -- both live in analysis/ now
from rl.league.league_runner import (HORIZON, build_deck_net, league_roster, load_vintage_agent,
                                     _play_paired_eval_games)





def _anchor_agent(deck_ctx, seed):
    """A fully untrained agent -- random encoder, random heads.

    Constructed with DeckNetwork's default trunk_hidden, which is what
    _run_eval_vs_gauntlet also assumes when it loads live.pt, so the anchor is
    architecture-matched to the thing it is scoring and the only difference
    between them is training. It is a STRICTLY weaker floor than it used to
    be: the anchor once sat on the real pretrained frozen stack, so it had
    trained perception and only random heads. With per-deck encoders there is
    no pretrained perception to borrow, so the whole network is random -- the
    scale's zero moved, and anchor win rates are not comparable across that
    change.
    """
    vocab, fixed_table = deck_ctx
    torch.manual_seed(seed)  # reproducible anchor: same random weights every run
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
                # Fresh rng per cell off the same base seed: every (deck, vintage,
                # mode) cell sees the SAME shuffle sequence, so differences between
                # rows are policy, not deck order.
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
