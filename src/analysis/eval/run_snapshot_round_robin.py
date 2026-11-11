"""Is this league's strategy space transitive or cyclic?

For each deck: pick K snapshots spanning the run, play all K(K-1)/2 pairs,
and ask three questions of the resulting matrix.

  1. 3-CYCLES. Any triple where A beats B beats C beats A? Reported both raw
     and restricted to triples whose three pairs are all significantly off
     50% -- the raw count alone is mostly noise (coin-flip pairs produce
     spurious cycles at a steady rate).
  2. ELO FIT + RESIDUAL. A one-dimensional strength model (Bradley-Terry, fit
     by minorization-maximization) can represent any transitive ordering and
     no cyclic one, so its residual is the intransitivity measurement.
     Printed next to the residual pure sampling noise alone would produce;
     at or below that floor, the transitive model fits as well as anything could.
  3. MONOTONICITY IN AGE. Does Elo rise with snapshot number? A flat rating
     curve on a transitive matrix means the deck is sitting still; a rising
     one means it genuinely improved.

Every pair is played through evaluation.play_matchups (greedy, paired seeds; who
starts is each game's own coin flip, so neither side is systematically on the play).

Evaluation-only: loads frozen snapshots, trains nothing, writes no checkpoint.

Usage:
  python analysis/eval/run_snapshot_round_robin.py [--snapshots 0,58,116,174,232,289]
                                     [--games 100] [--out ../logs/rr.json]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/
import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

from evaluation import AgentRef, Matchup, play_matchups
from evaluation.matchups import available_decks
from repo_paths import CHECKPOINTS_DIR


def _fit_bradley_terry(labels, pair_wins, iters=500, prior=1.0):
    """Bradley-Terry strengths by minorization-maximization, returned as Elo.

    MM update: p_i <- W_i / sum_j (n_ij / (p_i + p_j)). No learning rate, no
    divergence, monotone in likelihood.

    `prior` adds that many games split evenly to every pair, keeping a
    snapshot that went 0-for-everything (or won everything) from driving its
    rating to +/-infinity.
    """
    p = dict.fromkeys(labels, 1.0)
    wins, n = {}, {}
    for (a, b), (wa, total) in pair_wins.items():
        wins[(a, b)] = wa + prior / 2
        wins[(b, a)] = (total - wa) + prior / 2
        n[(a, b)] = n[(b, a)] = total + prior
    for _ in range(iters):
        new = {}
        for i in labels:
            num = sum(wins[(i, j)] for j in labels if (i, j) in wins)
            den = sum(n[(i, j)] / (p[i] + p[j]) for j in labels if (i, j) in n)
            new[i] = num / den if den else p[i]
        geo = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {k: v / geo for k, v in new.items()}  # anchor scale-invariant BT
    return {k: 400 * math.log10(v) for k, v in p.items()}


def _analyze(labels, pair_wins):
    """(elo, raw 3-cycles, significant 3-cycles, residual, noise floor)."""
    elo = _fit_bradley_terry(labels, pair_wins)

    def rate(i, j):
        if (i, j) in pair_wins:
            w, n = pair_wins[(i, j)]
            return w / n, n
        w, n = pair_wins[(j, i)]
        return (n - w) / n, n

    raw = sig = 0
    for trio in itertools.combinations(labels, 3):
        i, j, k = trio
        rs = [rate(i, j), rate(j, k), rate(k, i)]
        fwd = all(r > 0.5 for r, _ in rs)
        rev = all(r < 0.5 for r, _ in rs)
        if fwd or rev:
            raw += 1
            # significant only if all three legs are 2 sigma off 50%
            if all(abs(r - 0.5) > 2 * math.sqrt(0.25 / n) for r, n in rs):
                sig += 1

    resid, floor = [], []
    for (a, b), (w, n) in pair_wins.items():
        obs = w / n
        pred = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
        resid.append(abs(obs - pred))
        # residual a perfect model would still show from sampling n games
        floor.append(math.sqrt(pred * (1 - pred) / n) * math.sqrt(2 / math.pi))
    return elo, raw, sig, sum(resid) / len(resid), sum(floor) / len(floor)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--snapshots", default="0,58,116,174,232,289",
                   help="comma-separated snapshot ids and/or 'live', spanning the run")
    p.add_argument("--games", type=int, default=100, help="games per pair")
    p.add_argument("--workers", type=int, default=1, help="processes to spread the games over (1 = in-process)")
    p.add_argument("--decks", default="", help="comma-separated subset; default all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../logs/snapshot_round_robin.json")
    args = p.parse_args()

    league_dir = str(CHECKPOINTS_DIR / args.league)
    labels = args.snapshots.split(",")
    decks = args.decks.split(",") if args.decks else available_decks(league_dir)
    assert decks, f"no trained decks (no live.pt) under {league_dir}"
    pairs = list(itertools.combinations(labels, 2))

    print(f"{args.league}: {len(labels)} vintages, {len(pairs)} pairs, {args.games} games/pair, "
          f"{len(decks)} decks = {len(pairs) * args.games * len(decks)} games")
    print(f"vintages: {', '.join(labels)}  (snapshot N ~ N*200 games/deck)\n")

    t0, out = time.time(), {}
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    for deck in decks:
        # a vintage is a snapshot id, or "live" for the current checkpoint pair
        refs = {v: AgentRef(deck, league_dir, v, None if v == "live" else int(v)) for v in labels}
        results = play_matchups([Matchup(refs[a], refs[b]) for a, b in pairs], args.games, seed=args.seed,
                                executor=executor)
        pair_wins = {pair: (r.a_wins, r.games) for pair, r in zip(pairs, results)}
        elo, raw, sig, resid, floor = _analyze(labels, pair_wins)

        print(f"=== {deck} ===")
        print(f"{'':<10}" + "".join(f"{b:>9}" for b in labels) + f"{'elo':>9}")
        for a in labels:
            row = f"{a:<10}"
            for b in labels:
                if a == b:
                    row += f"{'-':>9}"
                elif (a, b) in pair_wins:
                    w, n = pair_wins[(a, b)]
                    row += f"{100 * w / n:>8.0f}%"
                else:
                    w, n = pair_wins[(b, a)]
                    row += f"{100 * (n - w) / n:>8.0f}%"
            print(row + f"{elo[a]:>+9.0f}")
        order = [elo[v] for v in labels]
        rising = sum(1 for i in range(len(order) - 1) if order[i + 1] > order[i])
        print(f"  3-cycles: {raw} raw, {sig} significant (of {math.comb(len(labels), 3)} triples)")
        print(f"  elo residual: {100 * resid:.2f}pp vs {100 * floor:.2f}pp noise floor "
              f"({'transitive' if resid <= floor * 1.5 else 'MISFIT -- check cycles'})")
        print(f"  elo span: {max(order) - min(order):.0f} points, "
              f"rising on {rising}/{len(order) - 1} steps\n")

        out[deck] = {"labels": labels, "elo": elo, "cycles_raw": raw, "cycles_significant": sig,
                     "residual_pp": 100 * resid, "noise_floor_pp": 100 * floor,
                     "pairs": {f"{a}|{b}": pair_wins[(a, b)] for a, b in pairs}}

    if executor is not None:
        executor.shutdown()
    with open(args.out, "w") as f:
        json.dump({"league": args.league, "games_per_pair": args.games, "decks": out}, f, indent=1)
    print(f"done in {(time.time() - t0) * 1000:,.0f}ms -> {args.out}")
    print("A transitive matrix with a FLAT elo span is the 'sitting still' result:\n"
          "no cycling, and no strength gained either.")


if __name__ == "__main__":
    main()
