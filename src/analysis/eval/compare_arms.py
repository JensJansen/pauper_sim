"""Head-to-head of training arms at matched age: for each deck, arm A's snapshot vs arm B's snapshot of the SAME id
(same games/deck trained), pooled over the decks and over the given snapshot ids, greedy play, paired seeds.
Score is A's: (wins + 0.5 ties) / games. Per-deck rows show whether the sign agrees across decks.

--b-ids pairs A's --ids with DIFFERENT B snapshot ids (position by position; one id is used for every A id). With
--a and --b the same league this is the continue-vs-stop anchor test: the league's later snapshots against its own
earlier ones, e.g. --a small_baseline --b small_baseline --ids 288 --b-ids 259 (~55.5k vs ~49.9k games/deck).

Usage: python analysis/eval/compare_arms.py --a small_lrdecay --b small_baseline --ids 127,128,129 [--games 300]
                                            [--b-ids 259] [--workers 6] [--out ../logs/compare.json]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import history
from evaluation.matchups import AgentRef, Matchup, available_decks
from evaluation.play import play_matchups
from evaluation.stats import tally, wilson_interval
from repo_paths import CHECKPOINTS_DIR


def _fmt(results):
    t = tally(results)
    lo, hi = wilson_interval(t)
    return f"{t.score:5.1%} [{lo:5.1%},{hi:5.1%}] n={t.games}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ids", required=True, help="comma list of snapshot ids (id k = (k+1)*192 games/deck)")
    ap.add_argument("--b-ids", help="comma list of B snapshot ids, paired with --ids (default: the same ids as A)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args()
    dir_a, dir_b = str(CHECKPOINTS_DIR / args.a), str(CHECKPOINTS_DIR / args.b)
    decks = sorted(set(available_decks(dir_a)) & set(available_decks(dir_b)))
    ids = [int(i) for i in args.ids.split(",")]
    b_ids = [int(i) for i in args.b_ids.split(",")] if args.b_ids else ids
    if len(b_ids) == 1:
        b_ids = b_ids * len(ids)
    assert len(b_ids) == len(ids), "--b-ids must be one id or as many as --ids"
    for d in decks:
        for i, j in zip(ids, b_ids):
            history.snapshot_path(dir_a, d, i)
            history.snapshot_path(dir_b, d, j)
    matchups = [Matchup(AgentRef(d, dir_a, args.a, snapshot=i), AgentRef(d, dir_b, args.b, snapshot=j))
                for d in decks for i, j in zip(ids, b_ids)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = play_matchups(matchups, args.games, executor=ex)
    print(f"{args.a} vs {args.b}, snapshot ids {ids} (~{[history.games_at(i, 192) for i in ids]} games/deck)"
          + (f" vs B ids {b_ids} (~{[history.games_at(j, 192) for j in b_ids]})" if b_ids != ids else ""))
    for d in decks:
        print(f"  {d:16s} {_fmt([r for r in results if r.a.deck == d])}")
    print(f"  POOLED           {_fmt(results)}", flush=True)
    if args.out:
        rows = [{"deck": r.a.deck, "snapshot": r.a.snapshot, "games": r.games, "a_wins": r.a_wins,
                 "b_wins": r.b_wins, "no_winner": r.no_winner} for r in results]
        Path(args.out).write_text(json.dumps({"a": args.a, "b": args.b, "ids": ids, "b_ids": b_ids, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
