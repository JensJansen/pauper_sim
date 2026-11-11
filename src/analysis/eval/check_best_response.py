"""Reads a best-response run (run_best_response.py): the fresh DECK agent at a few ages against (1) the source league's
frozen DECK agent head-to-head and (2) the frozen field, next to that same frozen DECK agent against the same field.
Score = (wins + 0.5 ties) / games, greedy play, paired seeds, 95% Wilson intervals.

Usage: python analysis/eval/check_best_response.py --deck elves [--source small_baseline] [--ages 2000,5000,...]
                                                   [--games 200] [--workers 3]
"""
import argparse
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--source", default="small_baseline")
    ap.add_argument("--ages", default="", help="comma list of games/deck; default: ~5 evenly spaced up to the newest")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    src, br = str(CHECKPOINTS_DIR / args.source), str(CHECKPOINTS_DIR / f"br_{args.deck}")
    field = [d for d in available_decks(br) if d != args.deck]
    ids = history.snapshot_ids(br, args.deck)
    if args.ages:
        wanted = [int(a) for a in args.ages.split(",")]
        ids = [min(ids, key=lambda i: abs(history.games_at(i) - a)) for a in wanted]
    else:
        ids = sorted({ids[round(k * (len(ids) - 1) / 5)] for k in range(1, 6)})
    frozen = AgentRef(args.deck, src, "frozen")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        base = play_matchups([Matchup(frozen, AgentRef(f, br)) for f in field], args.games, executor=ex)
        print(f"frozen {args.deck} vs frozen field {field}: {_fmt(base)}")
        for i in ids:
            fresh = AgentRef(args.deck, br, "fresh", snapshot=i)
            vs_field = play_matchups([Matchup(fresh, AgentRef(f, br)) for f in field], args.games, executor=ex)
            h2h = play_matchups([Matchup(fresh, frozen)], args.games * len(field), executor=ex)
            print(f"fresh@{history.games_at(i):>6} games/deck  vs field {_fmt(vs_field)}  |  head-to-head vs frozen {_fmt(h2h)}",
                  flush=True)


if __name__ == "__main__":
    main()
