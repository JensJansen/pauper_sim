"""Retrospective strength sweeps over a league's archived snapshots (see evaluation/history.py).

  python retro_sweep.py matrix --league DIR [--decks a,b] [--ages 1000,5000,...,final] [--games 100]
      Metric 1. Per deck, every older snapshot (age a) against every newer one (age t > a): the newer agent's score.
      A steadily improving deck beats older and older selves ever more clearly; a plateau is rows of the matrix that
      stop changing (the agent at 80k beats every anchor exactly as the agent at 40k does).
  python retro_sweep.py field --league DIR [--ages ...] [--games 40]
      Metric 2. Each deck's FINAL agent against every deck's historical agent (age a): A_final vs B@a. As B ages its
      curve falls to where B stops improving; a deck whose final agent beats its own history but not the others'
      is not generalizing. Read it from both sides: by B (how each deck's history looks to the final field) and
      by A (how each final agent does against history).
  python retro_sweep.py report --in FILE    re-print a saved sweep's tables.

Common options: --seed N (0), --workers N (6; 1 = in-process), --out FILE (raw counts as JSON), --games-per-snapshot
(180). Greedy play; paired seeds (evaluation.play.matchup_seed), so cells that share decks share their deals.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from repo_paths import SRC_DIR

DEFAULT_AGES = "1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,final"


def _parse_ages(arg):
    return [a if a == "final" else int(a) for a in arg.split(",")]


def build_matrix(league, decks, ages, games_per_snapshot, AgentRef, Matchup, history):
    """[(Matchup(newer, older), deck, t_games, a_games)] for every age pair of every deck."""
    out = []
    for deck in decks:
        pts = history.at_ages(league, deck, ages, games_per_snapshot)
        for i, (_, sid_a, ga) in enumerate(pts):
            for _, sid_t, gt in pts[i + 1:]:
                out.append((Matchup(AgentRef(deck, league, f"t{gt}", sid_t), AgentRef(deck, league, f"a{ga}", sid_a)),
                            deck, gt, ga))
    return out


def build_field(league, decks, ages, games_per_snapshot, AgentRef, Matchup, history):
    """[(Matchup(A final, B@age), A, B, B's games)] for every deck pair and age (mirrors included)."""
    finals = {d: history.at_ages(league, d, ["final"], games_per_snapshot)[0] for d in decks}
    out = []
    for b in decks:
        for _, sid_b, gb in history.at_ages(league, b, ages, games_per_snapshot):
            for a in decks:
                out.append((Matchup(AgentRef(a, league, "final", finals[a][1]), AgentRef(b, league, f"@{gb}", sid_b)),
                            a, b, gb))
    return out


def report(mode, rows):
    from evaluation import Tally, wilson_interval

    def cell(rs):
        if not rs:
            return ""
        t = Tally()
        for r in rs:
            t += Tally(r["a_wins"], r["no_winner"], r["games"])
        lo, hi = wilson_interval(t)
        return f"{100 * t.score:5.1f}+-{100 * (hi - lo) / 2:3.1f}"

    def table(title, row_names, col_names, cell_of, head="", total=None):
        print(f"\n{title}")
        print(f"{head:20s}" + "".join(f"{c:>12}" for c in col_names))
        for rn in row_names:
            print(f"{str(rn):20s}" + "".join(f"{cell_of(rn, c):>12s}" for c in col_names))
        if total:
            print(f"{'ALL':20s}" + "".join(f"{total(c):>12s}" for c in col_names))

    if mode == "matrix":
        decks = sorted({r["deck"] for r in rows})
        ages = sorted({r["a_games"] for r in rows} | {r["t_games"] for r in rows})
        newest = ages[-1]
        table("POOLED over decks: score (%+-95% half-width) of the NEWER agent (row) against the older one (column)",
              ages[1:], ages[:-1],
              lambda t, a: cell([r for r in rows if r["t_games"] == t and r["a_games"] == a]), head="newer \\ older")
        table(f"PER DECK: the final agent ({newest} games) against each older age", decks, ages[:-1],
              lambda d, a: cell([r for r in rows if r["deck"] == d and r["t_games"] == newest and r["a_games"] == a]),
              head="deck")
        table("PER DECK: each age against the age before it (50 = no progress between the two)", decks, ages[1:],
              lambda d, t: cell([r for r in rows if r["deck"] == d and r["t_games"] == t
                                 and r["a_games"] == ages[ages.index(t) - 1]]), head="deck")
    else:
        decks = sorted({r["deck_a"] for r in rows})
        ages = sorted({r["b_games"] for r in rows})
        table("BY OPPONENT B: every deck's FINAL agent (pooled over A) against B at each age. Falls as B improves; "
              "flat = B stopped (or never started) improving", decks, ages,
              lambda b, g: cell([r for r in rows if r["deck_b"] == b and r["b_games"] == g]), head="B",
              total=lambda g: cell([r for r in rows if r["b_games"] == g]))
        table("BY FINAL AGENT A: A's final agent (pooled over B) against the whole field at each age", decks, ages,
              lambda a, g: cell([r for r in rows if r["deck_a"] == a and r["b_games"] == g]), head="A")
        table("MIRROR CELLS: A's final agent against A's own history", decks, ages,
              lambda a, g: cell([r for r in rows if r["deck_a"] == a == r["deck_b"] and r["b_games"] == g]), head="A")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["matrix", "field", "report"])
    ap.add_argument("--league")
    ap.add_argument("--decks")
    ap.add_argument("--ages", default=DEFAULT_AGES)
    ap.add_argument("--games", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games-per-snapshot", type=int, default=180)
    ap.add_argument("--out")
    ap.add_argument("--in", dest="infile")
    args = ap.parse_args(argv)

    infile = os.path.abspath(args.infile) if args.infile else None
    league = os.path.abspath(args.league) if args.league else None
    out = os.path.abspath(args.out) if args.out else None
    os.chdir(SRC_DIR)  # build_pool's data paths are relative to src/
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if args.mode == "report":
        with open(infile) as f:
            saved = json.load(f)
        report(saved["mode"], saved["rows"])
        return

    from evaluation import AgentRef, Matchup, history, play_matchups
    from evaluation.matchups import available_decks

    decks = args.decks.split(",") if args.decks else available_decks(league)
    for d in decks:
        history.verify(league, d, args.games_per_snapshot)
    ages = _parse_ages(args.ages)
    games = args.games or (100 if args.mode == "matrix" else 40)
    build = build_matrix if args.mode == "matrix" else build_field
    plan = build(league, decks, ages, args.games_per_snapshot, AgentRef, Matchup, history)
    print(f"{args.mode}: {len(plan)} cells x {games} games = {len(plan) * games} games ({len(decks)} decks, "
          f"ages {ages}, seed {args.seed}, {args.workers} workers)", flush=True)
    matchups = [p[0] for p in plan]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = play_matchups(matchups, games, seed=args.seed, executor=ex)
    else:
        results = play_matchups(matchups, games, seed=args.seed)
    rows = []
    for (_m, *meta), r in zip(plan, results):
        base = {"games": r.games, "a_wins": r.a_wins, "b_wins": r.b_wins, "no_winner": r.no_winner}
        if args.mode == "matrix":
            rows.append({"deck": meta[0], "t_games": meta[1], "a_games": meta[2], **base})
        else:
            rows.append({"deck_a": meta[0], "deck_b": meta[1], "b_games": meta[2], **base})
    if out:
        with open(out, "w") as f:
            json.dump({"mode": args.mode, "league": league, "games": games, "seed": args.seed, "rows": rows}, f)
        print(f"wrote {out}", flush=True)
    report(args.mode, rows)


if __name__ == "__main__":
    main()
