"""Evaluate trained agents from any checkpoint directories against each other.

  python evaluate_agents.py gauntlet --subject S=DIR [--subject L1=DIR ...] --field DIR [--field-label start]
      Every subject deck plays every field deck (mirrors included), --games games per pairing. Each subject's
      per-deck score is pooled over the field; with several subjects, every later one is also compared with the
      first (score difference + 95% interval). All subjects face the same field with the same seeds, so the
      comparison is paired (see evaluation.play.matchup_seed).
  python evaluate_agents.py head2head --a LABEL=DIR --b LABEL=DIR
      The same deck from two directories against itself, for each deck both have.
  python evaluate_agents.py roundrobin --league DIR
      Every pair from one directory, mirrors included.

Common options: --subject-decks/--field-decks/--decks (comma lists, default: every deck found), --games N (100),
--seed N (0), --sampled (sample from the policy instead of greedy argmax), --workers N (6; 1 = in-process),
--out FILE (full per-matchup results as JSON). Score = wins + 0.5 * timeouts over games.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from repo_paths import SRC_DIR


def _labelled(spec):
    """'LABEL=DIR' or a bare 'DIR' -> (label, absolute dir)."""
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = "", spec
    return label or os.path.basename(os.path.normpath(path)), os.path.abspath(path)


def _decks(arg):
    return arg.split(",") if arg else None


def _fmt(t, wilson_interval):
    lo, hi = wilson_interval(t)
    return f"{100 * t.score:5.1f}% [{100 * lo:4.1f},{100 * hi:4.1f}] n={t.games}"


def _gauntlet_report(results, labels, tally, wilson_interval, diff_interval):
    by_label = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_label[r.a.label][r.a.deck].append(r)
    decks = sorted(by_label[labels[0]])
    print(f"\n{'deck':20s}" + "".join(f"{label:>34s}" for label in labels))
    for d in decks:
        print(f"{d:20s}" + "".join(f"{_fmt(tally(by_label[label][d]), wilson_interval):>34s}" for label in labels))
    overall = {label: tally([r for rs in by_label[label].values() for r in rs]) for label in labels}
    print(f"{'ALL DECKS':20s}" + "".join(f"{_fmt(overall[label], wilson_interval):>34s}" for label in labels))
    for label in labels[1:]:
        print(f"\n{label} minus {labels[0]} (score points, 95% interval):")
        for d in decks:
            diff, lo, hi = diff_interval(tally(by_label[label][d]), tally(by_label[labels[0]][d]))
            print(f"  {d:20s} {100 * diff:+6.1f}  [{100 * lo:+6.1f}, {100 * hi:+6.1f}]")
        diff, lo, hi = diff_interval(overall[label], overall[labels[0]])
        print(f"  {'ALL DECKS':20s} {100 * diff:+6.1f}  [{100 * lo:+6.1f}, {100 * hi:+6.1f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gauntlet", "head2head", "roundrobin"])
    ap.add_argument("--subject", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--field", metavar="DIR")
    ap.add_argument("--field-label", default="field")
    ap.add_argument("--a", metavar="LABEL=DIR")
    ap.add_argument("--b", metavar="LABEL=DIR")
    ap.add_argument("--league", metavar="DIR")
    ap.add_argument("--subject-decks")
    ap.add_argument("--field-decks")
    ap.add_argument("--decks")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampled", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    # Resolve every path before the chdir below: build_pool's data paths are relative to src/.
    subjects = [_labelled(s) for s in args.subject]
    field = os.path.abspath(args.field) if args.field else None
    a = _labelled(args.a) if args.a else None
    b = _labelled(args.b) if args.b else None
    league = os.path.abspath(args.league) if args.league else None
    out = os.path.abspath(args.out) if args.out else None
    os.chdir(SRC_DIR)
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from evaluation import gauntlet, head_to_head, play_matchups, round_robin, tally, diff_interval, wilson_interval

    if args.mode == "gauntlet":
        assert subjects and field, "gauntlet needs at least one --subject and a --field"
        assert len({label for label, _ in subjects}) == len(subjects), "subject labels must be distinct"
        matchups = [m for label, path in subjects
                    for m in gauntlet(path, field, _decks(args.subject_decks), _decks(args.field_decks),
                                      subject_label=label, field_label=args.field_label)]
    elif args.mode == "head2head":
        assert a and b, "head2head needs --a and --b"
        matchups = head_to_head(a[1], b[1], _decks(args.decks), label_a=a[0], label_b=b[0])
    else:
        assert league, "roundrobin needs --league"
        matchups = round_robin(league, _decks(args.decks))
    print(f"{args.mode}: {len(matchups)} matchups x {args.games} games = {len(matchups) * args.games} games "
          f"({'sampled' if args.sampled else 'greedy'}, seed {args.seed}, {args.workers} workers)", flush=True)

    kw = {"games": args.games, "seed": args.seed, "greedy": not args.sampled}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = play_matchups(matchups, executor=executor, **kw)
    else:
        results = play_matchups(matchups, **kw)

    if args.mode == "gauntlet":
        _gauntlet_report(results, [label for label, _ in subjects], tally, wilson_interval, diff_interval)
    else:
        print(f"\n{'a vs b':44s}  a's score")
        for r in results:
            print(f"{r.a.name + ' vs ' + r.b.name:44s}  {_fmt(tally([r]), wilson_interval)}")
        if args.mode == "head2head":
            print(f"{'ALL DECKS (' + a[0] + ' vs ' + b[0] + ')':44s}  {_fmt(tally(results), wilson_interval)}")
    if out:
        with open(out, "w") as f:
            json.dump([{"a": r.a.name, "b": r.b.name, "games": r.games, "a_wins": r.a_wins, "b_wins": r.b_wins,
                        "no_winner": r.no_winner} for r in results], f, indent=1)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
