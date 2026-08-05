"""Summarizes checkpoints/<league>/metrics.jsonl -- the per-iteration "ppo" /
"mulligan" / "vs_history" / "vs_gauntlet" / "vs_heuristic" records
rl.league_runner's _run_session appends during every training run (see
_append_metric). Plain-text report, stdlib only (no plotting library in
requirements.txt) -- enough to see whether entropy is collapsing, loss has
actually moved, a deck's win rate against its own archived past self has
changed, or (the two gauntlet checks -- see README's Gauntlet section) it
beats an independently-trained twin population and a hand-authored heuristic
opponent, without re-deriving any of it by eye from stdout scrollback.

Usage: python report_metrics.py <league_dir>
  e.g. python report_metrics.py ../checkpoints/4_deck_subleague_test
"""
import json
import sys
from collections import defaultdict


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def report(records):
    """Returns the report as a list of printed lines (also prints them) --
    returning them makes this testable without capturing stdout."""
    lines = []
    by_deck_kind = defaultdict(list)
    for r in records:
        by_deck_kind[(r["deck"], r["kind"])].append(r)

    for deck, kind in sorted(by_deck_kind):
        rows = by_deck_kind[(deck, kind)]
        if kind == "ppo":
            entropies = [r["entropy"] for r in rows]
            half = max(1, len(entropies) // 2)
            first_half = sum(entropies[:half]) / half
            second_half = sum(entropies[-half:]) / len(entropies[-half:])
            lines.append(
                f"{deck} [ppo] {len(rows)} iterations -- entropy {entropies[0]:.3f} -> {entropies[-1]:.3f} "
                f"(first-half mean {first_half:.3f}, second-half mean {second_half:.3f}), "
                f"latest policy_loss={rows[-1]['policy_loss']:.4f} value_loss={rows[-1]['value_loss']:.4f} "
                f"buffer_size={rows[-1]['buffer_size']} batch_size={rows[-1]['batch_size']}"
            )
        elif kind == "mulligan":
            lines.append(f"{deck} [mulligan] {len(rows)} updates -- latest loss={rows[-1]['loss']:.4f} (n={rows[-1]['n']})")
        elif kind == "vs_history":
            for r in rows:
                total = r["games"]
                win_rate = r["live_wins"] / total if total else float("nan")
                lines.append(
                    f"{deck} [vs_history:{r['label']}] live {r['live_wins']}/{total} ({win_rate:.0%}) "
                    f"vs its own archived past self (session {r['session']})"
                )
        elif kind == "vs_gauntlet":
            for r in rows:
                total = r["games"]
                win_rate = r["live_wins"] / total if total else float("nan")
                lines.append(
                    f"{deck} [vs_gauntlet] live {r['live_wins']}/{total} ({win_rate:.0%}) "
                    f"vs an independently-trained twin population (session {r['session']})"
                )
        elif kind == "vs_heuristic":
            for r in rows:
                total = r["games"]
                win_rate = r["live_wins"] / total if total else float("nan")
                lines.append(
                    f"{deck} [vs_heuristic] live {r['live_wins']}/{total} ({win_rate:.0%}) "
                    f"vs the hand-authored HeuristicAgent (session {r['session']})"
                )
    for line in lines:
        print(line)
    return lines


def main():
    if len(sys.argv) != 2:
        print("usage: python report_metrics.py <league_dir>")
        raise SystemExit(1)
    report(load(f"{sys.argv[1]}/metrics.jsonl"))


if __name__ == "__main__":
    main()
