"""Scan recorded engine event logs for the signature of the collection hang.

The hang (2026-08-19) is an unbounded priority round: game/turn.py's inner
`while True` only exits when every player passes consecutively, and ANY
non-pass action resets consecutive_passes to 0 while the actor KEEPS priority
(turn.py:546). So a policy that can keep taking some legal action never lets
the counter reach 2, state.turn_number never advances, horizon (which bounds
turns only) never fires, and every iteration appends another ~42.6 KB
transition to the rollout buffer. Measured growth of ~1 GB/min matches ~400
decisions/sec, i.e. one policy forward pass each -- a live loop, not a leak.

This script does not need to catch a live hang. A loop that runs forever in
training should leave a NEAR-miss in already-recorded games: an unusually long
single-turn decision run, or a long repetition of one action label. So it
reports, per game:

  - the longest run of consecutive decisions with no turn_number change
  - the longest run of the same chosen action label
  - total decisions, for outlier spotting

CAVEAT worth stating loudly: these logs come from --eval, which plays GREEDY.
Training SAMPLES from the policy. A loop reachable only by sampling may be
entirely absent here, so a clean scan does NOT clear the codebase -- it only
fails to convict. That is why this runs alongside the sampled soak, not
instead of it.

Usage:
  python analysis/scan_action_loops.py                    # all round-robin logs
  python analysis/scan_action_loops.py --top 15
"""
import sys
import json
import glob
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

REPO = Path(__file__).resolve().parent.parent.parent


def chosen_label(ev):
    """The label of the action this decision actually took, or None."""
    for c in ev.get("candidates") or ():
        if c.get("index") == ev.get("chosen_index"):
            lbl = c.get("fixed_label")
            if lbl:
                return lbl
            # pointer_identity is a DICT (card name + slot), not a string --
            # str() it so it can key a Counter and compare run-to-run. A
            # pointer action repeating on the SAME permanent is exactly the
            # signature being hunted, so identity must stay part of the key.
            ident = c.get("pointer_identity")
            return f"<ptr {ident}>" if ident is not None else "<ptr ?>"
    return None


def scan_game(events):
    """Returns (total_decisions, longest_same_turn_run, longest_repeat_run,
    repeated_label, turn_of_worst)."""
    total = 0
    run_turn, best_turn_run, worst_turn = 0, 0, None
    cur_turn = object()
    run_lbl, best_lbl_run, cur_lbl, worst_lbl = 0, 0, None, None
    for ev in events:
        if ev.get("kind") != "decision_weights":
            continue
        total += 1
        t = ev.get("turn")
        if t == cur_turn:
            run_turn += 1
        else:
            cur_turn, run_turn = t, 1
        if run_turn > best_turn_run:
            best_turn_run, worst_turn = run_turn, t
        lbl = chosen_label(ev)
        if lbl == cur_lbl:
            run_lbl += 1
        else:
            cur_lbl, run_lbl = lbl, 1
        if run_lbl > best_lbl_run:
            best_lbl_run, worst_lbl = run_lbl, lbl
    return total, best_turn_run, best_lbl_run, worst_lbl, worst_turn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=str(REPO / "logs" / "*double_round_robin*.json"))
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    rows, files = [], sorted(glob.glob(args.glob))
    label_hist = Counter()
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {Path(f).name}: {exc}")
            continue
        for g in d.get("games", []):
            total, turn_run, lbl_run, lbl, turn = scan_game(g.get("events", []))
            rows.append({"file": Path(f).name, "game": g.get("game_index"),
                         "decks": f"{g.get('deck_a')} vs {g.get('deck_b')}",
                         "decisions": total, "same_turn_run": turn_run,
                         "repeat_run": lbl_run, "label": lbl, "turn": turn})
            if lbl_run >= 5:
                label_hist[lbl] += 1

    print(f"scanned {len(files)} file(s), {len(rows)} games\n")
    if not rows:
        return
    dec = sorted(r["decisions"] for r in rows)
    print(f"decisions/game: median {dec[len(dec)//2]}  p90 {dec[int(.9*len(dec))]}  max {dec[-1]}")
    st = sorted(r["same_turn_run"] for r in rows)
    print(f"longest same-TURN decision run: median {st[len(st)//2]}  p90 {st[int(.9*len(st))]}  max {st[-1]}")
    rp = sorted(r["repeat_run"] for r in rows)
    print(f"longest same-ACTION repeat run : median {rp[len(rp)//2]}  p90 {rp[int(.9*len(rp))]}  max {rp[-1]}\n")

    print(f"--- top {args.top} by same-turn decision run (loop suspects) ---")
    for r in sorted(rows, key=lambda r: -r["same_turn_run"])[:args.top]:
        print(f"  {r['same_turn_run']:5d} decisions in turn {r['turn']}  "
              f"({r['decisions']:5d} total)  {r['decks']:36} {r['file'][-22:]}")
    print(f"\n--- top {args.top} by repeated identical action ---")
    for r in sorted(rows, key=lambda r: -r["repeat_run"])[:args.top]:
        print(f"  {r['repeat_run']:5d}x {str(r['label'])[:44]:46} turn {r['turn']}  {r['decks']}")
    if label_hist:
        print("\n--- actions repeated >=5x in a row, by how many games ---")
        for lbl, n in label_hist.most_common(15):
            print(f"  {n:4d} games  {lbl}")


if __name__ == "__main__":
    main()
