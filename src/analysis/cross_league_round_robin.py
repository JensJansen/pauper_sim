"""Full cross-league round robin: EVERY main-league deck vs EVERY twin-league
deck, seat-swapped.

What already existed and why it is not this. league_runner._run_eval_vs_gauntlet
plays each deck against the SAME-NAMED deck in the independently-trained twin
-- rakdos vs rakdos, elves vs elves. Four mirror pairings, and the vs_gauntlet
metric in metrics.jsonl is exactly that. It answers "has my rakdos out-trained
their rakdos", which is the right question for detecting a co-adapted bubble
but says nothing about cross-archetype matchups.

This plays the full 4x4 = 16 pairings, so the output is a matchup MATRIX:
main-league rakdos vs twin elves, vs twin dmir, and so on. That surfaces
archetype-level facts the mirror diagonal hides -- e.g. a deck that beats its
own twin while losing to a different twin archetype it never meets in-league.

Seat swapping is not optional. On-the-play is a real advantage in Magic, so
each pairing splits its games half with the main deck on seat 0 and half on
seat 1; a lopsided seat split would read as a strategy result. Both halves use
the same seed, so the two arms see common random numbers.

Greedy (argmax) play, matching rl.agent's own "greedy=True is for eval"
contract -- this measures the policy's best play, not an exploration sample.
No training, no checkpoint writes: nets load with optimizer=None and
collect_rollout runs with record=False.

    python -u analysis/cross_league_round_robin.py --games 40 \
        --main 4_deck_subleague_test --twin 4_deck_subleague_gauntlet \
        --out ../logs/cross_league_30k.json
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl import checkpoint as ckpt_io  # noqa: E402
from rl.agent import SeatAgent  # noqa: E402
from rl.league_runner import HORIZON, build_deck_net  # noqa: E402
from rl.mulligan import MulliganNet  # noqa: E402
from rl.pool import build_pool  # noqa: E402
from rl.train import _constant_pairing, collect_rollout  # noqa: E402
from repo_paths import CHECKPOINTS_DIR  # noqa: E402


def _load_league(league_dir, deck_names, vocab, fixed_tables):
    """Every deck's live net + mulligan net from one league directory, in eval
    mode. optimizer=None throughout: this only ever needs inference weights,
    and loading an optimizer would be the one way a read-only benchmark could
    touch training state."""
    nets, mulls = {}, {}
    for name in deck_names:
        live_path = f"{league_dir}/{name}/live.pt"
        if not os.path.exists(live_path):
            raise SystemExit(f"no live.pt for {name} in {league_dir} -- has that league trained this deck?")
        net = build_deck_net(vocab.size, len(fixed_tables[name]),
                             ckpt_io.trunk_hidden_from_deck_checkpoint(live_path))
        ckpt_io.load_deck_checkpoint(live_path, net)
        net.eval()
        nets[name] = net
        mnet = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        mnet.eval()
        mulls[name] = mnet
    return nets, mulls


def _play(main_agent, twin_agent, main_decklist, twin_decklist, n_games, seed, main_seat):
    """n_games with the main deck pinned to `main_seat`. Returns (wins, games)
    counted for the MAIN deck; horizon timeouts (state.winner is None) are
    excluded from both, never scored as a loss."""
    agents = [main_agent, twin_agent] if main_seat == 0 else [twin_agent, main_agent]
    decklists = [main_decklist, twin_decklist] if main_seat == 0 else [twin_decklist, main_decklist]
    pairing = _constant_pairing(agents, decklists, [None, None], [None, None])

    tally = {"wins": 0, "decided": 0}

    def on_game_end(state):
        if state.winner is None:
            return  # horizon timeout -- nobody won; excluded, not a loss
        tally["decided"] += 1
        if state.winner == main_seat:
            tally["wins"] += 1

    collect_rollout(pairing, n_games, HORIZON, random.Random(seed), device="cpu",
                    record=False, greedy=True, on_game_end=on_game_end)
    return tally["wins"], tally["decided"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="4_deck_subleague_test", help="main league checkpoint dir name")
    ap.add_argument("--twin", default="4_deck_subleague_gauntlet", help="twin league checkpoint dir name")
    ap.add_argument("--games", type=int, default=40,
                    help="games per pairing, split evenly across both seats (rounded down to even)")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", default=None, help="write the matrix here as JSON")
    args = ap.parse_args()

    games = max(2, args.games - args.games % 2)  # even, so the seat split is exact
    half = games // 2

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    main_dir = os.path.join(CHECKPOINTS_DIR, args.main)
    twin_dir = os.path.join(CHECKPOINTS_DIR, args.twin)

    # The roster is whichever decks BOTH leagues have actually trained, not
    # build_pool()'s full catalog -- that returns every deck in
    # data/league_decks.json, while a league's own config narrows it (these two
    # train a 4-deck subset). Deriving it from the checkpoints on disk means
    # this cannot drift out of sync with either config, and a deck the twin has
    # not reached yet is skipped rather than crashing the run.
    deck_names = [n for n in decklists
                  if os.path.exists(f"{main_dir}/{n}/live.pt") and os.path.exists(f"{twin_dir}/{n}/live.pt")]
    if not deck_names:
        raise SystemExit(f"no deck has a live.pt in BOTH {args.main} and {args.twin}")
    skipped = [n for n in decklists if n not in deck_names]
    if skipped:
        print(f"skipping {len(skipped)} deck(s) absent from one or both leagues: {', '.join(skipped)}")

    main_nets, main_mulls = _load_league(main_dir, deck_names, vocab, fixed_tables)
    twin_nets, twin_mulls = _load_league(twin_dir, deck_names, vocab, fixed_tables)

    print(f"cross-league round robin: {len(deck_names)}x{len(deck_names)} = "
          f"{len(deck_names) ** 2} pairings x {games} games ({half} per seat), greedy, seed={args.seed}")
    print(f"  main = {args.main}\n  twin = {args.twin}")

    t0 = time.time()
    rows = []
    for a, b in itertools.product(deck_names, deck_names):
        main_agent = SeatAgent(main_nets[a], main_mulls[a], deck_ctxs[a])
        twin_agent = SeatAgent(twin_nets[b], twin_mulls[b], deck_ctxs[b])
        # Same seed for both halves: common random numbers across the seat swap.
        w0, d0 = _play(main_agent, twin_agent, decklists[a], decklists[b], half, args.seed, main_seat=0)
        w1, d1 = _play(main_agent, twin_agent, decklists[a], decklists[b], half, args.seed, main_seat=1)
        wins, decided = w0 + w1, d0 + d1
        rate = wins / decided if decided else float("nan")
        rows.append({"main_deck": a, "twin_deck": b, "wins": wins, "decided": decided,
                     "win_rate": rate, "on_the_play_wins": w0, "on_the_draw_wins": w1,
                     "is_mirror_pairing": a == b})
        print(f"  main {a:16s} vs twin {b:16s}: {wins:3d}/{decided:3d} = {rate:.3f} "
              f"(seat0 {w0}/{d0}, seat1 {w1}/{d1})", flush=True)

    decided_total = sum(r["decided"] for r in rows)
    wins_total = sum(r["wins"] for r in rows)
    mirrors = [r for r in rows if r["is_mirror_pairing"]]
    crosses = [r for r in rows if not r["is_mirror_pairing"]]

    def _rate(subset):
        d = sum(r["decided"] for r in subset)
        return (sum(r["wins"] for r in subset) / d) if d else float("nan")

    print(f"\noverall main-league win rate: {wins_total}/{decided_total} = "
          f"{wins_total / decided_total if decided_total else float('nan'):.3f}")
    # The diagonal is what vs_gauntlet already reported; the off-diagonal is
    # what this tool adds. Printed apart so the new information is not averaged
    # into the number that already existed.
    print(f"  mirror pairings only (what vs_gauntlet already covers): {_rate(mirrors):.3f}")
    print(f"  cross-archetype pairings only (new here):               {_rate(crosses):.3f}")
    print(f"done in {time.time() - t0:.1f}s")

    if args.out:
        payload = {"main_league": args.main, "twin_league": args.twin, "games_per_pairing": games,
                   "seed": args.seed, "rows": rows,
                   "overall_win_rate": wins_total / decided_total if decided_total else None,
                   "mirror_win_rate": _rate(mirrors), "cross_win_rate": _rate(crosses)}
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
