"""Cross-population tournament: every deck in league A's checkpoint dir plays
every deck in league B's checkpoint dir -- the full cross product across both
rosters (deck_a x deck_b, including mismatched-name pairs), not just same-name
pairs the way rl.league.league_runner._run_eval_vs_gauntlet compares.
games_per_matchup games each, greedy, no training/checkpointing.

Also reports mana-burn rates per side (game_over's mana_burnt_total /
mana_burnt_total_single_pip, seat-indexed) to compare overtapping between two
reward-policy populations under identical opponents, not just win rate.

Usage:
  python analysis/eval/run_cross_league_eval.py LEAGUE_A LEAGUE_B [--games N] [--seed N] [--log PATH]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/
import argparse
import itertools
import json
import os
import random
import time

from repo_paths import CHECKPOINTS_DIR
from rl.model.mulligan import MulliganNet
from rl.decision.agent import SeatAgent
from rl.roster import build_pool
from rl.training.train import _constant_pairing, collect_rollout
from rl import checkpoint as ckpt_io
from rl.league.league_runner import HORIZON, build_deck_net, load_vintage_agent

DEFAULT_ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]


def _load_deck_nets(league_dir, names, vocab, fixed_tables):
    live_nets, mulligan_nets = {}, {}
    for name in names:
        net = build_deck_net(vocab.size, len(fixed_tables[name]))
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)  # eval only needs weights
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(net.encoder)
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
        mnet.eval()
        mulligan_nets[name] = mnet
    return live_nets, mulligan_nets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("league_a", help="checkpoints/<league_a>, e.g. 4_deck_subleague_test")
    p.add_argument("league_b", help="checkpoints/<league_b>, e.g. 4_deck_subleague_gauntlet")
    p.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                    help="Deck subset to test on both sides (default: the 4-deck subleague/gauntlet roster).")
    p.add_argument("--games", type=int, default=50, help="Games per matchup (default 50).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log", type=str, default=None, metavar="PATH", help="Write the summary JSON here.")
    # Vintages make the comparison budget-matched, since two populations are
    # rarely at the same games/deck at the same wall-clock moment.
    p.add_argument("--vintage-a", type=str, default="live", metavar="N|live",
                    help="Snapshot id for league_a (default live). snapshot N ~ N*192 games/deck.")
    p.add_argument("--vintage-b", type=str, default="live", metavar="N|live",
                    help="Snapshot id for league_b (default live).")
    args = p.parse_args()

    roster = args.roster.split(",") if args.roster else DEFAULT_ROSTER
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()

    def _load_side(league, vintage):
        if vintage == "live":
            return _load_deck_nets(CHECKPOINTS_DIR / league, roster, vocab, fixed_tables)
        # Unwrap the SeatAgent load_vintage_agent returns back into the (net,
        # mulligan) pair this script's pairing builder wants.
        vid = int(vintage)
        nets, mulls = {}, {}
        for name in roster:
            ag = load_vintage_agent(str(CHECKPOINTS_DIR / league), name, vid, deck_ctxs[name])
            nets[name], mulls[name] = ag.main, ag.mulligan
        return nets, mulls

    live_a, mull_a = _load_side(args.league_a, args.vintage_a)
    live_b, mull_b = _load_side(args.league_b, args.vintage_b)
    print(f"vintages: A={args.vintage_a}  B={args.vintage_b}", flush=True)

    rng = random.Random(args.seed)
    results = []
    burn_by_deck = {}  # (letter, name) -> [games, mana_burnt_total, mana_burnt_total_single_pip]

    def _record_burn(letter, name, n_games, total, total_single_pip):
        acc = burn_by_deck.setdefault((letter, name), [0, 0, 0])
        acc[0] += n_games
        acc[1] += total
        acc[2] += total_single_pip

    def _half(agents, decks, n_games, seed):
        """One orientation: agents[0]/decks[0] at seat 0. Returns the raw
        game_over events plus the count played."""
        pairing = _constant_pairing(agents, decks, [None, None], [None, None])
        game_logs = []
        _bufs, _mull, played = collect_rollout(pairing, n_games, HORIZON, random.Random(seed),
                                                device="cpu", record=False, greedy=True,
                                                game_logs=game_logs)
        return [e for ev in game_logs for e in ev if e["kind"] == "game_over"], played

    t0 = time.time()
    for a, b in itertools.product(roster, roster):
        agent_a = SeatAgent(live_a[a], mull_a[a], deck_ctxs[a])
        agent_b = SeatAgent(live_b[b], mull_b[b], deck_ctxs[b])
        # Common random numbers: half the games with A at seat 0, half with
        # seats swapped, both from the same seed -- since collect_rollout draws
        # starting_idx per game from that rng, this balances on-the-play exactly
        # rather than in expectation.
        pair_seed = rng.randrange(2 ** 31)
        half = args.games // 2
        fwd, played_f = _half([agent_a, agent_b], [decklists[a], decklists[b]], half, pair_seed)
        rev, played_r = _half([agent_b, agent_a], [decklists[b], decklists[a]], args.games - half, pair_seed)
        played = played_f + played_r

        # In `rev` seats are exchanged (seat 0 = league B), so per-seat reads flip.
        a_wins = sum(1 for e in fwd if e["winner"] == 0) + sum(1 for e in rev if e["winner"] == 1)
        b_wins = sum(1 for e in fwd if e["winner"] == 1) + sum(1 for e in rev if e["winner"] == 0)
        no_winner = played - a_wins - b_wins
        a_burnt = sum(e["mana_burnt_total"][0] for e in fwd) + sum(e["mana_burnt_total"][1] for e in rev)
        b_burnt = sum(e["mana_burnt_total"][1] for e in fwd) + sum(e["mana_burnt_total"][0] for e in rev)
        a_burnt_single_pip = (sum(e["mana_burnt_total_single_pip"][0] for e in fwd)
                              + sum(e["mana_burnt_total_single_pip"][1] for e in rev))
        b_burnt_single_pip = (sum(e["mana_burnt_total_single_pip"][1] for e in fwd)
                              + sum(e["mana_burnt_total_single_pip"][0] for e in rev))
        _record_burn("a", a, played, a_burnt, a_burnt_single_pip)
        _record_burn("b", b, played, b_burnt, b_burnt_single_pip)
        results.append({"deck_a": a, "deck_b": b, "games": played,
                         "a_wins": a_wins, "b_wins": b_wins, "no_winner": no_winner,
                         "a_mana_burnt_total": a_burnt, "b_mana_burnt_total": b_burnt,
                         "a_mana_burnt_total_single_pip": a_burnt_single_pip,
                         "b_mana_burnt_total_single_pip": b_burnt_single_pip})
        print(f"  {a} ({args.league_a}) vs {b} ({args.league_b}): "
              f"{a_wins}-{b_wins} ({no_winner} no-winner) of {played}, "
              f"mana burnt/game: {a_burnt / played:.2f} ({a_burnt_single_pip / played:.2f} tagged) vs. "
              f"{b_burnt / played:.2f} ({b_burnt_single_pip / played:.2f} tagged)", flush=True)

    total_games = sum(r["games"] for r in results)
    total_a = sum(r["a_wins"] for r in results)
    total_b = sum(r["b_wins"] for r in results)
    print(f"cross-league eval done: {total_games} games in {time.time() - t0:.1f}s")

    # Score matrix: cell = league_a deck's win rate for (row deck_a, col deck_b).
    by_pair = {(r["deck_a"], r["deck_b"]): r for r in results}
    col_w = max(len(n) for n in roster) + 1
    print(f"\nscore matrix ({args.league_a} row's win rate vs {args.league_b} col):")
    print(" " * col_w + "".join(f"{n:>{col_w}}" for n in roster))
    for a in roster:
        row = []
        for b in roster:
            r = by_pair[(a, b)]
            row.append(f"{r['a_wins'] / r['games']:>{col_w}.1%}")
        print(f"{a:<{col_w}}" + "".join(row))

    print(f"\nOVERALL: {args.league_a} {total_a}-{total_b} {args.league_b} "
          f"({total_a / total_games:.1%} - {total_b / total_games:.1%}) over {total_games} games")

    burn_summary = {}
    print("mana burnt/game by deck (raw, tagged-single-pip):")
    for (letter, name), (n, total, total_single_pip) in sorted(burn_by_deck.items()):
        league = args.league_a if letter == "a" else args.league_b
        burn_summary[f"{league}/{name}"] = {
            "games": n, "mana_burnt_total_per_game": total / n,
            "mana_burnt_total_single_pip_per_game": total_single_pip / n,
        }
        print(f"  {league}/{name}: {total / n:.2f} ({total_single_pip / n:.2f} tagged) over {n} games")

    out = {"league_a": args.league_a, "league_b": args.league_b, "roster": roster,
           "vintage_a": args.vintage_a, "vintage_b": args.vintage_b,
           "games_per_matchup": args.games, "seed": args.seed, "results": results,
           "total_games": total_games, "total_a_wins": total_a, "total_b_wins": total_b,
           "mana_burnt_by_deck": burn_summary}
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote summary to {args.log}")


if __name__ == "__main__":
    main()
