"""One-off cross-population tournament: every deck in league A's checkpoint
dir plays every deck in league B's checkpoint dir -- the full cross product
across BOTH rosters (deck_a x deck_b, including mismatched-name pairs), not
just same-name pairs the way rl.league_runner._run_eval_vs_gauntlet compares.
games_per_matchup games each, greedy (policy's actual best play), no
training/checkpointing. Written for benchmarking checkpoints/
4_deck_subleague_test (the actively-training subleague) against checkpoints/
4_deck_subleague_gauntlet (the frozen reference pod) before/after a training
batch, per the owner's own before/after comparison request.

Also reports mana-burn rates per side (game_over's mana_burnt_total/
mana_burnt_total_single_pip, seat-indexed -- rl.train.collect_rollout) --
added to compare overtapping between two reward-policy populations (e.g.
deploy_reward_v3 vs. the v2 gauntlet twin) head to head under identical
opponents, not just win rate.

Usage:
  python analysis/run_cross_league_eval.py LEAGUE_A LEAGUE_B [--games N] [--seed N] [--log PATH]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths` / `rl.*` -- these live one level up now that this script sits in analysis/
import argparse
import itertools
import json
import os
import random
import time

from repo_paths import CHECKPOINTS_DIR
from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet
from rl.agent import SeatAgent
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl import checkpoint as ckpt_io
from rl.league_runner import HORIZON, load_frozen_stack, D_MODEL

DEFAULT_ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]


def _load_deck_nets(league_dir, names, shared, fixed_tables):
    live_nets, mulligan_nets = {}, {}
    for name in names:
        net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
        ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)  # optimizer=None: eval only needs weights
        net.eval()
        live_nets[name] = net
        mnet = MulliganNet(shared)
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
    args = p.parse_args()

    roster = args.roster.split(",") if args.roster else DEFAULT_ROSTER
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)

    live_a, mull_a = _load_deck_nets(CHECKPOINTS_DIR / args.league_a, roster, shared, fixed_tables)
    live_b, mull_b = _load_deck_nets(CHECKPOINTS_DIR / args.league_b, roster, shared, fixed_tables)

    rng = random.Random(args.seed)
    results = []
    # Per-deck mana-burn accumulators, keyed by (league_letter, deck_name) --
    # a deck's own burn rate is compared across every opponent it faced, not
    # just averaged blindly into one league-wide number, since burn rate is
    # expected to vary a lot by archetype (Priest of Titania/Elves bursts).
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
        # COMMON RANDOM NUMBERS: half the games with league A's deck at seat 0,
        # half with the seats exchanged, BOTH driven from the same seed. Because
        # collect_rollout draws starting_idx per game from that rng, replaying
        # the seed with the seats swapped hands the play to the other league on
        # the very same shuffles -- on-the-play is then balanced exactly rather
        # than in expectation.
        #
        # The itertools.product grid does NOT already do this. It pairs every
        # ordered (deck_from_A, deck_from_B), which are different MATCHUPS, not
        # the same matchup from both seats -- so before this, league A's deck sat
        # at seat 0 in every single game, and the MIRRORS (a == b, the most
        # informative cells) were completely unpaired.
        pair_seed = rng.randrange(2 ** 31)
        half = args.games // 2
        fwd, played_f = _half([agent_a, agent_b], [decklists[a], decklists[b]], half, pair_seed)
        rev, played_r = _half([agent_b, agent_a], [decklists[b], decklists[a]], args.games - half, pair_seed)
        played = played_f + played_r

        # In `rev` the seats are exchanged, so seat 0 is league B and seat 1 is
        # league A -- every per-seat read below flips accordingly.
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
    print(f"OVERALL: {args.league_a} {total_a}-{total_b} {args.league_b} "
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
