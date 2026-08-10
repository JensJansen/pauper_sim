"""One-off: reconstruct a per-turn mana-burn timeline for one deck across
both reward-policy leagues (test/v3 vs gauntlet/v2), from mana_emptied
event logs (game.turn._empty_mana_pools) rather than the whole-game totals
run_cross_league_eval.py reports. Built to test a specific hypothesis raised
by that script's own output (dmir_terror's burn barely moved under v3's
harsher curve despite v3 clearly working for elves): does rl.rewards.
with_dense_mana_burn_penalty's WHOLE-GAME game_penalty_cap get exhausted by
one bad turn, after which the rest of that game floats mana penalty-free?
(with_dense_mana_burn_penalty's own docstring already names this as an
accepted, not fixed, tradeoff -- this checks whether it actually happens in
practice, and how often, rather than assuming it from the reward math alone.)

Replays the ACTUAL with_dense_mana_burn_penalty charge sequence (rl.rewards.
_hill, same c/p/cap constants deploy_reward_v2/v3 use) against each game's
real per-turn tagged-burn timeline, so "cap exhausted at turn N" is a real
reward-mechanics finding, not eyeballed off a raw burn histogram.

Usage: python analyze_mana_burn_by_turn.py DECK [--games N] [--seed N]
       (--games N = games per opponent; roster has 4 opponents, so total
       games per league = 4 * N)
"""
import argparse
import random
import statistics

from repo_paths import CHECKPOINTS_DIR
from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet
from rl.agent import SeatAgent
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl import checkpoint as ckpt_io
from rl.league_runner import load_frozen_stack, D_MODEL
from rl.rewards import _hill

DEFAULT_ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]

# (mana_burn_c, mana_burn_p, game_penalty_cap) -- rl.rewards deploy_reward_v2/v3's own values.
CURVES = {"v2 (gauntlet)": (3.3, 4.0, 2.0), "v3 (test)": (2.0, 2.5, 0.6)}


def _load_deck(league_dir, name, shared, fixed_tables):
    net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)
    net.eval()
    mnet = MulliganNet(shared)
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
    mnet.eval()
    return net, mnet


def _play_games(deck_name, deck_league_dir, opponent_league_dir, opponents, games_per_opp, rng,
                 shared, decklists, deck_ctxs, fixed_tables):
    """Plays deck_name (from deck_league_dir, always seat 0) against every
    name in opponents (from opponent_league_dir, seat 1), games_per_opp
    games each -- returns the flat list of full event_logs, one per game."""
    net, mnet = _load_deck(deck_league_dir, deck_name, shared, fixed_tables)
    all_logs = []
    for opp in opponents:
        onet, omnet = _load_deck(opponent_league_dir, opp, shared, fixed_tables)
        pairing = _constant_pairing(
            [SeatAgent(net, mnet, deck_ctxs[deck_name]), SeatAgent(onet, omnet, deck_ctxs[opp])],
            [decklists[deck_name], decklists[opp]], [None, None], [None, None])
        game_logs = []
        collect_rollout(pairing, games_per_opp, 120, rng, device="cpu",
                         record=False, greedy=True, game_logs=game_logs)
        all_logs.extend(game_logs)
    return all_logs


def _per_turn_tagged_burn(event_log, seat):
    """turn -> tagged pips burnt that turn, for `seat`, from mana_emptied events."""
    by_turn = {}
    for e in event_log:
        if e["kind"] != "mana_emptied":
            continue
        amt = e["pools_single_pip"].get(seat, 0)
        if amt:
            by_turn[e["turn"]] = by_turn.get(e["turn"], 0) + amt
    return by_turn


def _game_length(event_log):
    return max((e["turn"] for e in event_log), default=0)


def _replay_penalty(per_turn, c, p, cap):
    """Replays with_dense_mana_burn_penalty's actual charge logic against a
    real per-turn tagged-burn timeline: each turn's own charge is
    _hill(that turn's total, c, p) (mana_burn_penalty_credited always
    starts a turn at 0, and telescopes to exactly that by turn's end),
    clamped so the running whole-game total never exceeds `cap`. Returns
    (total_charged, turn cap was fully exhausted at, or None)."""
    total = 0.0
    exhausted_at = None
    for turn in sorted(per_turn):
        charge = min(_hill(per_turn[turn], c, p), cap - total)
        total += charge
        if exhausted_at is None and total >= cap - 1e-9:
            exhausted_at = turn
    return total, exhausted_at


def _summarize(label, logs, c, p, cap, max_turn_col=20):
    per_game = [(_game_length(log), _per_turn_tagged_burn(log, 0)) for log in logs]
    n = len(per_game)
    print(f"\n=== {label}: {n} games ===")

    print("turn: mean tagged pips burnt that turn (n games that reached it)")
    row = []
    for t in range(1, max_turn_col + 1):
        reached = [pt.get(t, 0) for length, pt in per_game if length >= t]
        if not reached:
            break
        row.append(f"t{t}={statistics.mean(reached):.2f}(n={len(reached)})")
    print("  " + "  ".join(row))

    worst = [(max(pt.values()) if pt else 0, max(pt, key=pt.get) if pt else None) for _, pt in per_game]
    worst_amounts = [w for w, _ in worst]
    worst_turns = [t for _, t in worst if t is not None]
    print(f"worst single turn: mean={statistics.mean(worst_amounts):.2f} pips, "
          f"median={statistics.median(worst_amounts):.1f}, "
          f">=3 pips in one turn: {sum(1 for w in worst_amounts if w >= 3) / n:.0%} of games, "
          f">=5 pips in one turn: {sum(1 for w in worst_amounts if w >= 5) / n:.0%} of games")
    if worst_turns:
        print(f"turn index of each game's own worst turn: mean={statistics.mean(worst_turns):.1f}, "
              f"median={statistics.median(worst_turns):.0f}")

    replayed = [_replay_penalty(pt, c, p, cap) for _, pt in per_game]
    totals = [t for t, _ in replayed]
    exhausted = [(length, ea) for (length, _pt), (_t, ea) in zip(per_game, replayed) if ea is not None]
    print(f"replayed whole-game mana-burn penalty charge (cap={cap}): "
          f"mean={statistics.mean(totals):.3f}/{cap}, "
          f"cap FULLY exhausted before game end in {len(exhausted) / n:.0%} of games")
    if exhausted:
        frac_of_game = [ea / length for length, ea in exhausted if length]
        print(f"  of those: exhausted at turn (mean={statistics.mean(ea for _, ea in exhausted):.1f}), "
              f"i.e. {statistics.mean(frac_of_game):.0%} of the way through the game on average -- "
              f"every turn after that floats mana with ZERO further reward penalty this game")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("deck", help="deck name, e.g. dmir_terror")
    p.add_argument("--games", type=int, default=50, help="games per opponent (default 50; roster is 4 decks)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    rng = random.Random(args.seed)

    test_dir = CHECKPOINTS_DIR / "4_deck_subleague_test"
    gauntlet_dir = CHECKPOINTS_DIR / "4_deck_subleague_gauntlet"

    print(f"playing {args.deck}: test/v3 vs gauntlet roster, {args.games}/opponent...")
    test_logs = _play_games(args.deck, test_dir, gauntlet_dir, DEFAULT_ROSTER, args.games, rng,
                             shared, decklists, deck_ctxs, fixed_tables)
    print(f"playing {args.deck}: gauntlet/v2 vs test roster, {args.games}/opponent...")
    gauntlet_logs = _play_games(args.deck, gauntlet_dir, test_dir, DEFAULT_ROSTER, args.games, rng,
                                 shared, decklists, deck_ctxs, fixed_tables)

    c, p_, cap = CURVES["v3 (test)"]
    _summarize(f"{args.deck} -- v3 (test league)", test_logs, c, p_, cap)
    c, p_, cap = CURVES["v2 (gauntlet)"]
    _summarize(f"{args.deck} -- v2 (gauntlet league)", gauntlet_logs, c, p_, cap)


if __name__ == "__main__":
    main()
