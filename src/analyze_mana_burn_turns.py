"""One-off: pull the worst single-turn tagged-mana-burn turns for one deck
and print their full event narrative (mana taps/spends, casts, delve
amount choices, the final burn). Written to test a specific hypothesis:
dmir_terror runs Tolarian Terror (4x, {6}{U}) and Gurmag Angler (2x,
{6}{B}), both Delve (702.66) -- their real cost swings from 7 mana down to
1 depending on graveyard size AND how much of it the agent chooses to
exile. Float-first means mana can be tapped before the delve amount is
even chosen, so a turn that taps for a big hard-cast then delves deep (or
vice versa) can strand floated mana structurally, not just from sloppy
play -- this pulls real turns to check whether that's actually happening.

Usage: python analyze_mana_burn_turns.py DECK [--games N] [--seed N] [--top N] [--min_burn N]
"""
import argparse
import random

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl.league_runner import load_frozen_stack, D_MODEL
from analyze_mana_burn_by_turn import DEFAULT_ROSTER, _load_deck, _per_turn_tagged_burn

_ENVELOPE_SKIP = {"kind", "turn", "turn_player_idx"}
_SKIP_KINDS = {"priority_flip", "decision_weights", "pass", "untap_step", "phase_change", "turn_start"}


def _fmt_event(e):
    fields = ", ".join(f"{k}={v}" for k, v in e.items() if k not in _ENVELOPE_SKIP)
    return f"  [T{e['turn']} {e.get('phase')}] {e['kind']}: {fields}"


def _play_games(deck_name, deck_league_dir, opponent_league_dir, opponents, games_per_opp, rng,
                 shared, decklists, deck_ctxs, fixed_tables):
    from rl.agent import SeatAgent
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("deck", help="deck name, e.g. dmir_terror")
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--opponent_league", default="4_deck_subleague_gauntlet")
    p.add_argument("--games", type=int, default=50, help="games per opponent (default 50; roster is 4 decks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=12, help="how many worst turns to print in full")
    p.add_argument("--min_burn", type=int, default=3, help="only print turns burning at least this many tagged pips")
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    rng = random.Random(args.seed)

    print(f"playing {args.deck} ({args.league}) vs {args.opponent_league} roster, {args.games}/opponent...")
    logs = _play_games(args.deck, CHECKPOINTS_DIR / args.league, CHECKPOINTS_DIR / args.opponent_league,
                        DEFAULT_ROSTER, args.games, rng, shared, decklists, deck_ctxs, fixed_tables)

    seat = 0
    worst = []  # (burn, game_index, turn)
    for gi, log in enumerate(logs):
        per_turn = _per_turn_tagged_burn(log, seat)
        for turn, burnt in per_turn.items():
            if burnt >= args.min_burn:
                worst.append((burnt, gi, turn))
    worst.sort(reverse=True)
    print(f"\n{len(worst)} turns with >= {args.min_burn} tagged pips burnt, across {len(logs)} games "
          f"({len(worst) / len(logs):.2f}/game)")

    for burnt, gi, turn in worst[:args.top]:
        log = logs[gi]
        window = [e for e in log if e["turn"] == turn and e["kind"] not in _SKIP_KINDS]
        print(f"\n--- game {gi}, turn {turn}: {burnt} tagged pips burnt this turn ---")
        for e in window:
            print(_fmt_event(e))


if __name__ == "__main__":
    main()
