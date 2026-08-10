"""One-off: dig into game.effects.casting._log_target_fizzle occurrences for
one deck -- how often a targeted spell fails to resolve because its target
left the battlefield first (real Magic 608.2b), and what actually happened
in the surrounding turn(s). Written to investigate dmir_terror's "held
mana, target died" hypothesis raised alongside analyze_mana_burn_by_turn.py
(a fizzled spell's OWN cost was already paid at cast time -- a fizzle is not
itself mana burn -- so this checks the real event trace instead of assuming
the two are linked from the console print frequency alone).

Usage: python analyze_target_fizzle.py DECK [--games N] [--seed N] [--examples N]
       (--games N = games per opponent; roster has 4 opponents)
"""
import argparse
import random
from collections import Counter

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl.league_runner import load_frozen_stack, D_MODEL
from analyze_mana_burn_by_turn import DEFAULT_ROSTER, _load_deck, _per_turn_tagged_burn

_ENVELOPE_SKIP = {"kind", "turn", "turn_player_idx"}


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
    p.add_argument("--league", default="4_deck_subleague_test", help="checkpoints/<league> the deck is loaded from")
    p.add_argument("--opponent_league", default="4_deck_subleague_gauntlet")
    p.add_argument("--games", type=int, default=5, help="games per opponent (default 5; roster is 4 decks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--examples", type=int, default=8, help="how many concrete fizzle narratives to print")
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    rng = random.Random(args.seed)

    print(f"playing {args.deck} ({args.league}) vs {args.opponent_league} roster, {args.games}/opponent...")
    logs = _play_games(args.deck, CHECKPOINTS_DIR / args.league, CHECKPOINTS_DIR / args.opponent_league,
                        DEFAULT_ROSTER, args.games, rng, shared, decklists, deck_ctxs, fixed_tables)

    n_games = len(logs)
    seat = 0  # deck-under-analysis is always seat 0, per _play_games/_constant_pairing
    total_fizzles = 0
    by_card = Counter()
    burn_on_fizzle_turn, burn_on_clean_turn = [], []
    examples = []

    for log in logs:
        per_turn_burn = _per_turn_tagged_burn(log, seat)
        # target_fizzle has no per-caster field of its own, but
        # effects.stack.resolve_top_of_stack sets active_idx = the spell's
        # controller before running its resolve closure (where the fizzle
        # check happens) -- active_idx at fizzle time IS the caster.
        game_fizzles = [(i, e) for i, e in enumerate(log)
                         if e["kind"] == "target_fizzle" and e["active_idx"] == seat]
        fizzle_turns = set()
        for i, e in game_fizzles:
            total_fizzles += 1
            by_card[e["card"]] += 1
            fizzle_turns.add(e["turn"])
            if len(examples) < args.examples:
                # Full same-turn(+prev-turn) window FIRST, find the fizzle's own
                # position in it, THEN trim to a trailing slice ending at the
                # fizzle -- trimming before locating it (the original bug here)
                # let a verbose turn's decision_weights spam evict the fizzle
                # itself before it was ever found, printing an unrelated
                # earlier chunk of the turn instead.
                window = [ev for ev in log if ev["turn"] in (e["turn"] - 1, e["turn"]) and ev["kind"] not in
                          ("resolution_begin", "resolution_complete", "priority_flip", "decision_weights")]
                cutoff = window.index(e) + 1 if e in window else len(window)
                examples.append((e, window[max(0, cutoff - 40):cutoff]))
        for turn, burnt in per_turn_burn.items():
            (burn_on_fizzle_turn if turn in fizzle_turns else burn_on_clean_turn).append(burnt)
        # Turns with zero recorded burn never appear in per_turn_burn at all --
        # count those explicitly too so the "clean turn" average isn't biased high.
        game_len = max((ev["turn"] for ev in log), default=0)
        for t in range(1, game_len + 1):
            if t not in per_turn_burn:
                (burn_on_fizzle_turn if t in fizzle_turns else burn_on_clean_turn).append(0)

    print(f"\n{total_fizzles} target-fizzle events across {n_games} games "
          f"({total_fizzles / n_games:.2f}/game)")
    print("by card:", dict(by_card.most_common()))

    import statistics
    print(f"\nmean tagged pips burnt: turns WITH a fizzle = "
          f"{statistics.mean(burn_on_fizzle_turn) if burn_on_fizzle_turn else 0:.2f} "
          f"(n={len(burn_on_fizzle_turn)}), turns with NO fizzle = "
          f"{statistics.mean(burn_on_clean_turn) if burn_on_clean_turn else 0:.2f} (n={len(burn_on_clean_turn)})")

    print(f"\n=== {len(examples)} example fizzles, with same-turn(+prev-turn) context ===")
    for e, window in examples:
        print(f"\n--- fizzle: {e['card']} -> {e['target']} (turn {e['turn']}) ---")
        for ev in window:
            print(_fmt_event(ev))


if __name__ == "__main__":
    main()
