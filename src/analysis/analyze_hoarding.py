"""Standing check for the ONE risk deploy_reward_v5 knowingly took (and v6
keeps unchanged): dropping
the cleanup-discard penalty q removed the only explicit penalty for hoarding
anywhere in the reward (v5's own comment: the bet is that a terminal win/loss
signal can attribute hoarding's cost on its own, since hoarded cards stay
visible in game state, unlike burnt mana). If that bet is wrong, it shows up
as decks holding cards and eating repeated forced cleanup discards.

Reports, per deck, over a matchup-log glob: mean cleanup-discard TURNS per
game (the exact quantity q used to be computed from -- PlayerState.
cleanup_discard_turns, one count per turn a player was over hand size at
cleanup), plus mean cards pitched. Run against two different runs' logs to
compare (e.g. v4's logs/freshreset_*.json vs v5's logs/v5_*.json).

Usage:
  python analysis/analyze_hoarding.py '../logs/v5_{target}_vs_*_25games.json'
"""
import glob
import json
import sys

DECKS = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]
DEFAULT_GLOB = "../logs/v5_{target}_vs_*_25games.json"


def stats_for(pattern, target):
    """(games, mean discard-turns/game, mean cards-pitched/game) for target's
    own seat. A discard TURN is a distinct turn with at least one cleanup
    'Choose:' pick -- matching how game.effects.state_based increments
    cleanup_discard_turns (once per turn over the limit, not per card)."""
    games = discard_turns = cards = 0
    for path in sorted(glob.glob(pattern.format(target=target))):
        for game_rec in json.load(open(path))["games"]:
            ev = game_rec["events"]
            seats = [s for s, key in ((0, "deck_a"), (1, "deck_b")) if game_rec[key] == target]
            for seat in seats:
                games += 1
                turns = set()
                for e in ev:
                    if (e["kind"] == "decision_weights" and e.get("network") == "main"
                            and e["active_idx"] == seat and e["phase"] == "end"):
                        chosen = next(c for c in e["candidates"] if c["index"] == e["chosen_index"])
                        if (chosen.get("fixed_label") or "").startswith("Choose:"):
                            turns.add(e["turn"])
                            cards += 1
                discard_turns += len(turns)
    if not games:
        return 0, float("nan"), float("nan")
    return games, discard_turns / games, cards / games


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    print(f"{'deck':<16} {'games':>7} {'discard_turns/game':>19} {'cards_pitched/game':>19}")
    for target in DECKS:
        games, turns, cards = stats_for(pattern, target)
        print(f"{target:<16} {games:>7} {turns:>19.2f} {cards:>19.2f}")


if __name__ == "__main__":
    main()
