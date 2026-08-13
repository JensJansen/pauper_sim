"""Standing check for the degenerate "never play lands" pattern: across a 4x4
matchup batch (25 games per pairing, all 4 decks as target), find games where a
deck's own seat played zero lands and made zero mana taps despite the game
running past turn 10 -- and for every one found, pull the main-network
decision_weights entropy at each point a land-play option existed but Pass was
chosen instead. Reports occurrence rate and entropy stats per deck.

Takes the log glob as argv[1] (it has moved three times now -- freshreset_*,
v5_*, v5_20k/* -- so it is an argument rather than a constant to edit). A
{target} placeholder is honored if present, but the default scans every file
and picks each deck up wherever it appears: deck_a is always seat 0, so a
per-target glob would sample each deck almost entirely on the play.

Usage:
  python analyze_land_pattern_all_decks.py '../logs/v5_20k/*_25games.json'
"""
import glob
import json
import sys

DEFAULT_GLOB = "../logs/v5_20k/*_25games.json"
DECKS = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]
MIN_TURNS = 10


def target_seats(game, target):
    seats = []
    if game["deck_a"] == target:
        seats.append(0)
    if game["deck_b"] == target:
        seats.append(1)
    return seats


def analyze(pattern, target):
    total_instances = 0
    flagged = []

    for path in sorted(glob.glob(pattern.format(target=target))):
        data = json.load(open(path))
        for game in data["games"]:
            opponent_label = game["deck_b"] if game["deck_a"] == target else game["deck_a"]
            ev = game["events"]
            over = next(e for e in ev if e["kind"] == "game_over")
            for seat in target_seats(game, target):
                total_instances += 1
                taps = [e for e in ev if e["kind"] == "mana_tap" and e["active_idx"] == seat]
                lands = [e for e in ev if e["kind"] == "zone_move" and e.get("to_zone") == "battlefield"
                         and e.get("card_type") == "LAND" and e["active_idx"] == seat]
                if taps or lands or over["turn_won"] <= MIN_TURNS:
                    continue
                flagged.append((path, opponent_label, game["game_index"], seat, over["turn_won"], ev))

    all_entropies = []
    for path, opponent_label, gi, seat, turn_won, ev in flagged:
        main_dw = [e for e in ev if e["kind"] == "decision_weights" and e.get("network") == "main"
                   and e["active_idx"] == seat]
        for e in main_dw:
            land_opts = [c for c in e["candidates"] if "Play land" in c.get("fixed_label", "")]
            if not land_opts:
                continue
            chosen = next(c for c in e["candidates"] if c["index"] == e["chosen_index"])
            if chosen["fixed_label"] != "Pass":
                continue
            all_entropies.append(e["entropy"])

    return total_instances, flagged, all_entropies


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLOB
    print(f"logs={pattern}")
    print(f"{'deck':<16} {'instances':>10} {'flagged':>8} {'rate':>7} {'n_decisions':>12} {'mean_entropy':>13}")
    for target in DECKS:
        total, flagged, entropies = analyze(pattern, target)
        rate = 100 * len(flagged) / total if total else 0.0
        mean_e = sum(entropies) / len(entropies) if entropies else float("nan")
        print(f"{target:<16} {total:>10} {len(flagged):>8} {rate:>6.1f}% {len(entropies):>12} {mean_e:>13.4f}")
        for path, opponent_label, gi, seat, turn_won, ev in flagged:
            print(f"    vs {opponent_label} game_index={gi} seat={seat} turn_won={turn_won}")


if __name__ == "__main__":
    main()
