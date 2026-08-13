"""Standing check on whether the dense mana-burn penalty is still a GRADIENT
or has degenerated into a flat toll.

with_dense_mana_burn_penalty's game_penalty_cap bounds the whole-game charge.
Once a game hits that cap, every later burnt pip is charged exactly 0.0 -- the
policy pays the same whether it wastes nothing more or wastes everything, so
the shaping term stops shaping. That failure mode is invisible in the
burn-per-game numbers run_cross_league_eval.py reports (a saturated game and a
merely-bad game look identical in the reward), which is why it needs its own
check. It is what gated the v5 -> v6 switch: at deploy_reward_v5's
20,065-games/deck checkpoint dmir_terror and elves were saturating in 71%/64%
of games, roughly two thirds of the way through each one, so most of each
game was being played with the burn term already maxed out and inert.

Reports per deck, over a matchup-log glob:
  burn/game, turns/game, burn/turn        -- raw waste rate
  own turns/game, burn/own turn           -- ~85% of burn happens on the
                                             burner's own turn, so this is the
                                             per-decision rate that the
                                             per-turn curve actually sees
  burn turns/game, pips/burn turn         -- burn is bursty, not spread evenly
  charge/game, % saturating, cap died at  -- the actual question

Charges are replayed through the named reward's OWN charge_single_pip_burn
rather than a copy of its constants, so this can never drift out of sync with
rl.rewards the way analyze_mana_burn_by_turn.py's hardcoded CURVES table did.
One call per turn with that turn's total: _charge_single_pip_burn telescopes
against mana_burn_penalty_credited (reset every turn by game.turn's cleanup,
alongside mana_burnt_this_turn_single_pip), so how a turn's burn was split
across phase boundaries cannot change its total charge.

The default glob deliberately scans EVERY matchup file rather than the
'..._{target}_vs_*' shape analyze_hoarding.py/analyze_land_pattern_all_decks.py
use: deck_a is always seat 0, so a per-target glob would sample each deck
almost entirely on the play, and both game length and burn depend on that.
A {target} placeholder is still honored if one is supplied.

Usage:
  python analyze_burn_saturation.py [--logs '../logs/v5_*_25games.json']
                                    [--reward deploy_reward_v6]
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict

from rl import rewards

DECKS = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]
DEFAULT_GLOB = "../logs/v5_*_25games.json"


class _ChargeShim:
    """The three PlayerState fields _charge_single_pip_burn reads/mutates."""

    def __init__(self):
        self.mana_burnt_this_turn_single_pip = 0
        self.mana_burn_penalty_credited = 0.0
        self.mana_burn_penalty_charged_total = 0.0


def _games_for(pattern, target):
    """[(total turns, own turns, {turn: tagged pips burnt}, raw pips burnt)]
    for target's own seat, over every log matching the glob."""
    out = []
    for path in sorted(glob.glob(pattern.format(target=target))):
        for game in json.load(open(path))["games"]:
            ev = game["events"]
            over = next(e for e in ev if e["kind"] == "game_over")
            for seat, key in ((0, "deck_a"), (1, "deck_b")):
                if game[key] != target:
                    continue
                by_turn = defaultdict(int)
                for e in ev:
                    if e["kind"] == "mana_emptied":
                        by_turn[e["turn"]] += e["pools_single_pip"].get(str(seat), 0)
                own = len({e["turn"] for e in ev if e.get("turn_player_idx") == seat})
                out.append((over["turn_won"], own, {t: n for t, n in by_turn.items() if n},
                            over["mana_burnt_total"][seat]))
    return out


def _replay(per_turn, charge, cap):
    """(total charged, turn the cap was fully exhausted at or None), replaying
    the real charge function turn by turn."""
    shim = _ChargeShim()
    exhausted_at = None
    for turn in sorted(per_turn):
        shim.mana_burnt_this_turn_single_pip = per_turn[turn]
        shim.mana_burn_penalty_credited = 0.0  # game.turn resets this each turn
        charge(shim)
        if exhausted_at is None and shim.mana_burn_penalty_charged_total >= cap - 1e-9:
            exhausted_at = turn
    return shim.mana_burn_penalty_charged_total, exhausted_at


def _cap_of(charge):
    """The reward's own game_penalty_cap, read back by charging a shim far past
    it -- the constant lives in a closure, so this is the only way to get it
    without duplicating it here."""
    shim = _ChargeShim()
    for _ in range(100):
        shim.mana_burnt_this_turn_single_pip = 99
        shim.mana_burn_penalty_credited = 0.0
        charge(shim)
    return shim.mana_burn_penalty_charged_total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default=DEFAULT_GLOB, help="log glob; {target} placeholder optional")
    p.add_argument("--reward", default="deploy_reward_v6", help="rl.rewards name to replay")
    args = p.parse_args()

    reward_fn = getattr(rewards, args.reward)
    charge = reward_fn.charge_single_pip_burn
    cap = _cap_of(charge)
    print(f"{args.reward}: game_penalty_cap={cap:.2f}  logs={args.logs}")
    print(f"{'deck':<16}{'games':>6}{'burn/gm':>9}{'turns':>7}{'/turn':>7}"
          f"{'own t':>7}{'/own t':>8}{'burn t':>8}{'pips/bt':>9}"
          f"{'charge':>8}{'sat':>6}{'died at':>9}")
    for target in DECKS:
        games = _games_for(args.logs, target)
        if not games:
            print(f"{target:<16}{'(no logs matched)':>60}")
            continue
        n = len(games)
        burn = sum(g[3] for g in games)
        turns = sum(g[0] for g in games)
        own = sum(g[1] for g in games)
        burn_turns = [x for _, _, pt, _ in games for x in pt.values()]
        totals, died = [], []
        for length, _own, pt, _raw in games:
            total, at = _replay(pt, charge, cap)
            totals.append(total)
            if at is not None and length:
                died.append(at / length)
        print(f"{target:<16}{n:>6}{burn / n:>9.2f}{turns / n:>7.1f}{burn / turns:>7.2f}"
              f"{own / n:>7.1f}{burn / own:>8.2f}{len(burn_turns) / n:>8.2f}"
              f"{statistics.mean(burn_turns):>9.2f}{statistics.mean(totals):>8.2f}"
              f"{len(died) / n:>5.0%}"
              f"{(f'{statistics.mean(died):.0%}' if died else '-'):>9}")
    print("sat = % of games that fully exhausted the cap (every later burnt pip free);\n"
          "died at = how far through those games it happened, as % of game length")


if __name__ == "__main__":
    main()
