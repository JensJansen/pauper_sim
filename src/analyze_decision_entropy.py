"""One-off: compare a deck's own policy-entropy (rl.agent._log_decision_weights'
now-logged dist.entropy(), nats -- same units/definition as PPO's own entropy
bonus and the training-time entropy trend in TRAINING_IMPROVEMENT_OPTIONS.md)
across turn phases, and against a second deck as a baseline. Built to test
whether dmir_terror's upkeep/draw-step reflexive tapping (analyze_mana_burn_
turns.py found 67% of its high-burn turns cast NOTHING at all, often tapping
out fully before the draw step) is a locked-in, near-zero-entropy habit
specific to that decision context, rather than a magnitude-of-penalty problem
(a much harsher dense curve already failed to move this deck's burn at all,
see rl.rewards deploy_reward_v3 vs v2).

Usage: python analyze_decision_entropy.py DECK [--baseline DECK2] [--games N] [--seed N]
"""
import argparse
import random
import statistics
from collections import defaultdict

from repo_paths import CHECKPOINTS_DIR
from rl.pool import build_pool
from rl.train import _constant_pairing, collect_rollout
from rl.league_runner import load_frozen_stack, D_MODEL
from analyze_mana_burn_by_turn import DEFAULT_ROSTER, _load_deck

_PHASE_ORDER = ["untap", "upkeep", "draw", "main1", "declare_attackers",
                "declare_blockers", "combat_damage", "main2", "end"]


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


def _report(label, logs):
    by_phase = defaultdict(list)
    tap_entropy, pass_entropy = defaultdict(list), defaultdict(list)
    for log in logs:
        for e in log:
            # network == "main": excludes the separate pregame mulligan
            # decision_weights events (rl.mulligan), which carry no entropy
            # field -- this investigation is about in-game tapping decisions.
            if e["kind"] != "decision_weights" or e["active_idx"] != 0 or e["network"] != "main":
                continue
            phase = e["phase"]
            ent = e["entropy"]
            by_phase[phase].append(ent)
            top_label = e["candidates"][0]["fixed_label"] if e["candidates"] else None
            chosen = next((c for c in e["candidates"] if c["index"] == e["chosen_index"]), None)
            chosen_label = chosen["fixed_label"] if chosen else None
            if chosen_label is not None and chosen_label.startswith("Tap"):
                tap_entropy[phase].append(ent)
            elif chosen_label == "Pass":
                pass_entropy[phase].append(ent)

    print(f"\n=== {label}: mean decision entropy (nats) by phase ===")
    for phase in _PHASE_ORDER:
        vals = by_phase.get(phase)
        if not vals:
            continue
        print(f"  {phase:18s} mean={statistics.mean(vals):.3f}  median={statistics.median(vals):.3f}  n={len(vals)}")

    print(f"  --- upkeep/draw only, split by chosen action ---")
    for phase in ("upkeep", "draw"):
        t, p = tap_entropy.get(phase, []), pass_entropy.get(phase, [])
        if t:
            print(f"  {phase} chose Tap*: mean entropy={statistics.mean(t):.3f}, n={len(t)}, "
                  f"frac < 0.05 nats (near-deterministic)={sum(1 for x in t if x < 0.05)/len(t):.0%}")
        if p:
            print(f"  {phase} chose Pass:  mean entropy={statistics.mean(p):.3f}, n={len(p)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("deck", help="deck name, e.g. dmir_terror")
    p.add_argument("--baseline", default=None, help="second deck to compare against, e.g. mono_red_rally")
    p.add_argument("--league", default="4_deck_subleague_test")
    p.add_argument("--opponent_league", default="4_deck_subleague_gauntlet")
    p.add_argument("--games", type=int, default=25, help="games per opponent (default 25; roster is 4 decks)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    shared = load_frozen_stack(vocab.size)
    rng = random.Random(args.seed)

    print(f"playing {args.deck} ({args.league}) vs {args.opponent_league} roster, {args.games}/opponent...")
    logs = _play_games(args.deck, CHECKPOINTS_DIR / args.league, CHECKPOINTS_DIR / args.opponent_league,
                        DEFAULT_ROSTER, args.games, rng, shared, decklists, deck_ctxs, fixed_tables)
    _report(args.deck, logs)

    if args.baseline:
        print(f"\nplaying {args.baseline} ({args.league}) vs {args.opponent_league} roster, {args.games}/opponent...")
        logs2 = _play_games(args.baseline, CHECKPOINTS_DIR / args.league, CHECKPOINTS_DIR / args.opponent_league,
                             DEFAULT_ROSTER, args.games, rng, shared, decklists, deck_ctxs, fixed_tables)
        _report(args.baseline, logs2)


if __name__ == "__main__":
    main()
