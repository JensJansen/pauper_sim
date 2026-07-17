"""Regression snapshot tooling for the multi-deck refactor
(docs/MULTI_DECK_PLAN.md Phase M0). Two distinct uses:

1. `main()` produced the ORIGINAL fixture (data/multi_deck_regression_
   snapshot.json), captured once, before Phase M1 touched anything,
   driven by uniform-random-legal-action choices rather than the
   heuristic (which won't exist after Phase M5). That fixture is checked
   in and never regenerated -- it has to keep reflecting pre-refactor
   behavior forever, as the one remaining way to verify accuracy once
   Phase M5 deletes the heuristic this project's other baselines were
   compared against. `main()` refuses to overwrite it if it already
   exists (see below) precisely so a later, careless re-run can't destroy
   that ground truth.

2. `run_one(seed)` is reused after every later phase (imported directly,
   not via `main()`) to re-capture CURRENT behavior on the same seeds, for
   comparison against the frozen fixture from (1). Its return shape tracks
   whatever GameState currently looks like -- e.g. Phase M3 renamed
   turn_assembled -> turn_won and retired turn_online entirely, so a
   diff script comparing current output against the frozen fixture has to
   know how to map between the two schemas itself, not expect them to
   match key-for-key forever.
"""

import json
import os

import game
import tron_env

SEEDS = range(50)
HORIZON = 6
ON_THE_PLAY = True
OUT_PATH = "data/multi_deck_regression_snapshot.json"


def _zone_snapshot(state):
    return {
        "hand": sorted(c.name for c in state.hand),
        "battlefield": sorted([p.card_def.name, p.tapped] for p in state.battlefield),
        "graveyard": sorted(c.name for c in state.graveyard),
    }


def run_one(seed):
    state_rng = game.random.Random(seed)
    action_rng = game.random.Random(seed)  # independent stream, same seed value
    action_log = []

    def choose_action(state):
        mask = tron_env.legal_action_mask(state)
        legal = [i for i, ok in enumerate(mask) if ok]
        action = action_rng.choice(legal)
        if action == tron_env.PASS_ACTION:
            action_log.append("Pass")
            return None
        name, _, execute_fn = tron_env.ACTIONS[action]
        action_log.append(name)
        return lambda: execute_fn(state)

    state = game.run_game(game.TRON_DECKLIST, game.tron_terminated, state_rng, ON_THE_PLAY, HORIZON, choose_action)
    return {
        "seed": seed,
        "actions": action_log,
        "turn_won": state.turn_won,
        "final_turn_number": state.turn_number,
        "final_zones": _zone_snapshot(state),
    }


def main():
    if os.path.exists(OUT_PATH):
        raise RuntimeError(
            f"{OUT_PATH} already exists -- refusing to overwrite the frozen Phase M0 "
            "baseline. Import and call run_one(seed) directly for phase-by-phase "
            "verification instead of running this script's main()."
        )
    snapshot = [run_one(seed) for seed in SEEDS]
    with open(OUT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Wrote {len(snapshot)} games to {OUT_PATH}")

    terminated = sum(1 for g in snapshot if g["turn_won"] is not None)
    print(f"{terminated}/{len(snapshot)} assembled within horizon {HORIZON} (uniform-random play, not heuristic -- low rate expected)")


if __name__ == "__main__":
    main()
