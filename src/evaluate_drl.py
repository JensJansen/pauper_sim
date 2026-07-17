"""Thin wiring script: load a previously saved model cold (no training
step) and run it through a batch of real games, exactly like every other
policy comparison in this project -- "train once, evaluate against many
shuffles later," independent of the run that produced the model.

Reward function, model class, and horizon must match what the model was
saved with -- TrainingHarness.load() checks this and raises clearly if not.
"""

import time

from sb3_contrib import MaskablePPO

import game
import rewards
from harness import TrainingHarness

REWARD_FN = rewards.assembled_with_resource_quality
SCORING_FNS = [rewards.tron_online_score]  # score 2 (MULTI_DECK_PLAN.md Phase M7) -- a live argument to load(), same as REWARD_FN, never restored from the saved model's own metadata
MODEL_CLS = MaskablePPO
HORIZON = 6
ON_THE_PLAY = True
LOAD_PATH = "models/pilot"
NUM_GAMES = 500
EVAL_SEED = 1

if __name__ == "__main__":
    harness = TrainingHarness.load(
        LOAD_PATH, reward_fn=REWARD_FN, model_cls=MODEL_CLS,
        horizon=HORIZON, on_the_play=ON_THE_PLAY, scoring_fns=SCORING_FNS,
    )
    print(f"Loaded {LOAD_PATH}/ -- trained for {harness.total_timesteps_trained} timesteps.")

    t0 = time.time()
    results = harness.evaluate(num_games=NUM_GAMES, horizon=HORIZON, seed=EVAL_SEED)
    dt = time.time() - t0

    print(f"\nEvaluated {NUM_GAMES} games in {dt:.1f}s ({dt / NUM_GAMES * 1000:.2f} ms/game).\n")
    game.print_report(results, HORIZON)
