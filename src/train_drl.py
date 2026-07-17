"""Thin wiring script: pick a reward function + model class + hyperparams,
train a TrainingHarness, save it. Every piece-selection choice is a plain
variable below -- swapping any of them means editing this file, never
harness.py or tron_env.py (dependency injection, per DRL_PLAN.md).

Same script for pilot and full runs (DRL_PLAN.md) -- only the constants
below change. Currently configured for the first large-scale run: 50,000
episodes (exact count via SB3's StopTrainingOnMaxEpisodes, not a timestep
approximation), same network/hyperparameters as the Phase D8 pilot so the
comparison is a clean scale-up. TOTAL_TIMESTEPS is just a generous safety
cap in case episodes run longer than the pilot's ~13.5-step average as the
policy changes -- MAX_EPISODES is the real stopping condition.
"""

import time

from sb3_contrib import MaskablePPO

import rewards
from harness import TrainingHarness

REWARD_FN = rewards.assembled_with_resource_quality
SCORING_FNS = [rewards.tron_online_score]  # score 2 (MULTI_DECK_PLAN.md Phase M7) -- eval/logging only, never used in training
MODEL_CLS = MaskablePPO
MODEL_KWARGS = {"policy_kwargs": {"net_arch": [64, 64]}, "verbose": 1}
HORIZON = 6
ON_THE_PLAY = True
SEED = 0
MAX_EPISODES = 50_000
TOTAL_TIMESTEPS = 3_000_000  # safety cap; MAX_EPISODES is the real stopping condition
SAVE_PATH = "models/run_50k"

if __name__ == "__main__":
    harness = TrainingHarness(
        reward_fn=REWARD_FN, model_cls=MODEL_CLS, model_kwargs=MODEL_KWARGS,
        horizon=HORIZON, on_the_play=ON_THE_PLAY, seed=SEED, scoring_fns=SCORING_FNS,
    )

    t0 = time.time()
    harness.train(total_timesteps=TOTAL_TIMESTEPS, save_path=SAVE_PATH, max_episodes=MAX_EPISODES)
    dt = time.time() - t0

    print(f"\nTrained {harness.total_timesteps_trained} timesteps ({MAX_EPISODES} episodes) "
          f"in {dt:.1f}s ({dt / harness.total_timesteps_trained * 1000:.2f} ms/step).")
    print(f"Saved to {SAVE_PATH}/ (model.zip + metadata.json).")
