"""The rollout loop and PPO update math: collect_rollout + RolloutBuffer
(train), GAE + the PPO update itself (ppo), and the ProcessPoolExecutor
multiprocessing plumbing for league collection (rollout_parallel)."""
