# Component: training (PPO update and rollout collection)

**Contract.** The update must optimise the policy that actually played: replaying a recorded rollout through `ppo_update`
reproduces the rollout's own log-probabilities (importance ratio exactly 1 at lr 0), for any minibatch grouping, including
the recurrent memory and the previous-pointer input.

Covered today: `test_rollout_update_equivalence.py` (with two negative controls proving the measure notices a perturbed
encoder and a dropped previous-pointer input).

**Cost.** ~30 seconds. **Planted bugs.** The previous-pointer and encoder mutations in `component/encoder/mutations.py` exercise it.

**Gaps.** Closed 2026-11-12: GAE, advantage normalisation, the loss terms, reward attribution and episode segmentation are now
tested in the [`agent_pipeline`](../agent_pipeline/README.md) component and its unit files; this one stays the
rollout/update equivalence check.
