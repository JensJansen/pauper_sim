# Component: training (PPO update and rollout collection)

**Contract.** The update must optimise the policy that actually played: replaying a recorded rollout through `ppo_update`
reproduces the rollout's own log-probabilities (importance ratio exactly 1 at lr 0), for any minibatch grouping, including
the recurrent memory and the previous-pointer input.

Covered today: `test_rollout_update_equivalence.py` (with two negative controls proving the measure notices a perturbed
encoder and a dropped previous-pointer input).

**Cost.** ~30 seconds. **Planted bugs.** The previous-pointer and encoder mutations in `component/encoder/mutations.py` exercise it.

**Gaps (next work).** GAE/returns/advantage normalisation against a plain reference implementation; PPO loss terms
(clipping, value loss, entropy, KL early stop) on constructed batches; reward attribution; episode segmentation.
