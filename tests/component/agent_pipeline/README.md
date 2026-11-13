# Component: agent pipeline (training, league, evaluation numbers, checkpoints)

The non-engine, non-replay code that turns games into a trained policy and judges it. Plan:
[`planning/AGENT_PIPELINE_TEST_PLAN.md`](../../../planning/AGENT_PIPELINE_TEST_PLAN.md) (decisions Q1-Q6 recorded there).

**Contract.** (A) the PPO update's maths equals an independent reference; (B) what rollouts record is lossless, attributed to the
right seat and bucket, and legal under its own mask, in process and across the worker boundary; (C) the league pool weights and
books opponents correctly; (D) schedules are keyed on cumulative games, so a split run behaves as one; (E) rollback restores
exactly the named snapshot and a session killed at any checkpoint write resumes; (G) the statistics decisions rest on are right;
(H) configs and entry points are valid; (I) the loop measurably learns on a task with a known optimum.

## Where the tests are

| File | Establishes |
|---|---|
| `tests/unit/rl/training/test_ppo_math.py` | GAE vs the textbook recursion; clipped surrogate, value loss, entropy, k3 KL, clip fraction vs references on constructed batches; grad clipping; `target_kl` early stop; update-wide `*_mean` stats (Q5); an action its own mask forbids raises (Q6) |
| `tests/unit/rl/training/test_rollout_parallel_boundary.py` | buffer round trip, identities dropped and only those, `_sanitize_events`, game split across workers, every keyword reaches every worker, `gather` merges nothing lost. |
| `test_worker_equivalence.py` | a game played by the worker equals the in-process game on the same weights and seed (real games, also through submit/gather) |
| `test_rollout_invariants.py` | closed episodes, reward only at the terminal step, each seat paid for its own result, a timeout scores -1 for both seats (Q3), bucket routing, every action legal under its own mask, forced decisions not recorded |
| `test_learning_signal.py` | a bandit written into buffers: the loop finds the rewarded action, the critic learns the return, credit reaches a delayed reward; negative controls |
| `tests/unit/rl/league/test_pfsp_reference.py`, `test_schedules_properties.py`, `test_runner_helpers.py` | exact PFSP weight, sampling frequencies, key round trip, corrupt stats file fails loudly; snapshot cadence over any session split, mulligan flush cadence, schedules, `progress.json` bookkeeping, flush and snapshot helpers |
| `test_resume_rollback.py` | `run_rollback` exactness, reversibility, refusal and dry-run; a session killed at each checkpoint write leaves a resumable directory with `progress.json` never ahead |
| `tests/unit/rl/decision/test_agent_helpers.py`, `tests/unit/rl/model/test_mulligan_update_reference.py` | padded masks, Pass test, all-False failure; the mulligan REINFORCE step vs a reference |
| `tests/unit/evaluation/test_statistics_reference.py`, `test_history_ladder.py` | Wilson, z, MDE, trend, verdicts vs scipy and by simulation (coverage, false-positive rate, power); the age ladder |
| `tests/unit/scripts/test_training_configs.py` | every key in every config is one a launcher reads; ranges; distinct leagues; the CLI spec |
| `test_metrics_schema.py` | a real session writes exactly the documented fields |
| `test_probe.py` | the policy-movement probe on real games: well-formed distributions, batching-independent, save/load, refuses a changed table |
| `test_pins.py` | golden pins: network parameter names/shapes, checkpoint and snapshot keys, every metrics kind's fields (`pins.json`; **regeneration needs the owner's sign-off**) |
| `tests/unit/test_pipeline_ledger.py` | every public function/class of the pipeline is named by a test |

## Findings ledger (both fixed 2026-11-13)

- **AP-1** (fixed) `submit_league_rollout` shipped `v.cpu()`, which on a CPU run returns the SAME tensor, so the dict aliased the
  live weights. `ProcessPoolExecutor` pickles a task on a feeder thread after `submit()` returns, so with `overlap_collect` the
  pickle could see weights the concurrent PPO update had already moved. Now ships `v.detach().cpu().clone()`. Test:
  `test_the_shipped_weights_are_cpu_copies_of_the_live_weights_at_call_time` in `test_rollout_parallel_boundary.py`.
- **AP-2** (fixed) `evaluation.probe.record_probe_set`'s docstring called the set "a pure function of (agents, seed)"; the agents
  sample from torch's global generator, which `seed` does not touch. The docstring now says to seed torch. Test: `test_probe.py`.

## Method notes

Decisions of 2026-11-12: a resumed run need only restore the saved state exactly (no bit-identical continuation); a horizon
timeout scores -1 for both seats and is terminal for GAE (a timeout should never occur: decking is an instant loss); `ppo_update`
keeps its last-minibatch numbers and ADDS `*_mean` fields; a stored action its own mask forbids raises. Reference implementations
are written from the maths, not copied from the code under test; the independence limit (same author) is the usual one.

**Cost.** ~4.5 min for the component plus the unit files it lists. **Planted bugs.** `mutations.py`
(`python tests/support/mutation_sweep.py agent_pipeline`). **Not built / deferred.** A real `coverage` measurement (the package is
not installed; the ledger gate is a name-level floor), GPU parity, a rigged real-engine learning matchup (bandit only), analysis
scripts (out of scope).
