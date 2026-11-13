"""Planted bugs for the AGENT-PIPELINE component (rl/training, rl/league, rl/model/mulligan, rl/decision, evaluation, run_rollback,
config_loader). Maintained jointly with the repo owner: when behavior changes, add the mutation that guards it, and treat any
survivor as a hole in the tests, not a nuisance. Plan: planning/AGENT_PIPELINE_TEST_PLAN.md (section 6).

Run:  python tests/support/mutation_sweep.py agent_pipeline          (all)  |  ... agent_pipeline 3 7  |  ... --baseline

Each entry: (file under src/, exact old text -- must occur exactly once --, replacement, what bug it models).

Equivalent mutants found and removed on 2026-11-12 (a second guard in the code makes the planted change a no-op, so no test can or
should catch them): the `dones_[t]` test in GAE's `next_value` (next_nonterminal already zeroes it), collect_rollout's
`if bucket is None: continue` (the recording gate before it already keeps those seats out), and the ladder's early `rung >` break
(the later `games <=` filter repeats it)."""

COMPONENT = "agent_pipeline"
TIMEOUT = 900  # seconds per mutation; a survivor runs the whole component, a hang is a catch
ENV = {}

U = "tests/unit"
AP = "tests/component/agent_pipeline"
TEST_PATHS = [
    f"{U}/rl/training/test_ppo_math.py", f"{U}/rl/training/test_rollout_parallel_boundary.py",
    f"{U}/rl/league/test_pfsp_reference.py", f"{U}/rl/league/test_schedules_properties.py",
    f"{U}/rl/league/test_runner_helpers.py", f"{U}/rl/league/test_league.py", f"{U}/rl/test_rewards.py",
    f"{U}/rl/decision/test_agent_helpers.py", f"{U}/rl/model/test_mulligan_update_reference.py",
    f"{U}/evaluation/test_statistics_reference.py", f"{U}/evaluation/test_history_ladder.py",
    f"{U}/scripts/test_config_loader.py", f"{U}/scripts/test_training_configs.py",
    f"{AP}/test_worker_equivalence.py", f"{AP}/test_rollout_invariants.py", f"{AP}/test_learning_signal.py",
    f"{AP}/test_pins.py", f"{AP}/test_resume_rollback.py", f"{AP}/test_metrics_schema.py", f"{AP}/test_probe.py",
]

PPO = "rl/training/ppo.py"
RP = "rl/training/rollout_parallel.py"
LG = "rl/league/league.py"
LR = "rl/league/league_runner.py"
TR = "rl/training/train.py"
RW = "rl/rewards.py"
ST = "evaluation/stats.py"
VD = "evaluation/verdicts.py"

MUTATIONS = [
    # --- PPO update ---
    (PPO, "next_value = 0.0 if dones_[t] or t + 1 >= n else values_[t + 1]", "next_value = 0.0 if dones_[t] or t + 1 >= n else values_[t]",
     "GAE bootstraps with this step's own value instead of the next step's"),
    (PPO, "next_nonterminal = 0.0 if dones_[t] else 1.0", "next_nonterminal = 1.0",
     "GAE carries the running advantage across a game boundary"),
    (PPO, "last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae", "last_gae = delta + gamma * next_nonterminal * last_gae",
     "GAE ignores gae_lambda"),
    (PPO, "delta = rewards_[t] + gamma * next_value * next_nonterminal - values_[t]", "delta = rewards_[t] + next_value * next_nonterminal - values_[t]",
     "GAE does not discount the bootstrapped value"),
    (PPO, "adv = (adv - adv.mean()) / (max(adv_std, adv_norm_floor) + 1e-8)", "adv = adv - adv.mean()", "advantages are not normalised"),
    (PPO, "max(adv_std, adv_norm_floor)", "adv_std", "adv_norm_floor is ignored"),
    (PPO, "torch.clamp(ratio, 1 - clip_range, 1 + clip_range)", "torch.clamp(ratio, 1 - clip_range, 1 + 2 * clip_range)", "the upper clip bound is too loose"),
    (PPO, "torch.min(surr1, surr2)", "torch.max(surr1, surr2)", "the surrogate takes the optimistic instead of the pessimistic term"),
    (PPO, "ratio = torch.exp(new_logp - old_logp_mb)", "ratio = torch.exp(old_logp_mb - new_logp)", "the importance ratio is inverted"),
    (PPO, "value_loss = (((values_pred - ret_mb) ** 2) * valid).sum() / n_valid", "value_loss = (((values_pred - ret_mb) ** 2) * valid).sum() / valid.numel()",
     "the value loss is diluted by padding"),
    (PPO, "entropy = (dist.entropy() * valid).sum() / n_valid", "entropy = dist.entropy().mean()", "the entropy mean includes padded steps"),
    (PPO, "loss = policy_loss + vf_coef * value_loss - ent_coef * entropy", "loss = policy_loss + vf_coef * value_loss + ent_coef * entropy",
     "the entropy bonus has the wrong sign"),
    (PPO, "loss = policy_loss + vf_coef * value_loss - ent_coef * entropy", "loss = policy_loss + value_loss - ent_coef * entropy", "vf_coef is ignored"),
    (PPO, "torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)", "pass", "gradient clipping removed"),
    (PPO, "float(epoch_kl_sum) / epoch_kl_n > target_kl", "float(epoch_kl_sum) / epoch_kl_n < target_kl", "the KL early stop fires when the policy moved LITTLE"),
    (PPO, "if target_kl is not None and epoch_kl_n and", "if False and", "the KL early stop never fires"),
    (PPO, "bounds.append((start_idx, i + 1))", "bounds.append((start_idx, i))", "episode boundary off by one"),
    (PPO, "approx_kl = (((ratio - 1) - torch.log(ratio)) * valid).sum() / n_valid", "approx_kl = ((ratio - 1) * valid).sum() / n_valid", "the KL estimate is the plain mean ratio"),
    (PPO, "sums[k] = sums[k] + v.detach() * n_valid", "sums[k] = v.detach() * n_valid", "the update-wide means are the last minibatch only"),
    (PPO, "        if not legal.all():", "        if False:", "an action its own mask forbids is trained on instead of raising"),
    # --- worker boundary ---
    (RP, "return [(idx, row, None) for idx, row, _identity in token_list]", "return token_list", "live identities are shipped across the process boundary"),
    (RP, "chunks = [base + (1 if i < remainder else 0) for i in range(n_workers)]", "chunks = [base for i in range(n_workers)]", "the remainder of the games is dropped"),
    (RP, "pfsp_power=pfsp_power,\n                         stratify_0land_pct=stratify_0land_pct)\n        for chunk", "pfsp_power=None,\n                         stratify_0land_pct=stratify_0land_pct)\n        for chunk", "pfsp_power is not forwarded to the workers"),
    (RP, "outcomes.extend(worker_outcomes)", "outcomes = list(worker_outcomes)", "only the last worker's outcomes survive the merge"),
    (RP, "games_played += played", "games_played = played", "only the last worker's game count survives"),
    (RP, "    for entry in entries:", "    for entry in entries[1:]:", "the first transition of a shipped buffer is lost"),
    (RP, "mull_by_deck.setdefault(name, []).extend(tr)", "mull_by_deck[name] = list(tr)", "mulligan transitions from earlier workers are overwritten"),
    # --- rewards and rollout attribution ---
    (RW, "return 1.0 if state.winner == state.active_idx else -1.0", "return -1.0 if state.winner == state.active_idx else 1.0", "winner and loser are paid the wrong way round"),
    (RW, "        if not done:\n            return 0.0\n        return 1.0", "        if not done:\n            return 0.01\n        return 1.0", "a reward is paid before the terminal step"),
    (TR, "game_buffers[seat].add(*pending[seat], reward, True)", "game_buffers[seat].add(*pending[seat], reward, False)", "the last transition of a game is not marked done"),
    (TR, "        elif opp_is_live:", "        elif True:", "a frozen snapshot opponent's transitions are recorded into a bucket"),
    # --- league pool and schedules ---
    (LG, "return PFSP_FLOOR + (1.0 - win_rate) ** self.pfsp_power", "return PFSP_FLOOR + win_rate ** self.pfsp_power", "PFSP weights the opponents the deck already beats"),
    (LG, "return PFSP_FLOOR + (1.0 - win_rate) ** self.pfsp_power", "return (1.0 - win_rate) ** self.pfsp_power", "the PFSP floor is removed"),
    (LG, "win_rate = wins / games if games else 0.5", "win_rate = wins / games if games else 0.0", "an unseen opponent is assumed unbeaten"),
    (LG, "stats[opponent_key] = (wins + int(won), games + 1)", "stats[opponent_key] = (wins + int(won), games)", "recorded games are not counted"),
    (LG, "rng.random() < checkpoint_rate", "rng.random() > checkpoint_rate", "checkpoint_rate selects snapshots at 1 minus the rate"),
    (LR, "(games_before + games_per_iteration) // snapshot_every_games > games_before // snapshot_every_games", "(games_before + games_per_iteration) // snapshot_every_games >= games_before // snapshot_every_games",
     "a snapshot is taken every iteration"),
    (LR, "or iteration == n_iterations - 1)", "or False)", "the mulligan flush skips the session's last iteration"),
    (LR, "mull_by_deck_accum[name] = []", "pass", "flushed mulligan transitions are trained on again"),
    (LR, "for name in ctx[\"train_decks\"]:  # only TRAIN decks change", "for name in ctx[\"deck_names\"]:  # only TRAIN decks change", "frozen decks get new snapshots"),
    (LR, "checkpoint_progress(league_dir, games_before + games_per_iteration)", "checkpoint_progress(league_dir, games_before + 2 * games_per_iteration)", "progress.json runs ahead of the weights"),
    (LR, "cumulative = base + played", "cumulative = base + 2 * played", "advance_progress double counts the session"),
    (TR, "doublings = int(progress * n_steps)", "doublings = int(progress * n_steps) + 1", "the minibatch ramp starts a step early"),
    (TR, "return start + (floor - start) * progress", "return start + (start - floor) * progress", "the entropy coefficient anneals upward"),
    # --- rollback ---
    ("run_rollback.py", "        if os.path.exists(path):\n            if os.path.exists(backup) and not force:", "        if os.path.exists(path):\n            if False:",
     "a second rollback silently overwrites the only backup"),
    ("run_rollback.py", "            shutil.copy2(path, backup)", "            pass", "no backup of the live pair is taken"),
    ("run_rollback.py", "src = history.snapshot_path(league_dir, deck, snapshot_id)", "src = history.snapshot_path(league_dir, deck, snapshot_id + 1)", "rollback restores the wrong snapshot"),
    ("run_rollback.py", "    if dry_run:", "    if False:", "dry-run changes the league"),
    # --- agent runtime, mulligan, evaluation ---
    ("rl/decision/agent.py", "out[n_fixed:n_fixed + len(ptr)] = torch.as_tensor(ptr, dtype=torch.bool, device=device)", "out[n_fixed + 1:n_fixed + 1 + len(ptr)] = torch.as_tensor(ptr, dtype=torch.bool, device=device)",
     "the padded mask shifts every pointer slot by one"),
    ("rl/decision/agent.py", "return action_idx < len(fixed_table) and fixed_table[action_idx][0] == \"Pass\"", "return fixed_table[action_idx][0] == \"Pass\"", "a pointer index is looked up in the fixed table"),
    ("rl/model/mulligan.py", "loss = (policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy) / max(1, n)", "loss = (policy_loss + 0.5 * value_loss + ENTROPY_COEF * entropy) / max(1, n)",
     "the mulligan entropy bonus has the wrong sign"),
    (ST, "return (centre - half, centre + half)", "return (centre - half, centre + 2 * half)", "the Wilson interval's upper bound is too wide"),
    (ST, "half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom", "half = z * math.sqrt(p * (1 - p) / n) / denom", "the Wilson interval is the Wald half-width over the shrunk centre"),
    (VD, "pooled = (w1 + w2) / (n1 + n2)", "pooled = w1 / n1", "the z-test pools nothing"),
    (VD, "return (alpha_z + power_z) * math.sqrt(2 * 0.25 / n_per_group)", "return alpha_z * math.sqrt(2 * 0.25 / n_per_group)", "the minimum detectable effect ignores power"),
    (VD, "if n_late and w_late / n_late >= SATURATED_AT:", "if n_late and w_late / n_late > 1.0:", "a saturated series is never labelled saturated"),
    ("evaluation/history.py", "if sid not in seen and games <= current_games - 2 * games_per_snapshot:", "if games <= current_games - 2 * games_per_snapshot:", "the ladder keeps duplicate snapshots"),
    ("config_loader.py", "return {**load_config(path.parent / base_name), **cfg}", "return {**cfg, **load_config(path.parent / base_name)}", "an extended config's base overrides the file's own keys"),
]
