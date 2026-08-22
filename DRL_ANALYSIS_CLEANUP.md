# DRL & Analysis Cleanup — Checklist

Repo-wide cleanup was scoped to `src/analysis/`, `src/rl/`, and `src/drl_env/`
first (2026-08-22). Five parallel reviews covered every file in those
directories; a second pass then restructured `src/rl/` and `src/analysis/`
into subdirectories by actual cohesion (not guessed from file names).

## Done — code cull (2026-08-22)

- [x] **`rl/rewards.py`** trimmed from 7 reward functions to 1: removed
      `deploy_reward_v1`–`v5`, the bare `deploy_reward` factory,
      `with_mana_mistake_penalty`, and `action_count_win_reward*` — all
      self-documented as superseded/reference-only with zero production
      callers. Only `deploy_reward_v6` (the live one) and its real
      dependencies remain.
- [x] Deleted **`src/analysis/cross_league_round_robin.py`** — a strict
      functional subset of `run_cross_league_eval.py`, written 13 hours
      later the same day, never referenced elsewhere.
- [x] Deleted **`training_configs/run_bench_cpu.json`** /
      **`run_bench_gpu.json`** — byte-identical to `run_bench.json` except
      league name, neither ever opened by any script.
- [x] Deleted **`rl/train.py`'s `train_selfplay`** — dead since the pretrain
      phase was removed 2026-08-17; its test went with it.
- [x] Fixed **`run_anchor_eval.py`**'s methodology bug — it used the
      unpaired `_play_eval_games` instead of the seat-swapped
      `_play_paired_eval_games` every sibling eval script uses.
- [x] Fixed **`rl/train.py`'s `_reward_for`** — the non-terminal branch
      skipped the seat-perspective flip. Harmless today, but a future dense
      reward would have silently read the wrong seat.
- [x] Removed the orphaned **`--decks`/`train_decks` CLI knob** from
      `run_league.py`/`league_cli_spec.py` — no config ever set it, no test
      exercised it, undocumented in README. Kept `--roster` (used in every
      active training config) and the internal `train_decks` parameter on
      `_run_session` (still load-bearing for `--matchup` mode).
- [x] Trimmed **`rl/agent.py`'s `_raise_all_false`** from ~140 lines to a
      compact safety-net raise. Kept the raise itself (deleting it entirely
      would silently reintroduce uniform-random sampling over an illegal
      action space) but cut the verbose diagnostic dump built for specific
      bugs the commit history shows are now fixed.
- [x] **Enabled GPU training** — `--device`/`"device"` was fully wired but
      no active config ever set it, so every real training run defaulted to
      CPU despite CUDA being available. Added `"device": "cuda"` to
      `run_default.json`, `league_main.json`, `run_gauntlet_twin.json`, and
      `run_bench.json`, and added `"device"` to `tests/test_run_league.py`'s
      `MECHANICS` parity list.
- [x] Fixed ~10 stale doc/comment references pointing at deleted
      functions/files across README.md, `drl_env/`, `game/`, `rl/`,
      `.claude/skills/train/SKILL.md`.
- [x] Full test suite green: 834 fast + 128 slow.

## Reviewed, no action needed

- [x] **Mulligan count cap** (`rl/model/mulligan.py`,
      `rl/decision/agent.py`) reuses `game.HAND_SIZE_LIMIT` (the
      cleanup-step hand-size rule) as a mulligan-count cap, rather than real
      London mulligan's uncapped rule. Owner call (2026-08-22): functionally
      identical in practice (an 8th mulligan is never worth taking), not a
      real rules violation worth an `AUTHORIZED SIMPLIFICATION` comment. No
      action needed.

## Needs owner input (flagged, not touched)

- [ ] **`analysis/mulligan_retrain/`** (`train_mulligan_self_mirror.py`,
      `train_mulligan_vs_twin.py`, `_mulligan_common.py`) — actively
      touched, mid an open mulligan-net investigation. Not dead, but the two
      scripts share real duplicated `_play_eval_arm` logic that could be
      factored into `_mulligan_common.py` once the investigation settles.
- [ ] **GPU path still unvalidated end-to-end.** Now that real configs
      default to `cuda`, only a per-`ppo_update`-call microbenchmark
      (`analysis/eval/bench_gpu_vs_cpu.py`) has ever confirmed correctness —
      no session-level test exercises the `cpu_nets`/`cpu_mulligan_nets`
      mirror-sync path on a real GPU run.
- [ ] **Real multiprocessing path has zero test coverage.** The only test
      of `rollout_parallel.collect_rollout_league_parallel` substitutes a
      `ThreadPoolExecutor` for the real `ProcessPoolExecutor` — a
      spawn/pickling regression would only surface in real training.

## Done — structure reorg (2026-08-22, second pass)

Both directories were flat; neither needed a reorg "just because," so this
only happened where real import-graph cohesion (not guessed from names)
supported it. `src/drl_env/` was reviewed and left alone: already one
cohesive package split by MTG action category, imported broadly outside RL
(catalog tests too), no benefit to further nesting.

- [x] **`src/rl/`** (16 files, flat) split into 4 subpackages by actual
      import edges:
  - `model/` — `features.py`, `arch.py`, `deck.py`, `mulligan.py` (network
    architecture / observation shape)
  - `decision/` — `action_bridge.py`, `agent.py`, `heuristic_agent.py`
    (turning an observation into a chosen action)
  - `training/` — `train.py`, `ppo.py`, `rollout_parallel.py` (rollout loop
    + PPO update math)
  - `league/` — `league.py`, `league_runner.py` (opponent pool + session
    orchestration)
  - Stayed top-level (used across 2+ clusters, no single natural home):
    `rewards.py`, `checkpoint.py`, `league_cli_spec.py`
  - **Renamed** `pool.py` → `roster.py` (owner-approved): resolves the
    naming collision with `league.py`'s `LeaguePool` class — two unrelated
    concepts that both happened to say "pool."
  - `tests/rl/` mirrored into the same 4 subdirs; `test_pool.py` →
    `test_roster.py`.
- [x] **`src/analysis/`** (9 files, flat) split into 2 subdirs by concern:
  - `eval/` — `report_metrics.py`, `run_anchor_eval.py`,
    `run_cross_league_eval.py`, `run_snapshot_round_robin.py`,
    `bench_gpu_vs_cpu.py` (standalone "play games, print a table" tools)
  - `mulligan_retrain/` — `train_mulligan_self_mirror.py`,
    `train_mulligan_vs_twin.py`, `_mulligan_common.py` (the active
    mulligan-net investigation, already only cross-referencing within
    itself)
- [x] Every import site repo-wide updated (~70 files touched: `rl/`
      internals, `analysis/` internals, tests, `run_league.py`,
      `run_rollback.py`, `benchmarking/training_run.py`, README.md,
      `.claude/skills/train/SKILL.md`) — verified by repo-wide grep sweeps
      for leftover old dotted-paths and old flat file-paths (none found)
      plus a full test-suite run.
- [x] `sys.path.insert` depth fixed in the 4 `analysis/eval/*.py` scripts
      and 2 `analysis/mulligan_retrain/train_mulligan_*.py` scripts (one
      extra directory level to reach `src/`).
- [x] README's "Repository layout" tree and `## The DRL system` section
      rewritten to match the new structure, including the `roster.py`
      rename rationale.
- [x] Full test suite green: 962 passed (834 fast + 128 slow) after the
      reorg.

## Next

The DRL (`rl/`, `drl_env/`) and analysis (`analysis/`, `benchmarking/`)
sections are now clean and reorganized. The original ask was a repo-wide
cull — remaining scope (other directories, top-level docs, `webapp/`, etc.)
hasn't been reviewed yet. See `project_repo_reorg_2026-08.md` in memory for
the standing reorg plan this continues.
