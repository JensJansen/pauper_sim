# Work checklist

Four large changes, taken **one at a time**. Each item's goals are agreed with the
repo owner *before* any code is written; the "Goals" section starts empty and is
filled in during that discussion, then the work is done against it.

Status: `[ ]` not started · `[~]` in progress · `[x]` done

---

## 1. `[~]` Cleanup

Repo-wide cleanup pass.

### Done (commit `f6a16b9`)

Deleted: `RL_METHODOLOGY_PLAN.md`; all of `checkpoints/` (1.4 GB — every population,
both stack backups, `vocab.json`, `shared_stack_frozen.pt`); all of `logs/` (80 MB);
the four concluded A/B configs; the seven one-off `src/analysis/` scripts and their
`_shared.py`. Every reference to a deleted *file* was repaired (`README.md`,
`rl/rewards.py`, `tests/test_run_league.py`, `tests/rl/test_train.py`,
`.claude/skills/train/SKILL.md`). 655 passed, 1 skipped.

**Consequence:** a fresh pretrain + freeze is now mandatory before any league run.

### Done (commit `71b31f1`)

- **A.** All 24 dangling `RL_METHODOLOGY_PLAN.md` citations stripped across 15 files.
  Each comment kept its conclusion; only the pointer went.
- **B.** `run_default.json` 5,741 → 2,042 chars, `league_main.json` 4,666 → 1,542.
  Every `_`-note about a deleted population is gone; what remains documents live
  values (`games_per_iteration=24`, `pfsp_power=0.5`) and current status.
- **F. BUG 3 fixed.** `_save_live_checkpoints` had always written the live nets at
  every snapshot point so a crash keeps its training; `progress.json` was written
  only after `_run_session` returned, so a crash left the counter *behind* the
  weights and it had to be hand-corrected three times. New
  `league_runner.checkpoint_progress()` writes the counter absolutely at the same
  snapshot points; `advance_progress()` takes `session_start_games` so the
  session-end write is idempotent rather than double-counting. Regression test
  pins both halves.
- **G.** `HeuristicAgent` + its nine scoring helpers moved to
  [src/rl/heuristic_agent.py](src/rl/heuristic_agent.py). One-way dependency
  (heuristic → agent), so nothing in the learned path can come to depend on it.
  Its tests stayed in `test_agent.py` — they share `_rally_ctx`/`_make_creature`
  fixtures with the SeatAgent tests, and duplicating those would be more fragile
  than the tidier split is worth.

656 passed, 1 skipped.

### Deferred / declined

| # | Item | Disposition |
|---|---|---|
| C | Gauntlet machinery (97 refs, 14 files). Both configs point `gauntlet_league_name` at deleted populations, so `vs_gauntlet` is a permanent silent no-op | **deferred by owner** — machinery and all three gauntlet configs left untouched |
| D | `deploy_reward_v1`–`v5` — production-dead, only tests reference them | **declined** — thin parameter bindings over machinery that stays anyway; their comments are load-bearing (v3's collapse is the cited reason v6's safety margin exists) |
| E | README narrative for deleted experiments (1,192 lines) | **deferred** — items 2–4 will rewrite exactly those sections; doing it now means doing it twice |
| H | Tooling whose target populations no longer exist | **declined** — all take league names as arguments, so they survive a fresh start unchanged. Only `src/benchmarking/` looks genuinely spent |

---

## 2. `[ ]` Cast-then-pay, with floating mana allowed only in main phases

Move from pay-then-cast to the real rule 601 sequence: announce the spell, choose
modes/targets/X, *then* activate mana abilities and pay. Floating mana is permitted
only during main phases.

**Goals:** _(to be defined with owner)_

**Open questions:**
- "Floating allowed only in main phases" is a deviation from real MTG (mana empties
  at every step/phase end, and floating is legal in any phase). Confirm this is an
  intentional owner-authorized simplification / action-space restriction, and it gets
  an `# AUTHORIZED SIMPLIFICATION:` marker — or whether the intent is instead that the
  *agent* is only offered the choice to float in main phases.
- Scope of the action-space change: the mana subdecision state machine
  (`choose_target` → `choose_color`), `action_bridge`, `_actions_mana`, and every
  legality read.
- Does this invalidate the frozen stack / existing checkpoints (action-space shape
  change ⇒ likely a fresh start)?

---

## 3. `[ ]` Pretrain the frozen stack on prior-generation *agent* games

Today `run_pretrain.py` trains the shared encoder from near-random self-play. Instead,
generate its training games with a prior generation of trained agents, so the encoder
learns to represent boards that actually occur in competent play.

**Goals:** _(to be defined with owner)_

**Open questions:**
- Which generation is the source — current league `live.pt`s, a fixed snapshot vintage,
  or a mix across vintages?
- Is this behavior cloning / representation learning off recorded trajectories, or
  still on-policy RL with the old agents merely as opponents?
- Bootstrapping: the prior generation was itself trained on the *old* frozen stack, so
  its checkpoints don't load onto the new one. What is the intended handoff?
- Does the cross-deck pairing setting from `--cross-deck` carry over?

---

## 4. `[ ]` Agent memory / historical information as input

Give the policy access to game history, not just the current board state.

**Goals:** _(to be defined with owner)_

**Open questions:**
- What history — cards seen in the opponent's deck this game, previous turns' plays,
  the full action log, or a learned recurrent state?
- Per-game memory only, or across games in a match/league (opponent modelling)?
- Where it enters: extra static features, extra tokens into the SetTransformer, or a
  recurrent/attention layer in the per-deck trunk?
- This grows `TOKEN_FEATURE_DIM` or the model shape ⇒ almost certainly requires a
  fresh pretrain + freeze. Sequencing vs. item 3 matters.

---

## Context that motivates these

`mono_red_rally` wins **85.5%** of its non-mirror games across the full 11-deck roster
(4,931/5,770 at cum 6,984 games/deck), up to 100% vs `spy_combo`. Owner's assessment:
the deck is functioning correctly — the other ten decks are missing *strategic facets*
the agents cannot currently express or perceive. Items 2–4 all widen what the agent can
do or see, which is the intended fix.

Ten prior hypotheses for the same plateau were tested and every one came back null:
self-play cycling, `adv_std`, degenerate matchups, entropy coefficient, Adam-step count,
KL truncation, trainable capacity, meta size, and a cross-deck-pretrained encoder. The
one durable methodological result from that work: **within-league elo does not measure
strength in either direction** — every strength claim needs a common external reference.
(The full writeup lived in `RL_METHODOLOGY_PLAN.md`, deleted 2026-08-17.)
