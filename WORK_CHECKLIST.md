# Work checklist

Four large changes, taken **one at a time**. Each item's goals are agreed with the
repo owner *before* any code is written; the "Goals" section starts empty and is
filled in during that discussion, then the work is done against it.

Status: `[ ]` not started · `[~]` in progress · `[x]` done

---

## 1. `[ ]` Cleanup

Repo-wide cleanup pass.

**Goals:** _(to be defined with owner)_

**Open questions:**
- What is in scope — dead code, stale docs, the archived checkpoint trees, the
  one-off experiment configs in `training_configs/`, the finished sections of
  `RL_METHODOLOGY_PLAN.md`, or all of it?
- Carry-over items already known: `advance_progress` counter cadence (advances at
  session end, should advance at snapshot cadence — hand-corrected 3× so far);
  `HeuristicAgent` move + Tier-4 naming from the repo-reorg plan.

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
do or see, which is the intended fix; ten prior optimizer/architecture hypotheses were
all null (`RL_METHODOLOGY_PLAN.md` §1A.5–1A.15).
