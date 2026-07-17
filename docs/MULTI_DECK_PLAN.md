# Multi-Deck Modularization — Implementation Plan

Companion to [DRL_PLAN.md](DRL_PLAN.md) and [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md).
Planning only — nothing gets built until this is confirmed and a checklist
is written, per this project's discuss -> plan -> checklist -> implement
discipline.

## Goal

Generalize the current Tron-only simulator into a deck-agnostic engine, so
a new deck (e.g. a pauper combo deck needing all lands + 3 creatures by
turn X) can be dropped in as pure *configuration* — decklist, termination
condition, reward/scoring functions, model config — with **zero new engine
code**, provided every card it uses is already implemented in the shared
card-effect library. A deck using a genuinely new mechanic still requires
implementing that one effect once, after which it's available to every
future deck too.

This also removes the heuristic-vs-DRL comparison entirely: hand-coded
strategic heuristics (`policy_choose_action` and everything it depends on)
are deleted from the codebase. Every deck, always, is played by a DRL
model — none are hand-tuned.

## Decisions locked in (this round of conversation)

1. **Single simulator module.** One `game.py` continues to handle all
   Magic rules for every deck — not a per-deck fork, not a family of
   simulator classes. "Modularize" means *parameterize*, not *split*.
2. **Single termination condition per deck**, injected as `terminated(state)
   -> bool`. Checked generically after every battlefield-mutating action —
   not hardcoded to one trigger site the way Tron's land-ETB check is
   today (`game.py`'s `enters_battlefield`, lines ~408-419).
3. **Score 1 is mandatory**, unchanged in spirit from today:
   `reward_fn(state, done, horizon) -> float`, called every env step
   during training, and the sort key for `evaluate()`'s logs.
4. **Scoring functions 2+ are optional and arbitrary-length** — no hard
   cap. Each has signature `(state) -> float`, computed exactly once, at
   game end, never mid-episode ("accept state, yield score; it can check
   the goal itself"). A game that never terminates by the horizon gets
   **every** score forced to `0.0`, enforced centrally by the harness —
   no individual scoring function has to remember a failure check.
5. **Cards are implemented deck-independently**, in a shared
   `EffectId`-keyed registry (mana sources, ETB triggers, activated
   abilities). Any deck's decklist can reference any already-implemented
   effect for free.
6. **No hand-coded strategic heuristics anywhere.** `policy_choose_action`
   and its support functions (`_rank_priority`, the land-drop priority
   logic in `choose_land_drop`, etc.) are deleted outright, not ported or
   generalized. Every deck is always played by a supplied DRL model.
7. **Search/scry/surveil/Ancient Stirrings become real model decisions.**
   `is_priority_land` (today's hardcoded "what's worth finding" rule) is
   deleted along with the rest of the heuristic layer. Two different
   mechanisms, split by whether hidden information is actually revealed:
   - **Search** (Expedition Map, Crop Rotation): no reveal step — the
     model already knows the deck's exact remaining composition by name.
     Becomes flat, one-shot actions: one per fetchable name, generated
     per deck (e.g. "Activate Expedition Map, fetch Forest").
   - **Scry / surveil / Ancient Stirrings**: genuinely reveals specific
     hidden cards from an unknown-order shuffle. Needs a new
     **pending-resolution state machine**: `GameState` gains a
     mid-resolution sub-state; while active, the only legal actions are a
     small fixed menu (keep-on-top / dispose) applied to the current
     revealed card, one at a time. Once every revealed card is decided,
     if 2+ were kept, a further sequence of "which kept card goes in the
     next top position" decisions builds the final order — scales to any
     reveal size without a factorial action space. Disposed cards
     (bottomed/binned) keep a random relative order, not modeled, since
     nothing in this simulator ever reads it again.
8. **Model compatibility is not preserved across this refactor.** The
   generic action-table/observation-builder is free to order things
   naturally; `models/run_50k` is retired and Tron's model gets retrained
   from scratch once this lands.
9. **8-phase migration order** (below), sequenced so the existing
   baseline stays independently verifiable for as long as possible,
   before the heuristic it's verified against gets deleted.

## What "dropping in a new deck" means, concretely

A deck becomes a bundle of:

- **Decklist** (data) — card name/qty/type/cost/effect_id, same shape as
  today's `DECKLIST`.
- **Termination condition** — `terminated(state) -> bool`.
- **Scores** — a mandatory `reward_fn` (score 1) plus any number of
  additional `(state) -> float` scoring functions.
- **Model config** — SB3-family class + kwargs (already fully generic
  today via `TrainingHarness`, no change needed here).

No heuristic to write. No new env file (per the earlier "would we need a
new spy_env?" question — no, under this design). No new card-art pipeline
(`fetch-card-art.mjs` is already keyed by card name, not by deck). A card
needing a genuinely new mechanic means one new entry in the shared
`EffectId` registry, available to every deck from then on.

## Architecture

```
   Shared card-effect library        Per-deck config             DRL model (SB3)
   (EffectId registry: mana          (decklist, terminated_fn,    (sb3 / sb3-contrib
    sources, ETB triggers,            reward_fn, scoring_fns,      class, unchanged)
    activated abilities --            model_cls/kwargs)
    game.py, shared across decks)            |
        \                                    |                         /
         \                                   |                        /
          v                                  v                       v
                     generic env: action table + observation space
                     generated from (decklist, registry), including the
                     pending-resolution state machine for scry/surveil/
                     Ancient Stirrings and flat fetch-by-name actions
                                       |
                                       v
                              harness.py (TrainingHarness)
                             .train() / .evaluate() / .save() / .load()
                                       |
                                       v
                          train_drl.py / evaluate_drl.py-style
                             scripts, one instantiation per deck
```

## Phase-by-phase plan

### Phase 1 — Registry-ize card effects (behavior-preserving)

Move Tron's card-effect functions (mana sources, ETB triggers, activated
abilities) into a shared `EffectId`-keyed registry format, capable of
describing: whether it's a land, its mana output, its ETB trigger, its
activated abilities (cost + which function resolves them). Still a single
hardcoded Tron deck at this point — `DECKLIST` stays global, the action
table doesn't change yet.

**Verify**: all of `game.py`'s existing heuristic-independent sanity
checks (phases 0-4) pass unchanged.

### Phase 2 — Parameterize deck/state

`GameState`/`new_game_state`/the turn loop take a decklist argument
instead of importing a global `DECKLIST`. Still only ever called with
Tron's decklist.

**Verify**: behavior is bit-for-bit identical to pre-refactor (same
sanity checks, now passing the decklist explicitly).

### Phase 3 — Generalize termination

Replace the land-ETB-hardcoded `controls_all_tron_types` check with an
injected `terminated(state) -> bool`, evaluated generically after every
battlefield mutation (not just land drops). Tron's own check becomes the
first instance of this pattern.

**Verify**: termination timing matches pre-refactor behavior exactly, on
a fixed seed set (see "Ensuring accuracy," below).

### Phase 4 — Generic action table + observation builder + pending-resolution machinery

The largest phase. Build: the generic action table (generated from
decklist + registry instead of hand-typed); the observation builder
(sized to the decklist); the new pending-resolution state machine for
scry/surveil/Ancient Stirrings; flat fetch-by-name actions for search
effects.

**Verify**: a large batch of uniform-random-legal-action rollouts (no
model yet) completes with zero illegal-action masking failures, zero
crashes, and a sane termination-turn distribution.

### Phase 5 — Delete heuristics

Remove `policy_choose_action`, `is_priority_land`, `_rank_priority`,
`choose_land_drop`'s priority logic, `simulate_many`'s heuristic path, and
every sanity check that depended on driving games via the heuristic
(roughly today's phases 5-8). Replace with a uniform-random-legal-action
driver for whichever engine-mechanics checks still need *some* driver to
run games at all (same pattern `tron_env.py`'s own D2.4 check already
uses).

**Verify**: replacement checks assert the same invariants the deleted
ones did (e.g. "termination can't happen before its own prerequisite
becomes true"), just driven differently.

### Phase 6 — Rework scoring + reporting

`reward_fn` stays the per-step-called, mandatory score 1. Add an
arbitrary-length `scoring_fns` list to `TrainingHarness`, each
`(state) -> float`, computed once at game end (in `_GameLogger.finalize()`
and wherever else a game's final score is produced); the harness
centrally zeroes every score for a non-terminated game. `aggregate_results`
/`print_report` move from the `(turn_assembled, turn_online)` 2-column
shape to "% terminated by turn X" (one generic column) plus a distribution
summary per configured score.

**Verify**: hand-computed example (matching the style of today's Phase 7
check) confirms the new aggregation math.

### Phase 7 — Migrate Tron itself into the new format

Tron becomes the first real deck config: its decklist, its `terminated_fn`
(three Tron land types on the battlefield), and its scores — score 1 =
today's `assembled_with_resource_quality` (formula unchanged), plus an
optional score 2 = "all three Tron types AND all untapped," a direct port
of what `turn_online` used to mean, now expressed as a scoring function
instead of a second termination tier.

**Verify**: fixed-seed equivalence test (see below) against the pre-Phase-1
snapshot.

### Phase 8 — Retrain + verify

Retrain Tron's model from scratch against the new generic env
(`models/run_50k` is not compatible — different action ordering). Compare
the resulting eval metrics against the pre-refactor DRL baseline (most
recent 1000-game eval: 83.0% assembled / 81.7% online by turn 6, mean
assembly turn 3.85) to confirm the new pipeline produces a policy of
comparable quality, not a regression introduced by re-plumbing.

## Ensuring accuracy through the refactor

Heuristic numbers stop being a valid regression target the moment the
heuristic is deleted (Phase 5), so the strategy has to lean on things that
survive:

- **Fixed-seed equivalence tests, taken before Phase 1 starts**: snapshot
  exact behavior (turn-by-turn termination, full action sequences) from
  the *current* system on a fixed set of seeds. After each phase, re-run
  the same seeds through the new code path and diff against the snapshot
  — right up until Phase 4 changes the action space shape, after which
  the comparison shifts to outcome-level (termination turn distributions)
  rather than action-sequence-level.
- **The current trained model's eval stats** (`models/run_50k`: 83.0%/
  81.7% assembled/online by turn 6) are a valid comparison point for "is
  the new pipeline's retrained model in the same ballpark" in Phase 8,
  even though the model file itself isn't reusable.
- **Existing phase-sanity-check discipline continues unchanged**: every
  new phase gets its own assert-based check before the next phase starts,
  same as every prior feature in this project.
- **New pending-resolution machinery gets dedicated, from-scratch
  regression coverage** — it's the one piece of Phase 4 with no
  historical baseline to diff against. Hand-built states with a known
  library order, exercising scry 2 / surveil 1 / Ancient Stirrings,
  asserting the exact sequence of pending decisions offered and the final
  resulting zone contents match hand-computed expectations.

## Explicitly out of scope for this plan

- Actually building the pauper combo deck (or any second deck) — this
  plan only covers making the engine capable of it.
- Hyperparameter tuning beyond "confirm the new pipeline produces a sane
  model" in Phase 8.
- Visualizer changes — `cardData.js`, the outcome banner's "Assembled/
  Online" wording, etc. all still assume Tron-specific concepts. Explicitly
  deferred per your earlier message; needs its own pass later.
- A config-file/registry-driven deck-*selection* mechanism beyond passing
  a deck-config-shaped object into constructors — still plain dependency
  injection, consistent with [DRL_PLAN.md](DRL_PLAN.md)'s original
  decision to decline a config-driven registry.
