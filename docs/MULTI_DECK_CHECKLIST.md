# Multi-Deck Modularization — Implementation Checklist

Companion to [MULTI_DECK_PLAN.md](MULTI_DECK_PLAN.md). Concrete build
order, phase by phase, each ending in a verification step before moving
on — same discipline as every prior feature in this project. **Planning
only — nothing below gets executed until you say go.** Each phase heading
states what that phase accomplishes toward the overarching goal (a
deck-agnostic engine where a new deck built from already-implemented cards
needs zero new code); each checklist item is a concrete, specific change,
not a restatement of the plan's prose.

## Phase M0 — Snapshot current behavior (regression baseline)

**Accomplishes**: creates the ground truth this entire refactor gets
checked against. Has to happen *before* Phase M1 touches anything, and
specifically before Phase M5 deletes the heuristic — after that, "compare
against the old heuristic" stops being possible, so this snapshot is the
only remaining way to prove Phases M2/M3/M7 didn't silently change
behavior.

- [ ] Pick a fixed, documented set of seeds (reuse the project's existing
      convention — e.g. seeds `0`–`49`).
- [ ] For each seed, drive a full game via `game.run_game` using a
      **uniform-random-legal-action** driver (not the heuristic — so this
      same snapshot mechanism keeps working after M5 deletes it). Record,
      per seed: the exact action sequence taken, the resulting
      `(turn_assembled, turn_online)` pair, and the final zone contents
      (hand/battlefield/graveyard, by card name).
- [ ] Save this as a checked-in fixture (e.g.
      `docs/_multi_deck_regression_snapshot.json`), not regenerated later
      — it has to reflect *pre-refactor* behavior permanently.
- [ ] Verify: re-running the snapshot generator twice on the same seed
      list produces byte-identical output (confirms the driver itself is
      deterministic before it becomes load-bearing for every later phase's
      regression check).

## Phase M1 — Registry-ize card effects (behavior-preserving)

**Accomplishes**: turns Tron's card-specific logic — currently scattered
as hardcoded `effect_id ==` conditionals inside `enters_battlefield`,
plus the three separate `SIMPLE_MANA_SOURCE_EFFECTS`/`_FIXED_SOURCE_COLOR`
/`_FLEXIBLE_SOURCE_CHOICES` dicts — into one shared, per-`EffectId`
registry. This is what makes a card reusable by a future deck: after this
phase, "is this card already implemented" has one place to check. Deck
data (`DECKLIST`) is *not* touched yet — still a single hardcoded Tron
deck.

- [ ] Design the registry entry shape: for each `EffectId`, whether it's a
      land, its mana output (fixed-color / flexible-color / none), its ETB
      trigger (if any), and its activated abilities (each: name, cost,
      resolve function).
- [ ] Fold `SIMPLE_MANA_SOURCE_EFFECTS`, `_FIXED_SOURCE_COLOR`,
      `_FLEXIBLE_SOURCE_CHOICES`, and `ENTERS_TAPPED_EFFECTS` into registry
      entries — these stop being separate module-level dicts.
    - [ ] Also fold in Tocasia's Dig Site's `{3}, T: Surveil 1` ability
      cost — today hardcoded in `tron_env.py` (`_TOCASIA_SURVEIL_COST`,
      documented there as not living in `card_def.extra` anywhere) — into
      its registry entry, closing that gap explicitly.
- [ ] Replace `enters_battlefield`'s inline `if effect_id == X: trigger()`
      chain with a single registry lookup + dispatch.
- [ ] Migrate each activated-ability function (`activate_expedition_map`,
      `activate_candy_trail_sac`, `activate_relic_of_progenitus`,
      `activate_bonders_ornament_draw`, `activate_tocasia_dig_site_surveil`)
      to be reachable via its `EffectId`'s registry entry rather than only
      by direct name.
- [ ] Verify: `game.py`'s existing Phase 0–4 sanity checks (deck totals,
      opening hand, mana payment, ETB/metric checks, always-pass policy)
      pass **unmodified** — proves this phase changed *organization* only,
      not behavior.

## Phase M2 — Parameterize deck/state

**Accomplishes**: removes the single global `DECKLIST` assumption from
`GameState` construction — the mechanical precondition for more than one
deck ever existing in the same process.

- [ ] `new_game_state(decklist, on_the_play, rng)` — takes a decklist
      argument instead of reading the module-level `DECKLIST` global.
- [ ] Rename the current global to `TRON_DECKLIST` (still the only
      decklist that exists after this phase — just no longer implicit).
- [ ] Decide and document: does `CARD_DEFS` (name → `CardDef` lookup)
      stay one shared global covering every implemented card regardless of
      deck (grows as new cards are registered), or become deck-scoped?
      (Leaning shared-global, since it's just a lookup table over the
      Phase M1 registry, not itself deck-specific data.)
- [ ] Audit `run_turn`/`run_game`/`continue_game` for any direct
      `DECKLIST`/`CARD_DEFS` reference beyond what's reached through
      `state` — expect none, but confirm by grep rather than assuming.
- [ ] Verify: fixed-seed diff against the Phase M0 snapshot — same seeds,
      same driver, `new_game_state(TRON_DECKLIST, ...)` in place of the
      old no-arg call, confirm identical shuffled-library contents and
      identical action-legality sequence (proves passing the decklist
      explicitly didn't change RNG consumption or behavior at all).

## Phase M3 — Generalize termination

**Accomplishes**: replaces Tron's hardcoded, single-trigger-site win check
with an injectable `terminated(state) -> bool`, the mechanism that makes a
non-Tron win condition (e.g. "all lands + 3 creatures") expressible at
all.

- [ ] Add `state.terminated_fn` (or pass it alongside the decklist to
      whatever constructs `GameState`) — a per-deck injected predicate.
- [ ] Replace the hardcoded `if effect_id == TRON_LAND and
      controls_all_tron_types(state): ...` block inside
      `enters_battlefield` with a generic call to `state.terminated_fn(state)`,
      checked after every permanent enters the battlefield.
    - [ ] Document the scope decision explicitly: checked on ETB only (not
      after every possible action) — correct for every win condition
      discussed so far (Tron's 3 lands; "3 creatures + all lands"), since
      both are permanent-count conditions that can only newly become true
      when a permanent enters. Flag as a documented limitation if a future
      deck's condition could depend on removal or tap-state alone.
- [ ] Replace `state.turn_assembled`/`state.turn_online` with a single
      generic field (e.g. `state.turn_won`), set once, the first time
      `terminated_fn` returns `True`.
- [ ] Port `controls_all_tron_types` into `tron_terminated(state) -> bool`
      — Tron's own `terminated_fn` — dropping the "online" half entirely
      (that becomes a Phase M7 scoring function, not a termination
      concept).
- [ ] Verify: fixed-seed diff against the Phase M0 snapshot — `turn_won`
      matches the old `turn_assembled` value exactly, game-by-game, across
      the full seed set.

## Phase M4 — Generic action table + observation builder + pending-resolution machinery

**Accomplishes**: the largest phase, and the one that actually delivers
"drop in a deck, zero new code" — replaces the hand-typed 22-entry
`ACTIONS` table and fixed 90-dim observation with generators driven by
(decklist, registry), and gives the model real control over
search/scry/surveil instead of the deleted heuristic auto-resolving them.

- [ ] Extend the Phase M1 registry with action-generation metadata per
      `EffectId`: land → auto land-drop action; castable nonland → auto
      cast action; each registered activated ability → auto activation
      action.
- [ ] Implement `build_action_table(decklist, registry) -> list[(name,
      legal_fn, execute_fn)]`, replacing the hand-typed `ACTIONS` tuple.
- [ ] Implement `build_observation(state, decklist, horizon) ->
      np.ndarray`, sized dynamically to the decklist's distinct-card
      count (replacing the hardcoded 90-dim vector).
- [ ] Design `state.pending_resolution`: `None`, or an object recording
      `{kind, revealed: [...], decisions_so_far: [...], kept: [...]}` for
      an in-progress scry/surveil/Ancient Stirrings resolution.
- [ ] Implement the state-machine transitions:
    - [ ] Entering a pending resolution (triggered by Candy Trail/Conduit
      Pylons/Tocasia's Dig Site/Ancient Stirrings, via their registry
      entries).
    - [ ] The per-card decision phase: while pending, `legal_action_mask`
      exposes only the small fixed menu (keep-on-top/dispose, or
      take/decline for Ancient Stirrings) for the current revealed card,
      and every normal action becomes illegal.
    - [ ] The ordering phase: once all revealed cards are decided, if 2+
      were kept, a further sequence of "which kept card goes in the next
      top position" decisions, one at a time, until fully ordered.
    - [ ] Exiting back to normal play once resolution completes.
- [ ] Add flat "fetch by name" actions for search effects (Expedition
      Map, Crop Rotation) — one action per fetchable land name present in
      the deck, no pending-resolution state needed (no hidden reveal).
- [ ] Verify (engine-level, no model yet): a large batch (thousands) of
      uniform-random-legal-action rollouts completes with zero
      illegal-action masking failures, no stuck pending-resolution states
      (every one eventually resolves), and a plausible termination-turn
      distribution.
- [ ] Verify (dedicated, from-scratch — this piece has no historical
      baseline to diff against): hand-built states with a known library
      order, walking through Candy Trail's scry 2, Tocasia's surveil 1,
      and Ancient Stirrings step by step, asserting the exact decision
      menu offered at each step and the final zone contents match
      hand-computed expectations.

## Phase M5 — Delete heuristics

**Accomplishes**: removes hand-coded strategy from the codebase entirely,
per the "every deck always played by a DRL model" decision — including
the deeper heuristic (`is_priority_land`) this planning conversation
surfaced, which Phase M4 already made obsolete by giving the model real
control over those decisions.

- [ ] Grep for zero remaining callers of `is_priority_land` before
      deleting it (should already be true post-M4 — this is a safety gate,
      not an assumption).
- [ ] Delete `policy_choose_action`, `_rank_priority`, `is_priority_land`,
      and the Tron-specific land-drop prioritization logic inside
      `choose_land_drop` (`has_untapped_green_source`, `produces_green`,
      `_land_drop_leftover_priority`).
- [ ] Delete `simulate_many` (existed only to drive `policy_choose_action`
      through many games — no purpose once there's no heuristic; all
      evaluation goes through `harness.evaluate()` with a real model from
      here on).
- [ ] Delete `_phase5_sanity_check`, `_phase6_sanity_check`,
      `_phase7_sanity_check`, `_phase8_invariant_check`,
      `_phase8_hand_fed_scenario` (all heuristic-driven).
- [ ] Implement a shared uniform-random-legal-action driver (promoting the
      pattern `tron_env.py`'s own random-rollout check already uses into a
      reusable helper), for whichever engine-mechanics checks still need
      *some* driver to run games at all.
- [ ] Rewrite the invariant coverage worth keeping (e.g. "hand-fed Tron
      pieces assemble deterministically by turn 3") using **direct
      hand-authored actions** (explicit calls to `play_land_from_hand` in
      sequence) instead of a heuristic — preserves the coverage without
      needing any decision-making code.
- [ ] Verify: full `game.py` check suite (post-deletion, replacements
      included) passes; grep confirms zero remaining references to any
      deleted function anywhere in the codebase, including `tron_env.py`
      and `harness.py`.

## Phase M6 — Rework scoring + reporting

**Accomplishes**: implements the locked-in scoring design (mandatory
score 1 + arbitrary-length additional scoring functions, centrally
zeroed on failure) and generalizes reporting away from the
Tron-specific `(turn_assembled, turn_online)` shape.

- [ ] `TrainingHarness.__init__` gains `scoring_fns: list[Callable[[state],
      float]] = []` (in addition to the existing mandatory `reward_fn`).
- [ ] Wherever a game's final score is produced (`_GameLogger.finalize()`
      and equivalent), compute `scores = [reward_fn(state, True, horizon)]
      + [fn(state) for fn in scoring_fns]`, then **centrally** zero every
      entry if the game never terminated by the horizon — one rule, not
      repeated per scoring function.
- [ ] Update the log schema: replace the fixed `score`/`score2` fields
      with a generic `scores: [float, ...]` list. Document this as a
      deliberate breaking change to the log format (the visualizer, which
      currently reads `score`/`score2` by name, is explicitly out of
      scope for this refactor and will need its own follow-up pass).
- [ ] Rework `aggregate_results(results, horizon)`/`print_report`:
      replace the `(turn_assembled, turn_online)` 2-tuple assumption with
      `(terminated_turn, scores)` per game — one generic "% terminated by
      turn X" column, plus a mean/median per configured score index among
      terminated games.
- [ ] Verify: a hand-computed aggregation example (same style as today's
      Phase 7 check) matches the new function's output exactly.

## Phase M7 — Migrate Tron itself into the new format

**Accomplishes**: proves the whole framework actually works by making
Tron its first real user — not a special case anymore, just the first
deck config.

- [ ] Assemble Tron's deck config: `TRON_DECKLIST` (from M2),
      `tron_terminated` (from M3), `tron_reward_fn` (the existing
      `assembled_with_resource_quality` formula, updated to read
      `state.turn_won` instead of `state.turn_assembled`), and a new
      `tron_online_score(state) -> float` scoring function — the
      successor to the old `turn_online` concept, now expressed as a
      post-hoc score rather than a second termination tier. (Exact
      formula for this one is new design, not a behavior-preserving port
      — decide precisely at implementation time.)
- [ ] Wire these into whichever concrete "pass the pieces in" mechanism
      Phase M4's generic env/harness ends up exposing (plain dependency
      injection, per the plan — no config-file/registry layer).
- [ ] Verify: fixed-seed diff against the Phase M0 snapshot, run all the
      way through the fully-reassembled pipeline — same shuffled hands,
      same termination turns as pre-refactor, on the original seed set.

## Phase M8 — Retrain + verify

**Accomplishes**: closes the loop — confirms the refactored pipeline
produces a real policy of comparable quality, not just code that runs.

- [ ] Update `train_drl.py` to construct the new generic
      harness/env using Tron's deck config (M7) instead of the current
      hardcoded `tron_env.TronEnv` construction.
- [ ] Train a fresh model from scratch at a scale comparable to the
      original 50,000-episode run, saved under a **new** path (e.g.
      `models/tron_run_2` — `models/run_50k` stays untouched as a
      historical artifact of the pre-refactor system, not overwritten).
- [ ] Update `evaluate_drl.py` similarly; run a 1000-game evaluation
      (matching the most recent eval methodology).
- [ ] Compare turn-by-turn termination percentages and mean/median
      termination turn against the pre-refactor baseline (83.0%
      assembled / 81.7% online by turn 6, mean assembly turn 3.85).
- [ ] Report the comparison back to you explicitly — this is a decision
      point requiring your sign-off (same pattern as every pilot-run
      decision point in this project), not an automatic pass/fail.
