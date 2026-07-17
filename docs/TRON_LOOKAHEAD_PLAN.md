# Lookahead Search — Implementation Plan

Companion to [TRON_SIMULATOR_PLAN.md](TRON_SIMULATOR_PLAN.md) /
[TRON_SIMULATOR_CHECKLIST.md](TRON_SIMULATOR_CHECKLIST.md). Planning only —
nothing in `game.py` changes yet.

## Goal

Replace the fixed-priority greedy policy at genuinely ambiguous decision
points with a Monte Carlo rollout search: try each legal candidate action,
simulate the rest of the game forward some number of times, and pick
whichever candidate produced the best outcome on average. Not
reinforcement learning — no training, no network, no persisted state
between games. This is "try it and see" at decision time.

## Decisions locked in (your answers)

1. **Scope — targeted, not full replacement.** The existing greedy
   heuristic (Phase 5) stays exactly as-is for every decision that's
   already forced (only one legal option). Search only activates at a
   decision point when **≥2 legal candidate actions exist right now** —
   that condition is also the precise, mechanical definition of
   "ambiguous" used below, not a separate judgment call.
2. **Rollout continuation policy — the existing greedy heuristic.**
   Search re-decides only the current branch point; every future decision
   within that rollout (including the rest of the current turn and all
   subsequent turns) is piloted by the already-verified `policy_choose_action`
   from Phase 5. Not recursive.
3. **Objective — maximize P(Tron assembled by the 6-turn horizon).** Ties
   broken toward whichever candidate also scores higher on P(online).
4. **Compute budget — keep ~50,000 outer games, accept a longer run.**
   See "Performance" below for what that implies and a mitigation if the
   actual runtime turns out to be unacceptable.

## Methodology decisions I'm resolving without asking (implementation
## detail, not high-level functionality — flag if you'd rather weigh in)

- **Determinization**: at a decision point, the true order of your
  remaining library is already fixed for the outer game (single seeded
  RNG stream), but the *player* doesn't know that order — only which
  specific cards remain unseen. So each rollout takes a **fresh reshuffle**
  of a copy of `state.library` (same cards, new random order) before
  simulating forward. This is the standard, statistically correct way to
  evaluate a decision under hidden information ("hindsight optimization").
  Minor acknowledged imprecision: Ancient Stirrings/scry/surveil bottom
  specific rejected cards, which leaks a sliver of order information a
  literal reshuffle discards — irrelevant in practice since nothing reads
  bottom-order within a 6-turn horizon.
- **Rollout RNG must be independent of the real game's RNG.** A rollout
  clone gets its own `random.Random` (seeded off a rollout-seed counter),
  never `state.rng` directly. Reusing the real stream would either desync
  the true game's future draws or make every rollout replay the same
  hypothetical future (defeating the point of sampling K of them).
- **Sub-choices inside a candidate action stay heuristic-pinned.** E.g.
  which specific land Crop Rotation sacrifices, or which card a search
  effect prioritizes, are not separately searched — only the top-level
  "which spell/land to play this instant" branches. Keeps the action
  space bounded to what Phase 5 already enumerates one at a time, rather
  than exploding into per-sub-choice combinatorics. Can be revisited if
  the top-level search alone doesn't move the numbers much.

## Required steps

### 1. Legal-action enumerator (replaces "first match wins")

Today's `MANA_SPEND_TRY_ORDER` walk stops at the first `try_X` that
returns an action. Search needs *all* currently-legal candidates at a
decision point, not just the top-priority one.

- [ ] Land-drop: enumerate one candidate action per **distinct land card
      in hand** (not just the heuristic's pick). Not searching "whether"
      to play a land at all — in this deck there's no downside to using
      an available land drop, so that's not a real branch.
- [ ] Mana-spend: run every `try_X` in `MANA_SPEND_TRY_ORDER`
      independently (not short-circuiting on the first hit), collect
      every one that returns a legal action this instant, plus "pass" as
      an always-available candidate.
- [ ] If the resulting candidate list has 0 or 1 entries, this decision
      point is unambiguous — return it directly, no rollout needed (the
      efficiency shortcut that makes "targeted" scope actually cheap).

### 2. Cheap state cloning

- [ ] `clone_state(state)`: new `GameState` with independent copies of
      `library`/`hand`/`graveyard` (shallow-copy is fine — `CardDef`
      instances are shared immutable singletons) and independent
      `Permanent` objects on `battlefield` (new object per permanent,
      copying `tapped` and a **copy of** `flags`, not the same dict
      reference). Scalar fields (`turn_number`, `lands_played_this_turn`,
      `on_the_play`, `turn_assembled`, `turn_online`) copy directly.
      `rng` is deliberately **not** carried over — see RNG note above.
- [ ] Sanity check: mutate a clone (tap a permanent, draw a card) and
      confirm the original `state` is untouched.

### 3. Rollout / value estimator

- [ ] `estimate_value(state, candidate_action, horizon, num_rollouts) ->
      float`: for `num_rollouts` iterations — clone `state`, give the
      clone a fresh independent RNG, reshuffle its library, apply
      `candidate_action` to the clone, then continue the turn/game to
      `horizon` using `policy_choose_action` (decision 2 above) exactly
      like today's `run_game`/`run_turn`. Score each rollout as `1.0` if
      `turn_assembled is not None` (decision 3 above), else `0.0`; return
      the mean.
- [ ] Tie-break data: also track mean `1.0 if turn_online is not None
      else 0.0` per candidate, used only to break exact ties on the
      primary score.

### 4. Search wrapper

- [ ] `lookahead_choose_action(state, horizon, num_rollouts)`: build the
      candidate list (step 1); if ≤1, return it as-is; else run
      `estimate_value` per candidate and return the action with the
      highest score (ties broken per above, then arbitrarily but
      deterministically).
- [ ] Same signature as `policy_choose_action(state)` (horizon/num_rollouts
      bound via a closure or partial) — drop-in replacement for
      `choose_action` in `run_game`/`run_turn`/`simulate_many`. **No
      changes needed to Phases 0–4 or 6–8.**

### 5. Performance / pilot run

Nested rollouts are meaningfully more expensive than the current engine
— every ambiguous decision point now costs `num_rollouts` extra simulated
partial-games instead of one heuristic lookup. Actual overhead depends on
how often ≥2-candidate decision points occur in practice, which isn't
known precisely until measured.

- [ ] Before committing to the full ~50,000-game report: run a small
      timing pilot (e.g. 100–500 games) with `lookahead_choose_action`,
      measure wall-clock, and extrapolate to 50,000.
- [ ] Pick `num_rollouts` (starting proposal: **30** — enough to average
      out single-game variance without being wasteful; tune after seeing
      the pilot's actual decision stability) and confirm the extrapolated
      full-run time is actually acceptable before running it for real.
- [ ] If the extrapolated time is too long: options in order of
      preference are (a) lower `num_rollouts`, (b) shrink outer game
      count for this comparison (you already have the option to do that,
      just chose not to by default), (c) parallelize across games with
      `multiprocessing` (games are fully independent, so this is a clean
      lever if it comes to that — not built unless actually needed).

### 6. Validation

- [ ] Run both policies (`policy_choose_action` vs
      `lookahead_choose_action`) over the **same seeds** and compare the
      Assembled%/Online% tables side by side.
- [ ] Spot-check a handful of games where the two policies' choices
      diverge with a manual turn-by-turn trace (same technique as the
      Phase 8 trace) — confirm the search's picks look like something a
      sensible pilot would actually do, not just "different because
      sampling noise broke a tie differently."
- [ ] Decide, based on the comparison, whether the improvement (if any)
      is worth keeping lookahead as the default vs. reverting to the
      heuristic for speed.

## Open item

`num_rollouts` and whether to lower it, plus what to do if the pilot's
extrapolated runtime is unacceptable, are flagged above as decisions to
make **after** the timing pilot runs — not blocking the start of
implementation, since they depend on a number we don't have yet.
