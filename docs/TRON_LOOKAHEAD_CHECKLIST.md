# Lookahead Search — Implementation Checklist

Companion to [TRON_LOOKAHEAD_PLAN.md](TRON_LOOKAHEAD_PLAN.md). Concrete
build order, phase by phase, each ending in a runnable check before moving
on — same discipline as the original engine build. Planning only; nothing
in `game.py` changes until you say go.

**Correction to the earlier plan**: it claimed "no changes needed to
Phases 0–4." That's mostly true but not quite — Phase 4's `run_turn` needs
one small, behavior-preserving extraction (Phase L2 below) so a rollout
can resume a game that's already mid-main-phase, not just start one fresh.
Flagging since it contradicts what I said earlier.

## Phase L1 — Decompose Phase 5 into granular try-functions (no behavior change)

Today, three of the existing `try_X` functions each silently pick *among*
multiple cards internally (loop, return the first that's legal):
`try_cast_search_artifact` (Map vs Candy Trail), `try_cast_value_artifact`
(Bonder's vs Barrels vs Relic), `try_extra_draw` (Candy Trail-sac vs Relic
vs Bonder's-draw). The enumerator (Phase L3) needs each of those as an
independently checkable candidate, not bundled.

- [ ] Split `try_cast_search_artifact` into `try_cast_expedition_map(state)`
      and `try_cast_candy_trail(state)`, each doing exactly what its slice
      of the original loop body did.
- [ ] Split `try_cast_value_artifact` into `try_cast_bonders_ornament(state)`,
      `try_cast_barrels(state)`, `try_cast_relic(state)`.
- [ ] Split `try_extra_draw` into `try_activate_candy_trail_sac(state)`,
      `try_activate_relic(state)`, `try_activate_bonders_draw(state)` —
      each repeats the `_tron_fully_online(state)` guard individually
      (small, acceptable duplication of a one-line check).
- [ ] Rewrite the three original functions as thin composites so
      `policy_choose_action`'s behavior is byte-for-byte unchanged:
      `try_cast_search_artifact = lambda s: try_cast_expedition_map(s) or try_cast_candy_trail(s)`
      (same pattern for the other two).
- [ ] Define `ALL_GRANULAR_TRY_FNS`, the 12-entry tuple of every granular
      function (the original 6 unsplit ones — `try_activate_expedition_map`,
      `try_ancient_stirrings`, `try_crop_rotation`, `try_forestcycle_generous_ent`
      — plus the 6 new ones above minus the 3 old composites they replace).
- [ ] Sanity check: rerun the existing Phase 5/6/7/8 sanity checks
      unmodified — they must still pass exactly as before, proving the
      split didn't change `policy_choose_action`'s behavior.

## Phase L2 — Resumable turn loop (Phase 4 extraction)

- [ ] Extract `run_turn`'s inner action loop into
      `run_main_phase(state, choose_action)`: the existing
      `for _ in range(MAX_MAIN_PHASE_ACTIONS): ...` block, unchanged, just
      pulled out of `run_turn`.
- [ ] `run_turn(state, choose_action)` becomes: increment turn, reset
      `lands_played_this_turn`, `untap_step`, `draw_step`, then call
      `run_main_phase(state, choose_action)`.
- [ ] Add `continue_game(state, horizon, choose_action)`: call
      `run_main_phase(state, choose_action)` once (finishes whatever turn
      is already in progress — a no-op if `state` is at a clean turn
      boundary), then `while state.turn_number < horizon and
      state.turn_assembled is None: run_turn(state, choose_action)`.
- [ ] `run_game` becomes `state = new_game_state(...); return
      continue_game(state, horizon, choose_action)`.
- [ ] Sanity check: rerun the Phase 4 sanity check unmodified — same
      assertions must hold, proving the extraction is behavior-preserving.

## Phase L3 — State cloning

- [ ] `Permanent.clone(self)`: new `Permanent(self.card_def,
      tapped=self.tapped)` with `flags = dict(self.flags)` (a real copy,
      not the same dict reference).
- [ ] `clone_state(state, rng)`: new `GameState(state.on_the_play,
      rng=rng)` (caller supplies the fresh rollout RNG — see Phase L4);
      `library`/`hand`/`graveyard` are shallow-copied lists (safe —
      `CardDef` instances are shared immutable singletons); `battlefield`
      is `[p.clone() for p in state.battlefield]`; scalar fields
      (`lands_played_this_turn`, `turn_number`, `turn_assembled`,
      `turn_online`) copy directly.
- [ ] Sanity check: clone a state, tap a permanent and draw a card on the
      clone, assert the original `state`'s battlefield/hand are untouched.

## Phase L4 — Legal-action enumeration

**Candidates are represented as references, never pre-built closures** —
a `try_fn` (from `ALL_GRANULAR_TRY_FNS`) or a `CardDef` (for a land-drop
choice), re-invoked fresh against whichever state (real or a rollout
clone) it's about to be applied to. This matters: the existing `try_X`
functions build closures that capture the `state` argument they were
called with; calling `try_ancient_stirrings(state)` once and then
executing that same closure against a *different* state object (a clone)
would silently mutate the wrong game. Always call `try_fn(target_state)`
fresh at the point of use, never cache and reuse a closure across states.

- [ ] `enumerate_land_drop_candidates(state)`: one `CardDef` per **distinct
      name** of land in `state.hand` (dedupe — two copies of Forest are
      one candidate, not two). Only meaningful when
      `state.lands_played_this_turn == 0`.
- [ ] `enumerate_mana_candidates(state)`: `[fn for fn in
      ALL_GRANULAR_TRY_FNS if fn(state) is not None]` — note this calls
      every `try_fn` once just to check legality; the resulting closures
      from *this* call are discarded, not reused (see above).
- [ ] **Ambiguity threshold, decided here**: pass does *not* count toward
      "≥2 candidates." If `enumerate_mana_candidates` returns 0 or 1 real
      candidates, that's unambiguous (matches today's heuristic exactly —
      do the one action, or pass if none). Only once ≥2 real candidates
      exist does it become a genuine branch point worth searching, and
      only then does "pass" get added as a 3rd-or-later option in the
      comparison. Same logic for land drop (≥2 distinct land names in
      hand → real branch; 0 or 1 → just play it, no search).

## Phase L5 — Rollout value estimator

- [ ] `estimate_value(state, candidate, is_land_drop, horizon,
      num_rollouts, seed_counter)`: for `num_rollouts` iterations —
      `rollout_rng = random.Random(next(seed_counter))`;
      `clone = clone_state(state, rollout_rng)`;
      `clone.rng.shuffle(clone.library)` (the determinization step — see
      plan); apply `candidate` to `clone` (`play_land_from_hand(clone,
      candidate)` if `is_land_drop`, else `candidate(clone)` re-derived
      fresh then called if not `None`, else do nothing for the "pass"
      candidate); `continue_game(clone, horizon, policy_choose_action)`;
      score `1.0`/`0.0` for `clone.turn_assembled is not None` and
      separately for `clone.turn_online is not None and clone.turn_online
      <= horizon`. Return `(mean_assembled, mean_online)`.
- [ ] `seed_counter`: a single `itertools.count()` created once per
      top-level `simulate_many`-style run and threaded down through every
      call, so every rollout across the *entire* run gets a unique,
      deterministic seed — the whole lookahead-driven run stays fully
      reproducible from one top-level seed, matching the existing
      engine's determinism.
- [ ] Sanity check: hand-build a state with 2 obviously-unequal candidate
      actions (e.g. one path finds a missing Tron land for certain, the
      other doesn't), run `estimate_value` with a small `num_rollouts` on
      each, and assert the better one scores higher.

## Phase L6 — Search wrapper

- [ ] `search_best(state, candidates, is_land_drop, horizon, num_rollouts,
      seed_counter)`: run `estimate_value` per candidate, return the one
      with the highest `mean_assembled`, ties broken by `mean_online`,
      further ties broken by candidate order (deterministic).
- [ ] `lookahead_choose_action(state, horizon, num_rollouts, seed_counter)`:
      if `state.lands_played_this_turn == 0`: get land candidates (Phase
      L4); if ≥2, return `search_best(...)`; if exactly 1, return it
      directly; if 0, fall through. Then: get mana candidates; if ≥2,
      return `search_best(state, candidates + [None], is_land_drop=False,
      ...)`; else return the single candidate's freshly-derived action (or
      `None` to pass).
- [ ] `make_lookahead_policy(horizon, num_rollouts, seed_counter)`: returns
      a one-argument `choose_action(state)` closure over the above —
      same call signature as `policy_choose_action`, so it's a drop-in
      replacement anywhere a `choose_action` is expected.

## Phase L7 — Wiring / config surface

- [ ] `simulate_many_lookahead(num_simulations, horizon, on_the_play, seed,
      num_rollouts)`: same structure as `simulate_many`, but builds one
      `itertools.count()` and one `make_lookahead_policy(...)` up front
      and reuses that policy object across all `num_simulations` calls to
      `run_game`.
- [ ] Sanity check: run 5–10 games through `simulate_many_lookahead` with
      a tiny `num_rollouts` (e.g. 3) and confirm it completes without
      error and returns well-formed `(turn_assembled, turn_online)` pairs,
      same shape as `simulate_many`'s output.

## Phase L8 — Timing pilot (must run before any full report)

- [ ] Run `simulate_many_lookahead` with `num_simulations=100–500`,
      `num_rollouts=30` (starting proposal), measure wall-clock.
- [ ] Extrapolate to `num_simulations=50_000` and report the estimate
      before running it for real.
- [ ] If unacceptable: try lower `num_rollouts` first (cheapest fix),
      then reconsider outer game count; `multiprocessing` across games is
      available as a later lever (games are independent) but not built
      unless the simpler options aren't enough.

## Phase L9 — Validation

- [ ] Run `simulate_many` and `simulate_many_lookahead` with the **same
      seed** and compare the Assembled%/Online% tables (`print_report`
      works unchanged on either's output).
- [ ] Find 2–3 games where the two policies' turn_assembled differ, trace
      each turn-by-turn (same technique as the original Phase 8 trace) —
      confirm the lookahead's picks make real sense, not just noise.
- [ ] Decide whether the improvement justifies keeping lookahead as the
      default vs. reserving it as an opt-in given the runtime cost.
