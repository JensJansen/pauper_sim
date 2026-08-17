# Work checklist

Four large changes, taken **one at a time**. Each item's goals are agreed with the
repo owner *before* any code is written; the "Goals" section starts empty and is
filled in during that discussion, then the work is done against it.

Status: `[ ]` not started · `[~]` in progress · `[x]` done

**Order of work (changed 2026-08-17): 2 → 3 → 4 → 1.** Cleanup moves to the end.
The first pass of it is already done and committed, but the rest is deferred until
items 2–4 have landed — each of them rewrites code and docs that a cleanup pass
would otherwise have to touch twice. Item numbers below are left as they were so
existing commit messages keep pointing at the right thing.

---

## 1. `[~]` Cleanup — **deferred to last**

Repo-wide cleanup pass.

### Done (commit `f6a16b9`)

Deleted: `RL_METHODOLOGY_PLAN.md`; all of `checkpoints/` (1.4 GB — every population,
both stack backups, `vocab.json`, `shared_stack_frozen.pt`); all of `logs/` (80 MB);
the four concluded A/B configs; the seven one-off `src/analysis/` scripts and their
`_shared.py`. Every reference to a deleted *file* was repaired (`README.md`,
`rl/rewards.py`, `tests/test_run_league.py`, `tests/rl/test_train.py`,
`.claude/skills/train/SKILL.md`). 655 passed, 1 skipped.

**Consequence:** a fresh pretrain + freeze is now mandatory before any league run.

### Done (commit `7812e33`)

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

## 2. `[x]` Cast-then-pay, with floating mana allowed only in main phases

**DONE** — commits `9cc470d` (core), `774fe5b` (500.4 + delve hole), plus the
test/doc pass. 675 passed, 1 skipped.

### What actually happened vs. the plan

The plan held. Two things it did **not** predict, both found by running the code
rather than reading it:

1. **A third hazard, and it was a faithfulness bug rather than a guard gap.**
   CR 601.2f activates mana abilities at *one* point — after modes, X, delve and
   which-copy are settled. The engine allowed them throughout, which is correct
   for a priority window but wrong mid-cast, since no player receives priority
   during 601.2. Left alone it strands: each "Delve N" button re-checks
   affordability, so a tap taken mid-choice can leave them *all* illegal. Seen
   live (Gurmag Angler, the black source tapped for blue after announcing).
   Refusing the whole window is more faithful *and* strictly safer than guarding
   individual actions inside it, and it subsumed the planned `choose_cast_copy`
   `reserved_cost` guard entirely.
2. **The delve *exile* needed a per-call flag, not a pending-kind entry.**
   `choose_graveyard_card` is equally used at resolution time (Masked Vandal,
   Relic), where mana abilities are perfectly legal, so being mid-cast is a fact
   about the *call*, not the kind.

Three of my own bugs, worth recording since each was caught by a mechanism the
plan put in place rather than by review:

- A state-keyed cache on `available_mana_units` — `plan_payment` is not
  sweep-only (cards and tests call it directly), so nothing reset it and it
  reported a just-floated pool as still empty. Removed; measured at ~1.5× the
  pool-only check it replaced, which is not worth a correctness hazard.
- Widening a *multi-symbol* source by an Abundant Growth grant, which claims the
  native output **and** a granted colour from one tap. Overstates supply, so it
  would strand. Caught while writing the code, before running it.
- `_filter_would_strand_payment` referencing a source name its signature did not
  take — it now has to resolve the filter permanent, which the pool-only version
  never needed.

**The diagnostic paid for itself immediately.** It named the delve bug on the
first capture: `announced == remaining`, ten units all `{U}`,
`can_pay(units, announced)=False` ⇒ cause (a), wrong at announce time, with all
four black-capable duals showing `OUT: tapped`. Adding an assertion to
`begin_pay_cost` then named the exact call site instead of failing several
actions later inside the agent.

### Deviation ledger: **two removed, one added**

| | |
|---|---|
| **removed** | `state.in_cleanup`'s mana ban (2026-08-10) — subsumed by the main-phase rule |
| **removed** | combat's shared mana window — CR 500.4 now applies at every boundary with no exception |
| **added** | speculative floating restricted to the active player's own main phase |

Also fixed, unplanned and free: `pay_unless` (Ward, Spell Pierce) could only be
paid from already-floating mana. CR 605.3a allows mana abilities "whenever a
rule or effect asks for a mana payment", so the payer may now tap for it.

### Not done, deliberately

`PRIORITY_ROUND_ACTION_CAP` is left at 20. Cast-then-pay does not change the
action *count* of a cast (taps merely moved from before the cast to inside it),
but a cap-exhausted payment now abandons an announced spell rather than only
wasting float. Watch for truncations in training before raising it.

---

## 2-original. Plan as agreed (kept for reference)

Move from pay-then-cast to the real rule 601 sequence: announce the spell, choose
modes/targets/X, *then* activate mana abilities and pay. Floating mana is permitted
only during main phases.

### Decided (owner, 2026-08-17)

- **Option 2.** Payment-time mana activation (CR 601.2f) is fully faithful and works in
  *every* phase. Only **speculative** floating — activating a mana ability with no
  payment in progress — is restricted, to the **active player's own main phase**
  (`phase ∈ {MAIN1, MAIN2}` *and* `active_idx == turn_player_idx`). This is the sole
  authorized deviation and carries an `# AUTHORIZED SIMPLIFICATION:` marker.
- **Restore CR 500.4 fully.** The `_COMBAT_PHASES` "combat is one mana window"
  exception is removed; mana dissipates between every step and phase.
- **Cleanup needs no special case.** Cleanup is not a main phase, so the three
  `not state.in_cleanup` mana checks are subsumed by the new gate and are deleted
  along with their own `AUTHORIZED SIMPLIFICATION` marker.
- **Mana-burn penalty stays wired.** Speculative floating still exists in main phases,
  so over-floating is still a real, punishable misplay — just a much narrower one.
- **Uniform per-action gate (owner's formulation, adopted over the two special-case
  guards originally planned):** every action taken while a payment is open is legal
  only if the payment is *still completable afterwards*. This covers taps, filters and
  both stages of Saruli's subdecision with one rule, and subsumes
  `_filter_would_strand_payment` entirely.
- **Saruli Caretaker is excluded from the affordability count.** Its cost taps another
  creature that may itself be a mana source, so counting it as free supply overstates
  (in `spy_combo`, Saruli + Llanowar Elves is 1 unit, not 2 — using Saruli *consumes*
  the Elves). Exclusion is conservative and safe; the agent can still float Saruli by
  hand in its own main phase. Recorded as a known false negative.

Net effect on the mandate ledger: **two authorized deviations removed, one added.**

### Why the solver must be exact, not greedy

A greedy solver is *sound but incomplete*: if it finds a way to pay, that way is real,
so it never strands — but it says "no" to costs that are genuinely payable, hiding a
legal cast, which is itself a faithfulness violation. The pre-float-first greedy needed
two passes of colour-preference heuristics to be even that good.

The uniform per-action gate does **not** rescue an unsound cast check: if legality says
"castable" when it is not, every subsequent action fails its own gate, the mask goes
all-False, and that is exactly the crash being avoided. So the cast check has to be
sound on its own; and since the exact test is *shorter and cheaper* than the greedy one
here, it should also be complete.

### What the investigation found

- **The engine is float-first**, not an approximation of 601.2: `begin_pay_cost` is
  entered *only* when the pool already covers the cost, and "no source is ever tapped
  during payment" ([mana.py:186](src/game/mana.py#L186)).
- **CR 500.4 is already correct** — `_empty_mana_pools` empties both players' pools at
  every step/phase end. There is no mana burn (the pre-2010 life-loss rule is
  correctly absent); `mana_mistake_burn` is an RL signal only.
- **601.2f already works mechanically.** Mana abilities carry no `_pending_gate` and
  `legal_action_mask` calls every closure unconditionally, so a mana ability is
  *already* legal during an open `pay_cost`. Nothing new is needed to allow tapping
  mid-payment.
- **Every affordability gate in the codebase funnels through one function**,
  `game.plan_payment(state, cost)` — ~25 call sites, all of the form
  `plan_payment(...) is not None`. Changing what that function *means* is the change.
- **A prior design had a real tap-solver** (pre-`68e4f64`), removed by float-first
  because it *auto-tapped* — it took the decision away from the agent. The synthesis
  wanted here is: solver for **legality only**, agent still chooses every tap.
- **The catalog is small**: 32 mana specs — 17 `fixed`, 12 `flexible`, and one each of
  `count` / `count_all` / `fixed_multi` / `tron`. Every source produces a fixed
  multiset except `flexible`, which produces one symbol from a colour set. Wall of
  Roots is once-per-turn (`used_this_turn`), so no source is repeatable.

### Two new hazards the change introduces (both reachable in the real decklists)

Float-first guarantees "a payment, once begun, can always be completed" — there is no
*Abandon payment* action, so an unpayable payment is an all-False mask and a crash.
Letting a payment begin before the mana exists creates two ways to break it:

1. **Conduit Pylons** is *both* a filter and a mana source (`fixed C`). Filtering taps
   it, removing a unit the affordability check had counted. `monster_tron`.
   The existing `_filter_would_strand_payment` does **not** catch this — its reasoning
   assumes the pool size is invariant, which is true, but the *source* it taps is now
   part of affordability.
2. **Saruli Caretaker**'s extra cost taps another creature, which may itself be a mana
   source, and its colour is chosen afterwards — so a wrong colour choice can strand a
   coloured requirement. `spy_combo` (which also runs Overgrown Battlement, Wall of
   Roots and Lotus Petal, so the tapped creature is very often a mana source).

### Goals

1. `plan_payment` means "pool **plus** mana still available from untapped sources",
   checked **exactly** — no false positives (which strand and crash) and no false
   negatives (which hide a legal cast, a faithfulness violation).
2. Agent keeps every tap decision. The solver decides *whether* a cost is payable,
   never *how* it is paid.
3. Speculative floating restricted to main phases; payment-time activation unrestricted.
4. No stranding remains reachable: every mid-payment choice that can reduce
   affordability is gated on the payment staying completable.
5. Net simplification, not net addition — the mana-burn reward machinery exists to
   punish a misplay that this change makes structurally impossible.

### File-by-file plan

**`src/game/mana.py` — the only file with genuinely new logic.**

- `available_mana_units(state)` → `list[frozenset[str]]`. One entry per mana symbol the
  active player could still put toward a cost: each floating pool pip as `{c}`, plus
  each symbol every untapped/available source would produce. `flexible` → one entry
  with its whole colour set; `fixed_multi`/`tron`/`count`/`count_all` → one entry per
  symbol; Utopia Sprawl's bonus → extra entries; Abundant Growth's grant widens the
  land's own entry. Reuses the existing per-permanent gates (`tapped`,
  `tap_summoning_locked`, `mana_extra_available`).
- `can_pay(units, cost)` → bool. Exact, via the deficiency form of Hall's theorem:
  feasible **iff** `len(units) >= total pips needed` **and**, for every non-empty
  subset `S` of the colours actually demanded, `sum(need[c] for c in S) <=` the number
  of units whose colour set meets `S`. At most 6 demanded colours ⇒ ≤63 subsets, and
  real costs demand 1–3 ⇒ ≤7. Exact both ways, which is the requirement: a false
  positive strands the payment and crashes; a false negative hides a legal cast.
- `plan_payment(state, cost)` becomes `can_pay(available_mana_units(state), cost)`,
  keeping its truthy-or-`None` return so **all ~25 call sites are untouched**.
- `pool_can_pay` stays as-is — still the right question for "pool alone", still
  directly tested, and now one input among several.
- `payment_in_progress(state)` — one-liner (`pending_resolution["kind"] == "pay_cost"`)
  shared by the three main-phase gates below, so the rule lives in one place.
- Per-sweep cache for `available_mana_units`, cleared by the existing
  `reset_mana_cache()` that `legal_action_mask` already calls before and after every
  sweep — same lifecycle as `_enchanting_cache`.
- Docstrings: `begin_pay_cost`, `plan_payment`, `pool_can_pay` and the module header
  all currently assert float-first semantics and must be rewritten.

**`src/drl_env/_actions_mana.py` — the action-space restriction and one guard.**

- `_mana_ability_legal`, `_mana_extra_choose_legal`, `_filter_mana_legal`: add
  `payment_in_progress(state) or state.phase in SORCERY_SPEED_PHASES`. This is the
  `# AUTHORIZED SIMPLIFICATION` and carries the marker.
- `_filter_would_strand_payment`: replace its bespoke colour-only reasoning with
  "rebuild the units list as it would be *after* this conversion — pool pip removed,
  output unit added, **the filter source's own mana units removed because it taps** —
  and re-ask `can_pay`". Strictly simpler than what's there now, catches the Conduit
  Pylons hazard the current version misses, and subsumes its `choose_cast_copy`
  special case.

**`src/game/resolution/handlers_casting.py` — the Saruli guard.**

- `begin_mana_subdecision` / `execute_mana_subdecision_target`: while a payment is in
  progress, only offer creatures whose tapping leaves the payment completable.
- `begin_mana_color_choice`'s `can_produce`: same, per colour. Both reuse `can_pay`.

**`src/drl_env/_actions_cast.py`, `_actions_cast_altzone.py`, `_actions_resolution.py`**

- **No logic changes** — they all route through `plan_payment`. Comment/docstring
  updates only, where text asserts pool-only semantics (`_delve_execute`'s "pure pool
  spend", `_pay_unless_pay_legal`, the alt-cost notes).
- Behavioural note, no code: `pay_unless` (Ward, Spell Pierce) currently can only be
  paid from already-floating mana. CR 605.3a explicitly allows activating mana
  abilities "whenever a rule or effect asks for a mana payment", so this change fixes
  an existing deviation for free.

**`src/drl_env/_actions_table.py`**

- The 16-line "Float-first: NO 'Abandon payment'" block is now wrong in its premise
  ("affordability was checked exactly … before the payment began, so spending alone can
  never strand"). Rewrite to state the new invariant and both guards.

**`src/game/turn.py`**

- `_empty_mana_pools` — no change (500.4 is already right).
- Two *consequences* to decide, not code I'd write unasked: the `_COMBAT_PHASES`
  one-mana-window simplification becomes moot, and `_tally_mana_mistake` loses most of
  its reason to exist. See the questions below.

**Tests**

- New in `tests/game/test_mana.py`: `can_pay` unit tests, adversarial —
  `{G:2}` against `[{G,R},{R}]` must be **False**; `{G:1,R:1}` against `[{G,R},{G,R}]`
  must be **True**; `{G:1,generic:1}` against `[{G}]` must be **False**.
- New: cast-then-pay end to end — empty pool, untapped lands, cast is legal, tap
  during the payment, spell resolves.
- New: speculative float illegal outside a main phase; payment-time float legal in
  every phase.
- New: one regression per hazard (Conduit Pylons; Saruli tapping a mana creature).
- Existing: catalog tests mostly pre-fill `state.mana_pool`, which still satisfies the
  broader check, so most pass untouched. The ones that will flip are those asserting a
  cast is *illegal* with an empty pool but untapped lands present — each needs review,
  since some of those become correctly legal.

**`src/rl/agent.py` — the stranded-payment bug report.**

`_raise_all_false` already dumps both pools and every mana source. What it cannot show
is **what the solver itself saw**, which is the only thing that localises a mask flaw.
Add, for the `pay_cost` case specifically:

- a distinct error message naming this failure — a stranded payment is a *legality-mask
  bug*, not a generic unrepresentable state, and should never be confused with one;
- the **original cost** the cast was gated on (stashed on the pending at
  `begin_pay_cost`) beside the **remaining** cost now, so it is immediately visible
  whether the solver was wrong at announce time or something consumed supply after;
- `available_mana_units(state)` as the solver sees it *right now*, plus the `can_pay`
  verdict against the remaining cost;
- one line per battlefield permanent with a mana/filter spec giving the reason it is or
  is not in that unit list — `tapped`, summoning-locked, `mana_extra_available` false,
  or excluded as `mana_extra_choose`.

Those four together separate the two possible causes: a **unit-construction** bug (a
source wrongly counted or wrongly omitted) or a **`can_pay`** bug (the feasibility test
itself). Without them the dump shows the board but not the model of the board, and the
model is where the bug would be.

**`src/game/turn.py` — priority during the cast window (verification, not a change).**

Owner's requirement: no enemy action during a cast window, only once the spell is on
the stack. **Already true, by construction** — a player who takes an action keeps
priority (`_priority_round`, `active_idx` unchanged; only a Pass flips it), and Pass is
illegal while a pending is open (`_pass_legal._pending_gate = _GATE_NO_PENDING`).
`push_to_stack` fires in `_after_pay`, so the opponent's first chance to respond is
after the spell is cast, per CR 601.2i/117.3c. Cast-then-pay lengthens that window a
lot, so the invariant gets an explicit test rather than being left implicit.

Also here: the `PRIORITY_ROUND_ACTION_CAP` exhaustion path drops an open pending. Its
comment ("float-first has no undo … any mana already spent is gone") needs rewriting —
under cast-then-pay a dropped payment abandons an *announced spell*, not just floated
mana. The cap value (20) is left alone; flagged to watch for truncations in training.

**Docs** — README's mana section, the module docstrings above, and
`.claude/skills/train/SKILL.md`'s reward paragraph if the burn penalty is unwired.

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
