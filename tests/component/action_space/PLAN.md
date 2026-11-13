# Action space: end-to-end test plan

Status: **layers A0-A6 built (see README.md); L7-L13 below are still a plan.**

## 1. What "correct" means (the contract)

| # | Property | Failure it prevents |
|---|---|---|
| A | **Sound** - every offered action is legal in real Magic, executes without error, and does what its label says | engine crash, illegal play, a rules violation the agent learns to exploit |
| B | **Complete** - every legal action of the modeled game is offered, exactly once | a line of play the agent can never find (silent; shows up nowhere) |
| C | **Live** - no reachable decision has an empty mask; no legal action is a no-op loop; games end | all-False crash, infinite priority loop |
| D | **Pure** - computing the mask changes nothing, depends on nothing but the position, and is invariant to order / slot numbers / seat / table variant | state corruption, cache leaks, order-dependent behavior |
| E | **Addressed** - pointer position <-> engine object, (name, slot) <-> permanent, row <-> its own execute closure, row order pinned | an action executing as a different action |
| F | **Consistent across the stack** - the mask stored at rollout is the mask used in the update; the sampled index runs the closure at that index | training on actions the agent did not take |

## 2. What the investigation found (these shape the plan)

1. **The engine has no independent legality oracle.** The `legal(state)` closures *are* the engine's legality, and the `execute`
   closures do not re-validate. So "execute it and see if the engine objects" proves soundness only for crashes, not for rules,
   and cannot prove completeness at all. The oracle has to be written separately, from the Comprehensive Rules and the card text.
2. **Random self-play does not reach most rows.** 660 games / 188,565 decisions (68 s): 663 of 1,423 table rows (all 11 decks)
   were never legal once. Cast is covered (3 never) but Choose, Tap, Enter room, Activate, Spend, Decline, Dispose, Shuffle, Copy
   and X=n are not. Coverage needs a *constructed* scenario matrix, not just more games.
3. **Throughput is not the constraint.** ~2,800 decisions/s, so 1M fuzz decisions is ~6 min.
4. **Forking state by `deepcopy` is unsafe in 1.7% of states** (5 of 289 sampled). Closures in `pending_resolution` / the
   stack capture live `Permanent` / `dict` / `list` objects (e.g. `_modal_execute..._after_pay`, `nihil_spellbomb_sac._on_player`),
   and `deepcopy` leaves those pointing at the *original* game. A branch from such a fork would silently mutate or read the
   original. Fork = `deepcopy` **plus** a closure-capture gate that falls back to replaying the recorded actions from the
   seed (the engine is seeded and deterministic - verified by the fork-fidelity test below). 1.4 ms per copy.
5. **Three hand-maintained copies of one dispatch** in `action_bridge.py` (`pointer_legal_mask`, `any_pointer_legal`,
   `pointer_kind`) plus `_ID_MATCHED_KINDS`. They must agree for every pending kind; nothing checks that.
6. **The full action table is a free second implementation.** It still contains own-side `Attack / Assign Blocker / Choose
   target` rows (the production table drops them for the pointer head). Two independent addressings of the same decisions is a
   differential oracle for the pointer bridge.
7. **Existing tests are scenario-by-scenario regressions** (about 2,200 lines, ~60 hand-built cases). Nothing checks the mask
   over real decisions, nothing plants bugs, and the README records "Planted bugs: none".
8. About 31 pending-resolution kinds exist (`begin_resolution` sites); 141 cards, 109 with a cast spec, 34 with mana
   abilities. No label is duplicated within a production table (A0 enforces it); the oracle maps each action
   to its label.

## 3. Layers

Each layer is a file; `support/action_oracle.py` and `support/action_corpus.py` are the shared pieces (like the encoder's).

**L0 - structure and registry audit** (static, seconds). The table content is derived a second time from the registry
(every `cast / cast_modes / x_cast_modes / delve / alt_cast / flashback / escape / plot / omen / prototype / cycle /
forestcycle / activated_abilities / graveyard_ability / mana*` spec -> expected row labels) and compared with the built table
- this checks that the order *pin* pins the right thing. Every `begin_resolution` kind is accounted for: table rows, a
pointer kind, or an explicit "no agent choice" entry. The three bridge dispatches agree on every kind. `_TARGETING_PREFIXES`
matches what the table emits. Labels well-formed, no two rows with identical (label, legal, execute) identity.

**L1 - purity and determinism.** The mask leaves a full state fingerprint unchanged (incl. `active_idx`, pools, caches).
Calling twice, after the other seat's mask, or after a `deepcopy` gives the same mask. `legal_action_mask` equals calling every
closure one at a time under reset caches. Table variants (with opponent / tokens / extra names) agree on every shared row.

**L2 - independent legality oracle over a real-decision corpus** (the core). `support/action_oracle.py` lists the legal
actions of a state in a canonical form (`("cast", card, variant)`, `("attack", perm)`, `("tap", source, colour)`, ...), written
from the rules, sharing nothing with `drl_env` / `plan_payment`:
timing (307.1 / 117.1a, speed), land drop, cast affordability (brute-force search over tap orders, not `plan_payment`),
alternative costs and zones, cycling, activations, pool spending, pass, attack / block eligibility, each pending kind's option
set. Compared both ways with the mask: *offered but illegal* (A), *legal but not offered* (B), *offered twice or two
actions on one index* (B, E). Corpus: random-legal self-play over **all 121 ordered deck pairs**, stratified by pending kind,
>=300k decisions (cached with the same `source_hash` key as the encoder corpus).

**L3 - execute every legal action** (soundness by trying). Fork at ~20k states stratified by pending kind; execute each legal
action on a fork. Checks: no exception; card conservation per player across zones; non-negative pools; no orphaned pending;
**one-step liveness** (the mask after the action is not all-False, unless the game ended); a post-condition table per
row family (`Cast X`: X left hand and the stack or pending payment grew; `Attack`: permanent is in `attackers`; `Play land`:
battlefield +1 and `lands_played` +1; ...); and **no-op detection** - every non-Pass action must change the observable
fingerprint (generalizes the declare-blockers property to the whole table). ~12 min.
*Fork-fidelity test*: deepcopy-fork vs replay-fork give identical fingerprints after the same action; the capture gate
fires on a planted capturing closure.

**L4 - try the illegal (discovery, not a gate).** For sampled masked-out rows, execute on a fork and classify (raises / no-op
/ mutates). A row that *progresses the game legitimately while masked out* is a suspected over-tight mask -> adjudicated
by the oracle and, if unclear, by the owner. Used to find oracle blind spots; findings graduate into L2/L7.

**L5 - mana and payment, exhaustive.** Grid of costs (generic/coloured/hybrid-free combos up to 6 pips, X, delve,
reductions from all 7 `cost_reduction` cards) x source sets (every mana card alone and in pairs/triples, incl. Wall of Roots,
Saruli Caretaker, filter lands, tron lands, Bloom Tender-style grants). Properties: `plan_payment` feasible <=> brute-force
feasible; **no strand** - DFS over *every* mask-legal tap sequence always completes the payment; the mask never offers a tap
that makes the cost unpayable; pool-spend legality; mana-burn bookkeeping.

**L6 - combat, exhaustive.** Random boards drawn from `CARD_DEFS` (extends `test_declare_blockers_no_op_property`): attack /
block eligibility vs the rules (summoning sickness, haste, defender, flying/reach, menace, can't-block effects, goad);
every legal declare sequence terminates; assign-combat-damage mask respects lethal caps; blocker<->attacker pointer mapping.

**L7 - per-card scenario matrix (coverage engine).** For each of the 141 cards x each applicable row family x contexts
(phase, own/other turn, empty/non-empty stack, 0/1/2+ copies, mana short / exact / ample per colour, targets present / absent
/ opponent-only, graveyard and exile states), build the state, ask the oracle, compare with the mask. Same for every pending
kind with generated option sets (0, 1, many; duplicates; four same-named slots; options on the opponent side only).
A **coverage gate** fails with the list of rows/kinds the corpus + matrix never made legal; the gate's allowlist holds only
rows that are dead by design (`Assign combat damage to opponent`) and needs owner sign-off.

**L8 - metamorphic.** Permute hand / battlefield / graveyard order -> pointer mask permutes, fixed mask identical. Relabel
slot numbers -> only slot rows change, consistently. Mirror the seats -> mirrored mask. Change the opponent's hand or either
library order -> *my* mask identical (no hidden-information leak through the mask). Add a non-interacting permanent -> unchanged.
Monotonicity: adding an untapped mana source never removes a cast option; removing a hand card removes only its rows.

**L9 - pointer bridge vs the legacy full table.** For every decision, the legal pointer set equals the legal set of the full
table's own-side targeting rows mapped by (side, name, slot); executing via pointer and via the row gives the same resulting
fingerprint. Token position <-> identity alignment is already proven by encoder L7; it is referenced, not repeated.

**L10 - liveness fuzz.** >=1M decisions with adversarial policies (uniform, first-legal, last-legal, always-pass,
never-pass-if-possible, attack-with-everything, never-block, max-cast) x all pairs x 0-land / 7-land stratified hands.
No all-False, no exception, no non-progress loop (a test-side priority-round watchdog, since the engine deliberately has no
cap), every game ends or hits the horizon. ~6 min.

**L11 - cross-path consistency (with `training/`).** The `full_mask` stored in the buffer equals a recomputation on the same
state; the padded mask used by PPO keeps legal positions and identities; the index sampled runs the closure at that index
(the `n_fixed` boundary); the forced-move shortcut (`n_legal == 1`) picks what the full path would; a masked position can
never be sampled.

**L12 - golden pins.** Keeps the table-order pin. Adds: per-family legal-action counts on ~40 fixed hand-built positions;
the registry -> row ledger from L0. Regeneration needs the owner's sign-off.

**L13 - expressiveness probes (efficacy).** Per deck, 3-5 scripted winning lines (Spy combo, Tron assembly, Bogles aura stack,
Madness discard loops, Elves mana chain, ...) driven *only* through the mask and table API. Each must be completable. This
proves the action space can express the lines that win the game, the thing B cannot show by enumeration alone.

**L14 - planted bugs** (`mutations.py`, run with `python tests/support/mutation_sweep.py action_space`). Initial list, ~45,
grown together with the owner:
timing: instant treated as sorcery / sorcery as instant / `active_idx` vs `turn_player_idx` swapped / Pass legal under a
pending / Pass illegal on an empty stack; land: `lands_played` check dropped; cast: recast guard `>=`->`>`, coloured pips ignored
in `plan_payment`, cost reduction applied to coloured pips, `_effective_cast_cost` ignores reduction (mask/execute mismatch),
flashback ignores graveyard, plot same-turn cast, delve max off by one, X button max off by one, mode button index shifted;
combat: summoning-sickness ignored, defender attacks, tapped blocker allowed, reach ignored, menace unenforced, lethal cap
off by one; pointer: battlefield-membership check removed (same-name collision), side swapped in opponent / any-target,
`id()` vs equality, `n_fixed` offset off by one, `_padded_full_mask` misplaces the pointer half, forced-move picks the wrong
index; mana: strand filter removed, sub-decision stage gate removed, pool spend ignores colour; hygiene: mask cache not reset,
mask mutates state (`_for_player` not restored), choose-name coverage check disabled; table: slot range off by one, two
rows swapped (pin), token slot pool size.
Each mutation must be caught by a fast subset (<2 min) so the full sweep stays ~1.5 h.

## 4. Triage protocol (rules-faithfulness mandate)

An oracle/mask disagreement is one of: oracle wrong, mask wrong, or **rules ambiguity**. I never settle a rules question
myself: each unresolved disagreement goes to the owner as (position, action, what the mask says, what I read the rule
as). An allowlist entry exists only with an `# OWNER-APPROVED` note and a date; a confirmed engine bug is fixed with a regression
test. Findings are logged in `FINDINGS.md` beside this file during the build.

## 5. Cost and order of work

Run time of the suite: ~35-40 min (corpus ~3 min cached; L3 ~12; L7 ~5; L5 ~3; L10 ~6; rest ~5) - under the 1-hour component
budget. Build order, each step shippable: **(1)** corpus + oracle skeleton + L0/L1 -> **(2)** L2 with the timing / land / cast
oracle, first mutation batch -> **(3)** fork + L3 -> **(4)** L5 mana + L6 combat -> **(5)** L7 matrix + coverage gate ->
**(6)** L8/L9/L11 -> **(7)** L10 fuzz, L13 probes, L12 pins -> **(8)** finish mutations, README.

## 6. Known limits (stated up front)

- The oracle is written by the same author as the engine's tests; it is independent *code*, not independent *understanding*.
  The triage protocol and L13 are the mitigations.
- Card *effect resolution* (what a spell does after it resolves) is checked only through post-conditions; that is the
  engine-invariants component's job.
- Mulligan decisions are owned by the MulliganNet and are out of scope here.
- Random-legal play is not the trained agent's distribution; L7 and L13 exist because of that.

## 7. Plan extension (2026-11-12): what the F-01 / F-02 / cost fixes add

Fixed by owner decision: Auras on the other player's land (F-01), choosing among same-named permanents (F-02), and per-ability
`taps` for activation costs. Testing now covers them (A4, and A2/A3 against the real-rules oracle with no mimic mode for them)
and the plan grows as follows.

| Item | Layer | What it checks |
|---|---|---|
| Copy-equivalence soundness | L8 | for random boards, two same-named permanents the engine calls interchangeable (equal `copy_signature`) give isomorphic positions whichever is tapped / activated; and every field that distinguishes copies (counters, flags, Auras on either side, tapped, sick, damage, temp effects) is in the signature (planted: drop a field -> caught) |
| Copy choice in the scenario matrix | L7 | every Aura (Utopia Sprawl, Abundant Growth) x who controls it x whose land x tapped/sick/untapped; every `taps` ability x {untapped, tapped, summoning-sick} x {1 copy, 2 interchangeable, 2 different} |
| `taps` declaration audit | A4 (done) + L7 | each cost_key ability's declaration is checked against what its resolve does; the matrix adds the no-cost_key abilities (Wellwisher, Timberwatch Elf, ...) the same way |
| Sub-decision stage coverage | A0/L9 | every `mana_subdecision` stage has a pointer mask, an executor and an `any_pointer_legal` entry (choose_target, choose_source); a new stage without them fails |
| F-03 (fixed) | A5b + A4 | affordability on a multi-symbol source with a granted ability: `EitherOr` units; `plan_payment` equals the oracle on 7,000 random boards with no mimic mode |
| Authorization ledger audit | A0 | every `AUTHORIZED SIMPLIFICATION` marker in `src/` has an entry in `action_oracle.AUTHORIZED` and vice versa |
| Beginning of combat | note | the engine has no such step; a goaded creature can be tapped by its own ability before attackers are declared. Real Magic allows that at beginning of combat, so no finding; listed so it is not rediscovered |
| Rules questions queued for the owner | - | none open |
