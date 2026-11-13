# Component: action space (`drl_env` + `rl/decision/action_bridge.py`)

**Contract.** The set of actions the policy can pick must be exactly the set of actions the engine allows: every legal
engine action is representable, nothing illegal is offered, no decision is ever left without an action, computing "what is
legal" changes nothing, and the fixed table's row order never silently changes (a policy head has one logit per row, so
reordering rows scrambles every saved checkpoint). The full plan, with the layers still to build, is in `PLAN.md`.

| File | What it establishes |
|---|---|
| `test_a0_structure.py` | static: unique labels; production table = full table minus exactly the pointer rows; every card's registry spec yields exactly the rows it should (derived independently of the table builder); every pending-resolution kind the engine opens has an answer path; the pointer dispatches agree |
| `test_a1_purity.py` | on live decisions: building / computing the mask (fixed + pointer half) never mutates the state (deep fingerprint incl. rng), is stable, equals the mask on a deepcopy, equals evaluating each row alone, and a bigger table agrees on every shared row |
| `test_a2_oracle.py` | ~360 games over every ordered deck pair: the mask vs an **independent rules oracle** (`support/action_oracle.py`) over pass, land, cast (all shapes), flashback/escape/plot, activate, cycle, impulse, mana taps, pool spend: unsound / incomplete / unrepresentable; plus the findings ledger |
| `test_a3_execute.py` | every legal action executed on a throw-away fork: no exception, never a no-op, no dead end after it, documented post-conditions, and the mask on the new position agrees with the oracle; the fork gate for closures that captured live objects (replay-fork fallback) |
| `test_a4_copies_and_costs.py` | the 2026-11-12 fixes: Auras on the other player's land (F-01); the player chooses among NON-interchangeable same-named permanents to tap / activate (F-02: `choose_source` mana sub-decision, `choose_permanent`); an activation's {T} is the per-ability registry key `taps`, cross-checked against what each ability's resolve actually does |
| `test_a5_mana.py` | `can_pay` vs brute force (exhaustive grid + random); `plan_payment` vs the oracle's per-card source model on ~7,000 random real boards; a depth-first search over EVERY sequence of mask-legal mana actions: no payment strands, every payable cost can be finished |
| `test_a6_combat.py` | attackers, blockers and damage vs the rules on random boards of real creatures (keywords granted at random): masks equal the rules; depth-first search reaches EXACTLY the legal attacker sets, EXACTLY the legal block declarations (menace, flying/reach, one block per creature), and EXACTLY the allowed damage splits |
| `test_action_table_order.py` | pins labels and row structure of every table (a change invalidates every checkpoint; owner sign-off to regenerate) |
| `test_action_table.py`, `test_action_bridge.py`, `test_declare_blockers_no_op_property.py` | hand-built regressions: mana sub-decisions, strand cases, pointer masks, blocking |

**Findings.** The oracle follows the REAL rules; where the engine differs and nothing authorizes it, the disagreement is
attributed to a named finding (mimic modes in `support/action_oracle.py`).
- Resolved 2026-11-12 (owner decision): **F-01** cross-controller Auras; **F-02** same-name copies (the action space could
  not choose which Forest to tap / which Kill-Ship to activate); **cost** every `cost_key` ability required an untapped copy
  even without {T} in its cost (Clue, Reckless Lackey, Relic's draw, Synthesizer, Moxite, Sewer-veillance Cam).
- Resolved 2026-11-12 (owner decision), also: **F-03** a granted ability on a source that makes several symbols (Abundant
  Growth on Rakdos Carnarium, Tron lands, Battlement/Priest with 2+ matches) was dropped when judging affordability. Such a
  source is now an `EitherOr` supply unit (`game/mana.py`): `can_pay` tries each way, exactly. No finding is open.

Authorized deviations the oracle follows are listed in `support/action_oracle.AUTHORIZED` (speculative floating only in your
own main phase; Saruli excluded from supply; the strand rule, owner-accepted 2026-11-12; no-undo blocking).

**Observed, not a finding:** the engine has no beginning-of-combat step, so a goaded creature can be tapped by its own
ability before attackers are declared (real Magic allows that in the beginning-of-combat step).

**Cost.** ~6 min (corpus ~1 min, cached under `tests/.cache/`, key = all of `src/` + decklists + corpus/oracle code;
A1 ~35 s; A3 ~1 min; A5 see below; A6 ~20 s). Knobs: `ACTION_CORPUS_GAMES`, `ACTION_CORPUS_NOCACHE`, `ACTION_PURITY_EVERY`,
`ACTION_FORK_EVERY`.

**Planted bugs.** `mutations.py` (`python tests/support/mutation_sweep.py action_space`). Entries marked KNOWN-GAP are bugs in
families the oracle does not judge yet and are expected to survive.

**Not built yet (see `PLAN.md`).** The per-card scenario matrix with a coverage gate (L7: includes the impulse-deadline
mutation); decision-button and "Choose: X" families in the oracle; metamorphic invariance (L8); pointer bridge vs the legacy
full table (L9); adversarial-policy liveness fuzz (L10); rollout/update mask consistency (L11); scripted winning lines (L13);
a test that every `AUTHORIZED SIMPLIFICATION` marker in `src/` has an entry in the oracle's ledger.
