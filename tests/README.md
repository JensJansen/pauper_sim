# Tests

Three tiers, each a folder. Within a tier, folders are ordered by specificity (unit tests mirror `src/`; components are
named for the subsystem they stress).

| Tier | What it is | Where | Time (this machine) |
|---|---|---|---|
| **Unit** | One card, function or rule, with fabricated inputs. Deterministic. | `tests/unit/` — mirrors `src/` (`unit/game/catalog/…`, `unit/rl/model/…`) | **~2 min** (680 tests) |
| **Component** | A subsystem tested against its *contract* over real games. Independent oracles, invariance checks, coverage gates, golden pins, planted bugs. | `tests/component/<name>/` | **~6 min total**; each component well under its 1-hour budget |
| **Integration** | Components wired together: whole sessions, the pipeline scripts, rollout/update plumbing. | `tests/integration/` | ~3 min |

The tier marker (`unit` / `component` / `integration`) is applied automatically from the folder, so `pytest -m unit`
and `pytest tests/unit` are the same. There is no nightly tier: long training runs happen overnight, so the suite is
sized to run on demand.

```
pytest tests/unit                        # the push gate
pytest tests/component/encoder           # one component
pytest tests/component                   # every component
pytest tests/integration
pytest tests                             # everything (budget: well under 12 hours; today ~12 minutes)
python tests/support/mutation_sweep.py encoder        # planted bugs for one component (see below)
```

## Budgets and policies (agreed with the repo owner)
- **Any single component: under 1 hour. The whole suite: under 12 hours.**
- **Push gate:** `.githooks/pre-push` runs `tests/unit`; a failure blocks the push. (Lint still gates commits.) Enable once
  per clone with `git config core.hooksPath .githooks`.
- **Running while training is fine, when asked for.** Don't start the suites on your own while a training job is using
  the CPU; they slow each other down.
- **Golden pins need the owner's sign-off.** A pin (`golden.json`, `action_table_snapshot.json`, …) failing means either
  an accident or a deliberate change that invalidates something downstream. Regenerating one is a deliberate, reviewed
  act, never a way to make a red test green.
- **Planted-bug lists are maintained together.** Each component keeps `mutations.py`: deliberate bugs, planted one at a
  time in a scratch copy of `src/` (the real tree is never touched). A surviving mutation is a hole in that component's
  tests. When behavior changes, add the mutation that guards it.

## Shared infrastructure: `tests/support/`
`paths` (repo, src, data — never compute them from `__file__`, files move between tiers), `fixtures` (the
`cwd = src/` fixture many tests rely on), `corpus` (real self-play decisions recorded with an independent oracle),
`scenario` (hand-built two-player positions), `synth` (synthetic token batches), `mutation_sweep` (the harness).
Import with `from support import …`.

## What a component test suite looks like
Built to catch regressions in *methodology*, not just code. The encoder suite is the worked example; reuse its layers:

1. **Contract** — shapes, ranges, finiteness, invariants that must hold for every real input.
2. **Independent oracle** — recompute the expected output from raw engine state with separate code and compare.
3. **Metamorphic / invariance** — transformations that must not change the answer (reorder, pad, relabel, mirror) and ones that must.
4. **Sensitivity** — no dead inputs: every input that should matter measurably does.
5. **Identifiability** — states that differ in a decision-relevant fact encode/behave differently; ones that don't, don't.
6. **Cross-component equivalence** — the same quantity computed two ways (e.g. rollout vs update) agrees.
7. **Coverage gate** — the corpus must contain what the checks claim to cover; otherwise the test says what's missing.
8. **Golden pin** — a deliberate, reviewed snapshot against silent drift.
9. **Efficacy probes** — can a model learn the thing the component exists to support?
10. **Planted bugs** — prove the above would notice real mistakes.

## Components today
`encoder` (all ten layers), `action_space` (table order pin, mask legality, pointer bridge), `training` (rollout/update
equivalence), `validation` (the check modules end to end). Each has a README with its contract and known gaps.
Candidates next, in rough order of silent-failure risk: action-space mask vs engine, PPO/GAE and rewards,
league/opponent sampling, engine invariants over random self-play, policy network, replay engine.
