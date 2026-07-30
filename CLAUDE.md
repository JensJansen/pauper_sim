# Magic: the Gathering engine + RL training harness

## RULES FAITHFULNESS MANDATE (standing instruction — highest priority for game behavior)

Every game-rules decision in this engine MUST be **totally faithful to the real
rules of Magic: the Gathering**, unless the repo owner has **explicitly**
authorized a simplification for that specific case.

- **No silent simplifications, approximations, or "reasonable shortcuts"** for
  any game behavior: targeting, the stack, priority passing, timing/speed,
  costs, zones, combat, replacement/triggered/activated/static abilities,
  state-based actions, layers, copying, counters — all of it.
- **When in doubt, do not assume — ask.** If you are uncertain, have a
  question, or have any doubt about how a rule works or how a specific card
  should behave, stop and ask the owner. A confident guess about a rule is a
  bug, even when it happens to be right.
- **A simplification is legal only when explicitly authorized** by the owner
  for that case. When granted, mark it in the code with a comment recording
  that it is an explicit, owner-authorized deviation (e.g.
  `# AUTHORIZED SIMPLIFICATION: <what> — approved <when/why>`) so it is never
  mistaken for an assumption or drift.
- This mandate governs **game/engine behavior only**. Non-gameplay engineering
  (training harness, tooling, performance, action-space encoding) still follows
  normal judgment — but any place those touch *observable game outcomes* is
  game behavior and falls under this mandate.

Known deviations still being resolved are tracked as work items; the goal state
is zero unauthorized deviations.

## COMMENT & DOC PARITY (standing instruction)

After every code change, verify that the surrounding documentation still matches
the code — a comment or doc that describes the old behavior is a bug.

- **Local comments.** Review the comments on and around the lines you changed,
  including the enclosing function/class docstring. Update any that no longer
  describe what the code does.
- **Non-local but related comments.** Comments elsewhere that reference the
  changed behavior — callers, related modules, self-check assertions that
  narrate an invariant, cross-references by name — must be checked and brought
  back into parity too. If in doubt whether a comment is related, grep for the
  symbol/behavior you changed.
- **README.** After each code change, review the related sections of the README
  and update them to match. Do not leave the README describing superseded
  behavior.
