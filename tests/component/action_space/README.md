# Component: action space (`drl_env` + `rl/decision/action_bridge.py`)

**Contract.** The set of actions the policy can pick must be exactly the set of actions the engine allows: every legal
engine action is representable, nothing illegal is offered, and the fixed table's row order never silently changes
(a policy head has one logit per row, so reordering rows scrambles every saved checkpoint).

Covered today: the table's labels, order and content pinned to `action_table_snapshot.json` (a change invalidates every
checkpoint trained on the old table -- owner sign-off to regenerate); category-by-category legality of the table rows; the
declare-blockers anti-loop property; the fixed-table / pointer-head bridge and its legality masks.

**Cost.** ~1 minute. **Planted bugs.** None yet.

**Gaps (next work).** An oracle that tries every legal engine action over thousands of real decisions and checks it maps to
exactly one row/pointer; the converse (every offered action executes without error); the all-False-mask crash class.
