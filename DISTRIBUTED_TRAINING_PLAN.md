# Distributed (WAN node-ownership) training — design plan

Status: **Phase 1 built and green, 2026-09-06** (branch
`distributed-training-phase-1`) — `src/distributed/` now has
`auth.py`/`hub_ownership.py`/`hub_store.py`/`version_gate.py`/`hub_server.py`,
38 new tests all passing, the full existing suite (937 tests,
`-m "not slow"`) still passing unmodified. This document is still the
design/decisions record; it lives outside `src/` on purpose (repo-root,
alongside `TODO.md` and `DRL_ANALYSIS_CLEANUP.md`).

**This is the design/decisions record — for the ordered, file-by-file build
sequence, see `DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md` (2026-09-06).**
That document resolves this one's remaining implementation-shaped gaps
(5/6/11/13/14/15) by scheduling each into the specific phase where it
actually gets built or measured, rather than leaving them as bare list
items here.

## Non-negotiables (from the owner)

1. **Does not disturb today's training.** Everything below is additive:
   `run_training_pipeline.py`, `run_league.py`, `league_runner.py`,
   `league.py`, `train.py`, `rollout_parallel.py`, `checkpoint.py` are not
   modified. The single-machine path keeps working exactly as it does today,
   indefinitely, until a deliberate later decision retires it.
2. **Lives in a separate subsection of the repo.** All new code under
   `src/distributed/` (new package), new configs under `distributed_configs/`
   (new directory), one new top-level doc (this file). Nothing existing is
   restructured to make room for it.
3. **Long-term direction, not immediate**: eventually the WAN system
   replaces local training. Not attempted now — every phase below ends in a
   working, disposable state.

## Confirmed assumptions (owner answers, 2026-09-01 – 2026-09-05)

- **Trust**: some nodes are third-party, but treated as non-adversarial for
  now. Real per-node auth is still in scope (cheap, and the "non-adversarial"
  assumption is explicitly named as provisional, not a permanent design
  constraint) — but sandboxing/output-verification against a hostile node is
  out of scope.
- **Code distribution**: each node runs its own `git clone`/`pull`; the hub
  never pushes code. Consequence: the hub must actively *gate* on version
  compatibility rather than assume it — mechanism **deferred pending the
  hosting-platform decision**, see "Version/compatibility gate" (which also
  flags an open question about how broadly that deferral should apply).
- **Hardware**: assume CPU-only nodes for now; keep the existing `device`
  config knob (already threaded through `_run_session`/PPO_DEFAULTS) exposed
  per-node so a node can opt into `"cuda"` later with no code change.
- **Uptime**: nodes are expected to start and stop *together*, as a batch
  ("training running" vs. "not running"), not churn independently. This
  removes the need for lease/heartbeat-driven auto-reassignment in v1 —
  reassignment is a manual, deliberate operator action. **Updated
  2026-09-04** to match the dynamic-registration design (see "Node
  lifecycle"): reassignment now means starting a replacement node, which
  registers for the deck and naturally invalidates the previous claimant's
  token — not editing a deck→node_id mapping, since that mapping no longer
  exists once identity moved to registration time. **Further updated
  2026-09-05**: liveness is no longer purely passive — the elastic/bounded
  drift design (see "Node lifecycle") actively excludes a node from the
  drift calculation once it's gone quiet longer than a liveness timeout, so
  a dead node can't become a permanent anchor everyone else waits on. Manual
  reassignment is still the only *ownership* action taken automatically —
  liveness now feeds a real mechanism, but that mechanism is pausing/
  resuming other nodes, not reassigning the quiet one's deck.
- **Node synchronization model (owner decision, 2026-09-05)**: neither a
  hard lockstep barrier (rejected — throughput capped at the slowest node,
  every iteration, forever, plus one dead node stalls the whole fleet) nor
  no bound at all — **elastic/bounded drift**. Nodes progress independently
  at their own pace; the hub only pauses a node once it's drifted too far
  ahead of the slowest *active* node. Full design under "Node lifecycle."
- **Ownership granularity (owner decision, 2026-09-02)**: exactly **one deck
  per node**, axiomatically — not "up to N decks, rebalanced later." A node
  is provisioned for a single named deck. This is now a hard assumption, not
  a default, and it removes most of the multi-deck-ownership complexity
  discussed below (see "Design review" for what it resolves).
- **Networking (owner decision, 2026-09-05)**: hub↔node reachability is via
  **Tailscale** — every node gets access through a Tailscale allowance,
  reaching the hub over the tailnet rather than the open internet. Resolves
  Design review gap #10 (WAN reachability) outright: no port forwarding,
  dynamic DNS, or public IP exposure on the hub needed. Does **not** resolve
  the version gate's still-deferred node-lifecycle question (what happens to
  a rejected node) — that depends on which platform hosts node *compute*
  (e.g. OVHcloud), a separate decision from this networking layer. Also
  meaningfully changes gap #9's calculus (see its resolution below) — the
  hub is never exposed to untrusted internet traffic, only to devices
  already in the tailnet.
- **Hub role (owner decision, 2026-09-05)**: the hub machine **never
  trains** — all training happens on nodes. Resolves Design review gap #12
  outright.
- **Data durability (owner decision, 2026-09-05)**: zero information loss is
  the target — nothing produced by a node stays localized except
  transiently, during in-flight work. Every category of data a node
  produces (checkpoints, `metrics.jsonl`, progress) pushes to the hub on the
  same regular sync cadence, not just weights — see Design review #7. Stated
  precisely so this isn't accidentally overclaimed: zero loss of anything
  already completed/written locally; *bounded*, not literally zero, loss of
  whatever's mid-flight in the currently in-flight `_run_session` call if a
  node dies at the wrong instant — a periodic-sync design can shrink that
  window (via call granularity, gap #14) but can't eliminate it without
  becoming a fully synchronous, per-event-networked system, which isn't
  what's being built here.
- **Config handling (owner decision, 2026-09-02)**: a node treats its
  training config file as **read-only** — never writes to it, never
  auto-selects a per-node override. The operator is solely responsible for
  each node's config (including infra-only values like `device`) being
  correct before that node launches; the code performs **no** config-drift
  detection or enforcement for training-regimen values. This is a distinct
  decision from the engine-code version gate below, which has **not** been
  waived — see "Version/compatibility gate."

## Architecture in one paragraph

The **hub** (this machine) is not a learner anymore — under node ownership,
each node runs its own local PPO update for the one deck it owns, exactly
the way `_run_session` already does today. The hub's job shrinks to: hold
the canonical merged `checkpoints/<league>/` tree, mint deck assignments and
scoped tokens at registration (see "Node lifecycle" below), enforce who is
allowed to write which deck's files, gate on code-version compatibility
(mechanism **deferred**, see "Version/compatibility gate"), and serve the
current state of the whole roster to any node that asks. A **node** is a
thin process that calls the *existing, unmodified* `_run_session` (not
`_train_to` — see "Node lifecycle" for why) against its own local checkpoint
directory, with `train_decks` set to its one owned deck and `roster` left at
the full manifest — driving its own small loop around that call so it can
sync with the hub and check for a stop request between calls, rather than
running one long, uninterruptible session.

## Evidence this fits with (almost) no changes to existing code

- `_run_session`'s `roster`/`train_decks` split already does the load-everyone-
  train-a-subset thing this needs: `roster` restricts the whole opponent pool
  (which nets get built at all), `train_decks` restricts which of those
  receive PPO updates and get new snapshots registered
  ([league_runner.py:239-262](src/rl/league/league_runner.py#L239-L262),
  `for name in deck_names:` at
  [league_runner.py:268-296](src/rl/league/league_runner.py#L268-L296) builds
  live nets for the *whole* roster regardless of `train_decks`). A node calls
  this with `roster=None` (full 11-deck roster, so it has real opponents to
  play/mirror against) and `train_decks=[its one owned deck]` — no new
  parameter needed.
- **Correction (2026-09-04): the `on_chunk` claim above's earlier version was
  wrong, caught during this review, not before.** `_train_to`'s `on_chunk`
  fires once per `chunk_games`-sized chunk, which defaults to
  `opts["cadence_games"]` — for `main_league.json` (extends
  `baseline_settings.json`'s `checks_cadence_pct: 10`, `total_games:
  100000`) that's `round(100000 * 0.10) = 10,000` games/deck, i.e. `on_chunk`
  fires roughly every **166 iterations**
  ([run_training_pipeline.py:119-155](src/run_training_pipeline.py#L119-L155)) —
  not every `snapshot_every_games` (200). Those are two genuinely different
  cadences that happened to get conflated in an earlier version of this
  document and in conversation. `_train_to` also has no hook for `on_chunk`
  to signal "stop, don't run the next chunk" — its only loop exit is
  `remaining <= 0` (the total target reached). Both problems are solved the
  same way: **the node agent does not call `_train_to` at all.** It calls
  `_run_session` directly, in its own small loop, sized independently of
  either of `_train_to`'s two cadences. `_run_session` itself is untouched
  either way — see "Node lifecycle" below for the actual design.
- `_run_session` unconditionally calls `_save_live_checkpoints` and
  `pool.save_opponent_stats()` at the end of *every* call, not only when a
  snapshot cadence boundary was crossed during that call
  ([league_runner.py:578-582](src/rl/league/league_runner.py#L578-L582)) —
  verified by reading the function's tail, not assumed. This is what makes
  it safe for the node agent to call `_run_session` in small increments and
  push current, durable state after each one.
- Workers (`_league_rollout_worker`) are already pure, picklable-in/
  picklable-out functions with no assumption baked in about running on the
  same machine as the caller
  ([rollout_parallel.py:59-137](src/rl/training/rollout_parallel.py#L59-L137))
  — a node's own local `ProcessPoolExecutor` usage is untouched; distribution
  happens one level up, at the hub/node boundary, not inside rollout
  collection.
- `LeaguePool`'s opponent stats are already sharded correctly for this by
  construction: `record_outcome` is only ever called by the process training
  that deck ([league_runner.py:461-466](src/rl/league/league_runner.py#L461-L466)),
  so under static per-node ownership there is never a write conflict on
  `opponent_stats[deck]` — exactly one node ever writes it, no merge logic
  needed.
- **[CORRECTED 2026-10-17: this bullet's conclusion was wrong -- `vocab.json` IS now synced; see below.]** `vocab.json` was believed not to need syncing at all. `CardVocab.__init__`
  computes `new_names = sorted(needed_names - set(existing))` purely from the
  roster's decklists + fixed token defs
  ([features.py:124-151](src/rl/model/features.py#L124-L151)) — deterministic
  and order-independent given the same manifest, so every node regenerates a
  byte-identical `vocab.json` from a clean checkout on its own. This *is*
  contingent on every node calling `build_pool()` with the full manifest
  (never a per-node deck subset) — called out explicitly as an invariant
  below, not assumed silently.
  **Correction (2026-10-17, review issue 12):** `checkpoints/vocab.json` is
  *append-only persisted history* (`CardVocab` never reassigns an index), not a
  pure function of the manifest -- a new card is appended at the end on the hub
  but placed in sorted position by a clean node, and removed cards stay on the
  hub. The two matched on the day this was checked only because the one extra
  name sorted last. A divergence silently corrupts any net (embedding rows read
  as the wrong cards). Resolution: the hub serves its own file (`GET /vocab`,
  registration-credential protected) and a node writes it locally before
  `build_pool`; the vocab hash in the version gate remains as verification.
- Real file sizes, measured against the currently-running (now stopped)
  `main-league` checkpoint tree: `live.pt` ~3.9MB, `mulligan.pt` ~164KB,
  one active `snapshot_N.pt` ~1.36MB, `opponent_stats.json` ~11KB, 32 active
  snapshots/deck (the `DEFAULT_MAX_SNAPSHOTS_PER_DECK` cap). That's
  **~47.6MB of state per deck** a node actually needs — × 11 decks ≈
  **~524MB** for a node's full-roster mirror. Trivial over WAN, even more so
  incrementally.
- **`archive/` must be excluded from the sync surface.** Same measurement:
  `boggles/archive/` alone is already 624MB (481 evicted snapshots, growing
  forever — `register_snapshot` moves evicted snapshots there and never
  deletes, [league.py:132-140](src/rl/league/league.py#L132-L140)) against
  the whole `main-league` dir's 15GB. `archive/` is read only by
  `_run_eval_vs_history`/`load_vintage_agent` for retrospective analysis, and
  never by `LeaguePool`'s own sampling (`__init__` only lists the live
  `deck_dir`, [league.py:68-76](src/rl/league/league.py#L68-L76)). Syncing it
  would make node bootstrap balloon without end and buys nothing for
  training. `checks/` (validation output) is excluded for the same
  reason — never read by anything in the training path.

## What the distributed design actually changes about training behavior

Everything about PFSP math, checkpoint-rate sampling, snapshot cadence, and
PPO/mulligan update math is unchanged **inside** a node — it's the same code
running the same way. The one real behavioral difference: a "live" opponent
for a deck a node doesn't own is no longer instantaneously live — it's that
deck's state as of the node's last hub sync. This is bounded, quantifiable
staleness — bounded by the node agent's own call granularity (see "Node
lifecycle" below), targeted at roughly the same `snapshot_every_games`=200
order of magnitude the single-machine run already treats as a meaningful
cadence, though the exact figure is an empirical tuning question, not yet
measured. **Not** bounded by `_train_to`'s `cadence_games` (~10,000 games) —
an earlier version of this section cited that mechanism, which was wrong;
see the correction under "Evidence" above. Not a new failure mode either
way, just worth being explicit that a "live" mirror-match under this design
is closer in spirit to a very-fresh checkpoint match than a true
same-process live match. Flagging this so it's a conscious tradeoff, not a
surprise later.

## Within-node CPU parallelism (2026-09-03) — already covered, was just unstated

"How many decks does a node train" and "how many cores does a node use" are
two independent axes. The second one is already fully solved by existing
code, unrelated to anything in this document so far: `_run_session` already
takes `executor`/`n_workers` and splits each iteration's games across a
local `ProcessPoolExecutor`
([league_runner.py:215-220](src/rl/league/league_runner.py#L215-L220),
[rollout_parallel.py:140-189](src/rl/training/rollout_parallel.py#L140-L189))
— exactly the mechanism `run_training_pipeline.py` already uses today to
saturate this machine's cores, nothing to do with deck count. Each worker
process forces itself to one CPU thread
([rollout_parallel.py:81](src/rl/training/rollout_parallel.py#L81)) so
workers don't contend with each other.

Since `node_agent.py` calls `_run_session` directly (see "Node lifecycle"
below for why not via `_train_to`) and `_run_session` itself is unmodified,
a node training its one deck inherits multi-core use for free — it just
needs its own
`n_workers` set to roughly its vCPU count (minus a little headroom for the
main process's own PPO-update/orchestration work), via the same
`n_workers` field `training_configs/*.json` already has today.

**Consequence for the read-only-config decision above**: `n_workers` (and
`device`) are the two fields *expected* to differ per node, sized to that
node's actual hardware — unlike `checkpoint_rate`/`pfsp_power`/etc., which
must be identical everywhere. Not a contradiction of "the operator manages
config correctness by hand" — just worth being explicit that this one field
is supposed to vary, not a drift risk to watch for.

## Node lifecycle (2026-09-04 – 2026-09-05): registration, graceful stop, call granularity, drift bounding

Four related design decisions made in conversation, recorded together since
they interact.

### Registration and dynamic tokens (replaces the earlier static-per-node-token idea)

A per-node, permanently-assigned bearer token was the original design. Owner
pushback, on reflection correct: that would make every node's provisioning
image differ (needs its own pre-baked token), which is exactly the
injection complexity being avoided. But a single token shared by *every*
node was considered and rejected — it would remove the hub's ability to
tell "a legitimate node" apart from "a node authorized to write *this*
deck," which is the specific mechanism that turns an operator mistake into
a loud rejection instead of silent cross-contamination (see Security
posture below). Resolution: **neither fixed option — tokens are minted
dynamically at registration.**

- Every node image is identical: baked in are only the hub's address and a
  shared **registration credential** (not a write-token — good for exactly
  one thing, registering).
- At boot, a node registers with that shared credential. The hub picks the
  next unclaimed deck from the ownership pool (see the revised config
  surface below) and mints a **fresh, deck-scoped token** on the spot,
  returning both the assignment and the token in one response.
- **A new registration for an already-claimed deck invalidates whatever
  token was issued for the previous claim.** This is a strictly stronger
  version of the "hub re-reads current ownership on every push" rule
  already settled in the Design review below — instead of only catching a
  stale writer the next time it happens to push, a fresh claim immediately
  revokes the old one.
- Named risk, not hidden: the shared registration credential is only safe
  under the current "non-adversarial for now" trust assumption — anyone
  holding it could claim an unclaimed deck. Same caveat that already exists
  for the trust model generally (see "Confirmed assumptions"), not a new
  hole. If the trust model tightens later, the fix is the same shape as
  today's: distribute a genuinely unique credential per (now-trusted) node
  operator out of band, just moved to provisioning time instead of being a
  permanent per-node secret.

### The node agent's own loop, not `_train_to`

Per the correction above: the node agent calls `_run_session` directly,
implementing its own loop around it (new code, `src/distributed/`, calling
`rl.league.league_runner._run_session`/`advance_progress`/`_load_progress`
exactly as `run_training_pipeline.py` already does — none of those need to
change). Each iteration of that loop: read current progress, call
`_run_session` for some small number of iterations, sync the result with
the hub (push this deck's state, pull any deck this node doesn't own that
changed), then check whether to keep going.

**Call granularity is an open, empirically-tunable parameter, not asserted
here.** Smaller `n_iterations` per `_run_session` call means fresher hub
syncs and a more responsive graceful stop; it also means paying
`_run_session`'s own setup cost more often — rebuilding all 11 decks' nets
from disk and re-scanning `LeaguePool`'s snapshot directories happens fresh
on *every* call, nothing is cached across calls. How expensive that
overhead actually is has not been measured. Proposed default to start
benchmarking from: roughly `snapshot_every` iterations (3, for
`main_league.json`'s current settings) — not because it's provably optimal,
but because it's a cadence the codebase's own authors already picked
deliberately, rather than inventing a new number. **This needs a real
benchmark in Phase 2** (the loopback smoke test already planned there is
the natural place to measure it) before treating any specific value as
settled.

### Graceful stop

"Finish this batch, report your data, terminate" — mapped onto the actual
loop above: finish the `_run_session` call already in flight (can't
interrupt a call mid-way without touching `_run_session`'s internals, which
stays off-limits), do the same push it would do anyway, then don't start
another call.

The stop signal itself: a flag the hub can set (per-deck, or for every deck
at once, for a full-batch teardown), discovered **as part of the sync call
the node is already making** — no new inbound channel to the node, which
preserves the outbound-only networking property noted in
`OVHCLOUD_SETUP_NOTES.md`'s private-network-plus-Gateway option. A local
`SIGTERM` handler sets the same flag as a second, independent trigger, for
when infrastructure kills a specific node directly rather than going
through the hub. Either way, the final push on the way out is a **normal**
push of current live/optimizer/progress state — explicitly *not* a forced
extra snapshot registration, since that would perturb the snapshot
window's spacing away from the cadence it was deliberately tuned to,
for a stop event that has nothing to do with that tuning.

### Elastic/bounded drift (2026-09-05)

The owner's original proposal was a hard lockstep barrier: every node
finishes an iteration, reports in, then polls every 5-10 seconds until
every *other* node has also finished, before any of them proceed. Pushed
back on directly — this caps the whole fleet's throughput at its single
slowest node, on every iteration, forever (different decks genuinely
simulate at different speeds, and nodes may have different hardware — both
already-accepted sources of variation elsewhere in this document), and a
crashed or slow node stalls every other node too, a fragility that doesn't
exist in the current independent-progress design. Resolution: **elastic,
bounded drift instead** — nodes keep progressing independently at full
speed by default; the hub only intervenes once a node has drifted too far
ahead of the rest.

This reuses almost everything already designed rather than inventing new
machinery:

- Every node already reports its current `cumulative_games_per_deck` as
  part of its regular sync (it's in `progress.json`, already pushed). The
  hub now *uses* that existing data for a new purpose: tracking the spread
  across all currently active nodes. No new data needed from nodes.
- If a node's own progress gets more than a threshold `K` (unasserted, see
  Design review #15) ahead of the **slowest active node**, the hub's
  response to that node's next sync carries a **pause** signal — discovered
  the same way the existing stop signal is (as part of the sync response,
  no new inbound channel to the node). A paused node finishes whatever
  `_run_session` call is already in flight (same rule as graceful stop),
  then — instead of exiting — polls the hub at a short interval until told
  to resume. This is where the owner's original 5-10-second-ping idea
  actually fits well: cheap, and only happening while a node is genuinely
  idle waiting, not as the default behavior for every node on every
  iteration.
- **A dead node can't be allowed to become a permanent anchor.** If a
  crashed node's last-reported progress were treated as the floor forever,
  every other node would eventually pause waiting for one that's never
  coming back. A node stops counting as "active" for this calculation once
  it's gone quiet longer than a liveness timeout (unasserted, see Design
  review #15) — at which point it's excluded from the "slowest" comparison,
  and flagged for operator attention rather than silently starving the
  fleet.
- Drift is measured in `cumulative_games_per_deck` — the same unit
  everything else in this design already keys on (hyperparameter schedules,
  snapshot cadence, PFSP) — not wall-clock time, which would add clock-skew
  concerns across machines for no real benefit here.
- Unlike a hard barrier, this does **not** require every node's call
  granularity (iterations per `_run_session` call, gap #14) to be
  identical — drift is measured in games, not iteration-boundary alignment,
  so per-node tunability there is preserved.
- Side effect worth naming honestly: since the hub now tracks every active
  node's live progress for this purpose anyway, a whole-league status view
  (gap #7) becomes nearly free to build later — the data it would need is
  already being collected. This doesn't solve #7 on its own (nothing
  displays it yet), but meaningfully lowers what #7 would cost to build.

### A cost this makes sharper, not new: mass-simultaneous-bootstrap bandwidth

Already known: a fresh node needs the full-roster mirror (~524MB, measured
earlier against the real `main-league` tree) before it can play a
meaningful game, since it needs opponent nets for the 10 decks it doesn't
own. That was priced in as a one-time cost per node's lifetime. It stops
being one-time the moment destroy-and-recreate becomes routine practice,
which is exactly the direction this conversation has been heading — and if
all 11 nodes are recreated together (matching the "spin up/torn down as a
batch" uptime model), that's ~5.7GB of hub egress at once, from a machine
that's plausibly on a residential connection, not a datacenter uplink.
**Not yet resolved — added to the open list below**, not a blocker, but
real upload-bandwidth numbers for the actual hub machine should be checked
before leaning on "destroy everything every batch" as the default pattern.

## Design review (2026-09-02 – 2026-09-04): gaps found, and resolutions

A careful pass over this plan (not just the summary above — checked against
actual code, not memory) surfaced 13 gaps in the first pass (2026-09-02) and
2 more in a follow-up pass (2026-09-04), 15 total. Four were structural
enough to need an owner decision before anything gets built; the rest are
real but don't block starting. Recorded here so a decision made once in
conversation doesn't quietly evaporate.

**Status as of 2026-09-06**: 12 of 15 numbered items are now resolved (1-4
from the original pass, 7/9/10/12 on 2026-09-05, 5/6/8/14 on 2026-09-06 as
Phase 1/2 were actually built — 14 via a real benchmark). Remaining: 11
needs development (low priority, an accepted inefficiency, not scheduled
urgently), 13/15 need benchmarking against a real second machine, not yet
possible on this machine alone. Each open item below is tagged
**[Owner decision]**, **[Needs development]**, or **[Needs benchmarking]**
so it's easy to filter to what actually needs input right now versus what's
implementation work for later.

**Numbering note**: items are numbered 1-14 below, not 1-15 — an earlier
edit folded old item 7's content into item 2 and removed its now-empty stub
without renumbering the rest of the list, leaving a gap between 6 and 8
uncaught until this pass. Fixed now: the list below is consecutive.

### Resolved

1. **`progress.json`'s single value applies to every deck a node trains at
   once** ([league_runner.py:110-125](src/rl/league/league_runner.py#L110-L125),
   [league_runner.py:406-412](src/rl/league/league_runner.py#L406-L412),
   [league_runner.py:545-546](src/rl/league/league_runner.py#L545-L546)) —
   correct only when everything in `train_decks` shares one history. Under
   the old "a node may own several decks, rebalanced later" model this broke
   the instant a node's owned decks had different histories (near-certain
   given multi-deck ownership was the expected norm). **Resolved by the
   one-deck-per-node axiom above**: `train_decks` is always a singleton set
   on every node, so this dissolves rather than needing a code change to
   `league_runner.py`. The one remaining requirement: a node's first sync for
   its deck must seed its local `progress.json` from the hub's current value
   for that deck (see #2 — same mechanism, one more file in the same pull).
2. **Loading pre-existing trained leagues, not just fresh ones.** The hub's
   canonical store *is* whatever's already in `checkpoints/<league>/` on
   this machine — including a league's real prior history (`main-league`
   already has 481 snapshots/deck on disk as of this review). A node's first
   sync always pulls whatever the hub currently has for its assigned deck;
   if the hub has nothing yet (a genuinely new league), the hub seeds one
   canonical fresh initialization first and the node pulls that. Same code
   path either way — "start fresh" and "resume existing" aren't a branch.
   **This same pull-before-training rule also resolves node disk-loss
   recovery**: a node whose local disk is lost/replaced just re-pulls its
   own deck from the hub on restart, same mechanism, no special case.
3. **Config-drift enforcement is explicitly not wanted.** A node reads its
   training config read-only; the operator manages per-node config
   correctness (including `device`) by hand before deployment. No
   node-local-override layer, no split config files, no automated drift
   detection for training-regimen values. Accepted tradeoff, stated
   plainly: if a node's config *does* drift (stale edit, wrong file copied),
   nothing will catch it — training on that deck silently diverges from its
   siblings with no error, discoverable only by noticing odd results later.
4. **Ownership-check freshness**: the hub re-reads the *current* ownership
   registration on every push, never a value cached from an earlier
   registration — closes the double-writer risk immediately on a reassignment
   rather than only after a stale node happens to restart. No real tradeoff
   here, just needed to be pinned down explicitly rather than left implicit.

### Deferred (not forgotten)

- **Engine code-version gate** (a node's `git rev-parse HEAD` + a vocab hash,
  checked at registration/sync, hub rejects on mismatch): distinct from the
  config-drift question above — this catches a node running *stale rules*
  (a card-behavior fix committed on the hub that a node hasn't pulled yet),
  which this repo's standing rules-faithfulness mandate treats as invalid
  training data, not a version-skew inconvenience. **Deferred until the
  hosting platform is chosen** (owner, 2026-09-02) — what's practical here
  depends on what control the platform gives over node start/stop (e.g. can
  the hub force a node to re-pull or restart on a rejection, or does a
  rejected node just sit there). **Update 2026-09-05**: gap #10 (WAN
  reachability), which this was originally tied to, is now resolved via
  Tailscale — but that only answers *routing* (how a node reaches the hub),
  not this deferral's actual question (what control exists over a node's
  *compute*, e.g. can it be forced to stop/restart). That still depends on
  the node-hosting platform (e.g. OVHcloud), a separate decision, not
  resolved by Tailscale. **Confirmed 2026-09-02: the deferral covers the
  whole gate**, not just the node-lifecycle piece — `version_gate.py` is
  not part of Phase 1.

### Still open, unresolved, not yet discussed with the owner

5. **[RESOLVED, 2026-09-06]** No failure isolation around the hub-sync
   call. Built in `node_sync.py`'s `_with_retry`: exponential backoff on a
   transient connection failure, capped at a bounded number of attempts,
   never retrying a real HTTP error response (401/403/409 are deliberate
   rejections, not transient — retrying can't fix them). Tested against
   both a genuinely unreachable host and a real 401, confirming retry only
   fires for the former.
6. **[RESOLVED, 2026-09-06]** Push atomicity across a bundle of files.
   Built in `hub_store.py`'s `push_deck_files`: every file writes to a
   `.tmp` sibling first; only once *all* writes succeed does it swap every
   temp file into place. A failure partway through leaves every temp file
   cleaned up and nothing already on disk touched. Honest residual gap,
   stated in the code's own docstring, not hidden: this is not a true
   cross-file transaction — a crash between two `os.replace` calls in the
   final swap loop could still leave a batch partially applied. Accepted
   rather than solved with a transaction log neither this scope nor
   Windows calls for.
7. **[RESOLVED, 2026-09-05]** No observability/aggregation design, split
   into two genuinely different things that got conflated in the original
   framing:
   - **Whole-league progress + quality view.** The hub tracks every active
     node's `cumulative_games_per_deck` (for elastic/bounded drift) and
     runs validation/round-robin checks itself (see Open questions #1)
     against its own synced copy.
   - **Raw per-iteration PPO training-dynamics metrics** (policy_loss,
     entropy, approx_kl, clip_fraction — what `report_metrics.py` reads
     from `metrics.jsonl`). **Resolved via the data-durability decision
     above**: nodes tail-forward new `metrics.jsonl` lines to the hub as
     part of their regular push, same cadence as the checkpoint files —
     not a separate mechanism, just one more thing in the same payload.
     `report_metrics.py` pointed at the hub's checkpoint tree now sees the
     real, whole-league picture (as fresh as the last sync from each node,
     same staleness bound as everything else).
8. **[RESOLVED, 2026-09-06]** `run_rollback.py`'s interaction with sync.
   **Resolution**: a rollback pushes to the hub immediately, not on the next
   scheduled sync. Consistent with the zero-information-loss principle —
   requiring the operator to separately remember to trigger or wait for a
   sync after a deliberate fix would be a footgun (the fleet stays on the
   bad state for up to one sync interval after the operator already thinks
   they fixed it). Rollback is rare and manual, so the extra push costs
   nothing meaningful.
9. **[RESOLVED, 2026-09-05]** Flask's dev server isn't a production server,
   and gunicorn doesn't run natively on Windows. **Resolution**: the Tailscale
   decision above changes the calculus — the dev server's "not for
   production" warning is mainly about exposure to arbitrary, untrusted
   internet traffic, and the hub is now never reachable from the open
   internet at all, only from devices already in the tailnet. Combined with
   the traffic pattern already being low-QPS (periodic node syncs, not
   heavy load), that removes the main justification for a hardened WSGI
   server. Decision: use Flask's built-in server as-is; revisit only if the
   traffic pattern changes materially.
10. **[RESOLVED, 2026-09-05]** WAN reachability. **Resolution**: Tailscale
    (see "Confirmed assumptions") — nodes reach the hub over the tailnet, no
    port forwarding/dynamic DNS/public IP needed. Still leaves node
    *provisioning* needing a Tailscale auth step at boot (alongside the
    registration credential already designed in "Node lifecycle") — that's
    implementation work for Phase 3, not an open decision.
11. **[Needs development]** Push/pull payload asymmetry: pulling another
    deck's state currently means the full `live.pt` (weights + Adam
    optimizer, ~3.9MB) when an opponent-only consumer never needs the
    optimizer half. Not a correctness bug, just an uncounted-for doubling of
    the bandwidth estimate for the common case.
12. **[RESOLVED, 2026-09-05]** Whether the hub machine also owns/trains a
    deck itself. **Resolution**: no — the hub never trains, all training
    happens on nodes (see "Confirmed assumptions"). No resource contention
    between `hub_server.py` and a local node-agent process, since there is
    no local node-agent process.
13. **[Needs benchmarking]** Mass-simultaneous-bootstrap bandwidth
    (2026-09-04): a full-roster mirror pull is ~524MB per node; if
    destroy-and-recreate becomes routine and all nodes restart together,
    that's ~5.7GB of hub egress at once. See "Node lifecycle" above. Not
    resolved by Tailscale — Tailscale changes routing, not the hub's
    underlying physical upload bandwidth. Needs real numbers for the hub
    machine's actual connection before this is a real problem or a
    non-issue.
14. **[RESOLVED, 2026-09-06]** `_run_session`'s per-call setup cost.
    **Measured** via `benchmarking/distributed_call_granularity.py`
    (5 calls of 4 games/deck each, real self-play, full 11-deck opponent
    pool, `n_workers=2`): marginal per-call overhead ≈ **47.5ms**, against
    the actual per-call cost of ~8,600ms (game collection + PPO update) —
    running in 5x smaller increments cost ~0% more wall-clock. Rebuilding
    11 decks' nets and rescanning `LeaguePool` is not the bottleneck this
    gap worried it might be. Setup cost therefore doesn't constrain the
    choice; **bandwidth does** (each call pushes the deck's whole sync
    surface, ~47MB incl. all active snapshots, and pulls the whole roster,
    ~700MB base64 -- NOT "a few MB" as first assumed). **Revised
    2026-10-17**: `DEFAULT_CALL_GRANULARITY_ITERATIONS` is now **3** in
    `node_agent.py` (was 1) -- equal to `main_league.json`'s `snapshot_every`
    (200 games / 60 per iteration), so each sync carries ~one new snapshot
    and syncing more often only re-sends unchanged files. 3x less traffic;
    staleness, the unsynced-loss window and stop latency are each ~180
    games. Delta sync (send only changed files) remains the real fix and is
    not built.
15. **[Needs benchmarking]** The elastic/bounded-drift threshold `K` and the
    liveness timeout (2026-09-05, see "Node lifecycle") are both unasserted.
    Unlike #14, this genuinely can't be tuned in Phase 2's loopback setup —
    a single machine has no real speed variance between "nodes" to measure
    drift against. Needs Phase 3, once at least one real remote node exists
    alongside continued local training.

## Components (new code, `src/distributed/`)

| Module | Role |
|---|---|
| `hub_server.py` | Flask app (flask is already a repo dependency, used by `src/webapp/` — no new dependency). Endpoints: registration (shared registration credential in, deck assignment + freshly-minted deck-scoped token out — see "Node lifecycle"), push (owned-deck files — `live.pt`/`mulligan.pt`/snapshots/`opponent_stats.json`/new `metrics.jsonl` lines, per the zero-information-loss decision in "Confirmed assumptions" — token-authorized), pull (full-roster current state), and a lightweight stop-flag surfaced as part of the push/pull response so a node discovers it via its own regular sync, not a new inbound channel. |
| `hub_store.py` | Canonical on-hub `checkpoints/<league>/` read/write. Reuses `checkpoint.py`'s existing file schema and atomic-write pattern (`_save_with_retry`'s temp-then-`os.replace` idiom) for hub-side writes — no new on-disk format invented. |
| `hub_ownership.py` | Loads the **static** pool of decks eligible for this distributed run (just a list of deck names checked into git — no node_id in it, since node identity isn't known until registration) and tracks the **dynamic**, hub-runtime-only record of which decks are currently claimed, by which live token, each claim's last-reported `cumulative_games_per_deck`, and each claim's last-sync timestamp (the last two feed the elastic/bounded-drift pause decision and liveness exclusion — see "Node lifecycle"). Not git-tracked, changes continuously as nodes sync. Pure functions where possible: easy to unit test without a server. |
| `version_gate.py` | Compares a registering node's reported `git rev-parse HEAD` (and vocab hash) against the hub's accepted set. **Deferred entirely** until the hosting platform is chosen — see "Version/compatibility gate" below. Not part of Phase 1. |
| `auth.py` | Verifies the shared registration credential at registration time; mints and verifies the per-deck tokens `hub_server.py` issues afterward. `secrets.token_hex` (stdlib) — no new dependency. |
| `node_agent.py` | CLI entrypoint. Registers with the hub, then drives its own loop calling `rl.league.league_runner._run_session`/`advance_progress`/`_load_progress` directly — **not** `_train_to` (see "Node lifecycle" for why). Checks a stop condition (hub-signaled or local `SIGTERM`) between calls. |
| `node_sync.py` | Push/pull client logic against `hub_server`'s API. |

New config surface, `distributed_configs/` (new directory, parallels
`training_configs/`):

- `<league>_ownership.json` — **revised, 2026-09-04**: now just a static
  list of deck names eligible for this distributed run, checked into git (no
  secrets, no node identity — see "Node lifecycle" for why identity moved
  to registration time instead of being pre-declared here). **Not every deck
  needs to be listed.** A deck left off simply isn't part of the distributed
  run — if you have fewer nodes than decks and want the remainder still
  progressing, that's the untouched local `run_training_pipeline.py` path's
  job, not this one. Which decks actually have a live claimant at any given
  moment is separate, dynamic, hub-runtime-only state (see `hub_ownership.py`
  in the Components table) — not part of this file.
- Per-node local config — now much smaller than originally scoped: just the
  hub's URL and the shared registration credential (see "Node lifecycle").
  No pre-assigned node id or deck-specific token anymore, since both are
  handed out dynamically at registration. Still **gitignored**, same
  treatment `checkpoints/` already gets, since the registration credential
  is still a secret even though it's shared across every node.

Tests under `tests/distributed/` (mirrors `src/distributed/`, matching every
existing `src/<pkg>/` ↔ `tests/<pkg>/` pairing): ownership-pool/claim
validation (a deck can only be claimed once; a second registration for an
already-claimed deck invalidates the first token), push-authorization (a
node's token can only write files under its one assigned deck — testable
via Flask's test client, no real network needed), and a `tmp_path`-based
push→pull round trip in the same style `test_run_league.py` already uses
for local progress/checkpoint plumbing. **Not** version-gate accept/reject
— `version_gate.py` is deferred out of Phase 1 entirely (see
"Version/compatibility gate"), so there's nothing to test yet; that test
category belongs in Phase 3 once the module exists. Explicitly **not**
re-testing PPO/game correctness either — that's already covered; these
tests are integration/glue only.

## Security posture (given "non-adversarial for now, some third-party nodes")

- **Revised, 2026-09-04**: per-node long-lived tokens replaced by dynamic
  registration — see "Node lifecycle." Every node image carries the same
  registration credential (generated by the hub operator, distributed
  out-of-band); the hub mints a fresh, deck-scoped token per registration,
  and a new registration for a deck invalidates whatever token was issued
  for the previous claim on it. Enough given the stated trust assumption —
  no need for sandboxing or output-verification against a hostile node yet,
  though the shared registration credential is explicitly named as only
  safe under that same provisional assumption (see "Node lifecycle").
- **TLS regardless of the trust assumption.** The risk isn't "will the node
  misbehave," it's "can a third party sniff a credential or token off the
  wire" — cheap to close, so it's in scope for v1 even though nothing else
  adversarial is.
- Push authorization is enforced per-request: a node's *currently valid*
  token maps to its one assigned deck; a push naming a file outside that
  deck, or using an invalidated token, is rejected, not silently ignored —
  turns an operator mistake (wrong node config) or a stale zombie process
  into a loud error instead of silent cross-contamination.

## Version/compatibility gate — **BUILT 2026-10-17** (originally deferred 2026-09-02; the text below is the original design record — see DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md section 7, item 11 for what was built)

This is the part the rules-faithfulness mandate makes non-optional, not
just good practice.

Because nodes `git pull` independently, a node can be running stale engine
code — different card behavior, different action-table shape — while the
hub has no way to know unless it checks. A node training games under
different rules than the hub believes isn't a version-skew inconvenience
under this repo's standing rules-faithfulness mandate, it's producing
invalid training data that looks valid. This is a distinct concern from the
config-drift question that *was* explicitly waived above (see "Design
review" #3) — this one has not been waived, only postponed.

**Owner decision (2026-09-02): hold off on the "what happens to a rejected
node" question until the hosting platform is chosen** — what's practical
there (can the hub force a rejected node to stop/restart, or does a
rejected node just sit there doing nothing useful?) depends on what control
that platform gives over node lifecycle. **Update 2026-09-05**: this is
about the platform hosting node *compute* (e.g. OVHcloud) specifically —
not resolved by the Tailscale networking decision, which only answers how a
node routes to the hub (see gap #10, now resolved).

**Confirmed 2026-09-02: the deferral covers the whole gate**, not only the
node-lifecycle question above — nothing in this section gets built until
the hosting platform is chosen.

Tentative shape, not yet committed to: exact commit-SHA match
against the hub's own `HEAD`, reject the sync request on any mismatch, with
an explicit opt-in operator override (an allowlist of extra accepted SHAs)
for when the operator personally knows a given diff is docs/tooling-only —
plus an independently compared vocab hash, since it catches the sharpest
failure mode (a silent shape mismatch) even in an edge case the commit
check might not cleanly cover.

## Open questions

Need an answer before implementation starts, not guessed at.

1. ~~Where do validation/round-robin checks run?~~ — **resolved
   2026-09-05**: the hub, on its own periodic timer, against its synced
   merged copy — independent of any node's chunk cadence, unlike today
   where `run_training_pipeline.py`'s `on_chunk=_validate` ties it to the
   primary league's own chunk boundary. Every node also holds a full-roster
   mirror, so the hub isn't unique in *having* the weights — it's the right
   home because (a) it has the freshest aggregate view (a node's own mirror
   of decks it doesn't own is only as fresh as that node's last pull, which
   can lag the hub's), and (b) it's the only place with CPU free to spend on
   validation without competing with actual training, since nodes commit
   their cores to training the one deck they own. `validation/` isn't in
   the protected-files list, so a new hub-side caller running it
   periodically needs zero changes to existing code.
2. ~~Sync cadence~~ — **superseded 2026-09-04** by "Node lifecycle"'s call-
   granularity discussion and Design review item #14: this turned out not
   to be a simple "confirm 200" question — it trades off against
   `_run_session`'s unmeasured per-call setup cost, and needs an actual
   Phase 2 benchmark rather than a yes/no answer now.
3. ~~`create_training_league`~~ — **finalized 2026-09-06**: stays
   single-machine-only, out of scope for the distributed system. No
   objection raised across many subsequent turns; it's a benchmarking tool,
   not core training.
4. ~~How many nodes / how many decks each~~ — **resolved 2026-09-02**: exactly
   one deck per node, axiomatically. See "Confirmed assumptions" and the
   ownership-file note above for the consequence (a deck with no node
   assigned isn't covered by this system at all).

See also "Design review" above for the fuller, evidence-checked gap list
(items 5-14 there are real open items too, just not phrased as
owner-decision questions the way 1-3 above are).

## Staged rollout (each phase additive, each phase independently disposable)

**See `DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md` for the file-by-file,
no-steps-skipped version of every phase below** (2026-09-06) — this section
stays as the high-level summary; that document is the one to actually build
from.

- [ ] **Phase 0** — this document. No code.
- [ ] **Phase 1** — `hub_server.py` + `hub_ownership.py` + `auth.py`, tested
      standalone (Flask test client, no real second machine). Hub can
      register a fake node against the shared registration credential, mint
      it a deck assignment + scoped token from the static deck pool, and
      answer token-authorized push/pull against a `tmp_path` checkpoint
      tree — including the case of a second registration for an
      already-claimed deck invalidating the first token. `version_gate.py`
      is explicitly excluded — deferred until the hosting platform is
      chosen (see "Version/compatibility gate").
- [ ] **Phase 2** [mechanics verified; the "statistically indistinguishable from `run_training_pipeline.py`" criterion below was never run and is waived 2026-10-17 -- see DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md section 7 item 13] — `node_agent.py` + `node_sync.py`. Smoke-tested with the
      hub and the "node" both running on *this* machine (loopback), one
      real deck owned, proving the node agent's own loop around
      `_run_session` (not `_train_to`) produces metrics statistically
      indistinguishable from today's `run_training_pipeline.py` on the same
      config — i.e., prove zero behavioral drift before trusting it with a
      real remote machine. Also where the still-unmeasured `_run_session`
      per-call setup cost gets an actual number, and where graceful stop
      (hub-signaled and local `SIGTERM`) gets exercised for the first time.
- [ ] **Phase 3** — first real second machine, owning one deck, running
      *alongside* continued single-machine training on the rest of the
      roster (both can coexist — the local script is untouched). Real WAN
      latency/auth exercised for the first time here, not guessed at
      earlier. Also where `version_gate.py` finally gets designed and
      built — choosing a hosting platform is itself a prerequisite for this
      phase, so the two land together naturally. Also the earliest point
      the elastic/bounded-drift threshold and liveness timeout (gap #15)
      can be meaningfully benchmarked — needs real speed variance between
      at least two genuinely different machines, which Phase 2's loopback
      setup can't provide.
- [ ] **Phase 4** — scale ownership to the full roster across whatever
      node count is settled on. Local `run_training_pipeline.py` usage for
      `main-league` specifically is phased out here — the code itself still
      isn't removed, per the non-negotiables above.
- [ ] **Phase 5 (explicitly deferred, not designed yet)** — auto-rebalancing
      / lease-heartbeat churn handling (only worth building if the
      "spin up/down together" uptime assumption stops holding), centralized
      `archive/` aggregation for retrospective analysis, a status dashboard
      (likely piggybacking on the existing `webapp_mirror.py` pattern once
      there's real multi-node data to show).
