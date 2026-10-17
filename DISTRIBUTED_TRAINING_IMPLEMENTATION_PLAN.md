# Distributed Training — Implementation Plan

**Relationship to `DISTRIBUTED_TRAINING_PLAN.md`**: that document is the
design/decisions record (architecture, gaps found, tradeoffs argued). This
document is the execution order — every file, in the order it gets built,
with what it does, what tests it needs, and exactly where the plan stops
because it needs OVHcloud (or whichever platform) login credentials we
don't have yet. Read the design doc for *why*; read this one for *what,
in what order*.

**Verification note (2026-09-06)**: before writing this, every load-bearing
code citation in the design doc was re-checked against the live repo, not
trusted from memory — `_run_session` (league_runner.py:215),
`_train_to` (run_training_pipeline.py:119), `save_deck_checkpoint`/
`load_deck_checkpoint` (checkpoint.py:94/104), and `main_league.json`/
`baseline_settings.json`'s actual current values all match exactly what's
cited. Nothing had drifted. One new thing surfaced by reading
`report_metrics.py` directly (not just its README description, which is as
far as earlier verification went): every `metrics.jsonl` record already
carries its own `deck`/`kind`/`cumulative_games` fields — self-describing.
A hub merging every node's forwarded lines into one file is a plain
append, no merge logic needed. Recorded here since it simplifies
`hub_store.py`'s metrics handling below.

---

## 0. Decisions finalized in this pass

Things this plan needed to be concrete and complete that weren't decided
yet — resolved here rather than left as gaps, per the instruction not to
skip steps.

1. **`requirements.txt` cannot be used as-is for a CPU-only node.** It's
   hardcoded to a CUDA build (`torch==2.11.0+cu128`, `--extra-index-url
   .../cu128`, deliberately pinned exactly per its own comment, for this
   machine's GPU). A CPU node needs a different torch wheel entirely.
   **Not modifying `requirements.txt`** — that's this machine's own pinned
   file for the existing single-machine GPU path, and editing it risks that
   path. Instead: a new file, `distributed_configs/requirements-cpu.txt`
   (same `numpy`/`pytest`/`flask` lines, `torch` swapped for the CPU wheel:
   `--index-url https://download.pytorch.org/whl/cpu` /
   `torch==2.11.0`). Lives under `distributed_configs/`, consistent with
   non-negotiable #2 (new subsection, nothing existing restructured).
2. **Hub runtime state (claims/tokens/progress) persists to local disk**,
   not just in-memory. Without this, a hub process restart would silently
   invalidate every node's token, forcing a mass re-registration the moment
   the hub itself needs a restart — a real, foreseeable event, not an edge
   case. Small JSON file, hub-local, gitignored, atomic temp-then-`os.replace`
   write (same pattern `checkpoint.py`'s `_save_with_retry` uses, not the
   same function — that one is torch-specific).
3. **`version_gate.py` ships in Phase 1 as an always-accept stub**, not
   omitted entirely. The registration *flow* (credential in, deck+token
   out, version check as one step of it) needs to exist and be testable now;
   the real commit-SHA/vocab-hash check is still correctly deferred to
   Phase 3 (per the design doc's confirmed decision), but as a drop-in
   replacement for the stub's function body — `hub_server.py`'s calling
   code never changes when the real check lands.
4. **Node provisioning uses OVHcloud's private-network + SNAT Gateway
   exclusively — no node ever gets a public IP.** `OVHCLOUD_SETUP_NOTES.md`
   hedged this as "arguably more correct, not obviously cheaper." With
   Tailscale as the confirmed networking layer, it's no longer a hedge: a
   node only ever needs *outbound* access (to Tailscale's coordination
   service, then the tailnet overlay to the hub) — it never needs to be
   reached from outside. A public IP would be pure unneeded exposure.
5. **`node_sync.py` uses the `requests` library**, a new dependency (added
   only to `distributed_configs/requirements-cpu.txt`, not the hub
   machine's own `requirements.txt`). Multipart file uploads (the
   checkpoint binaries) over stdlib `urllib` are workable but meaningfully
   more code for no real benefit; `requests` is small, standard, and the
   right tool here.
6. **Push/pull transport: multipart form data for binary files, not
   base64-in-JSON.** The checkpoint files are real binaries (~3.9MB
   `live.pt`, ~1.36MB per snapshot) — base64 would inflate that ~33% for
   nothing. Metadata (deck name, `cumulative_games`, the `metrics.jsonl`
   tail as text) rides alongside as regular multipart fields.
7. **Hub TLS**: evaluate Tailscale's own HTTPS certificate feature (it can
   issue real certs for a node's stable tailnet identity) before falling
   back to a self-signed adhoc cert. Not committing to one now — a Phase 1
   setup task, not a design decision, since it doesn't affect any code
   structure either way.
8b. **`HubRuntimeState` (2.3) serializes ownership operations with an
   explicit `threading.Lock`** around `claim_deck`/`record_progress`/drift
   computation — correct regardless of whether `hub_server.py` ever runs
   with `threaded=True`, not contingent on remembering to keep the server
   single-threaded.
8c. **`validation/` verified compatible with zero modification** (2026-09-06,
   read `_common.py` and `round_robin_training.py` directly, not inferred):
   `ValidationContext` only needs a `build_pool()` bundle and a
   caller-provided executor; `load_deck_net` reads `live.pt` off disk with
   no awareness of how it got there; `round_robin_training` already
   no-ops cleanly when `training_league_name=None`, exactly the state the
   hub passes since `create_training_league` stays out of scope.
8d. **Adding a deck to the ownership pool mid-run has a real, accepted
   fleet-wide cost**: a freshly-added deck starts at 0 progress and
   immediately becomes the "slowest active" node, pausing every other node
   until it catches up — consistent with "always gated by the slowest,"
   not a bug. Do deliberately, not casually.
8. **`node_agent.py` runs as a systemd service** on the node (standard
   Ubuntu image already has systemd, no new dependency) — auto-restarts on
   a transient crash without needing an operator to notice and intervene,
   consistent with nodes being meant to run unattended between operator
   check-ins.

Also finalized here, carried over from the design doc's open list:
**gap #8** (rollback auto-pushes immediately) and **Open Questions #3**
(`create_training_league` stays single-machine-only) — both marked resolved
there now.

---

## 1. One-time repo setup

- [ ] Create `src/distributed/`, `tests/distributed/`, `distributed_configs/`.
- [ ] `.gitignore` additions: per-node local config file pattern (hub URL +
      registration credential — decide the exact filename when
      `node_agent.py`'s CLI is written, Phase 2), and the hub's runtime
      state file from decision #2 above.
- [ ] `distributed_configs/requirements-cpu.txt` (decision #1 above).
- [ ] `distributed_configs/<league>_ownership.json` for `main-league`: the
      static pool of decks eligible for distributed training. Content is an
      operator call (which of the 11 decks to actually distribute first) —
      defaulting to the full 11-deck roster from `data/league_decks.json`
      unless told otherwise, since nothing stops a smaller pool later.

---

## 2. Phase 1 — Hub core

Buildable and fully testable right now. No real second machine, no hosting
platform, no credentials anywhere in this phase.

### 2.1 `src/distributed/__init__.py`
Empty, standard package marker.

### 2.2 `src/distributed/auth.py`
Pure crypto/secrets helpers, no I/O beyond what's handed to it:
- `generate_registration_credential() -> str` — `secrets.token_hex(32)`.
  Run once by the operator; the result goes into every node's provisioning
  template (decision from the design doc's "Node lifecycle").
- `verify_registration_credential(provided: str, expected: str) -> bool` —
  constant-time compare (`secrets.compare_digest`, stdlib — timing-attack
  resistance is free, no reason not to).
- `mint_deck_token() -> str` — `secrets.token_hex(32)`, same shape,
  different purpose (write-authorization, not registration).

**Tests** (`tests/distributed/test_auth.py`): credential verification
correctness (right value accepts, wrong rejects, wrong-length doesn't
crash), token uniqueness across many mints (birthday-bound sanity check,
not a proof).

### 2.3 `src/distributed/hub_ownership.py`
The static/dynamic split from the design doc, now concrete:
- `load_deck_pool(path) -> list[str]` — reads the ownership JSON from
  section 1.
- `HubRuntimeState` — holds `claims: dict[deck, {"token": str,
  "last_cumulative_games": int, "last_sync_ts": float}]`. Loaded from and
  saved to the local runtime-state file (decision #2), atomic write.
- `claim_deck(state, deck, pool) -> token` — mints a fresh token
  (`auth.mint_deck_token`), overwrites any existing claim for that deck
  (the "new registration invalidates the old one" rule), raises if `deck`
  isn't in the static pool.
- `verify_claim(state, deck, token) -> bool` — used by `hub_store.py`'s
  push path for authorization.
- `record_progress(state, deck, cumulative_games, now) -> None` — called on
  every push.
- `active_decks(state, now, liveness_timeout) -> set[str]` — decks whose
  `last_sync_ts` is within the timeout; used for the drift calculation, so
  a dead node's stale progress number can never anchor the "slowest active"
  comparison (the specific failure mode named in the design doc).
- `slowest_active_progress(state, now, liveness_timeout) -> int | None` —
  `None` when no deck is active yet (fresh league, nobody's synced).
- `drift_signal(state, deck, now, liveness_timeout, threshold) -> "continue"
  | "pause"` — the elastic/bounded-drift decision, called from the push
  handler. `threshold` is a parameter, not hardcoded — its real value is
  gap #15, benchmarked in Phase 3 (section 5); Phase 1's tests exercise the
  logic with injected values, not a asserted production number.

**Tests**: claim exclusivity (second claim on an already-claimed deck
invalidates the first token — verify the old token then fails
`verify_claim`), claiming a deck not in the pool raises, liveness exclusion
(a stale `last_sync_ts` drops out of `active_decks`), drift-signal
correctness at and around the threshold boundary (all with injected
timestamps/progress values — no real clock, no real network, matching how
`tests/test_run_league.py` already tests `league_runner.py`'s progress
plumbing with `tmp_path` and no real training).

### 2.4 `src/distributed/hub_store.py`
Canonical `checkpoints/<league>/` read/write, reusing the existing on-disk
schema (`checkpoint.py`'s file shapes — `live.pt`, `mulligan.pt`,
`snapshot_N.pt`, `opponent_stats.json`) without importing anything from a
protected file for the *write* path (this is new I/O code, not new game/
training logic):
- `push_deck_files(league_dir, deck, files: dict[str, bytes], metrics_lines:
  list[str]) -> None` — **this is where gap #6 (push atomicity) actually
  gets built**: every file in `files` written to a `.tmp` sibling first,
  `metrics.jsonl` lines appended to a `.tmp` copy of the current file too,
  then every temp file `os.replace()`'d into place in one tight loop — if
  any single write fails, nothing has been swapped in yet, so a
  connection-drop mid-push leaves the hub's prior state intact rather than
  half-updated. (Genuinely atomic *across the whole batch* would need a
  transaction log or filesystem-level transaction support neither Windows
  nor this scope calls for — the temp-then-replace-all-at-once ordering is
  the practical equivalent: the batch either lands as a whole or the
  failure happens before anything is visibly changed.)
- `pull_full_roster(league_dir) -> dict[deck, dict[str, bytes]]` — reads
  every deck's current `live.pt`/`mulligan.pt`/active `snapshot_*.pt`/
  `opponent_stats.json` (never `archive/` or `checks/`, per the design
  doc's explicit exclusion).
- `seed_fresh_deck(league_dir, deck, vocab, fixed_table) -> None` — the
  "no live.pt anywhere yet" bootstrap case (gap #2's resolution). Calls
  `rl.league.league_runner.build_deck_net` and
  `rl.checkpoint.save_deck_checkpoint` — importing and calling existing
  functions from new code, not editing the files they live in, which is
  exactly what non-negotiable #1 permits.

**Tests**: push-then-pull round trip on a `tmp_path` (matching
`test_run_league.py`'s own style), a simulated failure mid-push (kill the
write partway via a mocked file handle) leaves the previous state readable
and unchanged, `seed_fresh_deck` produces a net `checkpoint.load_deck_checkpoint`
can load back.

### 2.5 `src/distributed/version_gate.py`
Phase 1 stub, per decision #3:
```
def check(reported_sha, reported_vocab_hash):
    return True, None  # (accepted, reason) — always accepted in Phase 1
```
Real Phase 3 version compares `reported_sha` against the hub's own
`git rev-parse HEAD` (plus an operator-configurable allowlist) and
`reported_vocab_hash` against a hash computed from the hub's own
`build_pool()` — same function signature, so nothing calling it changes.

**Tests**: stub always returns accepted — trivial, exists to prove the
call-site wiring is correct before the real logic exists.

### 2.6 `src/distributed/hub_server.py`
Flask app (no new dependency — `flask` is already in this machine's
`requirements.txt`, used by `src/webapp/`). Ties 2.2–2.5 together:

- `POST /register` — body: `{credential}`. Verifies via `auth.py`; on
  success, calls `hub_ownership.claim_deck`, returns
  `{deck, token, version_gate_note}` (the stub's always-True result, still
  surfaced so the client-side logging shows what happened). Rejects with
  401 on a bad credential, 409 if the deck pool is fully claimed.
- `POST /push` — multipart body: `token`, `deck`, checkpoint files, a
  `metrics_lines` text field (newline-joined new `metrics.jsonl` records),
  `cumulative_games`. Verifies `hub_ownership.verify_claim`, calls
  `hub_store.push_deck_files`, `hub_ownership.record_progress`, computes
  `hub_ownership.drift_signal`, returns `{ack: true, signal: "continue" |
  "pause" | "stop"}`. The stop signal itself is operator-set (a small
  runtime flag alongside the claims state, flipped by a future admin
  action — no UI for this in Phase 1, a direct state-file edit or a Python
  REPL call is fine for now; a real control surface is Phase 5 territory
  per the design doc, not blocking here).
- `GET /pull` — query param `token` (to confirm the requester holds a live
  claim, though pull data itself isn't deck-scoped — every node needs the
  full roster). Returns `hub_store.pull_full_roster`.
- TLS: adhoc self-signed for local Phase 1/2 testing
  (`app.run(ssl_context="adhoc")`); Tailscale-issued cert evaluated before
  Phase 3's real deployment (decision #7).
- Dev server as-is (design doc gap #9's resolution) — no `waitress`, no
  production WSGI server, given Tailscale-only exposure and low traffic.

**Tests**: Flask test client, no real network. Register → push → pull round
trip; a second registration for an already-claimed deck invalidates the
first token (verified by the first token's next push now failing);
push-authorization rejection (wrong token, token for a different deck);
drift signal returned correctly for injected progress states.

**Phase 1 exit criteria**: `pytest tests/distributed/` green, covering
every file above, entirely offline. A developer can run the hub, register
a fake node via `curl`/`requests` from a second terminal on the same
machine, and see files land in a `tmp_path`-style checkpoint tree — proving
the whole hub-side flow works before any node code exists.

**[DONE, 2026-09-06]** All of section 2 built on branch
`distributed-training-phase-1`: `src/distributed/{__init__,auth,
hub_ownership,hub_store,version_gate,hub_server}.py`,
`tests/distributed/{test_auth,test_hub_ownership,test_hub_store,
test_hub_server}.py` (38 tests), `distributed_configs/{requirements-cpu.txt,
main_league_ownership.json}`, `.gitignore` additions. `pytest
tests/distributed/` green (38/38); full existing suite unaffected (937
passed, 1 skipped, `-m "not slow"`, unchanged from before this branch).

One real design refinement made during implementation, not just typing up
what was already decided: `hub_server.py` gained a `GET /poll` endpoint
(register/push/pull were the only three specified) — a paused node needs
*some* lightweight way to check "am I still paused?" without doing a full
push, and piggybacking that check onto `/pull` (a full-roster fetch) would
be wasteful for something meant to happen every 5-10 seconds while idle.

---

## 3. Phase 2 — Node agent

Still buildable and testable on this machine alone — "loopback," hub and
node both local processes, no remote machine yet.

### 3.1 `src/distributed/node_sync.py`
- `register(hub_url, credential) -> (deck, token)`
- `push(hub_url, token, deck, files, metrics_lines, cumulative_games) ->
  signal` — builds the multipart request (decision #6), via `requests`
  (decision #5).
- `pull(hub_url, token) -> dict[deck, dict[str, bytes]]`

**Tests**: against a `hub_server.py` Flask test client (no real HTTP
socket needed for these), or a locally-started dev-server instance on a
free port for a true over-the-wire check.

### 3.2 `src/distributed/node_agent.py`
CLI: `--config <training config path> --hub-url <url> --credential-file
<path>` (the credential file is the gitignored per-node local config from
section 1 — just the hub URL + registration credential, read-only, per the
design doc's already-settled config-handling decision).

Main loop, built as its own code rather than reusing `_train_to` (per the
design doc's correction — `_train_to` has neither the right cadence nor a
stop hook):
```
decklists, vocab, deck_ctxs, fixed_tables = build_pool()          # same call run_training_pipeline.py makes
deck, token = node_sync.register(hub_url, credential)
executor = ProcessPoolExecutor(max_workers=opts["n_workers"])      # this node's own local pool, unrelated to hub
while True:
    progress = _load_progress(local_league_dir)
    n_iterations = CALL_GRANULARITY                                # gap #14 — benchmarked in 3.3, not guessed here
    _run_session(n_iterations, opts["games_per_iteration"], opts["snapshot_every"],
                 executor, opts["n_workers"], league_dir=local_league_dir,
                 train_decks=[deck], roster=None, checkpoint_rate=opts["checkpoint_rate"],
                 cumulative_games=progress["cumulative_games_per_deck"], ...)
    advance_progress(local_league_dir, n_iterations, opts["games_per_iteration"],
                      auto_sizing=False, session_start_games=progress["cumulative_games_per_deck"])
    files, metrics_lines = collect_push_payload(local_league_dir, deck)   # new_agent-local helper: reads current checkpoint files + new metrics.jsonl tail since last push
    signal = node_sync.push(hub_url, token, deck, files, metrics_lines, new_progress["cumulative_games_per_deck"])
    pull_and_apply(node_sync.pull(hub_url, token), local_league_dir, exclude=deck)  # refresh the other 10 decks' mirrors
    if signal == "stop" or sigterm_received:
        break
    while signal == "pause":
        time.sleep(POLL_INTERVAL_SECONDS)   # the 5-10s ping, correctly scoped to only the paused case
        signal = node_sync.poll(hub_url, token)
```
All of `_run_session`/`advance_progress`/`_load_progress`/`build_pool`
imported from `rl.league.league_runner`/`rl.roster` exactly as
`run_training_pipeline.py` already does — zero changes to those modules.

**Retry/backoff (gap #5)**: the `node_sync.push`/`pull` calls wrapped in a
retry loop with exponential backoff (a handful of attempts, capped delay) —
new code, no protected file involved, so this is a straightforward addition
here rather than a design question. A sync failure after retries exhaust
logs and defers to the *next* loop iteration rather than crashing the
process (checkpoints from the just-completed `_run_session` call are
already safely on local disk regardless).

**SIGTERM handling**: a signal handler sets the same "don't start another
call" flag the hub-driven stop signal sets — one code path for both
triggers, per the design doc.

**Tests** (`tests/distributed/test_node_agent_loopback.py`): start
`hub_server.py` and `node_agent.py` both as local processes (or in-process
via Flask's test client + a directly-invoked loop function, whichever
proves cleaner once written), one deck, a small number of games. Compare
resulting metrics against a parallel plain `run_training_pipeline.py` run
on the same config — not bit-identical (self-play is stochastic) but
*statistically* indistinguishable. Concretely: reuse
`analysis/eval/report_metrics.py`'s own `two_proportion_z` helper (already
built for exactly this kind of comparison) to check the two runs' win-rate
distributions don't differ beyond noise. This is the "prove zero
behavioral drift" test the design doc calls for, made concrete rather than
left as a description.

### 3.3 Benchmark gap #14 — `_run_session` call granularity
A small script (not shipped as production code — a one-off under
`tests/distributed/` or `benchmarking/`, matching the existing
`benchmarking/training_run.py` convention) that runs the Phase 2 loop with
a few different `n_iterations`-per-call values (e.g. 1, 3, 10) against the
same config, measuring wall-clock per call. Produces a real number for
`CALL_GRANULARITY` above instead of the placeholder `snapshot_every`
default the design doc proposed as a starting point. **Deliverable**:
update `DISTRIBUTED_TRAINING_PLAN.md` gap #14 from "needs benchmarking" to
resolved, with the actual measured value and reasoning recorded.

**Phase 2 exit criteria**: hub + one node, both on this machine, training
one deck successfully end to end through the new path; ~~metrics
statistically match the old path~~ [**NOT MET, waived 2026-10-17 -- see section 7,
item 13**]; `CALL_GRANULARITY` is a measured number,
not a guess; graceful stop and pause both exercised at least once.

**[DONE, 2026-09-06 -- mechanics only; the statistical-equivalence criterion above was never run and is waived, see section 7 item 13]** `node_sync.py` + `node_agent.py` built and verified
against a *real* running hub over genuine HTTP (not just the Flask test
client) and a real self-play training pass through the unmodified
`_run_session` — 4 real games of `elves` vs. the full 11-deck roster,
real PPO/mulligan updates, resulting `live.pt` byte-identical between the
node's local disk and the hub's after push. 63 tests total in
`tests/distributed/` (up from 38), full existing suite still green (954
passed, `-m "not slow"`).

Two things caught only by actually running this, not by writing it:
- **`/register`'s response was missing `cumulative_games`.** Nothing in
  the original API design told a node what progress its assigned deck was
  actually at, which it needs to correctly seed local `progress.json` on
  bootstrap (gap #1/#2's own resolution). Added to both `hub_server.py`'s
  response and `node_sync.register`'s return value.
- **`os.kill(pid, SIGTERM)` doesn't invoke a Python signal handler on
  Windows** (this dev machine) — it terminates the process directly via
  `TerminateProcess`, unlike standard POSIX delivery on the Ubuntu nodes
  this code actually targets. The test now invokes the installed handler
  directly instead of depending on OS signal delivery this machine can't
  faithfully exercise; the handler's own logic is what's actually being
  verified, which is platform-independent.
- `node_agent.run()` gained a `league_dir` override parameter (mirroring
  `_run_session`'s own), so tests don't write into the real, shared
  `checkpoints/` tree — an omission that would have made any real test of
  this function unsafe to run.

---

## 4. Phase 2.5 — Provisioning artifacts

Still buildable without any hosting-platform account — this is the "get as
close to deployment as possible" section. Everything here can be written
and syntax/logic-checked now; only the very last step of *actually running*
it against a real account needs credentials, which is where Phase 3 (below)
draws the line explicitly.

### 4.1 Node bootstrap script (cloud-init user-data, bash)
Concrete steps, in order:
1. `apt-get update && apt-get install -y python3-venv git`
2. Install Tailscale: `curl -fsSL https://tailscale.com/install.sh | sh`,
   then `tailscale up --authkey=<injected>` — the auth key is a Tailscale
   concept distinct from the registration credential (decision area worth
   flagging plainly: **two separate secrets get baked into every node
   image** — a Tailscale auth key (network-level: "you're allowed on this
   tailnet") and the registration credential (application-level: "you're
   allowed to register with the hub"). Both are uniform-per-fleet, same
   injection mechanism, just two different values for two different
   layers. Not designing a unified secret for both — they're genuinely
   different concerns (network membership vs. application auth), and
   conflating them would make revoking one without the other impossible
   later.
3. `git clone --branch <pinned-tag-or-commit> <repo-url>` — pinning to a
   specific ref (not just `main`) is what actually makes the version-gate's
   eventual real check meaningful; a node that clones whatever `main`
   happens to be at boot time defeats the point even before the gate exists.
4. `python3 -m venv venv && venv/bin/pip install -r
   distributed_configs/requirements-cpu.txt`
5. Write the per-node local config (hub URL + registration credential,
   both cloud-init template variables) to the gitignored path from
   section 1.
6. Write and enable a systemd unit running `node_agent.py` (decision #8),
   `Restart=on-failure`.

### 4.2 Provisioning automation
Given the OVHcloud research already covers CLI/API/Terraform as realistic
options, and Terraform gives "bring up N nodes together, tear them down
together" most directly (a `for_each` over the ownership pool) — a
Terraform module, parameterized so the provider block is swappable if the
hosting decision changes: instance size (b3-8 per `OVHCLOUD_SETUP_NOTES.md`,
pending the still-unresolved live-price check that document already
flagged), private network + SNAT Gateway (decision #4), Ubuntu LTS image,
the cloud-init script from 4.1 as user-data. **This can be authored and
`terraform validate`'d for syntax now — applying it for real is the one
step in this entire plan that needs OVHcloud credentials.**

### 4.3 Hub-side setup (this machine)
- Install Tailscale on the hub, join the same tailnet.
- Run `hub_server.py` as a persistent background process — since the hub
  is this Windows machine, a Scheduled Task or simply a dedicated terminal
  window is enough for v1's traffic level; no need for a Windows-service
  wrapper (e.g. NSSM) unless unattended-restart-on-reboot becomes a real
  requirement later.

**Phase 2.5 exit criteria**: bootstrap script and Terraform module both
written and locally validated (syntax-checked, and the bootstrap script's
logic exercised against a local Ubuntu VM or container standing in for a
real node, if one's available — not required, but the cheapest way to
catch a bootstrap bug before it costs real cloud spend). Nothing here has
touched a real OVHcloud account yet.

**[DONE, 2026-09-06]** `distributed_configs/node_bootstrap.sh` (section
4.1) written: packages, Tailscale, pinned `git clone`, CPU-only venv,
node-local credential file, systemd unit with `KillSignal=SIGTERM`
(node_agent's own graceful-stop trigger — explicit so it's never
accidentally changed to `SIGKILL`, which would skip the final push).

**[DONE, 2026-09-06]** `hub_server.py` gained a real `main()`/CLI
(`python -m distributed.hub_server --league-name ... --ownership-pool ...
--credential-file ...`) — not in the original Phase 1 spec, but genuinely
needed: without it there was no way to actually *run* the hub outside a
test. **Verified against a real subprocess, not just argparse**: launched
`python -m distributed.hub_server` for real (throwaway league name, never
touching the real `checkpoints/main-league`), hit it with a real `curl
POST /register` over an actual socket, got back a real deck/token/
cumulative_games response, confirmed in the server's own log line. Also
caught by trying this for real rather than assuming: `--tls adhoc` needs
the `cryptography` package, **not installed in this environment** — would
have failed the moment anyone actually tried to run the hub. Fixed by not
defaulting to it: `--tls` defaults to `none` (a printed warning explains
why that's only safe behind Tailscale), `adhoc` is opt-in and documents
the missing dependency, consistent with decision #7 never having actually
committed to a TLS approach yet.

**[DONE, 2026-09-06]** `python -m distributed.auth` — a one-line CLI to
generate a registration credential, not in the original Phase 1 spec
either, but there was no way to actually produce the value the hub's
`--credential-file` and every node's provisioning template need.

**[DONE, 2026-09-06]** The Terraform module itself
(`distributed_configs/terraform/{main,network,instances,outputs}.tf` +
`node_bootstrap.sh.tftpl`), built only after verifying the real `ovh/ovh`
provider schema (`OVH_TERRAFORM_NOTES.md`) rather than from memory — that
research surfaced a genuine fork (native `ovh_cloud_project_instance`,
labeled Beta API, vs. OVHcloud's own officially-recommended
`openstack_compute_instance_v2` via a second provider) put to the owner
rather than picked by default; **owner chose `ovh_cloud_project_instance`**
for the single-provider/single-credential-set simplicity.

**Actually validated, not just written**: Terraform wasn't installed on
this machine — installed it (`winget install Hashicorp.Terraform`) rather
than claim "syntax-checked" without checking. `terraform validate` caught
a real bug on the first pass: the bootstrap template's own comment
*explaining* that the script has no Terraform-syntax collisions literally
contained that syntax, which `templatefile()` parses anywhere in the raw
file, comments included — fixed. A `terraform plan` with dummy credentials
then got all the way through local schema validation, variable wiring, and
template rendering to a genuine, live OVH API rejection ("invalid
application key," a real 403 from the real API) — confirming the module is
structurally sound right up to the exact boundary where real credentials
become necessary, which is as close as this plan can get without them.

Not yet done: OVHcloud pricing/availability re-verification (both prior
research docs already flagged several numbers as needing a live check) —
deferred to Phase 3, since it's only actionable once the owner is actually
choosing a region/instance size to pay for.

**[DONE, 2026-09-02]** Section 4.3 completed and the whole stack verified
end-to-end for real, not just via pytest fixtures. The owner's own machine
was already a Tailscale node, so the only remaining piece was running
`hub_server.py` persistently. `checkpoints/main-league-distributed/` was
created as an isolated hub tree, seeded per-deck from the real
`checkpoints/main-league/`'s current `live.pt`/`mulligan.pt`/
`opponent_stats.json` (never touching production directly), with a fresh
registration credential in the gitignored `distributed_configs/
hub_credential.local`. `python -m distributed.hub_server ... --no-seed-fresh-decks`
run in the background, reached over its real Tailscale address
(`http://100.113.155.84:5000`) with a real `curl -X POST /register` --
confirmed a genuine `{deck, token, cumulative_games}` response, not just
the earlier bare-GET 405 (proof of reachability only).

Then a full `node_agent.run()` loop was driven for real against that live
hub -- register, pull, two real `_run_session` calls (24 real self-play
games total), push, and the hub's own on-disk state inspected afterward to
confirm the round trip: `hub_runtime_state.json` shows
`mono_red_madness` at `last_cumulative_games: 24`; the hub's
`checkpoints/main-league-distributed/mono_red_madness/{live.pt,mulligan.pt,
opponent_stats.json}` are genuinely updated (real sizes/timestamps, not
copies); `metrics.jsonl` on the hub side carries every PPO/mulligan record
from both sessions, matching the zero-information-loss requirement. This is
the strongest verification possible on one machine -- a real second process
over a real network path, not two ends of the same in-memory test fixture.

Two real bugs surfaced and fixed along the way, both artifacts of this
same-machine smoke test rather than defects in `distributed/` itself:
1. The hub process from the earlier curl test wasn't kept running (the
   owner's own choice was to run it in a terminal they could close) --
   confirmed via `netstat`/`tasklist` that nothing was listening, then
   restarted it.
2. The throwaway smoke-test driver script called `node_agent.run()` at
   module level instead of inside `if __name__ == "__main__":`. Windows'
   `spawn` multiprocessing start method re-imports the entire top-level
   module in every worker process, so every rollout worker re-ran the
   whole script from scratch -- re-registering as a brand-new node each
   time and cascading into runaway registrations (observed: `
   mono_red_madness` -> `rakdos_madness` -> `spy_combo` claimed in
   quick succession by processes that immediately crashed). Killed the
   runaway process tree, added the guard, reran clean. The stray claims
   this left in `hub_runtime_state.json` needed no manual cleanup --
   they aged past `liveness_timeout` (300s) on their own and
   `pick_available_deck` released them automatically, confirming that
   part of gap resolution works under a real failure, not just its test.
   (A first attempt, before this, also hit `OSError: [WinError 1455] The
   paging file is too small` spawning 10 CUDA-torch-importing workers at
   once -- an artifact of testing on this GPU-equipped hub machine with
   its CUDA-pinned torch build, not something a real CPU-only node
   using `requirements-cpu.txt` would ever hit; worked around for this
   smoke test only by using a throwaway low-`n_workers` config, never by
   editing `main_league.json`.)

**Phase 2.5 is now fully closed.** Everything reachable without an
OVHcloud account has been built and verified for real. What's left is
exclusively Phase 3.

---

## 5. Phase 3 — First real remote node (credential-gated)

**Everything above this line needs zero hosting-platform credentials.**
This is the explicit boundary the plan was asked to get as close to as
possible without crossing.

- [ ] **Confirm the hosting platform** (OVHcloud per the existing research,
      or revisit if something changed) — this is the actual credential-gated
      step; nothing before it requires a decision here.
- [ ] Create the OVHcloud account, generate API credentials
      (`OVHCLOUD_SETUP_NOTES.md`'s step list already covers this).
- [ ] Spot-check the b3-8/Discovery-tier pricing and regional availability
      live (both were flagged as needing reconfirmation, not stale-trusted,
      in the original research).
- [ ] Run the Terraform module from 4.2 for real, for **one** node.
- [ ] Verify end to end: node boots, joins the tailnet, registers with the
      hub, begins training, pushes and pulls successfully — running
      *alongside* continued local `run_training_pipeline.py` training on
      the rest of the roster (both coexist, per non-negotiable #1).
- [ ] Replace `version_gate.py`'s stub with the real commit-SHA/vocab-hash
      check (decision #3) — this is the natural point for it, since a real
      second machine with its own independent `git clone` is what makes the
      check meaningful for the first time.
- [ ] Benchmark gap #15 (drift threshold, liveness timeout) using this
      node's real observed speed against local training's — the first
      point genuine speed variance exists to measure against.
- [ ] Verify the zero-information-loss decision holds in practice, not just
      on paper: kill this node mid-run, confirm the hub has everything up
      to its last completed push, bring up a replacement node claiming the
      same deck, confirm it resumes from exactly that point with no manual
      intervention.

**Phase 3 exit criteria**: one real remote node, fully operational,
indistinguishable in behavior from the Phase 2 loopback node except for
being on different hardware over a real network — at which point Phase 4
(scale to the rest of the roster) is "repeat section 4.2/5 per remaining
deck," not new design work.

---

## 6. Phases 4–5

Unchanged from `DISTRIBUTED_TRAINING_PLAN.md`'s own "Staged rollout"
section — scale ownership to the full roster, phase out local training for
`main-league` specifically (code stays, per non-negotiable #1), then the
explicitly-deferred Phase 5 items (auto-rebalancing, `archive/` aggregation,
a status dashboard) once there's real multi-node data to justify them.

---

## 7. Post-review fixes (2026-10-16 review session)

1. **[DONE] Node config.** Terraform defaulted nodes to `main_league.json`
   (`device: cuda`, `n_workers: 10`) -- a CPU b3-8 node would `SystemExit`
   and crash-loop. Added `training_configs/main_league_node.json` (extends
   `main_league.json`; overrides only `device: cpu`, `n_workers: 2`;
   `games_per_iteration` stays 60 -- owner decisions), pinned by
   `tests/distributed/test_node_config.py`; Terraform's `league_config_name`
   default now points at it.
2. **[DONE] Persistent node ID (restart keeps its deck).** A restart used to
   re-register as a stranger and get the next free deck while its old claim
   was still live. Now each node keeps a UUID in `/var/lib/pauper-sim/node_id`
   (systemd `StateDirectory`, outside the git checkout; written
   temp+fsync+replace *before* registering) and sends it with `/register`.
   The hub persists `node_id -> deck` (`HubRuntimeState.node_decks`;
   old bare-claims state files still load) and `assign_deck` returns the
   node's own deck even while its old claim is live (fresh token revokes the
   old one; the takeover is logged). If the deck was since claimed by another
   node the mapping is dropped and the node gets the next free deck. A
   recreated node (new disk, new ID) gets `409` while the pool is full and
   now retries registration with capped backoff (`register_with_backoff`)
   instead of crash-looping. Requests without `node_id` behave as before;
   a malformed `node_id` is a 400. DISTRIBUTED_TRAINING_PLAN.md's
   "Node lifecycle" registration section predates this: node identity is
   now node-generated and runtime-only, not pre-baked in the image.
3. **[DONE] Hub-failure handling in `node_agent.run`.** A push/pull that
   failed after `node_sync`'s own retries used to crash the process. Now:
   a connection failure is logged and retried next iteration; after
   `DEFAULT_MAX_FAILED_SYNCS` (5, owner decision, unmeasured placeholder)
   consecutive failed iterations the node stops training and polls until the
   hub answers. A 403 (superseded token) re-registers with the same node ID
   and continues if the hub returns the same deck; a different deck, or a
   second 403 with no successful sync between, exits (systemd restarts into
   a clean bootstrap; the second rule stops two processes sharing a node ID
   from ping-ponging the claim). Other HTTP errors (401, 400) still
   propagate. `metrics_offset` now advances only after a successful push, so
   deferred syncs never skip metrics lines (the restart-resets-offset-to-0
   duplicate-metrics problem is still open -- review issue 4).
4. **[DONE] Metrics de-dup (hub-authoritative).** The node's metrics offset
   used to reset to 0 on every restart, re-pushing the whole local
   `metrics.jsonl`, and a push whose ack was lost re-sent a batch; the hub
   appended blindly. Now each push carries `node_id` + `metrics_start_line`
   (index of its first line in the node's local file, non-blank lines
   only). The hub keeps per deck `metrics_source` (node ID, i.e. one disk)
   and `metrics_lines` (how many of that source's lines it holds) and
   appends only lines past that count (`hub_ownership.accept_metrics`),
   recording the new count after the append (a hub crash in between replays
   one batch -- duplicates, never loss). `/register` returns `metrics_acked`
   so a restarted node resumes at the right line (`_resume_metrics_position`);
   `node_sync.register` now returns a 4-tuple. A different source (new
   node ID/disk), a gap, or a local file shorter than the hub's count
   (`metrics_rebase`) re-anchors instead of skipping -- in the shorter-file
   case the node starts from the end of its local file (owner: "whichever
   most appropriate"). Also fixed: a partial trailing line (no newline yet)
   was previously counted as a complete line; only complete lines are read
   now. Pushes without `metrics_start_line` (older clients) are accepted
   undeduped.
5. **[DONE] Hub seeding with real progress (`python -m distributed.seed_hub`).**
   The hub reported `cumulative_games` 0 for any never-claimed deck, even one
   seeded with trained weights, so a node would restart its hyperparameter
   schedule/snapshot cadence at 0. `seed_hub` now copies each pool deck's sync
   surface from an existing league (live.pt, mulligan.pt, opponent_stats.json
   and the ACTIVE snapshots -- never archive/ or checks/) and records each
   deck's starting progress (the source league's `progress.json`, one counter
   shared by every deck it trained -- 92,340 for `main-league`; `--games`
   overrides) in `HubRuntimeState.seed_progress`, which `claim_deck` uses for
   a deck with no claim yet. Run with the hub stopped; refuses to overwrite a
   populated hub tree/state without `--force` (which also wipes claims/node
   mappings and stale snapshots). The hub warns at startup about pool decks
   that have weights but no recorded progress. Not copied: the source
   league's `metrics.jsonl` history (hub metrics start empty; add if
   `report_metrics.py` on the hub tree should show pre-distributed history).
   The existing `checkpoints/main-league-distributed/` smoke-test tree needs
   `--force` to be reseeded.
6. **[DONE, partial] Sync frequency lowered (bandwidth).** At granularity 1 a
   node pushed ~47MB (deck files incl. all ~32 active snapshots) and pulled
   ~700MB (full roster, base64-in-JSON) every iteration, on a single-threaded
   hub. Owner chose to lower the sync frequency (not delta sync):
   `DEFAULT_CALL_GRANULARITY_ITERATIONS` 1 -> 3 (= `snapshot_every`; sync
   happens once per `_run_session` call), a one-line change that cuts
   traffic 3x. Still open: delta sync/streaming/threaded hub would shrink the
   cost of each sync; the single-threaded hub still blocks all nodes during
   a large pull. Newly surfaced, not yet decided: an iteration on a 2-vCPU
   node probably takes minutes, so a 3-iteration call can exceed the systemd
   unit's `TimeoutStopSec=120`, SIGKILLing the node before its graceful-stop
   final push (iteration time on a 2-vCPU node is unmeasured).
7. **[DONE] Evicted snapshots are deleted on the hub and in node mirrors.**
   `push_deck_files` only ever wrote, so the hub kept every snapshot a node
   had evicted from its 32-wide window, and every node's pulled opponent pool
   grew past the cap too. Now each push carries `snapshot_names` (every
   snapshot_*.pt the node holds); after all file swaps succeed the hub deletes
   that deck's other snapshots (deleted, not archived -- owner decision
   2026-10-17; a push without the field prunes nothing). Node side,
   `_write_roster_to_disk` mirrors the same evictions for pulled decks (local
   snapshots absent from the hub are deleted); the node's own deck is exempt,
   including on the bootstrap pull. The mirror half is required for
   correctness, not a safety net. Known leftover: a node's
   `opponent_stats.json` can keep entries for snapshots another deck evicted
   (the single-machine run prunes those in `register_snapshot`, which only the
   evicting process can do); harmless -- sampling walks the present snapshots.
8. **[DONE] Operator control: stop / resume / status.** `request_stop(_all)`
   existed only as Python functions, and the hub keeps its state in memory and
   rewrites the file on every save, so there was no way to actually stop a
   node. Now (owner chose authenticated HTTP admin, 2026-10-17):
   `GET /admin/status`, `POST /admin/stop` (`{"deck": ...}` or `{"all": true}`),
   `POST /admin/resume` (cancels a stop not yet acted on), guarded by an
   `X-Admin-Token` header checked against `--admin-credential-file` -- a
   SEPARATE operator-only secret (the hub refuses to start if it equals the
   registration credential; omitting the flag disables /admin/* with 404).
   CLI: `python -m distributed.admin --hub-url ... --admin-credential-file ...
   status | stop --deck D|--all | resume --deck D|--all`. A node that gets
   "stop" (or SIGTERM) finishes its call, does its final push, then POSTs
   `/release`: token invalidated, stop flag cleared, deck immediately free
   (`last_sync_ts=0`), with progress, node-id mapping and metrics ack kept so
   the same or a replacement node resumes from real state. If `/release` is
   lost the deck frees via the liveness timeout instead. A manual
   hold/pause-without-stopping operation was offered and not requested; not
   built.
9. **[DONE] A node ends its own run at `total_games`.** Nothing checked the
   config's `total_games` (100,000 for `main_league.json`, confirmed by the
   owner as the distributed target too), so nodes trained and billed forever.
   `node_agent.run` now stops once its deck reaches `total_games`: the final
   call is clipped like `run_training_pipeline._train_to` (floor to whole
   iterations, min 1 -- within one iteration of the target), the final state
   is pushed, then it releases the deck and exits (owner: "exit it"). It never
   exits with that last state unsynced -- if the push fails it re-pushes
   without training further. A node restarted already past the target pushes
   once and exits without training. The hub does not track targets
   (the node's own config decides; a drifted config changes the target, per
   the accepted config-handling decision), so `admin status` shows a finished
   deck as `released`, not `finished`.
10. **[DONE, unverified on a real node] Nodes reachable via Tailscale SSH.**
    Nodes had no public IP and `--ssh=false`, so a misbehaving node could not
    be inspected at all. The bootstrap now runs `tailscale up ... --ssh` and
    makes the journal persistent; `TAILSCALE_SETUP_NOTES.md` has the tailnet
    policy (replaces the default allow-everything rule: your devices reach
    everything and may SSH into `tag:pauper-node` nodes; nodes reach only the
    hub at 100.113.155.84:5000), the auth-key settings (reusable, ephemeral,
    tagged, <=90 days) and a debugging cheat-sheet. Policy/key syntax was
    checked against Tailscale's docs, not run: the policy is not yet applied
    to the real tailnet, and `--ssh` on Ubuntu 24.04 / tag-from-key behavior
    need confirming on the first Phase 3 node. If Tailscale itself fails to
    start there is still no way in (destroy and recreate).
11. **[DONE] Version gate built (replaces the always-accept stub).** Owner
    accepted the recommendation (2026-10-17). Nodes report `git rev-parse HEAD`,
    whether any tracked file is modified, and a vocab hash
    (`version_gate.local_version`/`vocab_hash`). The hub accepts only SHAs
    (full, or >=7-char prefix) in an operator-maintained, gitignored,
    hub-local file (`distributed_configs/accepted_versions.local.json`,
    `{"shas": [...]}`), re-read on every registration so approving a SHA needs
    no restart; a missing/empty file rejects everyone (fail closed); the hub's
    own HEAD is NOT the reference (it never trains and has uncommitted work).
    Rejections return **426** (distinct from 409 pool-full, which the node
    retries quickly): the node logs loudly, never trains, and retries every
    300 s until the operator approves the SHA or fixes the node. The hub logs
    each rejection and `admin status` lists the last ones. `--no-version-gate`
    disables it (loud warning; development only). Bootstrap fix: `git clone
    --branch` accepts only a branch/tag although `git_ref` is documented as
    possibly a raw SHA -- now `git init` + shallow `git fetch origin <ref>` +
    checkout (verified against a local repo; GitHub allows SHA fetches).
    The tag/SHA a node boots from must resolve to a commit on the accepted list.
    Known limits: untracked files are ignored by the dirty check (venv and
    node-local config live in the checkout; non-adversarial trust model).
    **Surfaced while building this, not yet decided -- see review issue 12
    below:** `checkpoints/vocab.json` is append-only history on the hub but a
    clean node regenerates it fresh and sorted, so the two can diverge
    structurally; today they happen to match (verified), the vocab hash would
    reject (not repair) a divergence.
12. **[DONE] Hub serves the card vocabulary (nodes adopt it).** The design doc
    assumed every node regenerates an identical `vocab.json` from the manifest;
    wrong, since the hub's file is append-only history while a clean node builds
    it fresh and sorted (they matched only because the single missing name,
    "Insectile Aberration", sorts last). Divergence silently misreads cards in
    pulled weights, and the version gate could only reject, never repair it.
    Now `GET /vocab` (header `X-Registration-Credential`) serves the hub's file
    (`rl.roster.VOCAB_PATH`, the file `build_pool` itself uses) and
    `node_agent._sync_vocab` writes it before `build_pool`: absent/identical ->
    write/no-op; a local file that is a compatible older subset (the hub only
    appends) -> replaced; a local file assigning any name a different index ->
    refuse (SystemExit) rather than silently invalidate local weights. The
    vocab hash check stays as verification (a node whose roster needs a card
    the hub's vocab lacks appends locally and is rejected 426 -- the hub's
    operator must add it). Owner chose Option A; the served file is
    `checkpoints/vocab.json`, shared by every league in `checkpoints/` (not
    separately confirmed). Changing the roster so the hub's vocab grows still
    changes the embedding table size, i.e. is a retrain event, as before.
13. **[DONE as a correction, 2026-10-17] The Phase 2 "statistically
    indistinguishable from `run_training_pipeline.py`" criterion was never run
    and is waived by the owner.** Section 3.2 specified it and Phase 2 was
    recorded as done, but `test_node_agent_loopback.py` deliberately checks only
    mechanics (real training happens, files land on node and hub, progress and
    `live.pt` agree after a push). What IS verified: those mechanics; and by
    reading `_run_session`, that the loop's chunking is sound -- snapshot cadence
    and the PPO/entropy schedules key on cumulative games, not per-call counters,
    and production configs set no `seed`, so repeated calls don't replay one
    random stream. What is NOT verified: that a deck trained through the node loop
    learns as well as the same deck on the single-machine path (e.g. nothing
    about optimizer state surviving repeated calls, or a pull overwriting
    something it shouldn't, has been measured). Exact statistical equivalence is
    neither achievable nor the goal: in the real run each deck trains on its own
    node against opponents that lag by up to one sync (accepted design
    tradeoff). The offered controlled single-node comparison (same deck, same
    config, `train_decks=[deck]` direct vs node loop, compared with
    `report_metrics.two_proportion_z`) was declined; Phase 3's real nodes are the
    first evidence of learning behavior, uncontrolled.
14. **[DONE] Graceful stop vs systemd's stop timeout.** Owner delegated the
    choice ("best judgement", 2026-10-17); chose options A + C. Measured one
    node-style call on this dev machine (CPU, 2 workers, 60 games, one deck, full
    roster): 102 s per iteration, 310 s for the 3-iteration call (an OVH b3-8 is
    probably slower; unmeasured there). SIGTERM only sets a flag and
    `_run_session` can't be interrupted, so the node must finish its call before
    it can push, release and exit -- minutes, against `TimeoutStopSec=120`, which
    SIGKILLed it mid-call (losing up to a call of compute plus the final push and
    release, leaving the deck claimed until the 300 s liveness timeout).
    Now `TimeoutStopSec=1800` (~3x the measured call), and the docs/comments name
    **`admin stop` as THE clean path** for planned stops (hub-driven: lossless,
    nothing for systemd to time out); SIGTERM/`systemctl restart`/reboot remain
    lossless only where the shutdown window allows, and `terraform destroy` /
    instance delete never give a node any chance (run `admin stop --all`, wait for
    nodes to release, then destroy). Rejected: aborting the in-flight call on
    SIGTERM (loses a call every routine restart, needs interrupt handling around
    a protected function); 1-iteration calls (undoes the bandwidth saving).
    Scale note from the same measurement: ~100 s/iteration puts a fresh
    100,000-game deck at roughly 47 h on this machine (longer on a node) and the
    7,660 games left to the current 92,340 at roughly 3.6 h.
15. **[DONE] Hub concurrency correction + per-deck lock.** *Correction to this
    document and to review issue 6:* the hub is NOT single-threaded -- Flask's
    `app.run` sets `threaded=True` by default (verified in the installed Flask
    3.1.3), so the `hub_server.py` docstring and the claim that "one pull blocks
    every node" were wrong; requests run concurrently. Consequences: ownership
    state was already locked and `metrics.jsonl` appends already had a lock, but a
    pull could read a deck while a push swapped/pruned its files (new `live.pt`
    next to the old snapshot set). On Windows (the hub machine) it is worse: an
    unlocked `os.replace` onto a file a pull has open raises `PermissionError`,
    i.e. a push would 500. `hub_store` now takes a per-deck lock around the swap +
    prune and around each deck's read in `pull_full_roster` (held only for the
    fast disk work, not the upload/response). Test: a hammer test (concurrent
    pulls vs 200 pushes must always see one push's complete file set) plus a
    sensitivity check that it fails with the lock disabled.
16. **[DONE] Bootstrap makes the node's own deck an exact hub mirror.**
    Previously a node restarting after a crash with unpushed local work got the
    hub's OLDER `live.pt` written over its own, but kept its AHEAD local
    `progress.json` counter and its extra local snapshots: weights, snapshots and
    schedule position described different moments. Hub is canonical (also what a
    deliberate `seed_hub --force` reseed needs), so bootstrap now writes the
    hub's own-deck files, deletes local snapshots the hub lacks, and sets the
    local counter to the hub's value even if local was ahead
    (`_seed_local_progress`; the old "never regress" test was replaced). Cost:
    up to one call (~3 iterations) of unpushed work after a crash is discarded
    (it was already unrecoverable once `live.pt` was overwritten).
17. **[DONE] Terraform `hub_url` example** said `https://hub.<tailnet>.ts.net`
    but the hub serves plain HTTP by default; now `http://100.113.155.84:5000`
    with the reason (tailnet traffic is WireGuard-encrypted). The repo is
    public (`gh repo view`), so the bootstrap's unauthenticated clone/fetch
    needs no token.
