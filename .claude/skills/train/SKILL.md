---
name: train
description: Run monitored, escalating self-play training for this repo's MTG deck agents (the per-deck league DeckNetworks). Use when the user asks to train, run training, train the agents/models, start a training run, or "/train". Parses the invocation message for "fresh start" (wipe the league's checkpoints and train from scratch). Starts with a tiny batch, verifies it is healthy, then doubles the batch each clean step.
---

# train — monitored, escalating agent training

Orchestrates the EXISTING training script (`src/run_league.py`). Never reimplements
training. All commands run from `src/`.

There is ONE training phase. Until 2026-08-17 there were two -- `run_pretrain.py`
built and froze a single shared perception stack that the league then trained
per-deck heads on top of. Each deck now owns its encoder and trains it with its
policy, so that script, `checkpoints/shared_stack_frozen.pt`, and the whole
pretrain/freeze step no longer exist.

The whole point is **safety through small, monitored, escalating batches**: start
each phase tiny, confirm it ran clean (exit 0, expected "done" line, no traceback,
real progress — not hung), then roughly **double the games** each clean step. On any
error, STOP escalating, find the root cause, fix it, re-run the SAME size, and watch
specifically for that failure again before growing.

## 1. Parse the invocation message

- **`fresh start`** (also "from scratch", "start over", "wipe", "retrain everything")
  → MODE = `fresh`: wipe the league's checkpoints (and `vocab.json`), then train.
- **No flag** (or "per-deck only" / "league only", which now mean the same thing)
  → MODE = `resume`: train from whatever checkpoints exist. Never wipes anything.
- **Per-deck game target**: the LEAGUE trains until each deck has played ~this many
  games. If the message names a count ("train each deck 8000 games", "8000 per deck")
  use it as `TARGET_PER_DECK`; default `TARGET_PER_DECK = 3000`. Stop as soon as ANY
  deck's cumulative game count reaches it (in league mode the decks advance together and
  arrive ~simultaneously; "any" just guards against one racing ahead).
Restate the parsed MODE / TARGET_PER_DECK (and N_DECKS) back to the user in one line
before starting.

## 2. Preconditions

- `cd src` for every command. Ensure `../logs/` exists (`mkdir -p ../logs`).
- Confirm torch imports (`python -c "import torch"`); the game engine self-checks are
  assumed already green.
- **Deck roster is dynamic** — it is whatever `data/league_decks.json` lists (deck name
  → decklist file). Read the count once into `N_DECKS`
  (`python -c "import json; print(len(json.load(open('../data/league_decks.json'))))"`)
  and use it wherever a TOTAL game count or run-time estimate is needed. Never hardcode a
  deck count. Adding a deck (edit league_decks.json + its data/*.txt) needs no change to
  this skill — but it changes the vocab, so a new deck requires a `fresh start`.
- **Nothing has to exist before the first run.** A deck with no `live.pt` starts
  from a freshly-initialized net, encoder included, and `vocab.json` is written on
  demand by `build_pool()`. (There used to be a hard precondition here: the league
  could not start without `checkpoints/shared_stack_frozen.pt` from a prior pretrain
  phase.)
- **Which league/config — check BEFORE the first command, not after.** List
  `../training_configs/*.json` and check which checkpoint directories already have
  content (`../checkpoints/<name>/session.txt` or any `live.pt` under it). If the
  invocation names a league, or exactly one `training_configs/*.json` matches an
  existing checkpoint tree, use that config explicitly:
  `--run-config ../training_configs/<file>.json --league-config ../training_configs/<file>.json`
  (both flags MAY point at the same file — a config can carry both run-mechanics
  fields like `checkpoint_opponent_rate`/`snapshot_every_games`/`n_workers` and
  league-identity fields like `league_name`/`roster`/`total_games` at once; check the
  file's own contents, not its name or which flag you'd guess it belongs to).
  **Never fall back to the bare, config-less command form** (`run_league.py
  --n-iterations ... --n-workers ...` with no `--run-config`/`--league-config`) when a
  config file exists that matches an already-populated checkpoint directory — the bare
  form checkpoints to the hardcoded default `../checkpoints/league/` with hardcoded
  defaults (`checkpoint_opponent_rate=0.0` among them), which silently trains a
  DIFFERENT league under different settings instead of resuming the one that's
  actually there. If more than one config could plausibly be meant and it isn't
  obvious which, ASK — don't guess (same standing rule as everywhere else in this
  repo for an uncertain call). Only use the bare config-less form for a genuinely new,
  never-configured roster with no existing checkpoints.
- A vocab change (e.g. new cards/tokens) makes old checkpoints dimensionally
  incompatible. If any run aborts with a `vocab_size` / roster-mismatch assert, that
  is the signal to do a `fresh start`.
- **Feature-dim check (catches new cards even before they join the pool, or a change
  to `rl/features.py` itself).** Adding a card with a NEW keyword/type grows
  `STATIC_FEATURE_DIM`; a code change to the per-token dynamic layout (e.g. the
  own-hand-tokenization + `cost_reduction_delta` feature added 2026-08) grows
  `DYNAMIC_FEATURE_DIM` instead — either way `TOKEN_FEATURE_DIM` → the per-card
  feature vector → each encoder's `input_proj.weight` changes shape, but
  `vocab_size` stays unchanged, so the vocab assert does NOT catch it and the league
  crashes on the first checkpoint LOAD with `size mismatch for input_proj.weight`.
  If a league run aborts that way, either the catalog changed or `rl/features.py`'s
  own feature dim did → a `fresh start` is required. Note this now surfaces when
  RESUMING an existing league (its `live.pt` files carry the stale shapes), not at
  startup against a single frozen-stack file.

## 3. `fresh start`: wipe, then train

Wipe **the resolved league_dir** for this invocation (per the config precondition above
— `checkpoints/league/` ONLY in the genuine config-less case; e.g.
`checkpoints/4_deck_subleague_test/` for that config). Do not hardcode
`checkpoints/league/` here without checking. Keep `checkpoints/archive_2deck/` — an
unrelated top-level directory from an old experiment, NOT the same thing as the
per-deck `<league_dir>/<deck>/archive/` this wipe DOES need to clear (evicted-snapshot
history `rl.league.LeaguePool` now keeps instead of deleting — stale archived selves
left behind after a wipe would make a post-reset `vs_history` check compare the fresh
policy against pre-reset history, which is meaningless). Also wipe each deck's
`opponent_stats.json` — the PFSP win/loss tallies `LeaguePool` persists there; leaving it
would bias opponent sampling against a freshly reset, randomly-initialized policy using
history from the policy that no longer exists:

```
rm -f ../checkpoints/vocab.json \
rm -f ../<league_dir>/*/live.pt ../<league_dir>/*/mulligan.pt \
      ../<league_dir>/*/snapshot_*.pt ../<league_dir>/*/opponent_stats.json \
      ../<league_dir>/session.txt ../<league_dir>/progress.json \
      ../<league_dir>/metrics.jsonl
rm -rf ../<league_dir>/*/archive/
```

Then run the escalation loop (§5) to `TARGET_PER_DECK` games/deck.

## 4. `resume` (default)

The same loop without the wipe — league checkpoints self-resume and sessions increment.

## 5. The monitored escalation loop

Track `per_deck_cumulative = 0` (games EACH deck has played). Batch ladder, measured in
**games per deck per session**: `batch = 1, 2, 4, 8, 16, ...`, each step
`batch = min(batch * 2, 3000, TARGET_PER_DECK - per_deck_cumulative)`. The first batch
is the tiny **shakeout**. Stop when `per_deck_cumulative >= TARGET_PER_DECK` (i.e. as
soon as any deck reaches it). The 3000 cap is the max games-per-deck in a single batch.

For each batch:

1. **Map `batch` (games PER DECK this session) to script args.** The script gives every
   deck `n_iterations × games_per_iteration` games per session — the per-deck number,
   independent of how many decks there are — and plays `n_iterations × N_DECKS ×
   games_per_iteration` games in TOTAL. The total (not the per-deck number) is what sets
   run time, so estimate duration with `N_DECKS`. Read the ACTUAL per-deck counts back
   from the log afterward.
   - `python -u run_league.py --n-iterations <n_iter> --snapshot-every <snap>
     --n-workers <W>`, PLUS `--run-config <path> --league-config <path>` whenever a
     matching `training_configs/*.json` exists (see the precondition above) -- a config's
     own `checkpoint_opponent_rate` and `snapshot_every_games` then apply automatically;
     do not also pass `--checkpoint-opponent-rate` yourself in that case (the config's
     value IS the owner's already-made decision, baked in on purpose -- re-specifying it
     here would just be a second, redundant source of truth to drift out of sync). Only
     when there's genuinely no config for this league does `--checkpoint-opponent-rate`
     default to 0.0 (no checkpoint opponents, every game real-model-vs-real-model) --
     leave it unset in that config-less case unless the owner explicitly asks to
     reintroduce checkpoint-opponent diversity; don't pass a nonzero rate on your own
     initiative.
     `games_per_iteration` is no longer a flag (removed 2026-07-31) --
     run_league.py's `main()` always derives it as `gpi = max(1, W)`, one game per worker,
     which is what avoids the `n_games // n_workers` worker-starvation footgun a too-small
     gpi used to risk (see run_league.py's own comment on the benchmark that grounded this
     value). While `W = 1`: gpi=1, so `n_iter = batch` exactly -- no rounding, the cleanest
     possible mapping (divides the 1/2/4/8/16 shakeout ladder evenly by construction).
     Once `W = 6`: gpi=6, and `batch / gpi` no longer divides the doubling ladder evenly --
     same as before, stop tracking `batch` as a raw doubling sequence at that point and
     just double `n_iter` itself each step (1, 2, 4, 8, 16, ... -- picking up from whatever
     `n_iter` the batch that crossed the `W=1`->`W=6` seam used, not restarting at 1); past
     the seam `n_iter` IS the ladder, not a derived value. `snap = max(1, 200 // gpi)`
     — a FIXED ~200 games/deck between snapshots, independent of batch size, so early
     small batches don't flood the opponent pool with near-random, barely-trained
     early copies (gpi=1 -> snap=200; gpi=6 -> snap=33) -- used only in the config-less
     case; a config's own `snapshot_every_games` takes precedence when one applies.
     **Compute ramp**: `W = 1` (sequential CPU) for the shakeout and until one league
     batch giving each deck ≥ ~15 games has run clean; after that use `W = 6` (parallel
     collection) — the throughput sweet spot. **Pass `--device cuda`** for any real
     league batch. Measured 2026-08-19 on an RTX 5060 Ti, two arms from identical
     checkpoints with the same seed, run sequentially through the real
     `run_league.py` (768 games each): CPU 829.8s vs GPU 330.5s — **2.51x
     end-to-end**, from `ppo_update` alone running 3.37x faster (708.5s -> 210.0s).
     Collection is unchanged (121.1s vs 120.2s) because it is CPU-bound in both
     arms: rollout inference is batch-of-1 per decision, spread across the worker
     processes, and never touches the training device.

     Do NOT try to move collection onto the GPU — measured 2026-08-19, batch-of-1
     inference at realistic token counts is 1.8-2.6x SLOWER there (CPU
     1.1-1.6ms vs GPU a flat ~2.9ms, which is launch overhead: the model is far
     too small to amortise it). That is single-process; six collect workers
     sharing one GPU, each with its own CUDA context, would be worse again. The
     rule this leaves is simple — BATCHED work (ppo_update, the mulligan update)
     belongs on the GPU, per-decision work stays on CPU.

     This REPLACES the earlier "never use the GPU" directive, which reasoned that
     the per-update net + optimizer CPU<->GPU round-trip would cost more than the
     tiny matmuls save. The round-trip is real, but the update is ~85% of
     wall-clock and wins anyway. That directive set its own release condition — a
     benchmark saying otherwise — and this is it.

     Keep small batches sequential-CPU (`W = 1`, no `--device`) — easier to
     diagnose, and a shakeout is too short for the update to dominate.
2. **Launch harness-tracked, logged to a file**, so completion/errors auto-notify:
   run with `run_in_background: true`, as
   `python -u <cmd> > ../logs/<phase>_<batch>.log 2>&1; grep -q "session .* done"
   ../logs/<phase>_<batch>.log`. The trailing grep makes the COMMAND's exit code track
   real success — **a parallel `run_league` (`--n-workers > 1`) exits 1 on Windows from
   `ProcessPoolExecutor` teardown even when the session fully completed and checkpointed**
   ("session done" + saved live.pt, zero tracebacks, still exit 1). So
   never trust the raw python exit code for parallel runs; the grep wrapper (0 iff the
   session-done line was printed) is the reliable signal. Do NOT use `nohup ... &` — that
   detaches from the harness and loses the notification. One batch at a time; wait for
   the completion notification before starting the next.
3. **On the completion notification, health-check the log** (log content, NOT exit code):
   - the league's end-of-session summary line IS present,
     `session.txt` advanced, AND no `Traceback` / `Error` / `Exception` / `ALL-FALSE`.
     (Trailing `[target fizzle]` lines are normal late-flushed worker gameplay output.)
   - Read the actual PER-DECK games from the log (the per-mirror/per-deck `games=`
     counts, or total-games / `N_DECKS`) and add to `per_deck_cumulative`.
   - **Hang check**: if a run runs far longer than the per-game rate implies (≈10–15 s/
     game CPU) with no new log lines, treat it as a stall — inspect, and if truly hung,
     kill it and investigate before retrying.
4. **Log a double round-robin snapshot.** Once the batch
   is confirmed healthy (step 3) and `per_deck_cumulative` is updated, pause
   before escalating and run (foreground — it's fast relative to a training
   batch, and the loop is already sequential):
   `python -u run_league.py --eval --games 2 --league-config <same config
   used for this batch> --log ../logs/<league_name>_double_round_robin_
   <per_deck_cumulative>games.json`
   Every deck plays every deck, including mirrors, twice
   (`combinations_with_replacement` — a "double round robin"), using the
   CURRENT live checkpoints; `--eval` never trains or checkpoints, so this
   has no effect on the escalation ladder itself. The filename embeds the
   league's cumulative games/deck at that point, so successive snapshots
   never collide and stay ordered by training progress. Health-check the
   same way as a training batch (no Traceback/Exception/ALL-FALSE in its
   output) — a failure here is treated exactly like a training-batch
   failure (step 6 below): it's exercising the same engine and the same
   live checkpoints, so it's just as real a bug.
   Note the log path so it can be reported at the end (§6). Each game in the
   written log is self-labeled (`deck_a`/`deck_b` fields, not just a bare
   `game_index`) — open it in the webapp's `/replay` file picker to watch
   any specific pairing directly (e.g. `python src/webapp/app.py`, `/replay`).
5. **Healthy** → double `batch` (per the formula) and continue until
   `per_deck_cumulative >= TARGET`.
6. **Errored / crashed / hung** → do NOT escalate. Read the traceback, find the ROOT
   cause in the code (fix once where all callers route through, not the symptom),
   apply the fix, re-run the game-engine self-checks if the fix touched engine code,
   then re-run the SAME batch size and grep the new log specifically for that failure
   mode. Only resume doubling once that size is clean. This applies equally to a
   failure in step 4's round-robin snapshot.

## 6. Report

When training reaches its target (or you stop on a blocking error), summarize:
mode, N_DECKS, per-deck games trained, number of sessions/batches, any errors
found + fixed, `per_deck_cumulative` vs target, where the checkpoints live (the resolved
league_dir -- `<league_dir>/<deck>/live.pt`; do NOT assume `checkpoints/league/` when a
config's own `league_name` pointed elsewhere), and two DIFFERENT log sources, both worth
reporting:
- the double round-robin snapshot logs written by §5 step 4
  (`logs/<league_name>_double_round_robin_<per_deck_cumulative>games.json`) -- manual,
  triggered once per escalation step by this skill, for the replay viewer;
- `<league_dir>/metrics.jsonl` -- automatic, appended by `rl.league_runner`'s own
  `_run_session` on EVERY iteration of every session this skill ran (not just at
  escalation boundaries): per-iteration policy/value loss and entropy, and (once any
  deck has been through a snapshot cycle) a `vs_history` win-rate-vs-its-own-archived-
  past-self check at the end of each session -- plus, only when the resolved config
  carries a `gauntlet_league_name` (or `--gauntlet-league-name` was passed), a
  `vs_gauntlet` win-rate-vs-an-independently-trained-twin-league check on the same
  cadence, once that twin has a checkpoint for the deck. Run `python analysis/report_metrics.py
  <league_dir>` for a plain-text summary of it -- mention this to the owner rather than
  re-deriving trends from stdout scrollback by hand.

Never claim a run succeeded without having seen its "done" line and exit 0 in the log.

## Notes

- The deck roster is whatever `data/league_decks.json` lists, further narrowed by a
  config's own `roster` (or `--roster`) when one is used -- the skill reads `N_DECKS` at
  runtime, so a changed roster needs no change here (only a `fresh start` if it's the
  FULL vocab-defining roster that changed, since that changes the vocab). There is no
  per-deck selection flag beyond a config's `roster`/`train_decks` -- training always
  covers every deck in whatever roster is in effect.
- Reward is terminal (near-zero early is expected, not a failure — watch losses moving
  and games completing instead): league play uses `deploy_reward_v6` (a FLAT +1 win / -1 loss — the
  cleanup-discard sloppiness penalty `q` that v3/v4 carried is gone from both bands as of
  2026-08-11 — see `rl/rewards.py`). `deploy_reward_v6` also wraps a dense, per-transition mana-burn penalty
  (`with_dense_mana_burn_penalty`, whole-game cap 1.5 against a per-turn curve weighted to
  0.5 — v5 used 1.5 and saturated that cap in 64-71% of games, making the penalty a flat
  toll rather than a gradient), charged to
  the WINNER only — a losing seat pays exactly -1.0 however it played,
  which is what fixed the "losing quietly is cheaper than trying" asymmetry that drove
  agents into doing nothing. It reads
  a PER-PIP single-pip-tagged subset of burnt mana (`PlayerState.
  mana_burnt_this_turn_single_pip`) that excludes
  board-state-scaled burst sources (Priest of Titania, Overgrown Battlement) from the
  penalty by construction, pip by pip, rather than by whole-phase exclusion -- an earlier
  unconditional version was tried and reverted in 2026-08 over an archetype-biased
  regression, then a whole-phase-exclusion fix, then this per-pip version (see
  `rl/rewards.py`'s own comment on `deploy_reward_v2` for the full history) -- if a
  future session finds it unwired, that's a deliberate regression to check the reasoning
  on, not the norm.
- Each session checkpoints at its end and resumes on the next run, so escalating across
  many separate sessions accumulates training without loss (a mid-session crash loses
  only that session's uncheckpointed games).
