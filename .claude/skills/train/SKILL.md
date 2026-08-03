---
name: train
description: Run monitored, escalating self-play training for this repo's MTG deck agents (the pretrain shared stack and/or the per-deck league DeckNetworks). Use when the user asks to train, run training, train the agents/models, start a training run, or "/train". Parses the invocation message for "fresh start" (wipe all checkpoints, run the full pipeline from scratch) and "per-deck only"/"league only" (train just the per-deck models). Default with no flag is league-only. Starts every phase with a tiny batch, verifies it is healthy, then doubles the batch each clean step.
---

# train — monitored, escalating agent training

Orchestrates the EXISTING training scripts (`src/run_pretrain.py`, `src/run_league.py`).
Never reimplements training. All commands run from `src/`.

The whole point is **safety through small, monitored, escalating batches**: start
each phase tiny, confirm it ran clean (exit 0, expected "done" line, no traceback,
real progress — not hung), then roughly **double the games** each clean step. On any
error, STOP escalating, find the root cause, fix it, re-run the SAME size, and watch
specifically for that failure again before growing.

## 1. Parse the invocation message

- **`fresh start`** (also "from scratch", "start over", "wipe", "retrain everything")
  → MODE = `fresh`: wipe checkpoints and run the FULL pipeline (regenerate vocab →
  pretrain → `--freeze` → league).
- **`per-deck only`** / **"league only"** / "only the deck models" → MODE = `league`.
- **No flag** → MODE = `league` (the default). Never wipes anything.
- **Per-deck game target**: the LEAGUE trains until each deck has played ~this many
  games. If the message names a count ("train each deck 8000 games", "8000 per deck")
  use it as `TARGET_PER_DECK`; default `TARGET_PER_DECK = 3000`. Stop as soon as ANY
  deck's cumulative game count reaches it (in league mode the decks advance together and
  arrive ~simultaneously; "any" just guards against one racing ahead).
- **Pretrain budget** (fresh only): per-deck games the shared stack sees before it is
  frozen — a prerequisite, not the goal. If named ("pretrain 300 per deck") use it;
  default `PRETRAIN_PER_DECK = 100`.

Restate the parsed MODE / TARGET_PER_DECK / PRETRAIN_PER_DECK (and N_DECKS) back to the
user in one line before starting.

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
- **League needs a frozen stack.** If MODE = `league` and
  `../checkpoints/shared_stack_frozen.pt` does NOT exist, stop and tell the user:
  "No frozen shared stack — run `/train fresh start` (or pretrain) first." Do not try
  to run the league without it (`run_league.load_frozen_stack` hard-asserts it).
- A vocab change (e.g. new cards/tokens) makes old checkpoints dimensionally
  incompatible. If any run aborts with a `vocab_size` / roster-mismatch assert, that
  is the signal to do a `fresh start`.
- **Feature-dim check (catches new cards even before they join the pool).** Adding a
  card with a NEW keyword/type grows `STATIC_FEATURE_DIM` → the per-card feature vector →
  the frozen stack's `input_proj.weight` — but leaves `vocab_size` unchanged, so the
  vocab assert does NOT catch it and league crashes at startup with
  `size mismatch for input_proj.weight`. If a league run aborts that way, the catalog
  changed since the freeze → a `fresh start` is required (re-pretrain rebuilds the stack
  at the new feature dim).

## 3. `fresh start`: wipe, then full pipeline

Wipe (keep `checkpoints/archive_2deck/` — it is an unrelated archive):

```
rm -f ../checkpoints/vocab.json ../checkpoints/pretrain_shared_stack.pt \
      ../checkpoints/shared_stack_frozen.pt
rm -f ../checkpoints/league/*/live.pt ../checkpoints/league/*/snapshot_*.pt \
      ../checkpoints/league/session.txt
```

Then, in order:
1. **Pretrain phase** — escalate to `PRETRAIN_PER_DECK` games/deck (§5, phase=`pretrain`).
2. **Freeze** — once pretrain is healthy and at/near budget, freeze the shared stack:
   `python -u run_pretrain.py 1 1 --freeze` (a tiny final session that also writes
   `../checkpoints/shared_stack_frozen.pt`). Confirm that file now exists.
3. **League phase** — escalate to `TARGET_PER_DECK` games/deck (§5, phase=`league`).

## 4. `league` (default): league phase only

Just run the League phase (§5, phase=`league`) to `TARGET_PER_DECK` games/deck, resuming
from whatever league checkpoints already exist (they self-resume; sessions increment).

## 5. The monitored escalation loop (used by both phases)

Track `per_deck_cumulative = 0` (games EACH deck has played this phase). Batch ladder,
measured in **games per deck per session**: `batch = 1, 2, 4, 8, 16, ...`, each step
`batch = min(batch * 2, 3000, TARGET - per_deck_cumulative)` where TARGET is
`PRETRAIN_PER_DECK` or `TARGET_PER_DECK` for the phase. The first batch is the tiny
**shakeout**. Stop the phase when `per_deck_cumulative >= TARGET` (i.e. as soon as any
deck reaches TARGET). The 3000 cap is the max games-per-deck in a single batch.

For each batch:

1. **Map `batch` (games PER DECK this session) to script args.** Each script gives every
   deck `n_iterations × games_per_iteration` games per session — the per-deck number,
   independent of how many decks there are — and plays `n_iterations × N_DECKS ×
   games_per_iteration` games in TOTAL. The total (not the per-deck number) is what sets
   run time, so estimate duration with `N_DECKS`. Read the ACTUAL per-deck counts back
   from the log afterward.
   - **pretrain**: `python -u run_pretrain.py <n_iter> <gpi>`, choosing `n_iter`, `gpi`
     so `n_iter * gpi ≈ batch`. Shakeout / small: `n_iter = 1`, `gpi = batch`. Large:
     `n_iter = 4`, `gpi = max(1, round(batch / 4))` (spreads updates).
   - **league**: `python -u run_league.py --n-iterations <n_iter> --snapshot-every <snap>
     --n-workers <W>`. `games_per_iteration` is no longer a flag (removed 2026-07-31) --
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
     early copies (gpi=1 -> snap=200; gpi=6 -> snap=33).
     `--checkpoint-opponent-rate` defaults to 0.0 (no checkpoint opponents at all, every
     game real-model-vs-real-model) — leave it unset unless the owner explicitly asks to
     reintroduce checkpoint-opponent diversity; don't pass a nonzero rate on your own
     initiative.
     **Compute ramp**: `W = 1` (sequential CPU) for the shakeout and until one league
     batch giving each deck ≥ ~15 games has run clean; after that use `W = 6` (parallel
     collection) — the throughput sweet spot. **NEVER use the GPU** — do NOT pass
     `--gpu-threshold` (leave it unset so every update stays on CPU). Owner directive:
     treat GPU as axiomatically slower at this model size — the per-update net +
     optimizer CPU↔GPU round-trip costs more than the tiny matmuls save. Not to be used
     until the owner's own benchmark says otherwise. Keep small batches sequential-CPU —
     easier to diagnose.
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
   - a `session N done` (pretrain) or league end-of-session summary line IS present,
     `session.txt` advanced, AND no `Traceback` / `Error` / `Exception` / `ALL-FALSE`.
     (Trailing `[target fizzle]` lines are normal late-flushed worker gameplay output.)
   - Read the actual PER-DECK games from the log (the per-mirror/per-deck `games=`
     counts, or total-games / `N_DECKS`) and add to `per_deck_cumulative`.
   - **Hang check**: if a run runs far longer than the per-game rate implies (≈10–15 s/
     game CPU) with no new log lines, treat it as a stall — inspect, and if truly hung,
     kill it and investigate before retrying.
4. **League phase only — log a double round-robin snapshot.** Once the batch
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
   live checkpoints, so it's just as real a bug. Skip this step for the
   pretrain phase — pretrain has no cross-deck round-robin concept (mirror-
   only self-play into one shared stack, nothing to pair decks against).
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

When the phase(s) reach their target (or you stop on a blocking error), summarize:
mode, N_DECKS, per-deck games trained per phase, number of sessions/batches, any errors
found + fixed, `per_deck_cumulative` vs target, where the checkpoints live
(`checkpoints/shared_stack_frozen.pt`, `checkpoints/league/<deck>/live.pt`), and (league
phase) the list of double round-robin snapshot logs written by §5 step 4
(`logs/<league_name>_double_round_robin_<per_deck_cumulative>games.json`) so the owner
can jump straight to a specific checkpoint's games in the replay viewer. Never
claim a run succeeded without having seen its "done" line and exit 0 in the log.

## Notes

- The deck roster is whatever `data/league_decks.json` lists (currently 5); the skill
  reads `N_DECKS` at runtime, so more decks need no change here (only a `fresh start`,
  since a new deck changes the vocab). There is no per-deck selection flag — training
  always covers every deck in the roster. "Per-deck only" means the league DeckNetworks
  vs. the pretrain shared stack, NOT a subset of decks.
- Reward is pure win/loss (`action_count_win_reward_200_floor02`); a near-zero
  mean_reward early is expected (sparse signal, barely-trained mirror policies) and is
  NOT a failure — watch losses moving and games completing instead.
- Each session checkpoints at its end and resumes on the next run, so escalating across
  many separate sessions accumulates training without loss (a mid-session crash loses
  only that session's uncheckpointed games).
