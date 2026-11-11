# Training-quality improvement plan (opened 2026-11-11)

Why this exists: the 11-deck main league's gains were concentrated before ~50k games/deck, and the A/B that
trained further from the final checkpoint showed no gain (see `docs/OVERLAP_AB_RESULTS.md` and the project memory).
This file tracks every mechanism we are trying, in order, with its status, so a result is never separated from the
question it answers. All experiments run on the 4-deck league (`training_configs/small_league.json`).

Two questions run through all of it:
1. **How do we make training produce a stronger agent per game?** (mechanisms, section 2)
2. **How do we know when more training no longer helps?** (detectors, section 3)

## 1. Baseline (control)

`training_configs/small_baseline.json`, league `small_baseline`, 50,016 games/deck, from scratch, run 2026-11-10/11.
Progress indicator at the last ticks (score of current agents vs their ~0.6x-age selves, pooled, 95% Wilson):

| games/deck | label | score | interval | policy ratio |
|---|---|---|---|---|
| 34,944 | IMPROVING | 55.2% | 52.7-57.6 | 0.40 |
| 39,936 | CHURNING | 52.4% | 50.0-54.9 | 0.45 |
| 44,928 | IMPROVING | 53.2% | 50.7-55.6 | 0.46 |
| 49,992 | IMPROVING | 53.4% | 51.0-55.9 | 0.33 |

Reading: still gaining, slowly (a few points per ~30k games of age), policy still moving. Every other arm is judged
against this run with the same history checks.

## 2. Mechanisms

### 2.1 Best-response diagnostic -- RUNNING (elves first)
*Question:* is the league's own deck agent close to the best possible play against its frozen field, or is there
headroom that self-play is not finding?
*Method:* `src/analysis/eval/run_best_response.py` trains a from-scratch agent for one deck against the other three
decks' **frozen** final agents (no snapshots, no adaptation; checkpoint_opponent_rate 0; same PPO settings and
snapshot cadence as the baseline). `src/analysis/eval/check_best_response.py` plays the fresh agent's snapshots at several ages
against (a) the frozen baseline agent of that deck head-to-head and (b) the frozen field, next to the frozen agent's
own score against the same field.
*Reading it:* fresh agent plateaus **below** the frozen agent -> training from scratch under a fixed field is the
harder problem and the baseline agent is not leaving a large gap on the table; fresh agent **exceeds** it -> the
league has headroom that its own dynamics (cycling opponents, non-stationary field) do not reach, which makes
opponent-pool design / training schedule the lever.
*Caveats:* the fresh agent never meets the old agent of its own deck (its mirror opponent is itself), and the field
does not adapt, so it is a headroom probe, not a reproduction of league conditions.
*Stop rule (owner, 2026-11-11):* run elves only. If it shows no improvement over the frozen version, do **not** run
the other three decks unless a specific reason to is documented here first.
*Status:* started 2026-11-11 10:04, ~5.6 s/iteration, 2,083 iterations, estimated finish ~13:30. Early reading
(~1k games trained): frozen elves scores 46.7% [38.0,55.6] against the field; the fresh agent is at 31% and rising.
Checkpoint plan: read at ~5k, 10k, 20k, 30k, 50k games.

### 2.2 Mulligan model -- HYPOTHESIS, NOT YET VERIFIED
*Hypothesis (owner):* the mulligan net is a bottleneck / variance source, and not fruitful early in training.
*What the code and data already say (facts, not yet tests):*
- The mulligan net (`rl/model/mulligan.py`) is trained by its own REINFORCE on a +/-1 game outcome, flushed every 8
  iterations (~190 decisions per deck per flush in the 4-deck league), and **reads the main net's encoder without
  training it** (stop-gradient), so its input representation moves under PPO between flushes.
- In the baseline's final audit the learned mulligan almost never mulligans: mulligan rate 0.0% (mono_red_rally),
  0.2% (dmir_terror), 3.0% (rakdos_madness), 9.6% (elves). `analysis/mulligan_retrain` already documents a
  "collapse-to-always-keep" failure mode, and `mulligan.py` records `ENTROPY_COEF` being raised because the head kept
  re-collapsing.
- So late in training the mulligan is mostly a constant "keep"; its direct variance contribution is small there.
  Early on it is a different story: a random-init net mulligans often, which lowers hand size and raises game-outcome
  noise for the deck net's PPO signal (pregame decisions are not in PPO; their effect is environment noise to it).
  The baseline's audit starts at the first validation tick (~5k games), so the early period is unmeasured.
*Planned verification (cheap first, training last):*
1. **Early behaviour, no training:** run the mulligan audit on the baseline's archived snapshots at ages 192 ... 5k
   (snapshots carry their era-matched mulligan net): mulligan rate and keep-by-land-count vs age.
2. **Value of the learned mulligan at the end, no training:** play the final baseline agents with their learned
   mulligan vs a fixed rule (e.g. keep any 2-5-land 7; London-bottom by a fixed rule), same seeds. A null difference
   means the learned mulligan is worth ~nothing at convergence.
3. **Variance / early-training arm (training):** 4-deck arm with a fixed mulligan rule and no mulligan training for the
   first N games vs the baseline; read with the history checks, and evaluate both arms' deck nets under the same
   fixed rule so the comparison is not confounded by the mulligan itself.
*Rules note:* a fixed mulligan rule is a training-policy choice, not a game rule, but evaluation of an arm must say
which mulligan policy played. Nothing here changes the engine's London-mulligan implementation.

### 2.3 Optimiser / schedule ablations -- QUEUED
One change per arm on the 4-deck league, each read against the baseline with the history checks:
- learning-rate decay (baseline lr is flat 2e-4 for the whole run; entropy and batch schedules end by 50k);
- larger `games_per_iteration` (fewer, less noisy updates);
- `gamma = 1`, `lambda ~= 0.97` (baseline gamma-lambda is 0.9405, short for ~100-step games).

### 2.4 Capacity -- QUEUED (after 2.1-2.3)
- larger network (the baseline: d_model 64, 2 layers, trunk 128x128);
- one shared FiLM-conditioned network across decks instead of one net per deck.
Any structural change invalidates the recorded probe-state suite (`data/probe_states/small_league`, it refuses to
load on an action-table mismatch) and needs re-recording first.

## 3. Detecting that more training no longer helps

| detector | status | what it tells us | weakness |
|---|---|---|---|
| progress indicator (current vs ~0.6x-age selves, Wilson) | built, report-only, thresholds provisional | gain per fixed age gap | flickers IMPROVING/CHURNING near 50% once per-interval gain is below what ~1,600 games resolve |
| strength-vs-age ladder, own + field | built | shape of the learning curve, not just a label | self-referential: all selves are the league's own |
| policy-movement probe | built | moving vs frozen policy | movement is not progress |
| **best-response gap** (2.1) | running | headroom against a fixed field | one deck, one field |
| **fixed external yardstick** | proposed | strength vs agents that never train (e.g. `reference_agents/`, the old main-league nets, or a scripted bot) | needs agents that are not trivially beaten, or it saturates |
| **continue-vs-stop A/B** | done once (old league), proposed as a periodic gate | direct test of "does more training pay": branch from a checkpoint, train further, head-to-head | costs a training branch per test |
| stagnation rule | **deferred by owner** | would turn the above into an automatic stop | needs the ladder data from several runs before picking thresholds |

A candidate stagnation rule worth testing against these runs once they exist: stop when the indicator's interval
includes 50% at **two consecutive** ticks *and* the continue-vs-stop A/B gain's interval includes 0. Not adopted.

## 4. Change log
- 2026-11-11: created; baseline finished; best-response (elves) started; mulligan hypothesis written up, unverified.
