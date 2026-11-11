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

### 2.1 Best-response diagnostic -- DONE for elves
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
*Status:* DONE for elves (2026-11-11, 10:04 -> 13:29; ~4.8 s/iteration, 50k games). Other three decks NOT run (stop rule).
*Result* (greedy, 300 games per matchup; frozen = `small_baseline` elves at 50,016 games/deck; field = the other three frozen decks):

| agent | vs frozen field | head-to-head vs frozen elves |
|---|---|---|
| frozen baseline elves | 47.8% [44.5, 51.0] (n=900) | n/a |
| fresh @ 5k games | 29.0% | 35.3% |
| fresh @ 10k | 40.0% | 46.0% |
| fresh @ 15k | 36.0% | 44.0% |
| fresh @ 18k (n=300 cells) | 39.7% | 45.0% |
| fresh @ 25k | 40.2% [37.1, 43.5] | 47.3% [44.1, 50.6] |
| fresh @ 35k | 45.9% [42.7, 49.2] | 53.4% [50.2, 56.7] |
| fresh @ 42k | 44.0% [40.8, 47.3] | 46.3% [43.1, 49.6] |
| last 6 snapshots (45.9k-46.8k) | 46.3-50.3% | 53.7-55.7% (all six lower bounds >= 50.4%) |
| final live net (49,992) | 46.6% [43.3, 49.8] | 52.1% [48.8, 55.4] |
| **pooled, last 6 snapshots + live (n=6,300)** | **47.6% [46.4, 48.9]** | **54.2% [52.9, 55.4]** |

(The pooled intervals treat the seven agents as independent; they are adjacent snapshots 192 games apart, so the real
intervals are wider. The head-to-head edge is nevertheless consistent across all seven.)
*Reading:*
- **Against the field -- the best-response question -- the fresh agent only ties the frozen one** (47.6% vs 47.8%). It
  needed ~35-45k games to get there from scratch (about 40% by 10k, flat through ~25k, then up).
- It beats the frozen elves head-to-head by ~4 points (54.2%). That edge is probably mirror specialisation, not
  headroom: the fresh agent's only same-deck opponent in training was itself, whereas the frozen agent trained against
  a mixed population. It is not evidence of better play against the other decks.
- So for elves the league agent is at about the best-response level against its field, and nothing here says the
  league's opponent-pool dynamics cost it strength. That moves weight toward the optimisation / capacity / critic
  levers in section 2.5 and away from the opponent-pool lever.
- Snapshot-to-snapshot variation (53 / 46 / 55% head-to-head between ages) is larger than single-cell sampling noise:
  never read one snapshot.
*Caveats:* one deck, one field; elves is a deck whose frozen agent scores below 50% against this field, so "ties the
frozen agent" says little about how much a stronger deck could gain. The fresh agent trained 50k games against a field
that never adapts, which is an easier problem than the league's.
*Decision:* by the owner's stop rule (no clear improvement over the frozen agent against the field), the other three
decks were not run. Re-open only with a specific reason recorded here.
*Tools:* `check_best_response.py --live` also tests the final live net and prints the pooled row when more than one agent is tested.
Raw logs: `logs/br_elves.log`, `logs/br_elves_pooled_check.log`; weights `checkpoints/br_elves/`.

### 2.2 Mulligan model -- QUEUED FOR LATER, NOT ACTIONED (owner, 2026-11-11)
*Recorded as candidate improvements only: nothing below is to be run while the best-response test is in motion. Scope note: the owner's
interest is the whole agent (section 2.5 lists the rest); the mulligan net is one candidate among them, not the focus.*
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
*Candidate verification, if/when picked up (cheap first, training last):*
1. **Early behaviour, no training:** run the mulligan audit on the baseline's archived snapshots at ages 192 ... 5k
   (snapshots carry their era-matched mulligan net): mulligan rate and keep-by-land-count vs age.
2. **Value of the learned mulligan at the end, no training:** play the final baseline agents with their learned
   mulligan vs a fixed rule (e.g. keep any 2-5-land 7; London-bottom by a fixed rule), same seeds. A null difference
   means the learned mulligan is worth ~nothing at convergence.
3. **Variance / early-training arm (training):** 4-deck arm with a fixed mulligan rule and no mulligan training for the
   first N games vs the baseline; read with the history checks, and evaluate both arms' deck nets under the same
   fixed rule so the comparison is not confounded by the mulligan itself.
*Open question for the owner before step 3:* is a fixed mulligan rule acceptable as a training policy? (Not a game rule; the engine's London mulligan is unchanged.)
*Rules note:* a fixed mulligan rule is a training-policy choice, not a game rule, but evaluation of an arm must say
which mulligan policy played. Nothing here changes the engine's London-mulligan implementation.

### 2.3 Optimiser / schedule ablations -- QUEUED
One change per arm on the 4-deck league, each read against the baseline with the history checks:
- learning-rate decay (baseline lr is flat 2e-4 for the whole run; entropy and batch schedules end by 50k);
- larger `games_per_iteration` (fewer, less noisy updates);
- `gamma = 1`, `lambda ~= 0.97` (baseline gamma-lambda is 0.9405, short for ~100-step games).

#### 2.3a Learning-rate decay -- FIRST ARM, DESIGN AGREED 2026-11-11 (not started)
*Decided (owner):* decay target **2e-4 -> 2e-5, linear**, over the run's length; control replicate wanted; twin league skipped.
*Arms (4-deck league, from scratch, 25k games/deck each, run one after the other, ~5 h each):*
`small_lrdecay` (decay) and `small_control` (identical to `small_baseline`, no seed -- runs are unseeded, so it differs from the baseline by chance only).
The 25k decay horizon ends at 25k, so the arm is not a prefix of a 50k run; a 50k-horizon arm is a later run if this wins.
Twin skipped by config only (`create_training_league: false` in the arm configs); no code change for that.
*Reading (rule fixed before looking):* each arm's 25k snapshot vs the baseline's 25k snapshot head-to-head, pooled over the 4 decks
(900 games per deck); the control-vs-baseline number is the noise floor. Decay wins only if its lower bound is above 50%, above the
control's value, and the sign agrees in >= 3 of 4 decks. Secondary: each arm's decks vs the frozen baseline-final field;
mechanism check from `metrics.jsonl` (late `approx_kl` / clip fraction / entropy trend vs baseline).
*Needs:* an lr schedule in the training loop (not present: lr is a flat 2e-4), logged per iteration.

### 2.4 Capacity -- QUEUED (after 2.1-2.3)
- larger network (the baseline: d_model 64, 2 layers, trunk 128x128);
- one shared FiLM-conditioned network across decks instead of one net per deck.
Any structural change invalidates the recorded probe-state suite (`data/probe_states/small_league`, it refuses to
load on an action-table mismatch) and needs re-recording first.

### 2.5 Whole-agent levers -- CANDIDATES, QUEUED (none run yet)
Ordered by how much existing evidence points at them. Evidence is from the old 11-deck run and the 4-deck baseline;
"?" means we have no measurement yet.

| lever | baseline value | why it is a candidate | evidence status |
|---|---|---|---|
| learning rate | flat 2e-4 all run | late approx_kl and clip fraction rise while gains shrink -> updates may be too large near convergence | seen in old run; no decay arm yet |
| games per update (`games_per_iteration`, minibatch) | 24 games/deck/iter, batch ramps 4 -> 16 by ~50k | larger batches cut gradient noise; the ramp ends by 50k and nothing grows after | ? |
| gamma / lambda | gamma-lambda product 0.9405 | games are ~100 decision steps; terminal reward may barely reach early decisions | zero horizon timeouts; effect unmeasured |
| value fit | explained variance ~0.82, flat | a flat critic ceiling can cap advantage quality; shared-trunk value head, value-loss weight, separate critic are all untried | EV flat across the run |
| entropy schedule | 0.02 -> 0.005 by 50k, then flat; policy entropy 0.5 -> 0.28 | exploration pressure ends exactly when gains thin out; either too early or irrelevant | ? |
| PPO epochs / clip / target_kl | 4 epochs, early-stopped by target_kl | clip fraction rising late suggests the trust region is binding | seen in old run |
| reward | `deploy_reward_v6` (shaped) | shaping can bias play away from winning; a flat win/loss arm tests it | ? (the mulligan net already uses +/-1) |
| opponent pool | PFSP, rate-0.15 frozen-snapshot opponents, 32-snapshot window, PFSP power default | cycling decks on a rising population; pool design decides what the agent is forced to beat | best-response test (2.1) bears on this |
| network capacity | d_model 64, 2 layers, 4 heads, trunk 128x128 | EV flat and slow late gains could be capacity-limited | ? (queued, 2.4) |
| shared FiLM network across decks | one net per deck | 4x the data per parameter set; transfer between decks; also changes the "decks cycle" dynamic | ? (queued, 2.4) |
| sequence model | GRU over own previous actions | longer or attention memory might matter for combo decks | ? |
| data efficiency | salvaged transitions from other decks' games | check the salvage is not skewing the training distribution | ? |

Method note: each arm costs ~9 h at 50k games/deck on this machine, and the history checks resolve only a few points of
strength. A cheaper screening pass (e.g. 20-25k games/deck, where the baseline gained the most) followed by full-length
runs of the survivors is the likely shape; one change per arm, same seed family, read against `small_baseline`. Not scheduled.

## 3. Detecting that more training no longer helps

| detector | status | what it tells us | weakness |
|---|---|---|---|
| progress indicator (current vs ~0.6x-age selves, Wilson) | built, report-only, thresholds provisional | gain per fixed age gap | flickers IMPROVING/CHURNING near 50% once per-interval gain is below what ~1,600 games resolve |
| strength-vs-age ladder, own + field | built | shape of the learning curve, not just a label | self-referential: all selves are the league's own |
| policy-movement probe | built | moving vs frozen policy | movement is not progress |
| **best-response gap** (2.1) | done for elves: gap ~0 against the field | headroom against a fixed field | one deck, one field; took ~35-45k games to reach parity |
| **fixed external yardstick** | proposed | strength vs agents that never train (e.g. `reference_agents/`, the old main-league nets, or a scripted bot) | needs agents that are not trivially beaten, or it saturates |
| **continue-vs-stop A/B** | done once (old league), proposed as a periodic gate | direct test of "does more training pay": branch from a checkpoint, train further, head-to-head | costs a training branch per test |
| stagnation rule | **deferred by owner** | would turn the above into an automatic stop | needs the ladder data from several runs before picking thresholds |

A candidate stagnation rule worth testing against these runs once they exist: stop when the indicator's interval
includes 50% at **two consecutive** ticks *and* the continue-vs-stop A/B gain's interval includes 0. Not adopted.

## 4. Change log
- 2026-11-11: created; baseline finished; best-response (elves) started; mulligan hypothesis written up, unverified.
- 2026-11-11: best-response (elves) finished: ties the frozen agent against the field (47.6% vs 47.8%), +4 head-to-head (mirror effect); other three decks not run.
- 2026-11-11: owner agreed the first arm: lr decay 2e-4 -> 2e-5 linear, with a control replicate, no twin league (2.3a).
- 2026-11-11: owner: scope is the whole agent, not the mulligan net; mulligan work recorded as queued (2.2); added the whole-agent lever table (2.5). No arm other than best-response to run until it reports.
