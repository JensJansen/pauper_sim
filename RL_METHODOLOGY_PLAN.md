# RL Methodology Plan — 2026-08-12

Status: **PLAN ONLY. Nothing here is implemented.**

Supersedes the diagnosis in `MEMORY.md` and extends `TRAINING_IMPROVEMENT_OPTIONS.md`
(2026-08-06), whose league-dynamics hypotheses this document partly overturns.

Produced from six parallel code-reading investigations, each briefed to argue
against its own workstream. **Five of six returned "no" or "last" on the
workstream they were asked to champion.** That is the single most important
fact in this document: the interventions we were about to build are not the
ones the evidence supports.

---

## 0. TL;DR

The league has trained 4 decks to 60,001 games/deck and produced **no
measurable improvement since ~6,000 games/deck**. The cause is **not** what the
last several days of work assumed.

- **It is not the reward function.** v5/v6 fixed real bugs (passivity 11.2%→0%,
  cap saturation 71%→36%, hoarding, entropy collapse). Those fixes hold. They
  were not the plateau.
- **It is not self-play cycling.** Cycling requires an intransitive strategy
  space. The measured matchup matrix is *strictly transitive* and 9 of 12
  cross-deck pairings moved by 0.000 excess SD across 50,000 games. The
  population is not going in circles; it is sitting still.
- **It is not obviously a capacity ceiling.** A saturated model converges. This
  one takes ~670,000 large, trust-region-clipped Adam steps per deck and does
  not move.

What the evidence actually supports: **three configuration defects that
silently disabled the training schedules, an entropy coefficient 2.5–10× below
the empirically best range, a credit-assignment horizon that hides the first
half of every game from the reward, an update signal-to-noise ratio of six
games per policy update, and instrumentation too coarse to have noticed any of
it.**

**Revised 2026-08-13 after the Wave 0 measurements (§1A).** Evidence is now
ranked by *where it comes from*, and measurements on this run outrank results
imported from a benchmark:

| rank | finding | source |
|---|---|---|
| 1 | **This is a regression, not a plateau.** The newest snapshot is strongest for 1 deck of 4; elves and rakdos both **lose to their own 200-game-old selves** | round robin, §1A.3 |
| 2 | **The one deck with a balanced training distribution is the one deck that improved.** The other three spend 58–77% of training in matchups they win <25% of | `opponent_stats.json`, §1A.4 |
| 3 | Advantage normalization (`ppo.py:122`) rescales those degenerate batches' noise to **unit variance** before ~67 Adam steps are taken on it | code, verified |
| 4 | Updates too large for the evidence behind them: `approx_kl` 0.026 → 0.044 vs `target_kl=0.03`, `clip_fraction` 0.18–0.25, `epochs_run` 3.8 → 2.8, on **6 games per update** | our telemetry, §1A.2 |
| 5 | `value_loss` ~0.01 and flat — *not* a healthy critic; the signature of the §4.1 horizon collapse | our telemetry |
| 6 | BUG 1 / BUG 3 | code, verified |
| 7 | `ent_coef` 2.5–10× below the empirically best range | arXiv:2502.08938 |

**BUG 2 is no longer a bug on a list. It is the leading explanation**, with a
verified mechanism (2 + 3 together) and a cheap falsification test.

The earlier claim that *"the single highest-evidence intervention available is
one line: raise `ent_coef`"* is **withdrawn**. It remains a good idea (§4.3),
but it is imported evidence, and our own instrumentation now points first at
**what the agent trains against** and **how much real evidence backs each step**.

Cycling is **ruled out by measurement**: 0 significant 3-cycles in 80 triples,
all four matrices transitive with residuals at or below their own noise floors.

Sequence: measure the baseline → fix what is broken → make the instruments
readable → change two orthogonal knobs → run the experiment that discriminates
between the remaining hypotheses → only then choose a large workstream.

---

## 1. The evidence

Cross-league eval vs the frozen gauntlet twin, 800 games per checkpoint
(SE 1.77pp), from `logs/cross_league_eval_v*.json`:

| deck | 10k | 20k | 30k | 40k | 50k | 60k |
|---|---|---|---|---|---|---|
| rakdos_madness | 49.5 | 62.5 | 62.5 | 61.0 | 68.5 | 55.0 |
| dmir_terror | 16.5 | 21.5 | 21.5 | 26.0 | 25.0 | 14.5 |
| elves | 29.0 | 20.5 | 16.0 | 32.0 | 28.0 | 20.5 |
| mono_red_rally | 78.0 | 85.0 | 80.0 | 87.0 | 82.0 | 87.5 |
| **aggregate** | **43.2** | **47.4** | **45.0** | **51.5** | **50.9** | **44.4** |

z(10k→60k) = **+0.45**. Mirror-only is 46.5% — statistically even against a twin
frozen at ~10,005 games/deck, i.e. one sixth the training.

`vs_history`, pooled over the whole run, **split by label** (these were being
read as a single averaged number, which flattered the result):

| deck | vs its **200-game-old** self (`archive_oldest`) | vs its 6,400-game-old self (`active_oldest`) |
|---|---|---|
| rakdos_madness | 52.6% | 52.4% |
| dmir_terror | 62.4% | 52.9% |
| elves | **50.9%** | 48.1% |
| mono_red_rally | 72.2% | 49.7% |

`archive_oldest` is **permanently `snapshot_0`** — `_list_snapshot_ids` returns
sorted ids and `_run_eval_vs_history` takes `[0]`, while eviction is oldest-first,
so the archive minimum is 0 forever (`src/rl/league_runner.py:533-540,585-587`).
At `snapshot_every_games: 200` that is a policy from ~200 games.

**Elves, after 60,001 games, cannot beat its own 200-game-old self.**

Early-vs-late z-test across every existing instrument: **1 of 12 deck×instrument
cells moves.** 40,000 games/deck produced no detectable change on anything we
were already recording.

> **Superseded by §1A.3 — and an apparent contradiction between the two
> resolved as a POOLING ARTIFACT, committed while writing this document.**
>
> The round robin measures snapshot_289 vs snapshot_0 at 42% (dmir). Pooling
> the last six `vs_history archive_oldest` records gave 57.5%, and that 15pp
> gap was written up here as a possible instrument failure. It was not. Read
> **per record instead of pooled**, dmir vs snapshot_0 runs:
>
> ```
> s24:60% s25:80% s26:85% s27:65% s28:80% s29:60% s30:60%
> s31:80% s32:75% s33:55% s34:50% s35:60% s36:60% s37:45%
> ```
>
> A clear decline from ~85% to 45%. The round robin's 42% (n=100) and the
> latest record's 45% (n=20) agree (z=0.55). **The two instruments never
> disagreed; pooling a monotone decline into one average manufactured the
> disagreement** — precisely the failure §2.1 describes and Step 1.1 exists to
> fix, reproduced here by the author of both.
>
> Two consequences, both raising the priority of Phase 1:
>
> 1. **`vs_history archive_oldest` recorded the regression all along.** A second
>    independent instrument, already on disk since session 24, corroborates
>    §1A.3. Nobody read it as a trend because `analysis/report_metrics.py` prints one
>    record at a time and every summary of it in this project pooled.
> 2. **Confirming §1A.3 needed no new compute at all.** The 4,296s round robin
>    was worth running — it gives the full matrix, the cycle test, and Elo — but
>    the regression itself was legible in `metrics.jsonl` for 13 sessions.
>
> Separately confirmed by `logs/rr_live_check.json`: `live` is **worse than
> snapshot_289**, which is worse than snapshot_0 (Elo +46 / −4 / −42, rising on
> 0/2 steps). The decline continues past the last snapshot.

### Confounds in the series above, for the record

- It spans a reward change: `deploy_reward_v5` at 10k/20k, `deploy_reward_v6`
  at 30k+. It is not one continuous algorithm run.
- The 10k point used `seed=42`; every later point used `seed=0`.

---

## 1A. Wave 0 measurements — 2026-08-13

### 1A.1 Absolute scale: the random-init anchor

`src/analysis/run_anchor_eval.py`, 150 games/cell, greedy (the sampled column agreed
throughout — argmax over randomly initialized heads did **not** degenerate, so
that concern is retired). Opponent is an untrained `DeckNetwork` on the real
frozen stack: the first fixed point in this repo whose strength is known by
construction.

| deck | snapshot_0 (~200 games) | snapshot_116 (~23k) | live (60,001) | 0 → live |
|---|---|---|---|---|
| dmir_terror | 78.0% | 76.7% | 83.3% | +5.3pp, z=+1.17 |
| elves | 84.0% | 88.0% | 74.0% | **−10.0pp, z=−2.13** |
| mono_red_rally | 97.3% | 100.0% | 100.0% | +2.7pp, z=+2.01 *(ceiling)* |
| rakdos_madness | 100.0% | 100.0% | 98.7% | −1.3pp, z=−1.42 |

**A ~200-game policy already beats a random-init policy 78–100% of the time,
and another 59,800 games/deck bought nothing measurable.**

*Read this with its limits.* The anchor is a **floor** instrument. Beating a
random policy only requires "play lands, cast things, attack," so it saturates
as soon as a policy stops being embarrassing — `mono_red_rally` and
`rakdos_madness` are pinned at 97–100% at every vintage and the anchor simply
cannot speak about them (the +2.7pp / z=+2.01 on mono_red is a ceiling artifact,
not a result). Even on the two decks with headroom, some of the residual 17–26%
is draw variance a perfect policy would also lose. The anchor is consistent with
"learning stopped early"; it does not establish it. **`analysis/run_snapshot_round_robin.py`
is the instrument that does** — head-to-head between trained policies, no ceiling.

Elves reads −10.0pp greedy but −1.3pp sampled. A policy whose *argmax* degraded
while its *distribution* held is a specific, checkable thing; not worth
interpreting ahead of the round robin.

### 1A.2 PPO telemetry — 40,104 iterations already on disk

Free to compute; nothing had ever read it. First decile vs last:

| deck | value_loss | entropy | approx_kl | clip_frac | epochs_run |
|---|---|---|---|---|---|
| dmir_terror | 0.0156 → 0.0111 | 0.677 → 0.433 | 0.0269 → 0.0376 | 0.25 → 0.20 | 3.84 → 2.88 |
| elves | 0.0118 → 0.0090 | 0.624 → 0.442 | 0.0255 → 0.0444 | 0.23 → 0.23 | 3.88 → 2.77 |
| mono_red_rally | 0.0338 → 0.0329 | 0.477 → 0.336 | 0.0311 → 0.0452 | 0.22 → 0.18 | 3.58 → 2.65 |
| rakdos_madness | 0.0251 → 0.0239 | 0.589 → 0.386 | 0.0330 → 0.0443 | 0.25 → 0.19 | 3.69 → 2.79 |

**(a) The steps are too large for the evidence behind them.** `approx_kl`
exceeds `target_kl=0.03` at *both ends* of the run and rises to 0.044;
`epochs_run` falling 3.8 → 2.8 confirms early stopping actually fires, and
increasingly. `clip_fraction` sits at 0.18–0.25 throughout. All of this on
`games_per_iteration = max(1, n_workers) = 6`.

*Caveat, stated because it matters:* part of the `approx_kl` rise may be
mechanical — KL between two peaked distributions is larger for the same logit
change, and entropy fell 30% over the same window. The robust claim is not "the
steps got bigger" but "**the steps exceeded the trust region for the entire
run, at both ends.**"

**(b) The policy converged, to nothing better.** Entropy fell ~30% on every
deck. Convergence plus no strength gain is the whole finding in one line.

**(c) `value_loss` looks healthy and is not evidence that it is.** Plain MSE
against GAE returns, no normalization (`ppo.py:206`), sitting at 0.01–0.03 and
flat. Taken at face value that is a critic explaining nearly all return
variance — implausible in a game with this much draw variance. The consistent
reading is §4.1: at 50 decisions out the advantage weight is 0.047, so
early-game returns are all ≈0 and trivially predictable. **A low value loss
here is a symptom of the horizon collapse, not a refutation of it.**

**Does this already settle H1 vs H2?** Partly, and in H1's direction: a learner
at its capacity ceiling converges and *settles*, whereas this one keeps taking
trust-region-saturating steps for 10,000 updates. But (a)'s caveat weakens that,
so it is a lean, not a verdict. Step 2.3 still runs.

### 1A.3 Snapshot round robin — the decisive measurement

`src/analysis/run_snapshot_round_robin.py`, 6 vintages × 15 pairs × 100 games × 4 decks,
sides swapped. 4,296s. Elo from a Bradley-Terry MM fit; residual printed against
the residual sampling noise alone would produce.

| deck | 0 | 58 | 116 | 174 | 232 | 289 | span | rising | residual vs floor | sig. 3-cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| dmir_terror | −32 | −0 | **+56** | −47 | +54 | −32 | 103 | 3/5 | 4.00 / 3.92 | 0 |
| elves | +1 | **+89** | +24 | −24 | −35 | −56 | 145 | 1/5 | 2.63 / 3.90 | 0 |
| mono_red_rally | −141 | −28 | +38 | −3 | +34 | **+100** | 241 | 4/5 | 3.39 / 3.79 | 0 |
| rakdos_madness | −10 | +12 | −24 | +41 | **+58** | −76 | 133 | 3/5 | 2.44 / 3.92 | 0 |

**(a) Cycling is dead.** **0 significant 3-cycles in 80 triples.** Every matrix
is transitive with a residual at or below its own noise floor. §6's rejection of
R-NaD/NFSP on premise-failure grounds is now measured, not argued.

**(b) It is not a plateau. It is a regression.** The newest snapshot is the
strongest for exactly one deck of four. Peaks sit at snapshot **116**
(dmir, ~23k games), **58** (elves, ~11.6k), **289** (mono_red), **232**
(rakdos, ~46k). Head-to-head, newest vs oldest:

- **elves: 289 loses to its own 200-game-old self, 40%–60%.**
- **rakdos_madness: 289 loses to snapshot_0, 33%–67%** — and it crashed 134 Elo
  in the last 11,400 games alone.
- dmir_terror: −32 → −32, no net change over 60,001 games.
- mono_red_rally: **+241 Elo**, 289 beats 0 by 77%–23%.

This retires "plateau" as the framing. Training is actively destroying policy
strength on three of four decks.

**(c) The anchor's ceiling caveat was right, and its elves reading was right.**
mono_red's +241 Elo was invisible to the anchor (97–100% at every vintage);
rakdos's collapse was invisible for the same reason. Elves' −10.0pp / z=−2.13,
which §1A.1 declined to interpret, is confirmed: elves genuinely got worse.
`vs_history` cross-validates too — it reported mono_red 72.2% vs
`archive_oldest`; this measures 77%.

### 1A.4 Why — the training distribution, measured

From `opponent_stats.json`, the real accumulated shares and win rates:

| training deck | share in matchups won **<25%** | mirror share | Elo 0→289 |
|---|---|---|---|
| **mono_red_rally** | **0.0%** | **51.3%** | **+241** |
| rakdos_madness | 58.0% | 25.5% | −66 |
| elves | 69.8% | 11.8% | −57 |
| dmir_terror | 76.5% | 14.7% | 0 |

**The one deck with a balanced training distribution is the one deck that
improved.** elves spends 36.9% of its games at a **1.1%** win rate and 33.0% at
5.9%; rakdos spends 58.0% at 15.8%. mono_red, which beats everyone, has every
PFSP weight pinned near `PFSP_FLOOR` and therefore trains on a near-uniform
distribution with 51.3% mirrors.

**The mechanism is BUG 2 plus `ppo.py:122`.** In a matchup lost 96% of the time
nearly every trajectory returns −1, so the raw advantage spread is almost
entirely noise — and `adv = (adv - adv.mean()) / (adv.std() + 1e-8)` rescales
that noise to **unit variance**, unconditionally, whereupon ~67 Adam steps are
taken on it. Unwinnable matchups do not merely waste compute; normalization
launders their noise into a full-scale gradient. This compounds with §1A.2(a):
6 games per update, of which ~70% are foregone conclusions.

*Confound, stated:* deck strength and PFSP share are causally linked — being
strong is *why* mono_red avoids unwinnable matchups — so "good distribution
causes improvement" cannot be separated from "mono_red is an easier learning
problem" on n=4 decks. What makes this more than a correlation is that the
mechanism is independently verified in code and BUG 2 was found independently
of these results. The falsification test is Step 0.2's own mechanism check.

---

## 2. Critical assessment of the existing methodology

Stated plainly, because the request was to be critical.

**2.1 The instrumentation cannot detect the thing it exists to detect.**
Per-session `vs_gauntlet` is n=20 → SE 11.2pp, 95% CI ±21.9pp. Thirteen such
readings are taken per session (4 decks × 3 instruments + heuristic) at a
measured 1.64 s/game — **~7.1 min/session, ~4.5 h over the run, ~19% of total
compute** — and spent as thirteen unusable readings instead of one usable one.
`analysis/report_metrics.py` prints them one record at a time, so no pooled number, no
confidence interval, and no trend test has ever been computed automatically.

**2.2 ~~Two of the three references are near-worthless as written.~~
CORRECTED by §1A.3.** The original claim — that `archive_oldest` being a
permanent 200-game policy makes it near-worthless — is **wrong, and backwards**.
A *fixed, weak* reference is exactly what exposes a regression: elves and rakdos
losing to a 200-game policy is the single most informative number in this
document, and it is the one `archive_oldest` was already measuring. The defect
was never the reference; it was (a) averaging it with `active_oldest` under one
label, and (b) pooling it across the whole run so a monotone decline showed up
as a flat ~50%.

What survives of 2.2: `vs_heuristic` really is saturated (20/20 for 5+ sessions,
for *both* populations), and `vs_gauntlet` really is a fixed, rather weak
reference. But **`archive_oldest` should be promoted, not demoted** — and
extended, since one fixed reference per deck is fewer than the six the round
robin showed are informative.

**2.3 ~~There is no absolute scale.~~ ADDRESSED** by `run_anchor_eval.py`
(§1A.1) — with the caveat that the anchor is a *floor* instrument and saturates
against any non-embarrassing policy. It answers "is this worth anything" and
cannot answer "is this better than that."

**2.4 There is no exploitability measure.** Every instrument measures play
against a *specific* opponent. None measures play against a *best response*.

**2.5 The health check cannot see a stalled run — and §1A.3 makes this worse
than stated.** `webapp/runs.py:180-188` `_batch_healthy` greps for `Traceback`
and a "session done" line. `total_games: 600000` in `run_default.json` is
**~247 hours ≈ 10.3 days** at the measured 0.37 s/game. The loop would run all
ten days on a flat curve without comment.

**And it is not a flat curve — it is a descending one.** This invalidates Step
1.5 *as originally specified*: a gate that stops when the last K readings are
"indistinguishable from the preceding K (|z| < 2)" would look at a deck losing
ground, see |z| > 2, conclude "not stalled," and keep going. **The gate as
designed would have run through the entire regression.** It must test for
`recent < earlier`, not `recent ≈ earlier`. See the revised Step 1.5.

**2.6 The aggregate metric is ~75% dead weight.** 12 of 16 matchups are
structurally decided (mono_red beats dmir 92–98% at *every* checkpoint). A 4pp
aggregate move is a 16pp move in the mirrors. Reporting the aggregate alone
hides the signal.

**2.7 Doc parity has drifted.** `README.md` documents the PPO minibatch ramp
(32→2048) as active. It has never executed (§3, BUG 1).

---

## 3. Confirmed defects

### BUG 1 — both 2026-08-06 anti-plateau schedules have never executed
*Found independently by four of six investigators; verified directly.*

Across **all 40,104 PPO iterations** of the 60,001-games/deck run:
`batch_size` ∈ {32}, `ent_coef` ∈ [0.0191, 0.0200].

| | designed | actual |
|---|---|---|
| minibatch | 32 → 2048 over 6 doublings by 50k games | **32, always** |
| `ent_coef` | 0.02 → 0.005 | **0.0191–0.0200** |

Chain: `.claude/skills/train/SKILL.md` always passes `--n-iterations` →
`run_league.py:135` leaves `auto_sizing=False` → `run_league.py:196` never calls
`_save_progress` → `checkpoints/4_deck_subleague_test/progress.json` **does not
exist** → `_load_progress` returns `cumulative_games_per_deck: 0` →
`batch_size_for_iteration` and `ent_coef_schedule` restart at the origin every
session and reach at most 3000/50000 = 6% of their horizon, which is
`int(0.06 × 6) = 0` doublings.

Every hyperparameter conclusion drawn since 2026-08-06 is against a baseline
that is not the one the code describes.

### BUG 2 — `PFSP_POWER = 2.0` is backwards
*Found independently by two investigators.*

`_pfsp_weight = FLOOR + (1 - win_rate) ** POWER` (`src/rl/league.py:146-149`).
Raising POWER **sharpens** concentration onto the hardest matchup. The comment
at `src/rl/league.py:63-70` claims the opposite. Predicted shares reproduce the
observed to within 1pp:

| POWER | elves: mono_red / rakdos / dmir / mirror | hardest:mirror ratio |
|---|---|---|
| 0.5 | 28.1 / 27.5 / 23.7 / 20.7 | 1.36 |
| 1.0 | 31.1 / 29.6 / 22.2 / 17.1 | 1.82 |
| **2.0 (current)** | **36.3 / 33.0 / 18.9 / 11.8** | **3.09** |
| observed on disk | 36.9 / 33.0 / 18.4 / 11.8 | — |

The 2026-08-06 change intended to *reduce* concentration on unwinnable
matchups and **increased** it. Consequence: **elves spends 69.9% of its games
in matchups it wins <6% of the time; dmir_terror 76.5% at <11%.** Those games
carry near-zero advantage variance, i.e. near-zero gradient. Mirrors — the only
matchups with real signal — get 11.8% and 14.7%.

### BUG 3 — `DeckNetwork` registers the shared stack as a child module
`deck.py:34` `self.shared_stack = shared_stack` (same in `mulligan.py:59`).
`nn.Module.__setattr__` registers it, so **every checkpoint on disk embeds a
full copy** (37 of 51 keys in `elves/live.pt`). Harmless today — all copies are
byte-identical — but the instant the stack is trainable, three call sites
silently corrupt it, most severely `league.load_snapshot_agent`, which rewinds
the live stack to a snapshot's era *and* sets `requires_grad=False` on it
permanently. This is a latent landmine under any future unfreeze.

---

## 4. Two structural causes (not bugs, but not intended either)

### 4.1 Credit-assignment horizon hides the first half of every game
`gamma=0.99, gae_lambda=0.95` (`src/rl/ppo.py:58`), never overridden. GAE decay
is 0.9405/step. Measured median: **112 recorded decisions per game ≈ 56 per
seat-game.**

| decisions before terminal reward | advantage weight |
|---|---|
| 10 | 0.54 |
| 25 | 0.22 |
| 50 | **0.047** |
| 100 | 0.002 |

After advantage normalization, roughly **the first half of each game is
indistinguishable from zero advantage** and trains only against the dense
mana-burn shaping term. In Magic, early decisions — mulligan, land drops, curve
— are frequently the decisive ones.

### 4.2 Six games per policy update
`games_per_iteration = max(1, n_workers) = 6` (`run_league.py:134`), and every
iteration runs a full `ppo_update`: buffer ~777, batch 32, `epochs_run` 2.74 →
**~67 Adam steps on the outcome of 6 games**, ~10,000 updates per deck.
`approx_kl` = 0.0443 against `target_kl` = 0.03, so early stopping fires on
nearly every update — the trust region is saturated on six games of evidence,
every time.

Measured consequence: over `snapshot_258 → snapshot_289` (6,600 games), head
parameters drifted **23.4% of their own magnitude** while the resulting policy
scores **50.0%** against the window's start. Large motion, zero strength
change — a noise-driven random walk in a flat basin.

### 4.3 The entropy coefficient is 2.5–10× too low

**arXiv:2502.08938 (ICLR 2026), "Reevaluating Policy Gradient Methods for
Imperfect-Information Games"** — 7,000 training runs across 5 large games, with
*exact* exploitability rather than a proxy. Two findings, both directly binding
here:

1. > *"the entropy coefficients with the best average performance are between
   > **0.05 and 0.2**, larger than any of the default entropy coefficients for
   > PPO in Stable Baselines, CleanRL, RLlib, OpenSpiel, PufferLib, RL-Games,
   > and Tianshou, which range between 0 and 0.01."*

   It was **the single most important hyperparameter** across the sweep. Ours
   sits at **0.0197** — inside the range they identify as the reason PPO
   underperformed in prior literature.

2. > *"NFSP and R-NaD approached or matched the performance of generic PG
   > methods in select cases, but typically underperformed them."*

**The regime matters and it favors us.** Their budget was 10M env steps per
run. At ~100–150 agent decisions per MTG game, our 60,000 games/deck ≈ **6–9M
env steps** — the *same* order. DeepNash, by contrast, used 7.21M learner steps
× 768 trajectories = **5.5×10⁹ games**, roughly **92,000×** our budget, on
768+256 TPU nodes. We are in the benchmark's regime, not DeepNash's, so the
benchmark's conclusions are the directly relevant evidence.

**This interacts badly with BUG 1 and changes what "fixing" it means.**
`ent_coef_schedule` is designed to anneal **0.02 → 0.005**. That is the wrong
direction relative to this evidence: repairing the schedule as written would
drive the coefficient from 2.5× too low to **10× too low**. Fix the plumbing
(Step 0.1) but retarget the schedule (Step 0.4) — do not restore it as designed.

---

## 5. The plan

### Phase 0 — Fix what is broken (hours; do regardless of everything else)

**Step 0.1 — `cumulative_games` advances unconditionally.**
`src/run_league.py:196-199`. Split the two things `if auto_sizing:` gates:
`last_batch_size` feedback stays gated (it protects the doubling ladder's
documented contract); `cumulative_games_per_deck` must advance always — it is
the horizon for two schedules unrelated to the ladder.
*Risk:* this is a genuine behavior change — `batch_size` will ramp to 2048 and
`ent_coef` anneal to 0.005 for the first time. Do not bundle with 0.2 if you
want attribution.
*Test:* `tests/test_run_league.py` (the `.py` appears deleted; the `.pyc`
survives — restore it) — run with `--n-iterations` set, assert
`cumulative_games_per_deck` advanced and `last_batch_size` did not.
*Doc parity:* `README.md` PPO section already claims the ramp is active.

**Step 0.2 — `PFSP_POWER` 2.0 → 0.5**, and make it config-driven rather than a
module constant so it can be recalibrated without a code edit.
`src/rl/league.py:72`, rationale comment at `:58-70` rewritten with the
predicted-vs-observed table from §3.
*Risk:* 0.5 is a first estimate exactly as 2.0 was. The table above is now a
calibration instrument — re-check observed shares at the next checkpoint.
*Test:* `tests/rl/test_league.py` — assert the hardest opponent's empirical
share is **monotone decreasing** in `PFSP_POWER` over {0.5, 1.0, 2.0}. This is
the invariant the 2026-08-06 change violated with no test to catch it.

**Step 0.4 — retarget the entropy coefficient to 0.05–0.2.**
`src/rl/train.py:555` `ent_coef_schedule`. Its current 0.02→0.005 target is
backwards relative to the ICLR 2026 sweep (§4.3). Options, in order of
preference: (a) constant 0.05 — simplest, matches the benchmark's tuned config,
and removes a schedule that has never run anyway; (b) 0.2 → 0.05 annealed, i.e.
high-to-lower but staying inside the good band throughout.
*Risk:* higher entropy trades immediate exploitation for exploration; expect
short-run win rate to dip before it recovers. This must be evaluated with
Phase 1 instrumentation in place, not against the current n=20 readings.
*Note:* this is also **the cheapest possible test of the MMD idea** — with a
uniform magnet, MMD's `α·KL(π, ρ)` term *is* an entropy bonus up to a constant.
If a large `ent_coef` helps, that is evidence for the regularization family
without building any of it.
*Test:* `tests/rl/test_train.py` — pin the new schedule's endpoints.
*Doc parity:* `README.md` PPO section documents the 0.02→0.005 anneal.

**Step 0.3 — de-register the shared stack.**
`object.__setattr__(self, "shared_stack", shared_stack)` in `deck.py:34` and
`mulligan.py:59`; strip `shared_stack.*` keys in `checkpoint.load_deck_checkpoint`
and `load_snapshot` so existing checkpoints still load.
*Benefit now:* live.pt drops 51→14 tensors; ~470 KB × ~800 files reclaimed;
`rollout_parallel` stops shipping 4 redundant stack copies per call.
*Test:* loading deck B's checkpoint leaves the shared stack bit-identical;
`load_snapshot_agent` leaves every `p.requires_grad` untouched.

### Phase 1 — Make the instruments readable (≈1 day, zero new compute)

**Step 1.1 — rewrite `analysis/report_metrics.report()`.** Pool over a trailing window;
print `w/n (pp) ±CI95`; **never merge `archive_oldest` with `active_oldest`**;
early-vs-late two-proportion z per deck per instrument; `SATURATED` flag at
≥95%; `.get` everything so old records still parse.

**Step 1.2 — enrich what is written.** `_run_eval_vs_history` records
`snapshot_id` and `is_archive`. All `_append_metric` sites record
`cumulative_games`. New `kind: "session_start"` record carrying league name,
roster, reward_fn_name, config — `metrics.jsonl` is currently not
self-describing.

**Step 1.3 — reallocate the eval budget, and fix its sample size.** Make
`eval_games` and `eval_every_sessions` configurable (currently hardcoded 20 at
`league_runner.py:564,606,648`). At `eval_games=200, eval_every_sessions=4` the
cost is **identical** and per-reading SE drops 11.2pp → 3.5pp.

**Two additions that came out of the literature review, and they matter more
than the reallocation:**

- **Common random numbers.** Evaluate with *paired seeds and sides swapped* —
  the same shuffles played from both seats. In a card game this removes the
  "who drew better" variance and is worth an estimated **2–4× effective N for
  free**. `collect_rollout` already randomizes `starting_idx` per game
  (`train.py:334`), so this is a change to how eval games are seeded, not to
  the engine.
- **Our 7pp "oscillation" is inside the noise floor.** Amplitude over a
  training curve is an *extreme-value* statistic: the expected range of 50
  i.i.d. normals is 4.498σ. At N=1000 games/eval, pure coin-flipping produces
  an expected **7.1pp** peak-to-trough range. Our cross-league evals are N=800.
  **The observed ~7pp is what noise predicts.** Meaningful amplitude claims
  need **N ≳ 4,000/eval**; a single 7pp gap between two points needs N ≥ 409
  each for bare 2σ.

  *This retires the "±7pp oscillation" language used throughout the earlier
  analysis, including in §1 of this document — that variation is not
  established as real.*

**Step 1.4 — `stack_id` guard.** Hash the frozen stack; write
`<league_dir>/stack_id.txt`; `_run_eval_vs_gauntlet` and
`analysis/run_cross_league_eval.py` **warn and return None** on mismatch (warn, not
assert — missing file is the legacy case). This exact silent mismatch already
invalidated 24,579 games/deck of `vs_gauntlet` numbers once
(`run_default.json:9`). **Hard prerequisite for any perception-stack work.**

**Step 1.5 — learning-stall gate. REVISED after §1A.3 — the original spec would
not have fired.** In `webapp/runs.py:_escalating_loop`, pool the last K vs the
preceding K `vs_history archive_oldest` records (that label specifically — it is
the fixed reference, §2.2) and test **two** conditions per deck:

- `STALLED` — indistinguishable, |z| < 2, across ≥N sessions
- `REGRESSING` — a linear decline (Cochran-Armitage trend z ≤ −2)
- **`PAST PEAK`** — the current window is significantly below the *best* window
  this run reached, **whatever the overall trend says**

The original spec tested only the first. A deck losing ground shows |z| > 2 and
would have been passed as healthy; the gate as written would have run through
the entire 60,001-game regression without comment.

**And `REGRESSING` alone is still not enough — verified against the real data
while implementing Step 1.1.** Every deck in this league *rose then fell*
(§1A.3: peaks at snapshot 116 / 58 / 289 / 232), and a linear trend test reads
a symmetric rise-then-fall as no trend. `dmir_terror` vs `archive_oldest`
scores trend z=**+0.35** over its full 27 records — nominally FLAT, slightly
positive — while sitting **2.9σ below its own peak**. Only the peak comparison
catches it, and it is the one that maps to the operator's actual question:
*is the checkpoint I am about to keep training worse than one I already saved?*

Implemented in `analysis/report_metrics.peak_comparison`, which the gate should call
rather than reimplement. Note the **Šidák correction** in it: the best of W
sliding windows is high by selection (the max of ~23 standard normals sits ~2σ
up), so an uncorrected −2 threshold fires on flat data routinely. With the
correction, 3 of 4 flags on the real league data drop out and the surviving one
is corroborated by the round robin.

~20 lines. Highest hours-saved-per-line change available.
*Open question for the owner:* stop, or warn only? Specified as stop-with-
explicit-restart; that is an operator-policy call. (`REGRESSING` is a stronger
case for stopping than `STALLED` — a stalled run wastes compute, a regressing
one destroys a policy that is on disk and recoverable only if you notice.)

### Phase 2 — Discriminate the remaining hypotheses (≈6 h compute, no new training code)

Two hypotheses remain, and they demand opposite fixes:

- **H1** — the learner is fine, the *training distribution/dynamics* are bad
  (BUG 1, BUG 2, §4.1, §4.2).
- **H2** — the learner is capped regardless of opponent (frozen stack,
  observation gaps, action space).

**Step 2.1 — snapshot round-robin. ✅ DONE (§1A.3), 4,296s.**
`src/analysis/run_snapshot_round_robin.py`, reusing `LeaguePool.load_snapshot_agent` +
`league_runner._play_eval_games`. Six snapshots, 15-pair round robin, 100 games
each, **sides swapped**, across all four decks.

Neither pre-registered branch occurred. Transitive (0 significant 3-cycles in 80
triples, residuals at/below noise floor) — but **descending**, not flat, for
three of four decks. Cycling ruled out; §4 is *part* of the story and §1A.4 is
the rest.

**This step is now permanent, and its role has changed from diagnostic to
acceptance test.** Re-run after Wave 2a; the target is `rising 5/5`.

**Step 2.2 — random-init anchor. ✅ DONE (§1A.1), 2,632s.**
`src/analysis/run_anchor_eval.py`. An **untrained** `DeckNetwork` on the real frozen
stack. First absolute scale in the repo.

Result: a ~200-game policy already beats random 78–100%, so the anchor
*saturates* against any non-embarrassing policy and could not see mono_red's
+241 Elo or rakdos's collapse. **It answers "is this worth anything," not "is
this better than that."** Its one non-obvious contribution stands: it
independently flagged elves as *worse* at live than at snapshot_0 (−10.0pp,
z=−2.13), which §1A.3 confirmed. Keep as a cheap floor check; do not use it to
compare trained policies.

**Step 2.3 — best-response / exploitability (~1 h per deck, ~6 h with control).**
**No new trainer is needed** — the existing machinery composes:
`checkpoint_rate=1.0` + single-deck roster + `snapshot_every=10**9` (load-bearing:
a second snapshot would become a second sample-able opponent) + no `live.pt`
(cold-starts from random). `_make_league_pairing` leaves `record_as[opp]=None`,
so the target is structurally never updated. New `src/run_exploitability.py`
(~120 lines) is orchestration only.
*Mandatory `--control` run* against `snapshot_0`: an approximate BR gives a
**lower bound**, so a small number proves nothing unless the control shows the
BR can decisively beat a weak policy.
*This is the direct H1/H2 test:* training against a frozen target removes
non-stationarity by construction. If a fresh net still cannot learn, H1 is dead.

### Phase 3 — Act on the discriminated cause

**If H1 (distribution/dynamics):**
- **Step 3.1 — credit-assignment horizon.** Raise `gae_lambda` toward 0.99 and/or
  `gamma` to 0.997 so decisions 50 steps out retain meaningful weight. Two
  parameters, plumbed through `_run_session`'s existing override block. Largest
  expected effect of anything in this document, and among the cheapest.
- **Step 3.2 — games per update.** Decouple `games_per_iteration` from
  `n_workers` (accumulate N iterations of rollout before one `ppo_update`), so
  the trust region is spent on 24–48 games rather than 6.

**If H2 (learner capped):**
- **Step 3.3 — unfreeze the perception stack.** Requires Step 0.3 and Step 1.4
  first. Options ranked: bigger pretrain ≈ per-deck private stack copies >
  low-LR fine-tune > full joint unfreeze > LoRA (dominated — the whole stack is
  117k params, *smaller than one deck head*). Expect update cost ×3–5,
  session wall-clock ×2–3, because `ppo.py:133` `cache_shared` auto-disables.
- **Step 3.4 — observation gaps** (below).

### Phase 4 — Observation fidelity (independent, small, do when convenient)

**Step 4.1 — Brainstorm blindness.** `dmir_terror` runs 4× Brainstorm, which
resolves through a real agent decision (`begin_put_on_top_from_hand`,
`handlers_library.py:309`) choosing which two cards go on top of its library and
in what order. **Library contents appear nowhere in the observation.** The agent
makes a choice whose entire value materializes 1–2 draws later with zero
observability of the consequence — structurally untrainable. A real player
obviously knows what they put on top, so this is an observation-fidelity gap,
not hidden information being respected. ~30 lines: `known_top: list[CardDef]` on
`PlayerState`, popped on draw, tokenized as a sixth `ZONES` entry visible only
to its owner. **Note: adds a zone → changes `TOKEN_FEATURE_DIM` → invalidates
every checkpoint.** Bundle with any other feature-dim change.

**Step 4.2 — three missing scalars.** `on_the_play` (the *mulligan* net already
sees it, `mulligan.py:56`; the main policy does not — a plain oversight),
`opponent.mulligans_taken`, `opponent.cleanup_discard_turns`. All public in real
Magic, all one-line reads off `PlayerState`. `SCALAR_FEATURE_DIM` 32→35, which
touches **one** tensor (`trunk_layers.0.weight`) and is **surgically migratable**
by zero-padding on the right — behavior-preserving, 60,001 games/deck of
training survives. Put the pad in `checkpoint.py` so both the live and snapshot
loaders get it.

---

## 6. Rejected, with reasons

| Option | Verdict | Reason |
|---|---|---|
| **R-NaD / DeepNash** | **No — now measured** | Three independent reasons. (1) **Direct empirical evidence against it at our budget**: arXiv:2502.08938 ran 7,000 runs with exact exploitability at 10M env steps — our regime — and found R-NaD and NFSP "typically underperformed" tuned PPO. (2) Premise fails, and this is no longer an inference from win-rate traces: the round robin (§1A.3) found **0 significant 3-cycles in 80 triples**, with all four Bradley-Terry residuals **at or below their own sampling-noise floors**. Payoff-matrix structure is how cycling is actually demonstrated in the literature, and it says the space is transitive. (3) **The multi-deck league is not two-player zero-sum** — 4 policies optimizing over a PFSP mixture that depends on their own results is a population game the theorem says nothing about; only the mirrors (11.8–51.3% of games) are a clean instance. Cost for reference: DeepNash = 5.5×10⁹ games on 1,024 TPU nodes, ~92,000× our budget, and its reward transform is a *per-timestep reward modification with per-player sign flipping inside v-trace* plus a NeuRD logit-space loss and four parameter sets — not a loss term you can bolt on. |
| **NFSP** | **No** | Its own staged plan estimates 15–17 GB resident for a *2-deck* pilot and warns it may need 10–100× more games. |
| **Oracle guiding (Suphx)** | **No** | Hard-blocked by the frozen stack, and it targets sample-efficiency, not the measured failure. Oracle tokens land on the "theirs" side → pooled by a **frozen** `theirs_pool` → reach the trunk only via FiLM. Feeding new information into a pooler that cannot adapt to it. |
| **Auxiliary belief head** | **No, as specified** | Fork, both branches bad: hang it off the trunk and it cannot recover what the frozen encoder discarded; give it its own pooling and it has zero effect on the policy. Also, on a Markov observation it can only learn `P(hand \| current public state)` — close to a function of its own input. Revisit only after unfreezing **and** recurrence. |
| **Full recurrence (LSTM/GRU)** | **No** | Memory is opponent modeling, and opponent modeling against a shallow snapshot window is the fastest route to fitting *this population* rather than the game. Predicted effect — `vs_history` up, `vs_gauntlet` flat — is precisely the axis we already measure as the failure. If ever built, per-turn summaries (~13 steps) beat per-decision (~56), need no BPTT, and stay iid-shufflable. |
| **Deeper snapshot window** | ~~**No**~~ **REOPENED by §1A.3** | The original reason — *"the 6,400-game-old self already scores ~50% against live, so adding older opponents adds opponents we already beat"* — is **now known to be false**. We do not beat our older selves; elves and rakdos **lose** to theirs. Older snapshots are, for three of four decks, **stronger opponents than live**, and PFSP would correctly weight them heavily. With `checkpoint_opponent_rate: 0.15` (`run_default.json`), 15% of training games already draw from that pool, so the lever is live. Re-evaluate after Wave 2a: if 0.2 stops the regression, "older = stronger" stops being true and this closes again on its own. |
| **Excluding structurally-lost matchups** | **No** | A hand-coded policy over the training distribution, against the standing autonomy constraint in spirit — and it would remove a deck's only exposure to the field it is scored against. The soft lever (flatter PFSP power) redistributes without ever zeroing a matchup; `PFSP_FLOOR=0.1` already prevents starvation. |
| **Handicapping the heuristic anchor** to de-saturate it | **No** | A game-rules deviation requiring explicit owner authorization under `CLAUDE.md`. Legitimate alternatives: extend `heuristic_decks` to all four, or demote it to a regression tripwire. |
| **Roster expansion to 11 decks** | **Later** | Vocab claim **verified** — `build_pool()` already spans all 11, no rebuild, no retrain of existing decks. But 2.75× compute, and PFSP's cold-start prior would send ~64% of games to 7 untrained decks. Revisit after Phase 3. |

---

## 7. Ordering and dependencies

```
Phase 0 (bugs)          hours    ─┐
Phase 1 (instruments)   ~1 day    ├─> gate everything; nothing below is
  1.4 stack_id guard ──────────┐  │    evaluable at n=20 / SE 11.2pp
                               │  │
Phase 2 (discriminators) ~6 h  │  ┘
  2.1 round-robin  → cyclic? ──┼──> (if yes) regularized PG returns
  2.2 anchor       → scale     │
  2.3 BR + control → H1 vs H2  │
                               │
        ┌──────────────────────┴──────────────┐
   H1 confirmed                          H2 confirmed
   3.1 GAE horizon                       3.3 unfreeze stack ←── needs 0.3 + 1.4
   3.2 games/update                      3.4 observation gaps
```

- **Phase 1.4 is a hard prerequisite for Phase 3.3.** Unfreezing changes the
  representation, which silently invalidates every frozen reference — the exact
  failure that already cost 24,579 games/deck of confounded numbers.
- **Phase 0.3 is a hard prerequisite for Phase 3.3.** Without it, unfreezing
  silently corrupts the stack via three call sites.
- **Phase 4.1 and 4.2 conflict on checkpoint compatibility.** 4.2 is
  zero-pad-migratable; 4.1 adds a zone and forces a fresh start. If both are
  wanted, do them together and set `PRETRAIN_PER_DECK` high at that point —
  which delivers the "bigger pretrain" option for free.
- **Do not run Phase 2 concurrently with training.** Both saturate the same 6
  cores.

---

## 8. Open questions for the owner

### Decided 2026-08-13

| question | decision |
|---|---|
| Stall gate: stop or warn? | **Warn.** `stop_on_regression` opts in. Plus: the gate now names the snapshot to roll back to, and `src/run_rollback.py` performs it. |
| `eval_games` / `eval_every_sessions` | **Keep 20 / 1.** Knobs exist; values unchanged. *Consequence: the in-training gate stays underpowered — it catches dmir_terror's decline but reads elves as merely "stalled" where the round robin measured 145 Elo. The round robin, not `vs_history`, is the acceptance test.* |
| `games_per_iteration` for Wave 2a | **24** (4x the evidence per update) |
| `PFSP_POWER` | **0.5** |
| `total_games: 600000` | **Leave it** — the learning-health gate is what made the number safe |
| Phase 3.1 ordering (#3) | Resolved by the wave reordering; stays last, now with an `explained_variance` mechanism check |

**Still open: the starting checkpoint for Wave 2a (below, #4).**



1. **Stall gate: stop or warn?** Specified as stop-with-explicit-restart.
2. **`total_games: 600000`** is ~10.3 days at measured throughput on a curve
   flat since 6k games. Keep, lower, or leave it and rely on the stall gate?
3. **Phase 3.1 (`gamma`/`gae_lambda`)** is arguably cheap enough to run
   *before* Phase 2 as a probe rather than after as a treatment. It costs one
   training run; it would confound the H1/H2 discrimination if run first.
4. **Fresh start vs. continue — now a three-way choice.** Several Phase 4 items
   force a full retrain, so bundling every feature-dim change into one reset is
   cheaper than paying twice. But §1A.3 adds a third option that did not exist
   before it was measured: **restart from each deck's peak snapshot** rather
   than from `live.pt`, since for three of four decks `live` is *not* the
   strongest policy on disk.

   | option | starting point | cost |
   |---|---|---|
   | continue | `live.pt` — degraded for 3 of 4 decks | free, but resumes from the worst point |
   | **restart from peak** | dmir **116**, elves **58**, mono_red **289**, rakdos **232** | free; recovers ~57–241 Elo of already-paid-for training |
   | fresh start | random init | full retrain, required by Phase 4.1 anyway |

   *Caveat on restart-from-peak:* the peaks are single measurements at n=100/pair
   and adjacent vintages are within noise of each other (elves 58 = +89 vs
   116 = +24 is ~65 Elo, real; dmir 116 = +56 vs 232 = +54 is not). Pick peaks
   only where the margin exceeds the noise floor, and re-measure the chosen set
   before committing. Also: restarting rakdos at 232 discards ~11,400 games and
   restarting elves at 58 discards ~46,000 — that is the *point*, but it should
   be a deliberate decision, not a side effect.

---

## 9. Autonomy compliance

Every item in this plan was checked against the standing constraint that agents
keep making their own decisions.

- **Nothing** adds a scripted rule, hand-coded policy, or heuristic override.
  Action selection remains `Categorical(masked_logits)` over the engine's own
  legality mask everywhere.
- Phase 0/1/2 are training-harness, telemetry, and evaluation only.
- Phase 3.1/3.2 change optimizer hyperparameters.
- Phase 4 adds *observations* — public information a real player has — never
  evaluations. The line, for future work: a feature is compliant iff it is a
  deterministic function of publicly observable game state/history that a real
  player could compute at the table, **and** it is consumed only as a network
  input. It becomes non-compliant the moment it encodes an *evaluation*
  ("is attacking good here") rather than an *observation*.
- `HeuristicAgent` stays evaluation-only. It must never become a
  behavior-cloning target or a rollout policy.
- The random-init anchor (2.2) is an untrained `DeckNetwork` sampling from its
  own distribution — a policy making its own decisions, not a scripted rule set,
  and evaluation-only.
- **Rules mandate (`CLAUDE.md`):** nothing in Phases 0–3 touches `game/`. Phase
  4.1 touches `PlayerState` but adds only bookkeeping of information the
  controlling player already has in real Magic; no rules behavior changes. The
  one rejected option that *would* have engaged the mandate (handicapping the
  heuristic) is rejected on exactly that ground.

---

## 10. Investigator verdicts

| # | Workstream | Verdict on its own area |
|---|---|---|
| 1 | Exploitability / evaluation | **Yes, rank 1** — but reframed: the value is discriminating H1 from H2, plus free aggregation of data already on disk |
| 2 | Oracle guiding + belief head | **No** — hard-blocked by the frozen stack; proposed an asymmetric critic instead |
| 3 | Frozen perception stack | **Leave it frozen** — a capacity-bound model converges; this one takes a million clipped steps and doesn't move |
| 4 | History / recurrence | **No** — memory amplifies the failure axis; recommended 3 scalars only |
| 5 | Regularized policy gradient | **No, rank 6 of 6** — falsified the cycling premise directly |
| 6 | League dynamics | **Partly** — found BUG 2; the snapshot window is *not* the bottleneck, contradicting the 08-06 audit |
| 7 | Literature (spawned by 5) | **Entropy coefficient is the finding** — and R-NaD/NFSP lose to tuned PPO at our exact budget |

### A correction to the theory framing used earlier

"Poincaré recurrence" was invoked to justify the cycling diagnosis. It does not
apply as stated: the recurrence theorem (arXiv:1709.02738) is **continuous-time
only** — the authors restrict to the FTRL *ODE* explicitly. The discrete-time
result is the *opposite*: Bailey & Piliouras (EC 2018) prove MWU at a finite
step size has a **nonnegative lower bound on the rate of KL increase** from an
interior equilibrium — an outward spiral, not closed orbits — and Cheung &
Piliouras (COLT 2019) show constant-step MWU is Lyapunov chaotic. So even had
the strategy space been intransitive, "recurrence" would have been the wrong
word. A useful corollary: in zero-sum settings, *"learning rate too high"* and
*"discrete-time cycling"* are the same phenomenon, so lowering the step size is
a legitimate first response either way.

---

## 11. Execution checklist

**Revises the ordering in §7.** Phases are a dependency graph, not a schedule.
Three rules decide what can move together:

1. **Behavior-neutral changes ship together** — no attribution cost, so batching
   them is free.
2. **Read-only measurements go FIRST** — they are worth more as a
   pre-intervention baseline than as a post-hoc one, and 2.1's input (the
   pre-change snapshot series) stops being generated the moment 0.2 lands.
3. **Behavior-changing knobs: bundle the orthogonal, isolate the big.**

### Wave 0 — baseline measurement (read-only, no trainer changes)

- [x] `_play_eval_games` gains a `greedy` param (default True — no change to any existing caller)
- [x] `league_runner.load_vintage_agent` — path resolution across `live.pt` / active dir / `archive/`
- [x] `league_runner.league_roster` — decks with a `live.pt`, because `build_pool()` spans all 11 manifest decks
- [x] `src/analysis/run_anchor_eval.py` (Step 2.2) — **run, §1A.1**
- [x] `src/analysis/run_snapshot_round_robin.py` (Step 2.1)
- [x] PPO telemetry read off the existing `metrics.jsonl` — **§1A.2**, free
- [x] Round robin run → `logs/snapshot_round_robin.json` — **§1A.3**
- [x] `opponent_stats.json` cross-check — **§1A.4**, free

Measured throughput: **0.77 s/game** anchor, ~2.9 s/game round robin (two
trained policies play far longer games than trained-vs-random).

**Outcome.** The pre-registered branches were: flat-transitive → proceed;
rising-transitive → re-aim at evaluation; significant cycles → regularized PG
returns. **None of the three happened.** The matrices are transitive (so no
cycling) but *descending* for three of four decks — an outcome the branch table
did not anticipate, and a stronger finding than any of them. See §1A.3.

**Re-run this after Wave 2a.** It is now the success criterion, not a diagnostic.

### Wave 1 — behavior-neutral (ship together)

- [ ] **0.3** de-register the shared stack — `object.__setattr__` in `deck.py:34`, `mulligan.py:59`; strip `shared_stack.*` in `checkpoint.load_deck_checkpoint`/`load_snapshot`
  - [ ] test: loading deck B's checkpoint leaves the stack bit-identical; `load_snapshot_agent` leaves every `requires_grad` untouched
  - [ ] drop the BUG 3 note in `load_vintage_agent`'s docstring once fixed
- [ ] **1.1** rewrite `analysis/report_metrics.report()` — pooled windows, `±CI95`, `SATURATED` flag, early-vs-late z, **never merge `archive_oldest` with `active_oldest`**
- [ ] **1.2** enrich records — `snapshot_id`/`is_archive` on vs_history; `cumulative_games` on every `_append_metric`; new `kind: "session_start"`
- [ ] **1.3** `eval_games`/`eval_every_sessions` configurable (hardcoded 20 at `league_runner.py:564,606,648`); paired seeds + swapped sides, reusing `analysis/run_snapshot_round_robin._play_pair`
- [ ] **1.4** `stack_id` guard — hash the frozen stack, write `<league_dir>/stack_id.txt`, warn-and-return-None on mismatch
- [ ] **1.5** learning-stall gate in `webapp/runs.py:_escalating_loop` (**open question #1: stop or warn?**)
- [ ] **1.6 (new, from §1A.2)** make the optimizer knobs configurable — `lr` is hardcoded `3e-4` at `league_runner.py:202`, `games_per_iteration` is welded to `n_workers` at `run_league.py:134`, `target_kl`/`gamma`/`gae_lambda` are `ppo.py` defaults. Exposing them changes no value, so it batches here at zero attribution cost; *changing* them is Wave 2. Same split as the 0.1 counter fix.
- [ ] **1.7 (new, from §1A.2)** record `explained_variance` alongside `value_loss`. §1A.2(c) had to be *argued* because the raw MSE is uninterpretable without the return variance next to it.
- [ ] delete `analyze_dmir_terror_land_pattern.py`, `analyze_mana_burn_by_turn.py`, orphaned `tests/__pycache__/test_run_league.*.pyc`
- [ ] restore `tests/test_run_league.py` — the test whose absence let BUG 1 live
- [ ] doc parity: README PPO section

### Wave 1 — SHIPPED 2026-08-13

- [x] **0.3** shared stack de-registered via `object.__setattr__` in `deck.py`/`mulligan.py`; `checkpoint.strip_shared_stack` strips it on load.
- [x] **1.1** `analysis/report_metrics.report()` rewritten — pooled windows, Wilson CI, `SATURATED` flag, early-vs-late `trend_z`, Šidák-corrected `peak_comparison`, `archive_oldest`/`active_oldest` never merged.
- [x] **1.2** records enriched — `cumulative_games` on every `_append_metric`, new `kind: "session_start"`.
- [x] **1.3** paired common-random-numbers eval (`league_runner._play_paired_eval_games`) wired into all three in-training eval callers; `analysis/run_snapshot_round_robin._play_pair` now calls the same shared function instead of re-deriving it.
- [x] **1.4** `stack_id` guard added — hash the frozen stack, warn-and-return-None on mismatch.
- [x] **1.5** learning-stall gate added in `webapp/runs.py:_escalating_loop`, default warn / `stop_on_regression` opt-in (open question #1 resolved: warn by default), backed by `run_rollback.py`.
- [x] **1.6** optimizer knobs (`lr`, `gamma`, `gae_lambda`, `target_kl`, `n_epochs`, `mulligan_lr`) moved to `league_runner.PPO_DEFAULTS`, overridable per league via a run-config `"ppo"` object.
- [x] **1.7** `explained_variance` now recorded alongside `value_loss` from `ppo_update`.
- [x] `analyze_mana_burn_by_turn.py` deleted; `tests/test_run_league.py` restored.
- [x] doc parity: README PPO section updated.

### Wave 2a — signal-to-noise per update (the measured cause)

**Reordered again 2026-08-13 after §1A.3/§1A.4.** These three are one problem
seen from three angles: **how much real evidence backs each gradient step.**
PFSP decides what fraction of games carry any signal, `games_per_iteration`
decides how many games are pooled, and `ppo.py:122` decides whether the
resulting noise gets rescaled to look like signal. Bundling them is right
*because* they share a target — and each has an independent mechanism check, so
a failure to fire is still attributable.

- [ ] **0.2 — the leading candidate.** `PFSP_POWER` 2.0 → 0.5, config-driven; rewrite the rationale comment at `league.py:58-70` with §1A.4's table
  - [ ] test: hardest opponent's empirical share is **monotone decreasing** in `PFSP_POWER` over {0.5, 1.0, 2.0}
  - [ ] **mechanism check** (one session): elves' share in <25% matchups falls from **69.8%**; mirror share rises from 11.8%. This is the falsification test for §1A.4.
- [ ] **3.2, promoted** decouple `games_per_iteration` from `n_workers`: accumulate N iterations of rollout before one `ppo_update`, so the trust region is spent on **24–48 games rather than 6**
  - [ ] **mechanism check**: `buffer_size` up ~4–8×, `epochs_run` recovers toward 4, `approx_kl` falls to or below `target_kl=0.03`
  - [ ] if `approx_kl` still exceeds target with 4–8× the evidence, *then* lower `lr` from 3e-4 (knob exists after 1.6) — in that order, since more evidence per step dominates smaller steps and this tells them apart
- [ ] **NEW — guard the advantage normalization at `ppo.py:122`.** `(adv - mean) / (std + 1e-8)` on a batch that is ~96% losses rescales pure noise to unit variance. Options, cheapest first: (a) skip normalization when `adv.std()` is below a floor; (b) normalize by a running estimate rather than per-buffer. **Do not bundle with 0.2/3.2** — those two shrink the degenerate fraction, which may make this moot. Measure first.
- [ ] **0.1 (counter only)** `cumulative_games_per_deck` advances unconditionally (`run_league.py:196`); `last_batch_size` feedback stays gated
  - [ ] **pin the minibatch schedule constant at 32 in the same commit.** `ppo.py:149` slices `range(0, total, batch_size)` against a ~777 buffer, so `batch_size=2048` collapses to one minibatch — **2.74 Adam steps per iteration instead of 67, a 24× cut.** A real experiment, not a bug fix. *(3.2 grows the buffer, which changes what any `batch_size` means — another reason to pin it.)*
  - [ ] test: with `--n-iterations` set, `cumulative_games_per_deck` advanced and `last_batch_size` did not

**Success criterion, and it is not a win rate.** Re-run
`analysis/run_snapshot_round_robin.py` over the *new* snapshots. The target is
**newest = strongest**, i.e. `rising on 5/5` and a positive span. Elves and
rakdos currently lose to their own 200-game-old selves; anything short of
reversing that is not a fix.

### Wave 2a — SHIPPED 2026-08-13

- [x] **0.1** `cumulative_games_per_deck` advances unconditionally; `last_batch_size` still gated. Extracted to `league_runner.advance_progress`; `tests/test_run_league.py` restored and pins it. Minibatch ramp **pinned at 32** (`batch_size_start == batch_size_cap`) so fixing the counter does not switch a 24x Adam-step cut on as a side effect.
- [x] **0.2** `PFSP_POWER` 2.0 → 0.5, config-driven, **threaded to the parallel workers** (each builds its own `LeaguePool`; missing that would have left 5 of every 6 games on the old weighting). Monotonicity test added.
- [x] **3.2** `games_per_iteration` 6 → 24.
- [x] **advantage-norm floor** — `adv_norm_floor` in `PPO_DEFAULTS`, **default 0.0 = exactly the historical behavior**, plus `adv_std` now recorded per update. Deliberately not guessed: the right value depends on this reward's advantage scale, which nothing had ever measured. First real reading: **adv_std ≈ 0.21** on an untrained elves mirror.

*Measured effect of 0.2, and it is bounded.* Simulated against elves' real win rates, the share of its games in matchups it wins <25% of: **69.5% → 55.6%** (mirror 11.8% → 20.6%), reproducing §3's predicted table. But **pure-uniform sampling floors at ~50%** because two of elves' four opponents are structurally unwinnable. Flattening PFSP cannot go further without excluding matchups, which §6 rejects on autonomy grounds. That is *why* the advantage-norm floor matters rather than being made moot.

### Phase 4 — SHIPPED 2026-08-13 (bundled into the fresh start)

- [x] **4.2** three public scalars — `on_the_play` (the mulligan net always saw it; the policy never did), opponent `mulligans_taken`, opponent `cleanup_discard_turns`. `SCALAR_FEATURE_DIM` 32 → 35.
- [x] **4.1** Brainstorm `known_top`. `TOKEN_FEATURE_DIM` 40 → 41.

**4.1's design in the plan was unsound and the audit caught it.** Two findings:

1. *No choke point existed* — 40+ raw library mutations across 9 files, so "pop on draw" would have left `known_top` claiming cards that Thought Scour milled, i.e. feeding the agent **false** information. Fixed by creating one: `game.effects.shared.shuffle_library`, which all **9** shuffle sites now route through, with a test asserting none escapes it.
2. *Value-matching leaks hidden information.* Verifying `known_top` against the real library looks safe, but library entries are **interned CardDefs** — with a 4-of, neither `==` nor `is` distinguishes "the Mountain I placed" from "a different Mountain now on top." After drawing both known cards off `[Bolt, Mountain, Mountain]` the third Mountain matches and the agent is told it knows the top. True, but not legitimately knowable. Fixed with a second independent check: `known_top_library_len` tracks how many cards left the top by **length delta**, not by value.

### Wave 2b — exploration

- [ ] **0.4** `ent_coef` → **constant 0.05** (§4.3 option (a)). A constant also keeps 0.1 free of hidden behavior.
  - [ ] test: pin the schedule endpoints
  - [ ] **mechanism check** (one session): logged `ent_coef` = 0.05, policy entropy rises
  - [ ] doc parity: README documents the 0.02→0.005 anneal

Separate arm, and now clearly second. §1A.2(b) measured entropy **falling** 30%
while strength *declined*, and 0.4 pushes entropy back up. Those are compatible
— premature convergence is what an entropy bonus fights — but if 2a fixes the
signal-to-noise problem then 0.4 may be pushing on a door that is no longer
stuck. Running 2a first is what makes that answerable.

### Wave 3 — the H1/H2 discriminator (needs the box to itself)

- [ ] **2.3** `src/run_exploitability.py` — `checkpoint_rate=1.0` + single-deck roster + `snapshot_every=10**9` + no `live.pt`
- [ ] **mandatory `--control`** vs `snapshot_0`: an approximate BR is a *lower* bound, so a small number proves nothing unless the control shows the BR can decisively beat a weak policy
- [ ] → H1 or H2, which selects Phase 3

Sequential for compute contention, not attribution. Wave 2's knobs do not
contaminate it: a single-deck roster makes 0.2 irrelevant, and 0.4 only makes
the best-response stronger, which is what you want from a lower bound.

§1A.2 already **leans** H1 — a capacity-bound learner converges and settles,
this one keeps saturating its trust region for 10,000 updates — but the lean
rests on a number with a known confound, so the discriminator still runs.

### Wave 4 — the horizon (alone, after the discriminator)

- [ ] **3.1** raise `gae_lambda` toward 0.99 and/or `gamma` to 0.997 so decisions 50 steps out retain meaningful weight
- [ ] **verification that did not exist before:** 1.7's `explained_variance`. §4.1 predicts the *current* critic scores well by predicting a discounted-to-zero constant; a longer horizon should make `value_loss` **rise** while `explained_variance` holds or improves. Without 1.7 that is unfalsifiable.

Still the largest expected effect and still deliberately last: it is the one
knob whose attribution is worth protecting, and §1A.2(c) now gives it a
mechanism check it lacked.

### Held back deliberately

| item | why it waits |
|---|---|
| minibatch ramp 32→2048 | 24× fewer Adam steps; its own experiment — and 2a changes buffer size, which changes what any `batch_size` means |
| **`lr` below 3e-4** | only if 2a's 4–8× evidence increase fails to bring `approx_kl` under target. More evidence per step dominates smaller steps |
| **Phase 4** | 4.1 changes `TOKEN_FEATURE_DIM` and forces a retrain (**open question #4**) |

### Still open

Questions in §8 are unanswered. #1 (stall gate: stop or warn) blocks Wave 1.5.
#3 is **resolved by the reordering**: 3.1 stays after the discriminator, now
with a real mechanism check (Wave 4). #4 (fresh start vs. continue) blocks all
of Phase 4 and should be decided before Wave 2 spends training time on a
checkpoint lineage that a reset would discard.

---

### Sources

[2502.08938](https://arxiv.org/abs/2502.08938) (ICLR 2026 benchmark — the
decisive one) · [2002.08456](https://arxiv.org/abs/2002.08456) (Perolat,
regularization theory) · [2206.15378](https://arxiv.org/abs/2206.15378)
(DeepNash/R-NaD) · [2206.05825](https://arxiv.org/abs/2206.05825) (Magnetic
Mirror Descent) · [2408.00751](https://arxiv.org/abs/2408.00751) (QFR) ·
[1709.02738](https://arxiv.org/abs/1709.02738) (Poincaré recurrence,
continuous-time) · [Bailey & Piliouras EC'18](https://dl.acm.org/doi/10.1145/3219166.3219235)
(discrete-time divergence) · [1806.02643](https://arxiv.org/abs/1806.02643) /
[2004.09468](https://arxiv.org/abs/2004.09468) (how cycling is actually
demonstrated — payoff-matrix structure, never a single win-rate trace) ·
[1709.06560](https://arxiv.org/abs/1709.06560) (RL evaluation variance)
