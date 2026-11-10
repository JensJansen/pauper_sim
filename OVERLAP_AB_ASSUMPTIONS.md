# Overlap (L1) vs sequential (S) learning A/B: design, assumptions, decision rule

Branch `exp/overlap-l1-ab`. Question: does `overlap_collect` (workers collect the next deck's round from weights one
update old, while the GPU update runs) hurt learning, or is it just faster? Speed is already measured (24.4% faster,
11.98 -> 9.06 s/round). This document fixes the design and the decision rule BEFORE any result was seen, and records
every judgement call made while running it unattended. Results go in `OVERLAP_AB_RESULTS.md`.

## Design (fixed before the run)

- Start: the saved `checkpoints/main-league` state at 92,340 games/deck (lean private copy in `C:\pauper_ab\start`;
  the real directory is never written; the driver asserts its `live.pt` hashes and `vocab.json` are unchanged).
- Arms, each a private copy of the start, one 167-iteration session (10,020 games/deck; owner asked for 10,000), production
  `training_configs/main_league.json` settings read through `_resolve_options` (6 workers, 60 games/iteration,
  checkpoint rate 0.3, stratify 0.33, PFSP 0.5, CUDA, mulligan training on), run one after another:
  - **S**: sequential, seed 1001.  **L1**: `overlap_collect`, seed 1001.  **S2**: sequential, seed 2002.
  - S2 differs from S only by seed: S2 - S is the size of a gap that pure chance produces. L1 is judged against it.
- Evaluation (`src/evaluate_agents.py`, greedy play, 120 games per pairing, seed 7): each arm's 11 decks play the 11
  START decks (121 pairings, mirrors included); the START decks also play themselves as the untrained reference.
  Plus per-deck head-to-head S vs L1, S vs S2, L1 vs S2 (200 games per deck). Paired seeding: every arm faces
  identical deals against the same opponent.

## Decision rule (fixed before the run)

Primary statistic: overall gauntlet score difference L1 - S, 95% interval (independent-tallies interval; conservative
for the paired design). Margin of concern: **-1.5 score points**.
- **Harmful**: interval entirely below 0 AND L1 - S is clearly beyond the chance gap (|L1 - S| > |S2 - S| interval).
- **Not harmful (efficiency only)**: lower bound of L1 - S above -1.5 points, and L1 - S not outside the S2 - S noise.
- **Inconclusive**: anything else (interval straddles -1.5). Reported as such with what extra games would be needed.
Secondary: per-deck differences (11 comparisons: expect ~1 chance exceedance at 95%, so a single deck is not
evidence), head-to-head matrices, per-update PPO diagnostics from each arm's `metrics.jsonl` (approx_kl, clip
fraction, epochs_run, explained variance), and the arms' wall-clock (a secondary timing check).
Also reported: START vs arms, to show whether 3,000 games moves strength at all. If neither S nor S2 beats START
detectably, the test can only bound harm, not show learning parity; the report will say so plainly.

## Judgement calls made unattended

(appended during the run; each entry says what happened, what I chose, and why)

1. **Length raised to 10,000 games/deck (owner request, after the first launch).** 167 iterations x 60 games = 10,020. At the
   measured round times this is ~6.1 h (S), ~4.7 h (L1), ~6.1 h (S2) of training, i.e. ~17 h + evaluation, longer than
   the owner's 12 h absence (the owner said waiting longer is acceptable). The first launch (50 iterations) was killed
   a few iterations in and its partial arm deleted.
2. **Staged order to get the primary answer first:** S, L1 -> evaluate (start/S/L1 gauntlet + S-vs-L1 head-to-head) -> S2 ->
   evaluate S2 (same gauntlet seed, so its scores are directly comparable) + S-vs-S2 and L1-vs-S2 head-to-heads. The
   S2 noise floor therefore arrives last; the L1-vs-S verdict is only provisional until it does.
