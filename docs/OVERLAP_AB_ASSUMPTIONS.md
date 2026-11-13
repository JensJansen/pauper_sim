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
Also reported: START vs arms, to show whether 10,000 games moves strength at all. If neither S nor S2 beats START
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
3. **L1 crashed at iteration 6 on a pre-existing engine bug (not caused by overlap).** S finished all 167 iterations
   (6.29 h). L1's first attempt died in a worker: `_sac_artifact_or_creature._on_chosen` received `choice=None`
   (Eviscerator's Insight cast with no creature/artifact to sacrifice; the caster's board was only tapped Swamps). This is
   the open "sac-fizzle" investigation (`_dump_sac_fizzle_diagnostic`, commit f068b00; the dump gained one more record).
   I did NOT patch engine rules behaviour (RULES FAITHFULNESS MANDATE: the legality fix is the owner's call and the root
   cause is still unconfirmed). Instead the harness retries a crashed arm from the pristine start copy, up to 3 times,
   adding 0/1/2 to the arm's seed so a deterministic crash is not replayed. The crashed partial L1 was deleted. If L1 (or
   S2) ends up with a bumped seed, that is stated in the results. The bug is rare (three production crashes over ~92k games/deck: 2026-08-27, ~08-31, and this one), so a bumped
   seed is unlikely to matter for the comparison. Candidate causes are listed in `engine/docs/SAC_FIZZLE_HYPOTHESES.md` (not yet tested).
4. **S2 cut short at 3,960 games/deck (owner decision, option 3).** From about iteration 45 the S2 arm slowed from ~2.5 to
   8-15 min/iteration (GPU at 100% and 14.8 of 16.3 GB VRAM in use, ours the only compute process; cause not confirmed),
   projecting 20+ h more. The owner chose to stop it and compare at the shorter length. I stopped it at an iteration
   boundary (iteration 67 finished 16:06). The weights on disk are the last periodic save, 66 iterations =
   3,960 games/deck (`progress.json` cumulative 96,300; all 11 decks saved together at 15:50). Consequences, which the
   results must carry: S2 is NOT the same length as S and L1 (167 iterations), so S2 - S is not a pure-seed noise gap
   and the decision rule's "within S2 - S noise" clause cannot be applied as pre-registered. S2 can only show (a) how far
   a seed-different run is from START after 3,960 games and (b) the size of between-run spread at that length. The
   verdict therefore rests on the primary L1 - S interval and the S-vs-L1 head-to-head, and is stated as provisional on
   the noise clause.
