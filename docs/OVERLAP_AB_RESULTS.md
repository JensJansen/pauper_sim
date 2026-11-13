# Overlap (L1) vs sequential (S): results

Branch `exp/overlap-l1-ab`. Design, decision rule and every judgement call: `OVERLAP_AB_ASSUMPTIONS.md`. Raw outputs: `C:\pauper_ab\eval_*.{log,json}`
(not in the repo).

## Answer

**`overlap_collect` is not harmful to learning on this evidence; it is a ~24% wall-clock saving.** Final strength after
10,020 games/deck: L1 - S = **+0.10 points, 95% interval [-1.05, +1.25]** (14,520 games per arm). The lower bound clears the
pre-registered -1.5 margin, and L1 vs S head-to-head is 51.6% for S (interval 49.5-53.7, includes 50%). One caveat on the
pre-registered rule is below (S2 was cut short), so the verdict is "not harmful, with a ~1-point resolution limit", not "exactly equal".

## Setup (as run)

Start: `main-league` at 92,340 games/deck (private copy; the real directory was verified untouched). S = sequential, seed 1001;
L1 = `overlap_collect`, seed 1001; both 167 iterations (10,020 games/deck), production settings. S2 = sequential, seed 2002,
**stopped at 66 iterations (3,960 games/deck)** because it slowed from ~2.5 to 8-15 min/iteration (GPU at 100% / 14.8 of 16.3 GB
VRAM, cause not confirmed); the owner chose to cut it short. Evaluation: greedy, 120 games per pairing, seed 7, each arm's 11 decks
vs the 11 start decks (14,520 games per arm), plus 200-game-per-deck head-to-heads.

## Results

| | overall score vs start field | vs start |
|---|---|---|
| start (untrained reference) | 50.68% | |
| S, 10,020 games | 49.34% | -1.3 |
| L1, 10,020 games | 49.44% | -1.2 |
| S2, 3,960 games | 47.69% | -3.0 |

Differences (95% interval +/-1.15 points each): **L1 - S +0.10**, S2 - S -1.65, S2 - L1 -1.74, S2 - start -2.99.

Head-to-head (same deck from two arms, 2,200 games): S vs L1 **51.6%** [49.5, 53.7]; S vs S2 54.8% [52.7, 56.9]; L1 vs S2 55.9% [53.8, 57.9].

PPO diagnostics over the matched first 66 iterations (726 updates per arm): approx_kl 0.0129 / 0.0117 / 0.0125 (S / L1 / S2), clip fraction
0.072 / 0.075 / 0.075, epochs run 4.0 / 4.0 / 4.0, entropy 0.284 / 0.282 / 0.289, explained variance 0.817 / 0.820 / 0.809. The one-update
policy lag leaves no visible trace in the update statistics.

Wall clock: S 6.29 h, L1 4.78 h (-24.0%), matching the earlier 24.4% measurement.

## What this does and does not show

- Not harmful: L1 matches S in strength to within about +/-1.2 points and in head-to-head play; update diagnostics are indistinguishable.
- Per-deck differences are scattered in both signs; with 11 comparisons a few exceedances are expected by chance (see `eval_gauntlet_1.log`).
- **Training moved strength less than the evaluation can resolve against the start field.** Both 10k arms sit ~1.3 below start, and the shorter
  S2 sits ~3.0 below. But the two 10k arms beat S2 head-to-head (~55%), so there is real between-checkpoint movement (a dip then recovery
  against the start field, consistent with decks cycling on a shifting population, e.g. mono_red_madness at -13.7 in S, flat in L1). The A/B can
  bound harm; it cannot show "learning is exactly preserved".
- **Pre-registered noise clause not applied as written.** S2 is a shorter run, so S2 - S (-1.65) mixes seed noise with training length and is
  not a pure noise floor. It is no evidence against the verdict: the question was whether L1 falls clearly below S, and L1 - S is +0.10.
  A clean noise floor needs a full-length S2 (about 6 h on a healthy GPU) if more certainty is wanted.
- One retry occurred: L1 first crashed at iteration 6 on a pre-existing sac-cost engine bug (`engine/docs/SAC_FIZZLE_HYPOTHESES.md`); the harness restarted it
  with the same seed 1001 (no seed bump was needed).

## Recommendation

Enabling `overlap_collect` for main-league training is safe on this evidence. It stays off by default until the owner decides; the flag is
opt-in and covered by tests. Resolution limit ~1.2 points at 14.5k games; tighter needs more games or the full-length S2.
