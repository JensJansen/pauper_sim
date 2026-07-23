"""Side-by-side comparison of two train_boggles_mirror_segments.py CSV runs --
e.g. a DummyVecEnv baseline vs. a SubprocVecEnv candidate on the same config
otherwise (see docs/GPU_VECENV_INVESTIGATION.md). Answers "which structure
trains ~target_episodes games fastest, at comparable effectiveness":

- speed: total training WALL-CLOCK time to process target_episodes games --
  not steps/sec. episodes_per_batch is only a FLOOR (run_segment keeps
  chunking until BOTH sides have at least that many episodes, so it can
  overshoot slightly and unevenly between two separate runs -- see
  train_boggles_mirror_segments.py's own docstring), so raw train_wall_s
  from two runs with different actual episode counts isn't apples-to-apples.
  Normalized to wall_s_per_episode first (train_wall_s / the smaller of the
  two sides' actual episode counts, the tighter guaranteed floor), then
  projected to target_episodes -- this is what's actually comparable.
- effectiveness: win-rate delta (should stay near 0 for two symmetric
  mirror-match runs) and avg_turns/avg_actions_per_turn deltas (a
  policy that degenerated -- e.g. an opponent stuck always-Passing -- shows
  up here as games running unnaturally long/short vs. the baseline).

    python compare_bench_runs.py <baseline_csv> <candidate_csv> [target_episodes]
"""

import csv
import sys

RAW_FIELDS = (
    "train_wall_s", "episodes_trained_a", "episodes_trained_b", "timesteps_per_sec",
    "win_rate_a", "win_rate_b", "draw_rate", "avg_turns", "avg_actions_per_turn",
)
EFFECTIVENESS_FIELDS = ("win_rate_a", "win_rate_b", "draw_rate", "avg_turns", "avg_actions_per_turn")

DEFAULT_TARGET_EPISODES = 1000


def _final_row(csv_path, target_episodes):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: no rows")
    row = {field: float(rows[-1][field]) for field in RAW_FIELDS}
    # The tighter of the two sides' actual counts -- run_segment's own floor
    # guarantee is "both sides reach AT LEAST episodes_per_batch," so this is
    # the largest episode count both sides are guaranteed to have hit.
    episodes_min = min(row["episodes_trained_a"], row["episodes_trained_b"])
    row["wall_s_per_episode"] = row["train_wall_s"] / episodes_min
    row["projected_s_for_target"] = row["wall_s_per_episode"] * target_episodes
    return row


def compare(baseline, candidate):
    """{field: candidate - baseline} for every effectiveness field, plus
    time_saved_s/time_saved_pct on projected_s_for_target -- positive means
    the candidate is FASTER (took less wall-clock time for the same number
    of games), matching this script's "total time to process N games" framing
    rather than a steps/sec ratio."""
    deltas = {field: candidate[field] - baseline[field] for field in EFFECTIVENESS_FIELDS}
    deltas["time_saved_s"] = baseline["projected_s_for_target"] - candidate["projected_s_for_target"]
    deltas["time_saved_pct"] = (
        deltas["time_saved_s"] / baseline["projected_s_for_target"] * 100.0
        if baseline["projected_s_for_target"] else 0.0
    )
    return deltas


def print_comparison(baseline_path, candidate_path, baseline, candidate, deltas, target_episodes):
    print(f"baseline:  {baseline_path}")
    print(f"candidate: {candidate_path}\n")
    print(f"projected wall-clock time to train {target_episodes} games:")
    print(f"  baseline:  {baseline['projected_s_for_target']:.1f}s")
    print(f"  candidate: {candidate['projected_s_for_target']:.1f}s")
    print(f"  -> candidate is {deltas['time_saved_pct']:+.1f}% ({deltas['time_saved_s']:+.1f}s) vs. baseline\n")
    print(f"{'metric':>22} {'baseline':>12} {'candidate':>12} {'delta':>10}")
    for field in EFFECTIVENESS_FIELDS:
        print(f"{field:>22} {baseline[field]:>12.3f} {candidate[field]:>12.3f} {deltas[field]:>+10.3f}")
    print(
        f"\neffectiveness: win_rate_a delta {deltas['win_rate_a']:+.1%}, "
        f"avg_turns delta {deltas['avg_turns']:+.1f}, avg_actions_per_turn delta {deltas['avg_actions_per_turn']:+.1f} "
        f"-- near zero on all three means the candidate learned a comparably strong policy, not just a faster one"
    )
    print(
        f"\n(fyi: raw steps/sec -- baseline {baseline['timesteps_per_sec']:.1f}, "
        f"candidate {candidate['timesteps_per_sec']:.1f} -- included for reference, not the headline number here)"
    )


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: python {sys.argv[0]} <baseline_csv> <candidate_csv> [target_episodes]")
    baseline_path, candidate_path = sys.argv[1], sys.argv[2]
    target_episodes = int(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_TARGET_EPISODES
    baseline = _final_row(baseline_path, target_episodes)
    candidate = _final_row(candidate_path, target_episodes)
    print_comparison(baseline_path, candidate_path, baseline, candidate, compare(baseline, candidate), target_episodes)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # ponytail self-check: no pytest in this project, mirrors the
        # assert-based demo convention every other module here uses.
        target = 1000
        baseline = _row = {
            "train_wall_s": 100.0, "episodes_trained_a": 1000.0, "episodes_trained_b": 1010.0,
            "timesteps_per_sec": 100.0, "win_rate_a": 0.52, "win_rate_b": 0.46, "draw_rate": 0.02,
            "avg_turns": 12.0, "avg_actions_per_turn": 3.0,
        }
        candidate = {
            "train_wall_s": 90.0, "episodes_trained_a": 1000.0, "episodes_trained_b": 1000.0,
            "timesteps_per_sec": 125.0, "win_rate_a": 0.50, "win_rate_b": 0.48, "draw_rate": 0.02,
            "avg_turns": 12.5, "avg_actions_per_turn": 3.1,
        }
        # Simulate what _final_row derives, without needing real CSV files on disk.
        for row in (baseline, candidate):
            episodes_min = min(row["episodes_trained_a"], row["episodes_trained_b"])
            row["wall_s_per_episode"] = row["train_wall_s"] / episodes_min
            row["projected_s_for_target"] = row["wall_s_per_episode"] * target
        # baseline: 100s / 1000 episodes (the tighter floor, not 1010) = 0.1 s/ep -> 100.0s projected
        assert abs(baseline["projected_s_for_target"] - 100.0) < 1e-9
        # candidate: 90s / 1000 episodes = 0.09 s/ep -> 90.0s projected
        assert abs(candidate["projected_s_for_target"] - 90.0) < 1e-9
        deltas = compare(baseline, candidate)
        assert abs(deltas["time_saved_s"] - 10.0) < 1e-9
        assert abs(deltas["time_saved_pct"] - 10.0) < 1e-9
        assert abs(deltas["win_rate_a"] - (-0.02)) < 1e-9
        print_comparison("baseline.csv", "candidate.csv", baseline, candidate, deltas, target)
        print("\ncompare_bench_runs.py self-check: OK")
