"""Segmented self-play training: trains a two-player config in fixed-size
batches (default 1000 episodes each), running a deterministic evaluation
pass after every batch to measure win rate/game length/actions-per-turn
without the noise of the exploration-driven training policy, and timing
both phases separately. Ten batches means ten rows of trend data (wall
time, throughput, win rate, actions/turn) instead of one opaque total.

    python train_boggles_mirror_segments.py <config_name> [--batches N] [--episodes-per-batch N] [--eval-games N]

Reuses run.py's config loading/harness building (same models/<config_name>/
agent_a, agent_b slots -- a plain `python run.py <config_name> <n> --train`
afterward just continues the same models) and harness.py's
train_two_player/evaluate_two_player -- no engine or training-loop changes,
this only adds the segmenting/timing/CSV wrapper around them.

train_two_player has no episode-exact stop (unlike 1-player TrainingHarness.
train(), which uses SB3's StopTrainingOnMaxEpisodes) -- it only takes a
total_timesteps budget. run.py's own 2-player CLI reuses TIMESTEPS_PER_RUN=
2000 ("<num_runs> is episodes when --train"), which is calibrated for
1-player decks under an exact episode cap and is NOT a real per-episode
step count for any 2-player deck. A live check against the real boggles
deck (`python run.py boggles_mirror 1 --train`) measured ep_len_mean ~60-69
steps/episode -- using TIMESTEPS_PER_RUN's 2000 as if it meant "steps per
episode" would train ~30x more games per batch than requested.

So run_segment guarantees the floor directly rather than estimating it once
and hoping: it calls train_two_player in a loop, chunk by chunk, re-checking
the EXACT episode count after every chunk via _episodes_completed (which
reads Monitor's own uncapped per-env episode_lengths list -- SB3's own
ep_info_buffer is capped to the last 100 and would undercount a multi-
thousand-episode batch), and keeps going until BOTH harness_a and harness_b
have each actually completed at least episodes_per_batch episodes. Every
episode is bounded by `horizon` (a hard per-turn cap), so this always
terminates; it can only ever overshoot the floor slightly (the last chunk's
own size), never undershoot it. steps_per_episode_estimate only sizes each
chunk (to minimize how many chunks are needed) and is refined both within a
batch (from that batch's own real rate so far) and across batches (from the
previous batch's final rate) -- it affects efficiency, never correctness.
"""

import argparse
import csv
import os
import statistics
import time

import run
from harness import TrainingHarness, evaluate_two_player, train_two_player

# A fresh, untrained boggles_mirror policy measured ep_len_mean 60.5/68.9
# (SB3's own rolling log) -- start a bit above that so batch 1 doesn't
# undertrain relative to episodes_per_batch; every later batch retargets
# from its own previous batch's real measurement (see run_segmented_training).
INITIAL_STEPS_PER_EPISODE_ESTIMATE = 90

CSV_FIELDS = [
    "batch", "episodes_target", "episodes_trained_a", "episodes_trained_b", "chunks_used",
    "timesteps_trained_a", "timesteps_trained_b", "timesteps_delta", "train_wall_s", "timesteps_per_sec",
    "steps_per_episode_used", "steps_per_episode_next", "eval_games", "eval_wall_s", "ms_per_eval_game",
    "win_rate_a", "win_rate_b", "draw_rate", "avg_turns", "avg_actions_per_game", "avg_actions_per_turn",
]


def _episodes_completed(harness):
    """Exact count of episodes finished so far by this harness's own model,
    summed across every parallel env (n_envs>1) -- Monitor.get_episode_rewards()
    is a plain, never-truncated list for the env's whole lifetime (unlike
    SB3's own ep_info_buffer, capped to the last 100), so this stays exact
    even across a batch of many thousands of episodes.

    Goes through env_method("get_episode_rewards") rather than .envs directly:
    .envs only exists on DummyVecEnv (the actual env objects, reachable because
    everything's one process) -- env_method is the generic VecEnv API that works
    identically for SubprocVecEnv too (see docs/GPU_VECENV_INVESTIGATION.md)."""
    return sum(len(r) for r in harness.model.get_env().env_method("get_episode_rewards"))


MIN_CHUNK_TIMESTEPS = 500  # floor on each retry chunk -- keeps the tail of the loop (small "remaining" counts) from thrashing through many tiny, overhead-dominated train_two_player calls


def run_segment(harness_a, harness_b, episodes_per_batch, eval_games, horizon, seed,
                 steps_per_episode_estimate=INITIAL_STEPS_PER_EPISODE_ESTIMATE, path_a=None, path_b=None):
    """One batch: loops train_two_player in chunks, re-measuring the EXACT
    episode count via _episodes_completed after each chunk (never an
    estimate), until BOTH harness_a and harness_b have each actually
    completed at least episodes_per_batch episodes -- a real floor, not a
    budget that might undershoot. Every episode is bounded by `horizon`
    (a hard per-turn cap), so this always terminates. Each chunk's size is
    itself sized from steps_per_episode_estimate (refined chunk-to-chunk
    from what's actually been observed THIS batch, not just carried over
    from the last one) purely to minimize how many chunks are needed --
    it can never cause an undercount, only extra (small, shrinking-as-you-
    approach-target) overshoot in the final chunk. Then evaluate_games
    deterministic games measure the resulting policy cleanly, isolated
    from training's own exploration noise. Returns a flat stats dict
    including steps_per_episode_next, the caller's cue for seeding the
    FOLLOWING batch's initial chunk-size estimate."""
    before_ts_a, before_ts_b = harness_a.total_timesteps_trained, harness_b.total_timesteps_trained
    before_ep_a, before_ep_b = _episodes_completed(harness_a), _episodes_completed(harness_b)
    rate = steps_per_episode_estimate
    chunks_used = 0

    t0 = time.time()
    while True:
        ep_delta_a = _episodes_completed(harness_a) - before_ep_a
        ep_delta_b = _episodes_completed(harness_b) - before_ep_b
        remaining = episodes_per_batch - min(ep_delta_a, ep_delta_b)
        if remaining <= 0:
            break
        chunk = max(MIN_CHUNK_TIMESTEPS, round(remaining * rate))
        train_two_player(harness_a, harness_b, total_timesteps=chunk)
        chunks_used += 1
        ts_so_far_a = harness_a.total_timesteps_trained - before_ts_a
        ep_so_far_a = _episodes_completed(harness_a) - before_ep_a
        if ep_so_far_a > 0:  # refine chunk sizing with this batch's own real rate, not just the carried-over guess
            rate = ts_so_far_a / ep_so_far_a
    train_wall_s = time.time() - t0

    if path_a:
        harness_a.save(path_a)
    if path_b:
        harness_b.save(path_b)

    t0 = time.time()
    wins_a, wins_b, draws, turn_counts, action_counts = evaluate_two_player(
        harness_a, harness_b, num_games=eval_games, horizon=horizon, seed=seed,
    )
    eval_wall_s = time.time() - t0

    ts_delta_a = harness_a.total_timesteps_trained - before_ts_a
    ts_delta_b = harness_b.total_timesteps_trained - before_ts_b
    ep_delta_a = _episodes_completed(harness_a) - before_ep_a
    ep_delta_b = _episodes_completed(harness_b) - before_ep_b
    assert ep_delta_a >= episodes_per_batch and ep_delta_b >= episodes_per_batch  # the floor this loop exists to guarantee

    steps_per_episode_next = steps_per_episode_estimate
    if ep_delta_a > 0 and ep_delta_b > 0:
        steps_per_episode_next = (ts_delta_a / ep_delta_a + ts_delta_b / ep_delta_b) / 2

    actions_per_turn = [a / t for a, t in zip(action_counts, turn_counts) if t > 0]

    return {
        "chunks_used": chunks_used,
        "episodes_trained_a": ep_delta_a,
        "episodes_trained_b": ep_delta_b,
        "timesteps_trained_a": harness_a.total_timesteps_trained,
        "timesteps_trained_b": harness_b.total_timesteps_trained,
        "timesteps_delta": ts_delta_a + ts_delta_b,
        "train_wall_s": train_wall_s,
        "timesteps_per_sec": (ts_delta_a + ts_delta_b) / train_wall_s if train_wall_s > 0 else 0.0,
        "steps_per_episode_used": steps_per_episode_estimate,
        "steps_per_episode_next": steps_per_episode_next,
        "eval_games": eval_games,
        "eval_wall_s": eval_wall_s,
        "ms_per_eval_game": eval_wall_s / eval_games * 1000 if eval_games else 0.0,
        "win_rate_a": wins_a / eval_games if eval_games else 0.0,
        "win_rate_b": wins_b / eval_games if eval_games else 0.0,
        "draw_rate": draws / eval_games if eval_games else 0.0,
        "avg_turns": statistics.mean(turn_counts) if turn_counts else 0.0,
        "avg_actions_per_game": statistics.mean(action_counts) if action_counts else 0.0,
        "avg_actions_per_turn": statistics.mean(actions_per_turn) if actions_per_turn else 0.0,
    }


def _print_batch_line(batch, num_batches, episodes_per_batch, stats):
    print(
        f"batch {batch}/{num_batches}: trained {stats['episodes_trained_a']}/{stats['episodes_trained_b']} "
        f"episodes (a/b, target {episodes_per_batch}; {stats['chunks_used']} chunk(s), "
        f"{stats['timesteps_delta']} steps total) in "
        f"{stats['train_wall_s']:.1f}s ({stats['timesteps_per_sec']:.1f} steps/s); "
        f"eval {stats['eval_games']} games in {stats['eval_wall_s']:.1f}s "
        f"({stats['ms_per_eval_game']:.1f} ms/game) -- "
        f"win_a={stats['win_rate_a']:.1%} win_b={stats['win_rate_b']:.1%} draw={stats['draw_rate']:.1%} "
        f"avg_turns={stats['avg_turns']:.1f} actions/turn={stats['avg_actions_per_turn']:.1f}"
    )


def _print_summary(rows):
    print("\n=== Segmented training summary ===")
    print(f"{'batch':>5} {'ep_a':>6} {'ep_b':>6} {'train_s':>8} {'steps/s':>8} {'eval_s':>7} "
          f"{'win_a':>6} {'win_b':>6} {'draw':>6} {'turns':>6} {'act/turn':>8}")
    for r in rows:
        print(
            f"{r['batch']:>5} {r['episodes_trained_a']:>6} {r['episodes_trained_b']:>6} {r['train_wall_s']:>8.1f} "
            f"{r['timesteps_per_sec']:>8.1f} {r['eval_wall_s']:>7.1f} {r['win_rate_a']:>6.1%} "
            f"{r['win_rate_b']:>6.1%} {r['draw_rate']:>6.1%} {r['avg_turns']:>6.1f} {r['avg_actions_per_turn']:>8.1f}"
        )
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print(
            f"\nTrend batch 1 -> {last['batch']}: win_a {first['win_rate_a']:.1%} -> {last['win_rate_a']:.1%}, "
            f"avg_turns {first['avg_turns']:.1f} -> {last['avg_turns']:.1f}, "
            f"actions/turn {first['avg_actions_per_turn']:.1f} -> {last['avg_actions_per_turn']:.1f}, "
            f"steps/s {first['timesteps_per_sec']:.1f} -> {last['timesteps_per_sec']:.1f}"
        )
    total_train_s = sum(r["train_wall_s"] for r in rows)
    total_eval_s = sum(r["eval_wall_s"] for r in rows)
    print(f"\nTotal wall time: {total_train_s + total_eval_s:.1f}s (train {total_train_s:.1f}s, eval {total_eval_s:.1f}s)")


def run_segmented_training(config_name, num_batches=10, episodes_per_batch=1000, eval_games=100, csv_path=None):
    cfg = run.load_config(config_name)
    if not cfg["two_player"]:
        raise SystemExit(f"configs/{config_name}.json has no decklist_2 -- this script is two-player-config only.")
    opp = cfg["opponent"]
    path_a, path_b = run.model_path(config_name, "agent_a"), run.model_path(config_name, "agent_b")

    if os.path.isdir(path_a) and os.path.isdir(path_b):
        print(f"Continuing: {path_a}/ and {path_b}/ already exist.")
        harness_a, harness_b = run._load_harness_pair(path_a, path_b, config_name, cfg, opp)
    else:
        print(f"Starting fresh: {path_a}/ and {path_b}/ do not exist yet.")
        harness_a = run._build_harness(cfg, opponent_cfg=opp, my_seat_idx=0)
        harness_b = run._build_harness(opp, opponent_cfg=cfg, my_seat_idx=1)

    os.makedirs(run.LOGS_DIR, exist_ok=True)
    csv_path = csv_path or os.path.join(run.LOGS_DIR, f"{config_name}_segments_{int(time.time())}.csv")
    rows = []
    steps_per_episode_estimate = INITIAL_STEPS_PER_EPISODE_ESTIMATE

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for batch in range(1, num_batches + 1):
            stats = run_segment(
                harness_a, harness_b, episodes_per_batch, eval_games, cfg["horizon"],
                seed=cfg["seed"] + 1 + batch, steps_per_episode_estimate=steps_per_episode_estimate,
                path_a=path_a, path_b=path_b,
            )
            row = {"batch": batch, "episodes_target": episodes_per_batch, **stats}
            rows.append(row)
            writer.writerow(row)
            f.flush()
            _print_batch_line(batch, num_batches, episodes_per_batch, stats)
            steps_per_episode_estimate = stats["steps_per_episode_next"]

    _print_summary(rows)
    print(f"\nWrote {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_name", nargs="?", default="boggles_mirror",
                         help="configs/<config_name>.json -- must have decklist_2 (default: boggles_mirror)")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--episodes-per-batch", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    args = parser.parse_args()
    run_segmented_training(args.config_name, args.batches, args.episodes_per_batch, args.eval_games)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # ponytail self-check: no pytest in this project, mirrors harness.py's
        # own assert-based __main__ convention. Exercises run_segment (the
        # actual new logic -- timing wrapper, exact episode counting via
        # Monitor, and the adaptive steps/episode retarget) end-to-end with
        # tiny real MaskablePPO models, at a scale that runs in seconds
        # rather than the real script's hours. Doesn't touch run.py's config-
        # loading path (that machinery is already covered by run.py/harness.py's
        # own tests) -- only what this file adds.
        import game
        import rewards

        deck_a = [("Mountain", 20), ("Lightning Bolt", 10)]
        deck_b = [("Mountain", 20)]
        pending_a = game.derive_pending_kinds(deck_a)
        pending_b = game.derive_pending_kinds(deck_b)
        tiny_model_kwargs = {
            "policy_kwargs": {"net_arch": [8, 8]}, "verbose": 0, "device": "cpu", "n_steps": 32, "batch_size": 16,
        }

        from sb3_contrib import MaskablePPO

        harness_a = TrainingHarness(
            reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_a,
            terminated_fn=lambda s: False, pending_kinds=pending_a, model_kwargs=tiny_model_kwargs,
            horizon=10, on_the_play=True, seed=0, opponent_decklist=deck_b,
            opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_b, my_seat_idx=0,
        )
        harness_b = TrainingHarness(
            reward_fn=rewards.strict_binary_reward, model_cls=MaskablePPO, decklist=deck_b,
            terminated_fn=lambda s: False, pending_kinds=pending_b, model_kwargs=tiny_model_kwargs,
            horizon=10, on_the_play=False, seed=1, opponent_decklist=deck_a,
            opponent_terminated_fn=lambda s: False, opponent_pending_kinds=pending_a, my_seat_idx=1,
        )

        rows = []
        estimate = 20  # tiny horizon=10 mountain-only deck: short games, small initial guess
        for batch in range(1, 3):
            stats = run_segment(
                harness_a, harness_b, episodes_per_batch=2, eval_games=2, horizon=10,
                seed=batch, steps_per_episode_estimate=estimate,
            )
            rows.append({"batch": batch, "episodes_target": 2, **stats})
            assert stats["timesteps_delta"] > 0
            assert stats["episodes_trained_a"] >= 0 and stats["episodes_trained_b"] >= 0
            assert 0.0 <= stats["win_rate_a"] <= 1.0 and 0.0 <= stats["win_rate_b"] <= 1.0
            assert stats["avg_turns"] > 0
            assert stats["avg_actions_per_turn"] >= 0.0
            estimate = stats["steps_per_episode_next"]
        _print_summary(rows)
        print("train_boggles_mirror_segments.py self-check: OK")
