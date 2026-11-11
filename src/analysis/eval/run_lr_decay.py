"""Runs run_training_pipeline.py unchanged, with a linear learning-rate decay on the deck nets' optimizers.

The training loop has no lr schedule (lr is a flat ppo.lr), and the main pipeline is not to be edited for an
experiment, so this wrapper wraps `rl.league.league_runner.ppo_update` in-process: just before every deck-net PPO
update it sets every param group's lr to start + (end - start) * min(1, games / decay_games), where `games` is the
league's cumulative games/deck read from <league>/progress.json (which the loop advances at every snapshot, so lr is
constant within a snapshot interval of ~192 games -- <1% of a 25k decay). Everything else -- collection, schedules,
checks, publishing -- is the stock pipeline. The mulligan net's lr is NOT decayed. Each lr change is appended to
logs/lr_<league>.jsonl.

Usage (from src/): python analysis/eval/run_lr_decay.py --config ../training_configs/small_lrdecay.json
                       [--lr-start 2e-4] [--lr-end 2e-5] [--decay-games 25000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_training_pipeline
from config_loader import load_config
from rl.league import league_runner
from repo_paths import CHECKPOINTS_DIR


def install(progress_path, lr_start, lr_end, decay_games, log_path):
    """Replaces league_runner.ppo_update with a wrapper that applies the decayed lr first."""
    real_update = league_runner.ppo_update
    last = [None]

    def lr_now():
        try:
            games = json.loads(Path(progress_path).read_text())["cumulative_games_per_deck"]
        except (OSError, ValueError, KeyError):
            games = 0
        return games, lr_start + (lr_end - lr_start) * min(1.0, games / decay_games)

    def update(net, optimizer, *args, **kwargs):
        games, lr = lr_now()
        for group in optimizer.param_groups:
            group["lr"] = lr
        if lr != last[0]:
            last[0] = lr
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "cumulative_games": games, "lr": lr}) + "\n")
        return real_update(net, optimizer, *args, **kwargs)

    league_runner.ppo_update = update


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr-start", type=float, default=2e-4)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--decay-games", type=int, default=25000)
    args = ap.parse_args()
    league = load_config(args.config)["league_name"]
    log_path = Path(__file__).resolve().parents[3] / "logs" / f"lr_{league}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    install(CHECKPOINTS_DIR / league / "progress.json", args.lr_start, args.lr_end, args.decay_games, log_path)
    sys.argv = ["run_training_pipeline.py", "--config", args.config]
    run_training_pipeline.main()


if __name__ == "__main__":
    main()
