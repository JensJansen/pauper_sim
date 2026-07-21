"""Unified train/eval runner: one entrypoint for any deck, keyed off a
JSON model config under configs/ instead of a bespoke script per deck.
Supersedes train_tron.py, train_spy_combo.py, evaluate_drl.py -- same
TrainingHarness underneath, just driven by data instead of hardcoded
constants. See docs/DECK_REGISTRY_REFRESH_PLAN.md.

    python run.py <config_name> <num_runs> [--train] [--log]

<config_name> resolves configs/<config_name>.json, which is also the
model identifier: models/<config_name>/ is that config's single
canonical, continuously-trained model slot -- decoupled from the deck
itself, so e.g. rakdos_madness_aggressive.json and rakdos_madness_
conservative.json can both play data/rakdos_madness.txt with different
tuning, each training its own model. <num_runs> is episodes when
--train, games otherwise. --log gates evaluate()'s existing per-game JSON
log (eval mode only -- training's own console output is unaffected,
shell-redirect it yourself if you want a file).

Training continues an existing model at that config's slot if one is
already there (true incremental training), or starts fresh if not.
Continuing against a config whose reward/horizon/card-set no longer
matches what's saved is not supported -- TrainingHarness.load()'s own
mismatch check fails loudly rather than silently producing a mismatched
model; make a new config file instead of editing one in place once it's
trained a model.
"""

import argparse
import json
import os
import time

from sb3_contrib import MaskablePPO

import game
import rewards
import terminated
from harness import TrainingHarness

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_CLS = MaskablePPO  # every config uses this today -- no JSON field for it, add one in this one place if that ever changes (YAGNI)

# Safety-cap multiplier for total_timesteps: max_episodes (SB3's
# StopTrainingOnMaxEpisodes) is the real stop condition, this only needs
# to be generously above whatever steps/episode a given deck actually
# takes. Historical range across every deck trained so far: ~13 (Tron) to
# ~104 (spy_combo) steps/episode -- no per-config field needed.
TIMESTEPS_PER_RUN = 2000


def load_config(config_name):
    path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    with open(path) as f:
        raw = json.load(f)

    decklist = game.parse_decklist_file(os.path.join(DATA_DIR, raw["decklist"]))
    reward_fn_names = raw["reward_fns"]  # first trains, rest are eval-only scoring_fns

    return {
        "decklist": decklist,
        "terminated_fn": getattr(terminated, raw["terminated_fn"]),
        "reward_fn": getattr(rewards, reward_fn_names[0]),
        "scoring_fns": [getattr(rewards, name) for name in reward_fn_names[1:]],
        "pending_kinds": game.derive_pending_kinds(decklist),
        "horizon": raw["horizon"],
        "on_the_play": raw["on_the_play"],
        "combat_enabled": raw.get("combat_enabled", False),
        "seed": raw.get("seed", 0),
        "n_envs": raw.get("n_envs", 1),
        "model_kwargs": raw.get("model_kwargs", {}),
    }


def model_path(config_name):
    return os.path.join(MODELS_DIR, config_name)


def _load_harness(path, config_name, cfg):
    try:
        return TrainingHarness.load(
            path, reward_fn=cfg["reward_fn"], model_cls=MODEL_CLS, decklist=cfg["decklist"],
            terminated_fn=cfg["terminated_fn"], pending_kinds=cfg["pending_kinds"],
            horizon=cfg["horizon"], on_the_play=cfg["on_the_play"], scoring_fns=cfg["scoring_fns"],
            combat_enabled=cfg["combat_enabled"],
        )
    except ValueError as e:
        raise ValueError(
            f"{e}\nconfigs/{config_name}.json no longer matches the model already saved at {path}/ -- "
            f"editing a config in place after it's trained a model isn't supported. Create a new config "
            f"file (e.g. configs/{config_name}_v2.json) for the new settings instead."
        ) from None


def train(config_name, cfg, num_runs):
    path = model_path(config_name)
    total_timesteps = num_runs * TIMESTEPS_PER_RUN

    if os.path.isdir(path):
        print(f"Continuing training: {path}/ already exists.")
        harness = _load_harness(path, config_name, cfg)
    else:
        print(f"Training fresh: {path}/ does not exist yet.")
        harness = TrainingHarness(
            reward_fn=cfg["reward_fn"], model_cls=MODEL_CLS, decklist=cfg["decklist"],
            terminated_fn=cfg["terminated_fn"], pending_kinds=cfg["pending_kinds"],
            model_kwargs=cfg["model_kwargs"], horizon=cfg["horizon"], on_the_play=cfg["on_the_play"],
            seed=cfg["seed"], scoring_fns=cfg["scoring_fns"], n_envs=cfg["n_envs"],
            combat_enabled=cfg["combat_enabled"],
        )

    t0 = time.time()
    harness.train(total_timesteps=total_timesteps, save_path=path, max_episodes=num_runs)
    dt = time.time() - t0
    print(f"\nTrained {harness.total_timesteps_trained} timesteps ({num_runs} episodes) "
          f"in {dt:.1f}s ({dt / harness.total_timesteps_trained * 1000:.2f} ms/step).")
    print(f"Saved to {path}/ (model.zip + metadata.json).")


def evaluate(config_name, cfg, num_runs, log):
    path = model_path(config_name)
    if not os.path.isdir(path):
        raise SystemExit(
            f"No trained model at {path}/ -- nothing to evaluate. Train it first: "
            f"python run.py {config_name} <runs> --train"
        )

    harness = _load_harness(path, config_name, cfg)
    print(f"Loaded {path}/ -- trained for {harness.total_timesteps_trained} timesteps.")

    log_path = None
    if log:
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = os.path.join(LOGS_DIR, f"{config_name}_{int(time.time())}.json")

    t0 = time.time()
    # Distinct from the training seed (cfg["seed"]) so evaluation always
    # plays fresh shuffles, never the exact games training already saw --
    # same "train once, evaluate against many shuffles later" convention
    # evaluate_drl.py used (EVAL_SEED=1, distinct from SEED=0).
    results = harness.evaluate(
        num_games=num_runs, horizon=cfg["horizon"], seed=cfg["seed"] + 1, log_path=log_path,
        config_name=config_name,
    )
    dt = time.time() - t0

    print(f"\nEvaluated {num_runs} games in {dt:.1f}s ({dt / num_runs * 1000:.2f} ms/game).\n")
    if log_path:
        print(f"Wrote per-game log to {log_path} (drop into src/viz to inspect).")
    game.print_report(results, cfg["horizon"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_name", help="configs/<config_name>.json -- also the models/<config_name>/ identifier")
    parser.add_argument("num_runs", type=int, help="episodes if --train, games otherwise")
    parser.add_argument("--train", action="store_true", help="train (continuing an existing model if one exists) instead of evaluating")
    parser.add_argument("--log", action="store_true", help="write a per-game JSON log (eval mode only)")
    args = parser.parse_args()

    cfg = load_config(args.config_name)
    if args.train:
        train(args.config_name, cfg, args.num_runs)
    else:
        evaluate(args.config_name, cfg, args.num_runs, args.log)


if __name__ == "__main__":
    main()
