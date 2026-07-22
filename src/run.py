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
from harness import TrainingHarness, evaluate_two_player, train_two_player

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

# Tokens a config's own "token_card_defs" list (names) resolves against --
# add only what an actual config needs (boggles' Eldrazi Spawn is the
# first real user; Blood/Robot/Warrior have no config referencing them
# yet, listed here anyway since they already exist and cost nothing to
# include).
TOKEN_CARD_DEFS_BY_NAME = {
    "Blood": game.BLOOD_TOKEN_CARD_DEF,
    "Robot": game.ROBOT_TOKEN_CARD_DEF,
    "Warrior": game.WARRIOR_TOKEN_CARD_DEF,
    "Eldrazi Spawn": game.ELDRAZI_SPAWN_TOKEN_CARD_DEF,
}


def _load_side(raw, suffix, primary_on_the_play=None):
    """Everything about ONE seat of a config -- decklist/reward/terminated/
    on_the_play/tokens/seed/model_kwargs -- suffix is "" for the primary
    seat (config's own top-level keys, unchanged from before 2-player mode
    existed) or "_2" for the opponent (docs/MULTIPLAYER_ENGINE_PLAN.md's
    harness pass: "decklist_2" present is the whole trigger for 2-player
    mode -- see load_config). Every `_2`-suffixed key is optional and
    falls back to a sensible mirror-match default: same reward/terminated
    function, opposite on_the_play, same model hyperparameters, its own
    (never inherited) tokens, decorrelated seed."""
    decklist = game.parse_decklist_file(os.path.join(DATA_DIR, raw[f"decklist{suffix}"]))
    reward_fn_names = raw.get(f"reward_fns{suffix}", raw["reward_fns"])
    if suffix == "":
        default_on_the_play = raw.get("on_the_play", True)
    else:
        # primary_on_the_play is None when the primary seat itself
        # randomizes each game (TwoPlayerDeckEnv.reset()) -- mirror that
        # (independent coin flip), not "not None" (== True), or the
        # opponent side would silently go back to always going first.
        default_on_the_play = None if primary_on_the_play is None else not primary_on_the_play
    return {
        "decklist": decklist,
        "terminated_fn": getattr(terminated, raw.get(f"terminated_fn{suffix}", raw["terminated_fn"])),
        "reward_fn": getattr(rewards, reward_fn_names[0]),
        "scoring_fns": [getattr(rewards, name) for name in reward_fn_names[1:]],
        "pending_kinds": game.derive_pending_kinds(decklist),
        "on_the_play": raw.get(f"on_the_play{suffix}", default_on_the_play),
        "token_card_defs": tuple(TOKEN_CARD_DEFS_BY_NAME[name] for name in raw.get(f"token_card_defs{suffix}", [])),
        "seed": raw.get(f"seed{suffix}", raw.get("seed", 0) + (1 if suffix else 0)),
        "model_kwargs": raw.get(f"model_kwargs{suffix}", raw.get("model_kwargs", {})),
    }


def load_config(config_name):
    path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    with open(path) as f:
        raw = json.load(f)

    two_player = "decklist_2" in raw
    cfg = _load_side(raw, "")
    cfg["two_player"] = two_player
    cfg["horizon"] = raw["horizon"]
    cfg["n_envs"] = raw.get("n_envs", 1)
    # "If both appear, all steps are always enabled" -- 2 decklists present
    # is this config's own multiplayer trigger, and combat is the only
    # path to a life_total win (the other new 2-player win condition,
    # alongside terminated_fn), so it's never optional once that trigger
    # fires -- unlike 1-player, where combat_enabled stays each config's
    # own opt-in choice.
    cfg["combat_enabled"] = True if two_player else raw.get("combat_enabled", False)
    if two_player:
        cfg["opponent"] = _load_side(raw, "_2", primary_on_the_play=cfg["on_the_play"])
        # horizon/n_envs/combat_enabled are genuinely shared, never per-side
        # (one game, one safety-cap horizon, one env-count, one combat
        # toggle) -- copied onto the opponent dict too so _build_harness/
        # _load_harness can read them identically regardless of which
        # side's dict they're given.
        cfg["opponent"]["horizon"] = cfg["horizon"]
        cfg["opponent"]["n_envs"] = cfg["n_envs"]
        cfg["opponent"]["combat_enabled"] = cfg["combat_enabled"]
        # Potential-based dense reward (MULTIPLAYER_GAPS.md) -- opt-in,
        # default 0.0 (no shaping, today's exact behavior). One shared
        # weight for both sides, same reasoning as horizon/n_envs/
        # combat_enabled above: it's a training-process knob, not a
        # per-deck rule.
        cfg["shaping_weight"] = raw.get("shaping_weight", 0.0)
        cfg["opponent"]["shaping_weight"] = cfg["shaping_weight"]
    return cfg


def model_path(config_name, seat=None):
    base = os.path.join(MODELS_DIR, config_name)
    return base if seat is None else os.path.join(base, seat)


def _opponent_kwargs(cfg, opponent_cfg, my_seat_idx):
    """TrainingHarness's opponent_* constructor kwargs, shared by
    _load_harness/_build_harness -- {} in 1-player mode (opponent_cfg is
    None), same shape either way."""
    if opponent_cfg is None:
        return {}
    return dict(
        opponent_decklist=opponent_cfg["decklist"], opponent_terminated_fn=opponent_cfg["terminated_fn"],
        opponent_pending_kinds=opponent_cfg["pending_kinds"],
        opponent_token_card_defs=opponent_cfg["token_card_defs"], my_seat_idx=my_seat_idx,
        shaping_weight=cfg["shaping_weight"],
    )


def _load_harness(path, config_name, cfg, opponent_cfg=None, my_seat_idx=0):
    """opponent_cfg/my_seat_idx: two-player mode only -- see cfg["opponent"]
    (load_config)/TrainingHarness's own opponent_* constructor kwargs."""
    kwargs = dict(
        reward_fn=cfg["reward_fn"], model_cls=MODEL_CLS, decklist=cfg["decklist"],
        terminated_fn=cfg["terminated_fn"], pending_kinds=cfg["pending_kinds"],
        horizon=cfg["horizon"], on_the_play=cfg["on_the_play"], scoring_fns=cfg["scoring_fns"],
        combat_enabled=cfg["combat_enabled"], token_card_defs=cfg["token_card_defs"],
        **_opponent_kwargs(cfg, opponent_cfg, my_seat_idx),
    )
    try:
        return TrainingHarness.load(path, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"{e}\nconfigs/{config_name}.json no longer matches the model already saved at {path}/ -- "
            f"editing a config in place after it's trained a model isn't supported. Create a new config "
            f"file (e.g. configs/{config_name}_v2.json) for the new settings instead."
        ) from None


def _build_harness(cfg, opponent_cfg=None, my_seat_idx=0):
    kwargs = dict(
        reward_fn=cfg["reward_fn"], model_cls=MODEL_CLS, decklist=cfg["decklist"],
        terminated_fn=cfg["terminated_fn"], pending_kinds=cfg["pending_kinds"],
        model_kwargs=cfg["model_kwargs"], horizon=cfg["horizon"], on_the_play=cfg["on_the_play"],
        seed=cfg["seed"], scoring_fns=cfg["scoring_fns"], n_envs=cfg["n_envs"],
        combat_enabled=cfg["combat_enabled"], token_card_defs=cfg["token_card_defs"],
        **_opponent_kwargs(cfg, opponent_cfg, my_seat_idx),
    )
    return TrainingHarness(**kwargs)


def _load_harness_pair(path_a, path_b, config_name, cfg, opp):
    """Both trained agents of a two-player config, loaded from disk --
    shared by _train_two_player's continuing-model branch and
    _evaluate_two_player (which is always a load: nothing to evaluate
    without a model already trained)."""
    harness_a = _load_harness(path_a, config_name, cfg, opponent_cfg=opp, my_seat_idx=0)
    harness_b = _load_harness(path_b, config_name, opp, opponent_cfg=cfg, my_seat_idx=1)
    return harness_a, harness_b


def train(config_name, cfg, num_runs):
    if cfg["two_player"]:
        _train_two_player(config_name, cfg, num_runs)
        return

    path = model_path(config_name)
    total_timesteps = num_runs * TIMESTEPS_PER_RUN

    if os.path.isdir(path):
        print(f"Continuing training: {path}/ already exists.")
        harness = _load_harness(path, config_name, cfg)
    else:
        print(f"Training fresh: {path}/ does not exist yet.")
        harness = _build_harness(cfg)

    t0 = time.time()
    harness.train(total_timesteps=total_timesteps, save_path=path, max_episodes=num_runs)
    dt = time.time() - t0
    print(f"\nTrained {harness.total_timesteps_trained} timesteps ({num_runs} episodes) "
          f"in {dt:.1f}s ({dt / harness.total_timesteps_trained * 1000:.2f} ms/step).")
    print(f"Saved to {path}/ (model.zip + metadata.json).")


def _train_two_player(config_name, cfg, num_runs):
    """decklist_2 present (load_config) is the whole trigger for this path
    -- each seat gets its OWN continuously-trained model slot
    (models/<config_name>/agent_a and .../agent_b), cross-wired against
    each other via harness.train_two_player's opponent-as-environment
    design (docs/MULTIPLAYER_ENGINE_PLAN.md)."""
    opp = cfg["opponent"]
    path_a, path_b = model_path(config_name, "agent_a"), model_path(config_name, "agent_b")
    total_timesteps = num_runs * TIMESTEPS_PER_RUN

    if os.path.isdir(path_a) and os.path.isdir(path_b):
        print(f"Continuing two-player training: {path_a}/ and {path_b}/ already exist.")
        harness_a, harness_b = _load_harness_pair(path_a, path_b, config_name, cfg, opp)
    else:
        print(f"Training fresh: {path_a}/ and {path_b}/ do not exist yet.")
        harness_a = _build_harness(cfg, opponent_cfg=opp, my_seat_idx=0)
        harness_b = _build_harness(opp, opponent_cfg=cfg, my_seat_idx=1)

    t0 = time.time()
    train_two_player(
        harness_a, harness_b, total_timesteps=total_timesteps, max_episodes=num_runs,
        save_path_a=path_a, save_path_b=path_b,
    )
    dt = time.time() - t0
    print(
        f"\nTrained {harness_a.total_timesteps_trained} timesteps per side "
        f"(target {num_runs} episodes; actually completed agent_a={harness_a.episode_count()}, "
        f"agent_b={harness_b.episode_count()}) in {dt:.1f}s."
    )
    print(f"Saved to {path_a}/ (agent_a) and {path_b}/ (agent_b) -- model.zip + metadata.json each.")


def evaluate(config_name, cfg, num_runs, log):
    if cfg["two_player"]:
        _evaluate_two_player(config_name, cfg, num_runs, log)
        return

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


def _evaluate_two_player(config_name, cfg, num_runs, log):
    opp = cfg["opponent"]
    path_a, path_b = model_path(config_name, "agent_a"), model_path(config_name, "agent_b")
    if not (os.path.isdir(path_a) and os.path.isdir(path_b)):
        raise SystemExit(
            f"No trained two-player model at {path_a}/ + {path_b}/ -- nothing to evaluate. Train it first: "
            f"python run.py {config_name} <runs> --train"
        )
    if log:
        # harness.evaluate_two_player has no per-game JSON log yet
        # (harness._snapshot_state is single-sided -- see
        # docs/MULTIPLAYER_ENGINE_PLAN.md's own "downstream impact" note),
        # so --log is silently a no-op here rather than a hard error.
        print("Note: --log isn't supported for two-player configs yet -- evaluating without one.")

    harness_a, harness_b = _load_harness_pair(path_a, path_b, config_name, cfg, opp)
    print(
        f"Loaded {path_a}/ (agent_a, {harness_a.total_timesteps_trained} steps) and "
        f"{path_b}/ (agent_b, {harness_b.total_timesteps_trained} steps)."
    )

    t0 = time.time()
    wins_a, wins_b, draws, turn_counts, action_counts = evaluate_two_player(
        harness_a, harness_b, num_games=num_runs, horizon=cfg["horizon"], seed=cfg["seed"] + 1,
    )
    dt = time.time() - t0
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0.0
    avg_actions_per_turn = (
        sum(a / t for a, t in zip(action_counts, turn_counts) if t > 0) / len(turn_counts) if turn_counts else 0.0
    )

    print(f"\nEvaluated {num_runs} games in {dt:.1f}s ({dt / num_runs * 1000:.2f} ms/game).\n")
    print(f"agent_a wins: {wins_a} ({wins_a / num_runs:.1%})")
    print(f"agent_b wins: {wins_b} ({wins_b / num_runs:.1%})")
    print(f"draws (safety-cap horizon reached): {draws} ({draws / num_runs:.1%})")
    print(f"average game length: {avg_turns:.1f} turns")
    print(f"average actions/turn: {avg_actions_per_turn:.1f}")


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
