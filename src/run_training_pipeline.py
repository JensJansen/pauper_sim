"""Single, complete training pipeline: trains a whole league from wherever
progress.json says it's at, to --config's total_games/deck, running the
full validation/ check suite every ~checks_cadence_pct of the way there
(including the final 100% segment that ends the run).

Supersedes manually re-invoking run_league.py session by session: this
script owns the whole loop internally. Chunks are fixed-size (~checks_cadence_pct
of total_games each, clipped on the final chunk), not the old doubling/
shakeout ladder -- there's no more per-invocation judgment call about
sequential-vs-parallel or when to trust a bigger batch, since one process
now runs the entire thing. One ProcessPoolExecutor is created once and
reused for every chunk, same reuse rationale run_league.py's own docstring
describes. A validation check that raises is logged and skipped -- see
validation.run_all -- training itself is never aborted by a bad check.

Config (--config, e.g. training_configs/main_league.json): the same schema
run_league.py reads (run-mechanics + league identity, "extends"-composable
via config_loader), plus training_league_name (was gauntlet_league_name),
checks_cadence_pct (default 5), checks_games (default 50).
--matchup training and one-off debug runs still go through run_league.py
directly -- this script is league training only.

Usage:
  python run_training_pipeline.py --config training_configs/main_league.json [--fresh]
"""
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

from config_loader import load_config
from repo_paths import CHECKPOINTS_DIR
from rl.league.league import PFSP_POWER
from rl.league.league_runner import TRUNK_HIDDEN, _load_progress, _run_session, advance_progress
from rl.roster import build_pool
from run_league import _resolve_device
from validation import ValidationContext, run_all


def _build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, metavar="PATH",
                    help="A league config (run_league.py's own schema, extends-composable) -- see "
                         "training_configs/main_league.json. Everything about this run is config-driven; "
                         "there is deliberately no other flag besides --fresh.")
    ap.add_argument("--fresh", action="store_true",
                    help="Wipe this league's entire checkpoint tree (live.pt/mulligan.pt/snapshots/archive/"
                         "progress.json/metrics.jsonl/checks/, everything) plus the shared vocab.json, so "
                         "every deck starts genuinely fresh. Destructive -- does not touch a training "
                         "league's own checkpoints, only this run's own primary league dir.")
    return ap


def _resolve_options(cfg):
    """Every value here is config-driven (see module docstring), resolved
    with the same fallback defaults main() would otherwise apply inline.
    Split out so the cadence/games-per-chunk arithmetic is testable without
    build_pool() or a real device -- device is returned unvalidated (a bare
    string); main() runs it through _resolve_device (a real CUDA check)
    separately."""
    league_name = cfg.get("league_name")
    assert league_name, "config has no league_name"
    total_games = cfg.get("total_games")
    assert total_games, "config has no total_games"

    n_workers = cfg.get("n_workers", 6)
    games_per_iteration = cfg.get("games_per_iteration") or max(1, n_workers)
    checks_cadence_pct = cfg.get("checks_cadence_pct", 5)
    checkpoint_rate = cfg.get("checkpoint_opponent_rate", 0.0)
    stratify_0land_pct = cfg.get("stratify_0land_pct", 0.0)
    if stratify_0land_pct > 0 and checkpoint_rate <= 0:
        # stratify only ever fires against a frozen-snapshot opponent (see
        # collect_rollout_league's own docstring) -- sample_opponent
        # (rl.league.league) can only return one when rng.random() <
        # checkpoint_rate, so at checkpoint_rate<=0 that branch is dead and
        # this config's stratify_0land_pct would silently never activate.
        print(f"Warning: stratify_0land_pct={stratify_0land_pct} is set but checkpoint_opponent_rate is "
              f"{checkpoint_rate} -- stratify only fires against a frozen-snapshot opponent, so it will "
              f"never activate at this rate.")

    return {
        "league_name": league_name,
        "total_games": total_games,
        "roster": cfg.get("roster"),
        "training_league_name": cfg.get("training_league_name"),
        "checks_cadence_pct": checks_cadence_pct,
        "checks_games": cfg.get("checks_games", 50),
        "n_workers": n_workers,
        "games_per_iteration": games_per_iteration,
        "snapshot_every": max(1, cfg.get("snapshot_every_games", 200) // games_per_iteration),
        "checkpoint_rate": checkpoint_rate,
        "stratify_0land_pct": stratify_0land_pct,
        "pfsp": cfg.get("pfsp", True),
        "pfsp_power": cfg.get("pfsp_power", PFSP_POWER),
        "device": cfg.get("device", "cpu"),
        "trunk_hidden": tuple(cfg.get("trunk_hidden", TRUNK_HIDDEN)),
        "ppo_hparams": cfg.get("ppo"),
        "seed": cfg.get("seed"),
        "cadence_games": max(1, round(total_games * checks_cadence_pct / 100)),
    }


def _wipe(league_dir):
    if league_dir.exists():
        print(f"--fresh: wiping {league_dir}")
        shutil.rmtree(league_dir)
    vocab_path = CHECKPOINTS_DIR / "vocab.json"
    if vocab_path.exists():
        print(f"--fresh: wiping {vocab_path}")
        vocab_path.unlink()


def main():
    args = _build_arg_parser().parse_args()
    opts = _resolve_options(load_config(args.config))
    league_dir = CHECKPOINTS_DIR / opts["league_name"]

    if args.fresh:
        _wipe(league_dir)

    device = _resolve_device(opts["device"])
    decklists, vocab, deck_ctxs, fixed_tables = build_pool()
    train_decks = list(opts["roster"]) if opts["roster"] is not None else sorted(decklists)

    print(f"training pipeline: league={opts['league_name']!r} total_games={opts['total_games']} "
          f"checks_cadence_pct={opts['checks_cadence_pct']} (~{opts['cadence_games']} games/deck/chunk) "
          f"checks_games={opts['checks_games']} training_league={opts['training_league_name']!r} "
          f"device={device} n_workers={opts['n_workers']}")

    executor_cm = ProcessPoolExecutor(max_workers=opts["n_workers"]) if opts["n_workers"] > 1 else nullcontext(None)
    with executor_cm as executor:
        while True:
            progress = _load_progress(str(league_dir))
            remaining = opts["total_games"] - progress["cumulative_games_per_deck"]
            if remaining <= 0:
                print(f"{opts['league_name']!r} already at {progress['cumulative_games_per_deck']}/"
                      f"{opts['total_games']} games/deck -- done")
                break
            chunk = min(opts["cadence_games"], remaining)
            n_iterations = max(1, chunk // opts["games_per_iteration"])

            _run_session(n_iterations, opts["games_per_iteration"], opts["snapshot_every"], executor,
                        opts["n_workers"], league_dir=str(league_dir), seed=opts["seed"], roster=opts["roster"],
                        pfsp=opts["pfsp"], checkpoint_rate=opts["checkpoint_rate"],
                        cumulative_games=progress["cumulative_games_per_deck"], ppo_hparams=opts["ppo_hparams"],
                        pfsp_power=opts["pfsp_power"], trunk_hidden=opts["trunk_hidden"], device=device,
                        stratify_0land_pct=opts["stratify_0land_pct"])

            # auto_sizing=False: fixed cadence-sized chunks, not the doubling
            # ladder -- last_batch_size is meaningless here and left untouched.
            advance_progress(str(league_dir), n_iterations, opts["games_per_iteration"], auto_sizing=False,
                             session_start_games=progress["cumulative_games_per_deck"])

            new_progress = _load_progress(str(league_dir))
            print(f"=== cadence checkpoint: {new_progress['cumulative_games_per_deck']}/{opts['total_games']} "
                  f"games/deck -- running validation checks ===", flush=True)
            ctx = ValidationContext(
                primary_league_name=opts["league_name"], train_decks=train_decks, decklists=decklists, vocab=vocab,
                deck_ctxs=deck_ctxs, fixed_tables=fixed_tables, games_per_check=opts["checks_games"],
                seed=opts["seed"], cumulative_games=new_progress["cumulative_games_per_deck"],
                training_league_name=opts["training_league_name"],
            )
            run_all(ctx)

    print(f"training pipeline done: {opts['league_name']!r} reached "
          f"{_load_progress(str(league_dir))['cumulative_games_per_deck']}/{opts['total_games']} games/deck")


if __name__ == "__main__":
    main()
