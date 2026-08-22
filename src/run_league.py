"""League driver: one continuous training loop. Every deck in the roster
(data/league_decks.json) trains every round against an opponent resampled
each game from a LeaguePool (rl.league.league) -- historical snapshots of
every deck plus everyone's current live weights.

Every deck owns its whole network, perception encoder included, and trains
it end to end. A deck with no checkpoint yet starts from a
freshly-initialized net. Each deck's own live net/optimizer persists in
checkpoints/league/<deck_name>/live.pt; historical opponents live alongside
as snapshot_<id>.pt (LeaguePool's own concern).

Parallel rollout collection (rl.training.rollout_parallel.collect_rollout_league_parallel)
is used whenever n_workers > 1 -- benchmarked at ~3.2-3.5x wall-clock
speedup at 6-8 worker processes on this machine (6 physical cores),
plateauing beyond that. Defaults to 6. The executor is created once for the
whole session and reused across every iteration, so process-spawn/import
overhead is paid once, not once per collection round.

--matchup DECK_A DECK_B --games N [--log PATH]: bypasses league opponent
sampling entirely -- runs N games as a direct, fixed pairing between two
named decks, still updating and checkpointing both decks' live nets
normally. --log PATH captures the game engine's own event log
(game/state.py's GameState.log_event) for every game played, written as
one JSON file. Logging works through both the sequential and parallel
worker paths (event dicts are picklable).

Usage:
  python run_league.py --run-config PATH --league-config PATH
      Normal training: run-mechanics defaults (training_configs/run_default.json) + one league's identity
      (training_configs/league_*.json: league_name, roster, total_games).
      No game count given -- it's computed automatically from the league's own progress.json (doubles each
      batch, 1/2/4/8/..., stopping once total_games is reached). Either config's
      individual values, or --league-name/--roster/--n-workers/etc. directly, still override for one call.
  python run_league.py --n-iterations N [--snapshot-every N] [--n-workers N]
      Debug / one-off: force an exact iteration count, bypassing auto-sizing. progress.json's
      cumulative_games_per_deck still advances (it is the horizon the PPO schedules ramp against);
      only last_batch_size stays untouched, so a forced size never becomes the next auto-sized batch's base.
      Config files still apply for anything not explicitly overridden.
  python run_league.py --matchup DECK_A DECK_B [--games N] [--log PATH]
      Fixed A-vs-B pairing, no league opponent sampling, no auto-sizing.

This script is a thin CLI wrapper: every reusable piece (session driving,
eval-mode functions, checkpoint/progress helpers) lives in
rl.league.league_runner -- benchmarking/training_run.py imports that
directly rather than importing this script for its side-effect-free
functions.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch  # needed for _resolve_device's CUDA check

from repo_paths import CHECKPOINTS_DIR
from rl.league.league import PFSP_POWER
from rl.league_cli_spec import build_arg_parser
from rl.league.league_runner import (
    EVAL_EVERY_SESSIONS, EVAL_GAMES, TRUNK_HIDDEN, advance_progress,
    _load_progress, _next_batch_games, _run_eval, _run_session, _save_progress, _write_event_log,
)


def _load_config(path):
    """Loads one training_configs/*.json, resolving an optional top-level
    "extends": "<sibling-file>.json" as a base layer merged underneath this
    file's own keys -- so a value shared with another config (most commonly
    run_default.json's run-mechanics) is inherited instead of retyped, and
    can never silently drift out of sync the way a manual copy can. Resolved
    relative to the extending file's own directory, and recursively (a base
    may itself extend another base)."""
    if not path:
        return {}
    path = Path(path)
    cfg = json.load(open(path))
    base_name = cfg.pop("extends", None)
    if base_name is None:
        return cfg
    return {**_load_config(path.parent / base_name), **cfg}


def _resolve_device(name):
    """Validates a device string at startup, before any net is built: an
    unrecognized name or a cuda request without CUDA available both fail
    immediately instead of surfacing deep inside ppo_update or silently
    falling back to CPU."""
    name = str(name).lower()
    if name == "cpu":
        return "cpu"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {name}: torch reports no CUDA available "
                             f"(torch {torch.__version__}). Use --device cpu.")
        return name
    raise SystemExit(f"--device {name}: expected 'cpu' or 'cuda[:N]'.")


def main():
    args = build_arg_parser(description=__doc__).parse_args()

    # Missing configs read as {}, so every .get() below falls through to
    # its own hardcoded default.
    run_cfg = _load_config(args.run_config)
    league_cfg = _load_config(args.league_config)

    train_deck = not args.train_mulligan_only
    train_mulligan = not args.train_deck_only
    assert train_deck or train_mulligan, "cannot freeze BOTH layers (--train-deck-only + --train-mulligan-only)"

    # Deck identity: explicit flag > --league-config. --matchup narrows
    # training to the two named decks (handled in _run_session).
    roster = args.roster.split(",") if args.roster else league_cfg.get("roster")
    train_decks = roster
    matchup = tuple(args.matchup) if args.matchup else None
    game_logs = [] if args.log else None
    league_name = args.league_name or league_cfg.get("league_name")
    league_dir = CHECKPOINTS_DIR / league_name if league_name else None
    gauntlet_league_name = args.gauntlet_league_name or league_cfg.get("gauntlet_league_name")
    gauntlet_league_dir = CHECKPOINTS_DIR / gauntlet_league_name if gauntlet_league_name else None
    heuristic_decks = args.heuristic_decks.split(",") if args.heuristic_decks else league_cfg.get("heuristic_decks", [])

    if args.eval:  # no training: round-robin (or single matchup) over current live agents
        resolved_decks, game_pairings = _run_eval(train_decks, args.games, not args.sampled, args.seed, game_logs,
                                                    matchup=matchup, league_dir=league_dir)
        if args.log:
            # resolved_decks is the roster _run_eval actually played, not
            # train_decks (which may be None even though decks were played).
            _write_event_log(args.log, game_logs, {"mode": "eval", "matchup": list(matchup) if matchup else None,
                                                   "decks": resolved_decks, "greedy": not args.sampled, "games_logged": len(game_logs)},
                              game_pairings=game_pairings)
        return

    # Same three-tier resolution as deck identity above (flag > config > default).
    n_workers = args.n_workers if args.n_workers is not None else run_cfg.get("n_workers", 6)
    checkpoint_rate = args.checkpoint_opponent_rate if args.checkpoint_opponent_rate is not None else run_cfg.get("checkpoint_opponent_rate", 0.0)
    pfsp = args.pfsp if args.pfsp is not None else run_cfg.get("pfsp", True)

    # Sizing: --matchup counts by --games; --n-iterations forces an exact
    # size (a debug escape hatch that still advances the league's cumulative
    # game count -- see league_runner.advance_progress -- but is never fed
    # back into the doubling sequence); otherwise this league's total_games
    # and how far it's already gotten (progress.json) determine the next
    # batch automatically.
    auto_sizing = False
    if matchup is not None:
        # Matchup mode is always sequential (see `sequential` below); a flat
        # floor of 10 is enough.
        games_per_iteration = min(10, args.games)
        n_iterations = max(1, args.games // games_per_iteration)
    else:
        # One game per worker: n_games // n_workers silently starves any
        # worker beyond what divides evenly (zero games, never submitted).
        # Overridable via run-config's games_per_iteration to raise the
        # games-per-policy-update ratio, at the cost of PPO update wall-clock
        # (a bigger buffer per update adds no extra collection parallelism
        # once every worker already has work).
        games_per_iteration = run_cfg.get("games_per_iteration") or max(1, n_workers)
        if args.n_iterations is not None:
            n_iterations = args.n_iterations
        else:
            total_games = args.total_games if args.total_games is not None else league_cfg.get("total_games")
            assert league_dir is not None, (
                "no --n-iterations given and no league to auto-size from -- pass --league-config (or at least "
                "--league-name together with --total-games), or force an exact size with "
                "--n-iterations for a one-off debug run"
            )
            assert total_games is not None, (
                f"auto-sizing needs total_games -- got {total_games!r} (from --league-config, or --total-games directly)"
            )
            next_batch_games = _next_batch_games(league_dir, total_games)
            if next_batch_games is None:
                progress = _load_progress(league_dir)
                print(f"{league_name!r} already at {progress['cumulative_games_per_deck']}/{total_games} "
                      f"games/deck -- nothing to run")
                return
            n_iterations = max(1, next_batch_games // games_per_iteration)
            auto_sizing = True

    # snapshot_every_games (a games-count) is preferred; --snapshot-every
    # overrides in iterations, converted here since games_per_iteration is
    # already resolved above.
    if args.snapshot_every is not None:
        snapshot_every = args.snapshot_every
    elif "snapshot_every_games" in run_cfg:
        snapshot_every = max(1, run_cfg["snapshot_every_games"] // games_per_iteration)
    else:
        snapshot_every = 20

    # cumulative_games_per_deck (0 for a brand-new or config-less league) is
    # the horizon rl.training.train.batch_size_for_iteration and
    # ent_coef_schedule ramp against.
    cumulative_games = _load_progress(league_dir)["cumulative_games_per_deck"] if league_dir is not None else 0

    schedule_kwargs = dict(seed=args.seed,
                            train_deck=train_deck, train_mulligan=train_mulligan, train_decks=train_decks,
                            matchup=matchup, game_logs=game_logs, checkpoint_rate=checkpoint_rate,
                            league_dir=league_dir, roster=roster, pfsp=pfsp,
                            gauntlet_league_dir=gauntlet_league_dir, heuristic_decks=heuristic_decks,
                            cumulative_games=cumulative_games,
                            ppo_hparams=run_cfg.get("ppo"),
                            eval_games=run_cfg.get("eval_games", EVAL_GAMES),
                            eval_every_sessions=run_cfg.get("eval_every_sessions", EVAL_EVERY_SESSIONS),
                            pfsp_power=run_cfg.get("pfsp_power", PFSP_POWER),
                            # DeckNetwork's trainable trunk widths. Applies only to a
                            # deck with no live.pt yet -- an existing checkpoint's own
                            # width wins on resume, so this can't shape-mismatch a
                            # league already on disk. JSON gives a list; DeckNetwork
                            # wants a tuple.
                            trunk_hidden=tuple(run_cfg.get("trunk_hidden", TRUNK_HIDDEN)),
                            # Same flag > config > default precedence as elsewhere;
                            # validated so a bad device name or missing CUDA fails at
                            # startup, not mid-session.
                            device=_resolve_device(args.device or run_cfg.get("device", "cpu")))

    sequential = matchup is not None or n_workers <= 1  # matchup uses collect_rollout directly (no worker path)
    if not sequential:
        # One executor reused for the whole session, so process-spawn/import
        # overhead is paid once, not per iteration.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers, **schedule_kwargs)
    else:
        _run_session(n_iterations, games_per_iteration, snapshot_every, None, 1, **schedule_kwargs)

    # cumulative_games_per_deck must advance for every batch, not only
    # auto-sized ones (see league_runner.advance_progress). league_runner.
    # checkpoint_progress already advanced the same counter at each snapshot
    # point during the session, so this final write passes the session's
    # starting count and computes the total absolutely rather than adding to
    # what those writes left on disk.
    if league_dir is not None:
        advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing,
                         session_start_games=cumulative_games)

    if args.log:
        meta = {"mode": "matchup" if matchup else "league", "matchup": list(matchup) if matchup else None,
                "train_decks": train_decks, "games_logged": len(game_logs)}
        # One fixed pairing for the whole run -- same schema as _run_eval's
        # per-game pairings, just constant.
        game_pairings = [matchup] * len(game_logs) if matchup else None
        _write_event_log(args.log, game_logs, meta, game_pairings=game_pairings)


if __name__ == "__main__":
    main()
