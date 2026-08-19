"""League driver: one continuous training loop, with no separate "Stage 1"/
"Stage 2" curriculum. Every deck in the roster (data/league_decks.json)
trains every round, against an opponent RESAMPLED each game from a
LeaguePool (rl.league) -- historical snapshots of every deck plus everyone's
current live weights, picked uniformly. No separate stages are needed:
early on, with few decks and no snapshots yet, most games are naturally
close to mirror play; cross-deck and cross-snapshot exposure grows
organically as the pool fills in, without a hardcoded phase boundary.

Every deck owns its whole network, perception encoder included, and trains it
end to end -- there is no pretrain phase and no shared stack (removed
2026-08-17; see rl.deck's module docstring). A deck with no checkpoint yet
starts from a freshly-initialized net, so nothing has to be prepared first.
Each deck's own live net/optimizer persists in checkpoints/league/
<deck_name>/live.pt; historical opponents live alongside as
snapshot_<id>.pt (LeaguePool's own concern).

Parallel rollout collection (rl.rollout_parallel.collect_rollout_league_parallel)
is used whenever n_workers > 1 -- benchmarked at ~3.2-3.5x wall-clock
speedup at 6-8 worker processes on this machine (6 physical cores),
plateauing beyond that (hyperthreaded/logical cores past the physical
count add nothing reliable). Defaults to 6. The executor is created ONCE
for the whole session and reused across every iteration, so process-
spawn/import overhead (each worker re-importing torch + the game engine)
is paid once, not once per collection round.

--matchup DECK_A DECK_B --games N [--log PATH]: bypasses league opponent
sampling entirely -- runs N games as a DIRECT, fixed pairing between two
named decks (train_selfplay's cross-matchup path), still updating and
checkpointing both decks' live nets normally. --log PATH captures the
game engine's own existing event log (game/state.py's GameState.
log_event, already instrumented across mana.py/turn.py/resolution/*.py/
game/effects/*.py -- see rl.train.collect_rollout's own docstring) for
every game played, written as one JSON file. Logging is threaded through
both the sequential and parallel worker paths (event dicts are picklable).

Usage:
  python run_league.py --run-config PATH --league-config PATH
      Normal training: run-mechanics defaults (training_configs/run_default.json) + one league's identity
      (training_configs/league_*.json: league_name, roster, optional train_decks, total_games).
      No game count given -- it's computed automatically from the league's own progress.json (doubles each
      batch, 1/2/4/8/..., stopping once total_games is reached). Either config's
      individual values, or --league-name/--roster/--n-workers/etc. directly, still override for one call.
  python run_league.py --n-iterations N [--snapshot-every N] [--n-workers N]
      Debug / one-off: force an exact iteration count, bypassing auto-SIZING. progress.json's
      cumulative_games_per_deck still advances (it is the horizon the PPO schedules ramp against and has
      nothing to do with the ladder -- gating it here was BUG 1); only last_batch_size stays untouched, so a
      forced size never becomes the next auto-sized batch's base. Config files still apply for anything not
      explicitly overridden.
  python run_league.py --matchup DECK_A DECK_B [--games N] [--log PATH]
      Fixed A-vs-B pairing, no league opponent sampling, no auto-sizing.

This script is a thin CLI wrapper: every reusable piece (session driving,
eval-mode functions, checkpoint/progress helpers) lives in rl.league_runner
-- benchmarking/training_run.py imports that directly rather than importing
this 954-line-turned-~140-line CLI script for its side-effect-free functions.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import torch  # _resolve_device: validate --device before any net is built

from repo_paths import CHECKPOINTS_DIR
from rl.league import PFSP_POWER
from rl.league_cli_spec import build_arg_parser
from rl.league_runner import (
    EVAL_EVERY_SESSIONS, EVAL_GAMES, TRUNK_HIDDEN, advance_progress,
    _load_progress, _next_batch_games, _run_eval, _run_session, _save_progress, _write_event_log,
)


def _resolve_device(name):
    """Validates a device string at STARTUP, before any net is built.

    Fails loudly on two things that would otherwise surface much later and
    much less legibly: an unrecognized device name (which torch reports from
    inside ppo_update's first tensor allocation, several minutes into a
    session), and asking for cuda on a machine without a working CUDA build
    (which silently is not something to fall back to -- a run launched with
    --device cuda that quietly trained on CPU would produce timings the owner
    would reasonably read as GPU timings)."""
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

    # Both configs optional -- {} when omitted, so every .get() below falls
    # through to its own hardcoded default exactly as if no config existed
    # at all (the pre-config-file interface keeps working unchanged).
    run_cfg = json.load(open(args.run_config)) if args.run_config else {}
    league_cfg = json.load(open(args.league_config)) if args.league_config else {}

    train_deck = not args.train_mulligan_only
    train_mulligan = not args.train_deck_only
    assert train_deck or train_mulligan, "cannot freeze BOTH layers (--train-deck-only + --train-mulligan-only)"

    # Deck identity: explicit flag > --league-config > (roster itself, for
    # train_decks -- omitting it from the config means "train everyone in
    # the roster," _run_session's own existing default).
    roster = args.roster.split(",") if args.roster else league_cfg.get("roster")
    train_decks = args.decks.split(",") if args.decks else league_cfg.get("train_decks", roster)
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
            # "decks" logs the RESOLVED roster _run_eval actually played, not the raw
            # --decks arg -- train_decks is None whenever --decks was omitted (the
            # common case), which is not the same as "no decks played."
            _write_event_log(args.log, game_logs, {"mode": "eval", "matchup": list(matchup) if matchup else None,
                                                   "decks": resolved_decks, "greedy": not args.sampled, "games_logged": len(game_logs)},
                              game_pairings=game_pairings)
        return

    # Run mechanics: explicit flag > --run-config > hardcoded default -- the
    # SAME three-tier resolution as deck identity above, just against
    # run_cfg instead of league_cfg.
    n_workers = args.n_workers if args.n_workers is not None else run_cfg.get("n_workers", 6)
    checkpoint_rate = args.checkpoint_opponent_rate if args.checkpoint_opponent_rate is not None else run_cfg.get("checkpoint_opponent_rate", 0.0)
    pfsp = args.pfsp if args.pfsp is not None else run_cfg.get("pfsp", True)

    # Sizing: --matchup counts by --games (per deck round, its own scheme,
    # unaffected by any of this); an explicit --n-iterations forces an exact
    # size and is a pure debug escape hatch (see its own --help text -- never
    # fed back into the doubling sequence, though it DOES advance the league's
    # cumulative game count; see league_runner.advance_progress);
    # otherwise this league's own total_games + how far it's already gotten
    # (progress.json) determine the next batch automatically.
    # Logging is threaded through BOTH the sequential and MP league paths
    # (event dicts are picklable), so --log does not force sequential collection.
    auto_sizing = False
    if matchup is not None:
        # Matchup mode never parallelizes across workers (always sequential,
        # see `sequential` below) -- a flat floor of 10 is all this has ever
        # needed; used to be a CLI-configurable value, but the clamp already
        # forced it to 10 for anyone who didn't explicitly push it higher.
        games_per_iteration = min(10, args.games)
        n_iterations = max(1, args.games // games_per_iteration)
    else:
        # One game per worker: collect_rollout_league_parallel splits n_games
        # across n_workers via plain `n_games // n_workers` -- fewer games than
        # workers silently starves the rest (they get zero games and are never
        # even submitted). Used to be an independent, CLI-tunable value
        # (default 2) that could silently conflict with --n-workers; benchmarked
        # (src/benchmarking/training_run.py) against 1x/2x/3x n_workers on this
        # machine and 1x was both the simplest (never under-provisions) and the
        # fastest measured (2x/3x add PPO-update-side cost -- a bigger buffer to
        # update on -- without adding real collection parallelism once every
        # worker already has work).
        # games_per_iteration: one game per worker by default (the benchmark
        # above). Overridable via run-config so the games-per-POLICY-UPDATE
        # ratio can be raised without a code edit -- at the default of 6, every
        # ppo_update spends its whole trust region on six games of evidence
        # (approx_kl 0.044 against target_kl 0.03, i.e. early stopping on most
        # updates). Raising it costs
        # wall-clock exactly as the benchmark comment above predicts -- a bigger
        # buffer per update, no extra collection parallelism -- which is a
        # deliberate trade, not an oversight.
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

    # snapshot_every_games (a fixed games-count) is the preferred run-config
    # path; the raw --snapshot-every flag (iterations) is a lower-level
    # override for either it or the config -- converted using games_per_
    # iteration ABOVE this line since it's already fully resolved by here.
    if args.snapshot_every is not None:
        snapshot_every = args.snapshot_every
    elif "snapshot_every_games" in run_cfg:
        snapshot_every = max(1, run_cfg["snapshot_every_games"] // games_per_iteration)
    else:
        snapshot_every = 20

    # cumulative_games_per_deck (progress.json, 0 for a brand-new or
    # config-less league): the horizon rl.train.batch_size_for_iteration and
    # rl.train.ent_coef_schedule ramp against -- see both docstrings for why
    # this replaced the old session-LOCAL iteration/n_iterations tracking
    # (a real run is many separate process invocations; the old version
    # reset both ramps back to their start value at the beginning of every
    # one of them, regardless of overall run progress).
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
                            # trunk_hidden: DeckNetwork's TRAINABLE trunk widths. Applies
                            # only to a deck with no live.pt yet -- _run_session reads the
                            # width back off an existing checkpoint when resuming, so
                            # editing this cannot shape-mismatch a league already on disk,
                            # it just does nothing. JSON gives a list; DeckNetwork wants a
                            # tuple.
                            trunk_hidden=tuple(run_cfg.get("trunk_hidden", TRUNK_HIDDEN)),
                            # Explicit flag wins over the run-config, which wins
                            # over "cpu" -- same precedence every other mechanic
                            # here uses. Validated rather than passed through
                            # blind: a typo'd device name would otherwise surface
                            # deep inside ppo_update's first tensor allocation,
                            # and asking for cuda on a box without it should say
                            # so at startup rather than after loading four nets.
                            device=_resolve_device(args.device or run_cfg.get("device", "cpu")))

    sequential = matchup is not None or n_workers <= 1  # matchup uses collect_rollout directly (no worker path)
    if not sequential:
        # ONE executor for the whole session, reused across every iteration --
        # process-spawn/import overhead is paid once, not per collection round.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            _run_session(n_iterations, games_per_iteration, snapshot_every, executor, n_workers, **schedule_kwargs)
    else:
        _run_session(n_iterations, games_per_iteration, snapshot_every, None, 1, **schedule_kwargs)

    # BUG 1 FIX (2026-08-13): cumulative_games_per_deck must advance for EVERY
    # batch, not only auto-sized ones -- see league_runner.advance_progress for
    # the full rationale and the invariant it protects.
    # BUG 3 FIX (2026-08-17): league_runner.checkpoint_progress has already been
    # advancing the same counter at every snapshot point during the session, so
    # this final write passes the session's own starting count and computes the
    # total absolutely rather than adding to what those writes left on disk.
    if league_dir is not None:
        advance_progress(league_dir, n_iterations, games_per_iteration, auto_sizing,
                         session_start_games=cumulative_games)

    if args.log:
        meta = {"mode": "matchup" if matchup else "league", "matchup": list(matchup) if matchup else None,
                "train_decks": train_decks, "games_logged": len(game_logs)}
        # One fixed pairing for the whole run (--log is only ever wired through
        # --matchup mode) -- same schema as _run_eval's per-game pairings, just
        # constant, so every log this script can produce is consistently shaped.
        game_pairings = [matchup] * len(game_logs) if matchup else None
        _write_event_log(args.log, game_logs, meta, game_pairings=game_pairings)


if __name__ == "__main__":
    main()
