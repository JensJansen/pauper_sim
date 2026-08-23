"""run_league.py's CLI surface, torch-free.

build_arg_parser() only calls argparse.ArgumentParser()/add_argument() -- it
never touches torch or rl.* -- so it lives in its own module importing
nothing but argparse/stdlib, independent of run_league.py itself (which
imports rl.model.arch/rl.model.deck/rl.league.league/rl.decision.agent/
rl.training.train/rl.model.mulligan and, through them, torch)."""
import argparse


def build_arg_parser(description=None):
    """description defaults to None; run_league.py's main() passes its own
    __doc__ explicitly so `python run_league.py --help` shows the real
    script's usage text."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Flags defaulting to None are CONFIG-BACKED: main() resolves them as
    # explicit-flag > --run-config/--league-config value > hardcoded fallback
    # (see main()'s own resolution block). None means "not given", never
    # "off" or "zero", so a config-backed value can still validly be 0.
    parser.add_argument("--n-iterations", type=int, default=None,
                         help="Force this exact number of iterations, bypassing auto-SIZING (debugging / "
                              "one-off runs only). Never fed back into the doubling ladder, so a forced "
                              "size cannot become the next auto-sized batch's base -- but the league's "
                              "cumulative_games_per_deck DOES still advance, since that is the horizon the "
                              "PPO minibatch/entropy schedules ramp against and has nothing to do with the "
                              "ladder (gating it here was BUG 1: it froze both schedules at their origin "
                              "for an entire 60,001-games/deck run). Omit this for normal training: the "
                              "size is computed from --league-config's total_games and how far that league "
                              "has already gotten (checkpoints/<league_name>/progress.json).")
    parser.add_argument("--snapshot-every", type=int, default=None,
                         help="Snapshot cadence in ITERATIONS. Prefer --run-config's snapshot_every_games (a fixed "
                              "games-count, independent of games_per_iteration) -- this raw flag is a lower-level "
                              "override. Default 20 if neither is given.")
    parser.add_argument("--n-workers", type=int, default=None, help="Default 6 (run config).")
    parser.add_argument("--device", type=str, default=None, metavar="cpu|cuda",
                         help="Where the PPO update runs. Default cpu (or --run-config's own `device`). "
                              "Collection is always CPU across n_workers processes regardless -- it is "
                              "single-game-at-a-time inference, which a GPU cannot help with -- so this moves "
                              "only the gradient work, which is ~86%% of session wall time. Measured on this "
                              "repo's own checkpoints (analysis/eval/bench_gpu_vs_cpu.py): cuda runs ppo_update "
                              "1.6-2.25x faster with epochs_run identical on both arms, the gap widening with "
                              "buffer size. Checkpoints are always written on CPU, so a league can move between "
                              "devices between sessions with no conversion.")
    parser.add_argument("--matchup", nargs=2, metavar=("DECK_A", "DECK_B"), default=None,
                         help="Fixed A-vs-B pairing instead of league opponent sampling (snapshotting off, no "
                              "auto-sizing). Trains both decks with their real mulligan models via the unified loop.")
    parser.add_argument("--games", type=int, default=50, help="Total games (per deck round) for --matchup mode.")
    parser.add_argument("--train-deck-only", action="store_true",
                         help="Train the per-deck policies only; freeze the mulligan models.")
    parser.add_argument("--train-mulligan-only", action="store_true",
                         help="Train the mulligan models only; freeze the per-deck policies (a clean bandit vs fixed skill).")
    parser.add_argument("--eval", action="store_true",
                         help="Eval / log-generation: play --games games with NO training, one A-vs-B pairing "
                              "over current live agents -- requires --matchup. Roster-wide round-robin eval "
                              "moved to validation.round_robin_primary, run via run_training_pipeline.py's own "
                              "cadence rather than this flag.")
    parser.add_argument("--sampled", action="store_true",
                         help="Eval: sample from the policy instead of argmaxing (default: greedy/argmax, so logged "
                              "eval games show the policy's actual best play; pass this for the old exploratory-"
                              "sampled behavior).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed the rng (reproducible eval logs; also reproducible training sampling).")
    parser.add_argument("--log", type=str, default=None, metavar="PATH",
                         help="Write the game engine's own event log for every game this session to PATH as JSON "
                              "(sequential collection only).")
    parser.add_argument("--checkpoint-opponent-rate", type=float, default=None,
                         help="Probability that a sampled opponent is a frozen historical snapshot rather than that "
                              "deck's current live net, independent of how many snapshots exist (rl.league.league.LeaguePool."
                              "sample_opponent). Default 0.0 (run config): every game is real-model-vs-real-model, no "
                              "checkpoint opponents at all -- deliberately off during early training, when the "
                              "snapshot pool is mostly barely-trained early copies. The one run-config value expected "
                              "to be overridden deliberately per-invocation (e.g. a scheduled 0.0 -> 0.20 switch "
                              "partway through a long run), rather than edited in the file.")
    parser.add_argument("--pfsp", action=argparse.BooleanOptionalAction, default=None,
                         help="Weight opponent sampling (both the deck choice and, within a deck, the snapshot "
                              "choice) toward whoever is CURRENTLY beating the training deck most often, instead of "
                              "uniform (rl.league.league.LeaguePool.sample_opponent's pfsp param; see its own docstring for "
                              "the weighting formula and the anti-starvation floor). --pfsp / --no-pfsp; default "
                              "True (run config, or hardcoded if no config) -- safe to leave on: a never-yet-played "
                              "opponent gets a neutral prior, so this is statistically uniform until real win/loss "
                              "data exists to weight by.")
    parser.add_argument("--league-name", type=str, default=None,
                         help="Store this session's checkpoints under checkpoints/<league-name>/ instead of the "
                              "default checkpoints/league/. Prefer --league-config, which sets this AND the roster "
                              "together; this flag is the lower-level override (or use it alone with no config at "
                              "all, matching the pre-config-file interface).")
    parser.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                         help="Restrict the ENTIRE opponent pool to this comma-separated subset: a true isolated "
                              "sub-league where no deck outside the set is ever loaded, trained, or sampled as an "
                              "opponent. Reuses the full roster's vocab/shared stack unchanged. Prefer "
                              "--league-config; this is the lower-level override.")
    parser.add_argument("--total-games", type=int, default=None,
                         help="Games/deck this league trains to in total -- drives auto-sizing and the stop "
                              "condition. Default: --league-config's own total_games.")
    parser.add_argument("--run-config", type=str, default=None, metavar="PATH",
                         help="JSON of run-mechanics defaults shared across leagues (n_workers, snapshot_every_games, "
                              "checkpoint_opponent_rate, pfsp) -- see training_configs/run_default.json. Any "
                              "individual value can still be overridden by passing its own flag explicitly for one "
                              "invocation. A league config can inherit these instead of retyping them via its own "
                              "top-level \"extends\": \"run_default.json\" -- see training_configs/league_main.json.")
    parser.add_argument("--league-config", type=str, default=None, metavar="PATH",
                         help="JSON describing one league (league_name, roster, total_games) -- see "
                              "training_configs/league_*.json. Drives automatic batch sizing whenever "
                              "--n-iterations is not given. May itself \"extend\" a run-mechanics config (see "
                              "--run-config) so both flags can point at the same self-sufficient file. A config's "
                              "own training_league_name/heuristic_decks fields, if present, are read by "
                              "run_training_pipeline.py, not this script.")
    return parser
