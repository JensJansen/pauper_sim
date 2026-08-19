"""run_league.py's CLI surface, torch-free.

build_arg_parser() (moved here verbatim from run_league.py) only calls
argparse.ArgumentParser()/add_argument() -- it never touches torch or rl.*
-- so it lives in its own module that imports nothing but argparse/stdlib,
independent of run_league.py itself, which imports rl.arch/rl.deck/
rl.league/rl.agent/rl.train/rl.mulligan and, through them, torch -- a
multi-second cost this module avoids paying just to read the flag spec.

This module used to also hold LEAGUE_MODES/LEAGUE_GLOBAL, hand-authored
UI-grouping metadata for the training-ops web UI's form -- removed along
with that UI (2026-08-19); run_league.py's real CLI (below) is the only
surface now."""
import argparse


def build_arg_parser(description=None):
    """description defaults to None (no --help description) since this
    module's own __doc__ is about league_cli_spec.py, not run_league.py --
    run_league.py's main() passes its own __doc__ explicitly so `python
    run_league.py --help` still shows the real script's usage text unchanged."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag below whose default is None (instead of a real value) is
    # CONFIG-BACKED: main() resolves it as explicit-flag > --run-config /
    # --league-config value > the hardcoded fallback commented alongside each
    # one -- see main()'s own resolution block, the ONE place all three tiers
    # are reconciled. None here means "the user didn't say," never "off" or
    # "zero" -- so a config-backed value can still validly BE 0 or empty
    # without being mistaken for "not given" (e.g. checkpoint_opponent_rate=0.0).
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
    parser.add_argument("--matchup", nargs=2, metavar=("DECK_A", "DECK_B"), default=None,
                         help="Fixed A-vs-B pairing instead of league opponent sampling (snapshotting off, no "
                              "auto-sizing). Trains both decks with their real mulligan models via the unified loop.")
    parser.add_argument("--games", type=int, default=50, help="Total games (per deck round) for --matchup mode.")
    parser.add_argument("--decks", type=str, default=None, metavar="A,B,...",
                         help="Train only this comma-separated subset of the roster; the rest stay loaded as FROZEN "
                              "opponents (onboarding a new deck / targeted retraining). Falls back to --league-config's "
                              "own train_decks, then to the full roster (train everyone in it).")
    parser.add_argument("--train-deck-only", action="store_true",
                         help="Train the per-deck policies only; freeze the mulligan models.")
    parser.add_argument("--train-mulligan-only", action="store_true",
                         help="Train the mulligan models only; freeze the per-deck policies (a clean bandit vs fixed skill).")
    parser.add_argument("--eval", action="store_true",
                         help="Eval / log-generation: play games with NO training over the current live agents "
                              "(round-robin with mirrors over --decks, or one A-vs-B pairing with --matchup). "
                              "--games games per pairing.")
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
                              "deck's current live net, independent of how many snapshots exist (rl.league.LeaguePool."
                              "sample_opponent). Default 0.0 (run config): every game is real-model-vs-real-model, no "
                              "checkpoint opponents at all -- deliberately off during early training, when the "
                              "snapshot pool is mostly barely-trained early copies. The one run-config value expected "
                              "to be overridden deliberately per-invocation (e.g. a scheduled 0.0 -> 0.20 switch "
                              "partway through a long run), rather than edited in the file.")
    parser.add_argument("--pfsp", action=argparse.BooleanOptionalAction, default=None,
                         help="Weight opponent sampling (both the deck choice and, within a deck, the snapshot "
                              "choice) toward whoever is CURRENTLY beating the training deck most often, instead of "
                              "uniform (rl.league.LeaguePool.sample_opponent's pfsp param; see its own docstring for "
                              "the weighting formula and the anti-starvation floor). --pfsp / --no-pfsp; default "
                              "True (run config, or hardcoded if no config) -- safe to leave on: a never-yet-played "
                              "opponent gets a neutral prior, so this is statistically uniform until real win/loss "
                              "data exists to weight by.")
    parser.add_argument("--league-name", type=str, default=None,
                         help="Store this session's checkpoints under checkpoints/<league-name>/ instead of the "
                              "default checkpoints/league/. Prefer --league-config, which sets this AND the roster "
                              "together; this flag is the lower-level override (or use it alone with no config at "
                              "all, matching the pre-config-file interface).")
    parser.add_argument("--gauntlet-league-name", type=str, default=None,
                         help="An INDEPENDENTLY-trained twin league (checkpoints/<name>/) to periodically measure "
                              "this league's live nets against (rl.league_runner._run_eval_vs_gauntlet) -- a genuinely "
                              "external reference, unlike this league's own historical snapshots. Optional; most "
                              "leagues won't have one. Prefer --league-config's own gauntlet_league_name field (see "
                              "training_configs/run_gauntlet.json's _note); this flag is the lower-level override.")
    parser.add_argument("--heuristic-decks", type=str, default=None, metavar="A,B,...",
                         help="Deck(s) to ALSO periodically measure against rl.heuristic_agent.HeuristicAgent -- the "
                              "gauntlet's hand-authored, non-learned tier-1 member (rl.league_runner._run_eval_vs_heuristic). "
                              "Default: none (most decks don't have an owner-authored heuristic opponent). Prefer "
                              "--league-config's own heuristic_decks field; this flag is the lower-level override.")
    parser.add_argument("--roster", type=str, default=None, metavar="A,B,...",
                         help="Restrict the ENTIRE opponent pool (not just which decks train -- see --decks) to "
                              "this comma-separated subset: a true isolated sub-league where no deck outside the "
                              "set is ever loaded, trained, or sampled as an opponent. Reuses the full roster's "
                              "vocab/shared stack unchanged. Prefer --league-config; this is the lower-level override.")
    parser.add_argument("--total-games", type=int, default=None,
                         help="Games/deck this league trains to in total -- drives auto-sizing and the stop "
                              "condition. Default: --league-config's own total_games.")
    parser.add_argument("--run-config", type=str, default=None, metavar="PATH",
                         help="JSON of run-mechanics defaults shared across leagues (n_workers, snapshot_every_games, "
                              "checkpoint_opponent_rate, pfsp) -- see training_configs/run_default.json. Any "
                              "individual value can still be overridden by passing its own flag explicitly for one "
                              "invocation.")
    parser.add_argument("--league-config", type=str, default=None, metavar="PATH",
                         help="JSON describing one league (league_name, roster, optional train_decks, total_games, "
                              "optional gauntlet_league_name, optional heuristic_decks) -- see "
                              "training_configs/league_*.json. Drives automatic batch sizing whenever "
                              "--n-iterations is not given.")
    return parser
