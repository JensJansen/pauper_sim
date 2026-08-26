"""Check: primary_round_robin_review -- every training deck in the primary
league plays every other, mirrors included, exactly 2 games per pairing
(fixed, unlike primary_vs_primary_round_robin's configurable
ctx.games_per_check), greedy, on current live weights.

Exists purely so a human can page through real, current games in the
webapp's replay viewer -- not for tracked win-rate metrics, so unlike every
other check it writes NOTHING under checkpoints/ and appends no
metrics.jsonl line. Its only output is a full event log written straight
into the webapp submodule's own logs/replays/ (browsable there
automatically, no "Open new file" needed -- see webapp/README.md, "Browse
server logs"), via the same _write_event_log run_league.py --log/--eval
already use. A no-op entirely when the webapp submodule isn't checked out
(webapp_mirror.webapp_ready()) -- there'd be nowhere to put the file.

Deliberately doesn't feed ctx.collected_game_logs/collected_deck_league:
these games are for visual review only, not folded into mulligan_audit's
sample (see validation/__init__.py's registry-order note on why the two
round-robin checks DO feed it -- this one is intentionally not a third
source).
"""
import time

import webapp_mirror
from rl.league.league_runner import _run_eval, _write_event_log

NAME = "primary_round_robin_review"
GAMES_PER_PAIRING = 2


def run(ctx):
    # Module-attribute access (not `from webapp_mirror import WEBAPP_DIR`) so
    # a test's monkeypatch.setattr(webapp_mirror, "WEBAPP_DIR", ...) is seen
    # here -- same reasoning as _common.py's own webapp_mirror usage.
    if not webapp_mirror.webapp_ready():
        return {"skipped": "webapp submodule not checked out"}

    game_logs = []
    t0 = time.time()
    decks, game_pairings = _run_eval(ctx.train_decks, GAMES_PER_PAIRING, greedy=True, seed=ctx.seed,
                                     game_logs=game_logs, league_dir=ctx.primary_league_dir,
                                     executor=ctx.executor, n_workers=ctx.n_workers)
    elapsed_ms = (time.time() - t0) * 1000

    log_path = webapp_mirror.WEBAPP_DIR / "logs" / "replays" / f"{ctx.primary_league_name}_round_robin_review_{ctx.cumulative_games}games.json"
    meta = {"check": NAME, "primary_league": ctx.primary_league_name, "decks": decks,
            "games_per_pairing": GAMES_PER_PAIRING, "cumulative_games": ctx.cumulative_games,
            "elapsed_ms": elapsed_ms}
    _write_event_log(str(log_path), game_logs, meta, game_pairings=game_pairings)

    return {"pairings": len(game_pairings) // GAMES_PER_PAIRING, "games": len(game_logs),
            "elapsed_ms": elapsed_ms, "wrote": str(log_path)}
