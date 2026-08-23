"""Every mid-run quality/stats check lives here, one module per check, run
through to completion by run_training_pipeline.py at each cadence point
(~every checks_cadence_pct of a league's total_games/deck).

To add a new check: write a module exposing NAME (str) and run(ctx) -> dict
(see _common.ValidationContext for what ctx carries, and any existing check
for the shape -- write your own detail JSON via _common.write_league_json/
write_deck_json and, if it's trend-worthy, a compact record per deck via
_common.append_metric), then add it to CHECKS below.

Registry order matters: mulligan_audit is a post-hoc analysis of games the
two round-robin checks already played (ctx.collected_game_logs), not a
separate batch of its own, so it must run after both.
"""
from . import mulligan_audit, round_robin_primary, round_robin_training, vs_history
from ._common import ValidationContext  # re-exported for run_training_pipeline.py

CHECKS = [round_robin_primary, round_robin_training, mulligan_audit, vs_history]


def run_all(ctx):
    """Runs every registered check in order. A check that raises is logged
    and skipped -- one bad checkpoint load or a torch error in a single
    check must never stop training, which keeps running regardless (see
    run_training_pipeline.py). Returns {check_name: summary_dict_or_None},
    None marking a check that errored."""
    results = {}
    for check in CHECKS:
        try:
            summary = check.run(ctx)
            print(f"  [{check.NAME}] {summary}", flush=True)
            results[check.NAME] = summary
        except Exception as e:  # noqa: BLE001 -- a validation failure must never abort training
            print(f"  [{check.NAME}] FAILED: {type(e).__name__}: {e}", flush=True)
            results[check.NAME] = None
    return results
