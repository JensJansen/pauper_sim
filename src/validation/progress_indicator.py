"""Check: progress_indicator -- the one-line answer to "is this league still learning?", built from what
strength_history and policy_movement just measured (registry order: it runs after both). REPORT ONLY: it prints a
banner and writes a record, and never stops or changes training.

Reference point: for each deck, its own snapshot at the largest ladder age no newer than REF_FRACTION of its current
age (and at least MIN_REF_GAMES old, below which the early-training noise swamps everything). Two readings:

  strength  the pooled score of the CURRENT agents against their reference selves, with a 95% interval. A league that
            is still improving clearly beats its 0.6x-age self (interval above 50%); at a plateau the score is 50.
  policy    how far the current policy sits from the reference self on the probe suite, as a fraction of the distance
            to a randomly initialised net (mean JS divergence ratio). Under MOVING_FRACTION the policy has barely
            moved; above it the policy is still changing a lot.

  IMPROVING  strength interval above 50%: still learning.
  CHURNING   no clear strength gain, policy still moving: changing without getting better (cycling / noise-walk).
  STUCK      no clear strength gain, policy frozen: not moving at all.
  FLAT       no clear strength gain, policy probe unavailable.
  WARMUP     no deck has a reference old enough yet.

The rule (assess/banner and the PROVISIONAL thresholds REF_FRACTION, MIN_REF_GAMES, MOVING_FRACTION) lives in
evaluation/verdicts.py, shared with analysis/eval/report_metrics.py, which re-derives this label for every tick of a
finished run from metrics.jsonl. The numbers behind the label are always printed and recorded, so it can be
re-derived under other thresholds."""
from evaluation import verdicts
from evaluation.verdicts import MOVING_FRACTION, REF_FRACTION, REF_GAMES_FACTOR, MIN_REF_GAMES, assess, banner  # noqa: F401

from . import _common

NAME = "progress_indicator"


def run(ctx):
    own = ctx.extras.get("strength_history", {}).get("own", [])
    if not own:
        return {"skipped": "strength_history produced no results"}
    result = assess(own, ctx.extras.get("policy_movement", {}), ctx.cumulative_games)
    print(banner(result, ctx.cumulative_games), flush=True)
    payload = {"check": NAME, "primary_league": ctx.primary_league_name, "cumulative_games": ctx.cumulative_games,
               "provisional": True, "ref_fraction": verdicts.REF_FRACTION,
               "min_ref_games": verdicts.MIN_REF_GAMES, "moving_fraction": verdicts.MOVING_FRACTION, **result}
    write_path = _common.write_league_json(ctx, NAME, payload)
    _common.append_metric(ctx, kind=NAME, deck="ALL", label=result["label"], score=result.get("score"),
                          lo=result.get("lo"), hi=result.get("hi"), games=result.get("games"),
                          policy_ratio=result.get("policy_ratio"))
    return {"label": result["label"], "wrote": write_path}
