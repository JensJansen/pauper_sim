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

THE THRESHOLDS ARE PROVISIONAL (REF_FRACTION, MIN_REF_GAMES, MOVING_FRACTION): chosen before seeing a league's
history, they need tuning against real runs. The numbers behind the label are always printed and recorded, so the
label can be re-derived under other thresholds from metrics.jsonl."""
from evaluation import Tally, wilson_interval

from . import _common

NAME = "progress_indicator"

REF_FRACTION = 0.6
MIN_REF_GAMES = 10000
MOVING_FRACTION = 0.25  # share of the current-vs-random policy distance the policy has moved from its reference self


def _ref_games(anchor_games, current_games):
    ok = [g for g in anchor_games if MIN_REF_GAMES <= g <= REF_FRACTION * current_games]
    return max(ok) if ok else None


def assess(own_rows, policy_by_deck, current_games):
    """Pure function of the two checks' results. own_rows: strength_history's "own" rows; policy_by_deck:
    policy_movement's {deck: result} (may be empty). Returns {"label", "score", "lo", "hi", "games", "policy_ratio",
    "decks": {deck: {...}}}."""
    tally, per_deck, ratios = Tally(), {}, []
    for deck in sorted({r["deck"] for r in own_rows}):
        rows = [r for r in own_rows if r["deck"] == deck]
        ref = _ref_games([r["anchor_games"] for r in rows], current_games)
        if ref is None:
            continue
        r = next(r for r in rows if r["anchor_games"] == ref)
        t = Tally(r["wins"], r["no_winner"], r["games"])
        tally += t
        entry = {"ref_games": ref, "score": t.score, "games": t.games}
        pol = policy_by_deck.get(deck)
        if pol and pol["random"]["js"] > 0:
            a = next((a for a in pol["anchors"] if a["anchor_games"] == ref), None)
            if a is not None:
                entry["policy_ratio"] = a["js"] / pol["random"]["js"]
                ratios.append(entry["policy_ratio"])
        per_deck[deck] = entry
    if not tally.games:
        return {"label": "WARMUP", "decks": per_deck}
    lo, hi = wilson_interval(tally)
    ratio = sum(ratios) / len(ratios) if ratios else None
    if lo > 0.5:
        label = "IMPROVING"
    elif ratio is None:
        label = "FLAT"
    else:
        label = "CHURNING" if ratio >= MOVING_FRACTION else "STUCK"
    return {"label": label, "score": tally.score, "lo": lo, "hi": hi, "games": tally.games, "policy_ratio": ratio,
            "decks": per_deck}


def banner(result, current_games):
    rule = "=" * 76
    lines = [rule, f" PROGRESS INDICATOR (provisional thresholds) at {current_games} games/deck: {result['label']}"]
    if result["label"] != "WARMUP":
        refs = sorted({d["ref_games"] for d in result["decks"].values()})
        lines.append(f"   strength: current agents beat their own ~{refs[0]}..{refs[-1]}-game-old selves "
                     f"{100 * result['score']:.1f}% [{100 * result['lo']:.1f}, {100 * result['hi']:.1f}] "
                     f"over {result['games']} games (50 = no progress)")
        if result["policy_ratio"] is not None:
            lines.append(f"   policy:   moved {100 * result['policy_ratio']:.0f}% of the way to an unrelated policy "
                         f"(moving above {100 * MOVING_FRACTION:.0f}%)")
        else:
            lines.append("   policy:   probe not available")
    lines.append(rule)
    return "\n".join(lines)


def run(ctx):
    own = ctx.extras.get("strength_history", {}).get("own", [])
    if not own:
        return {"skipped": "strength_history produced no results"}
    result = assess(own, ctx.extras.get("policy_movement", {}), ctx.cumulative_games)
    print(banner(result, ctx.cumulative_games), flush=True)
    payload = {"check": NAME, "primary_league": ctx.primary_league_name, "cumulative_games": ctx.cumulative_games,
               "provisional": True, "ref_fraction": REF_FRACTION, "min_ref_games": MIN_REF_GAMES,
               "moving_fraction": MOVING_FRACTION, **result}
    write_path = _common.write_league_json(ctx, NAME, payload)
    _common.append_metric(ctx, kind=NAME, deck="ALL", label=result["label"], score=result.get("score"),
                          lo=result.get("lo"), hi=result.get("hi"), games=result.get("games"),
                          policy_ratio=result.get("policy_ratio"))
    return {"label": result["label"], "wrote": write_path}
