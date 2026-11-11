"""Is it still improving? -- every judgement about a run's progress, in one place (it was split between
analysis/eval/report_metrics.py's per-series verdicts and validation/progress_indicator.py's tick label).

Two levels, both pure functions of metrics.jsonl records (rl.league.metrics.read):

  series   one measurement followed over cadence ticks (a deck's win rate, its score against a fixed older
           self, ...): trend_z, peak_comparison, series_verdict -> IMPROVING / FLAT / REGRESSING / PAST PEAK /
           SATURATED, with Wilson intervals and the minimum effect the sample could resolve.
  tick     the whole league at ONE cadence point: assess -> IMPROVING / CHURNING / STUCK / FLAT / WARMUP from the
           strength_history / policy_movement / policy_scale records of that tick (what the in-run
           progress_indicator check prints; analysis/eval/report_metrics.py re-derives it for every tick of a
           finished run, so a threshold change can be replayed over history).

A record's wins are `live_wins` or `wins` when present, else round(score * games) (strength_* kinds record a score).

THE TICK THRESHOLDS ARE PROVISIONAL (REF_FRACTION, MIN_REF_GAMES, MOVING_FRACTION). Replayed over the 4-deck history
of the 11-deck main-league run (92k games/deck), at 100 games per deck and cell, the label flickered between
IMPROVING and CHURNING from ~70k on because the pooled score (400 games, +-5 points) sat within noise of 50 -- hence
the larger reference cell (REF_GAMES_FACTOR). The policy ratio was 0.39-0.62 at every age of that run and the
one-snapshot-apart step is ~0.13, so MOVING_FRACTION=0.25 separates "frozen" from "wandering" with room on both sides."""
import math
from statistics import NormalDist

from .stats import Tally, wilson_interval

SATURATED_AT = 0.95

# ---- tick-level (progress indicator) thresholds -----------------------------------------------------------------
REF_FRACTION = 0.6      # the reference self: the largest ladder age no newer than this fraction of the current age
MIN_REF_GAMES = 10000   # ... and at least this old: below it early-training noise swamps everything
MOVING_FRACTION = 0.25  # share of the current-vs-random policy distance the policy must have moved to count as moving
REF_GAMES_FACTOR = 4    # strength_history plays the reference cell with this many times its usual games


def wins_of(record):
    """A record's win count: live_wins / wins, else round(score * games) for a score-style record."""
    if "live_wins" in record or "wins" in record:
        return record.get("live_wins", record.get("wins", 0))
    return round(record.get("score", 0.0) * record.get("games", 0))


def wilson(wins, n, z=1.96):
    """(point estimate, lo, hi) as fractions. Wilson interval, not normal
    approximation, since rates near 0/1 would otherwise run off [0, 1]."""
    if not n:
        return float("nan"), 0.0, 1.0
    lo, hi = wilson_interval(Tally(wins, 0, n), z)
    return wins / n, max(0.0, lo), min(1.0, hi)


def two_proportion_z(w1, n1, w2, n2):
    """z for (group 2 - group 1) under a pooled-proportion null. Returns 0.0
    when either group is empty or the pooled rate is degenerate."""
    if not n1 or not n2:
        return 0.0
    pooled = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return ((w2 / n2) - (w1 / n1)) / se if se else 0.0


def trend_z(rows):
    """Cochran-Armitage trend test across ordered records: z for a linear
    change in win rate with record index. This is the verdict statistic, not
    an early-vs-late split (which discards ordering and is underpowered
    against a monotone drift); the split is still printed as a readable
    summary but does not decide the verdict.

    Positive z = improving over time. 0.0 when there is no variation to
    regress against (one record, or an all-wins/all-losses series)."""
    pts = [(i, wins_of(r), r.get("games", 0)) for i, r in enumerate(rows) if r.get("games")]
    if len(pts) < 2:
        return 0.0
    total_n = sum(n for _, _, n in pts)
    total_w = sum(w for _, w, _ in pts)
    p = total_w / total_n
    if p in (0.0, 1.0):
        return 0.0
    x_bar = sum(x * n for x, _, n in pts) / total_n
    num = sum((x - x_bar) * (w - n * p) for x, w, n in pts)
    den = math.sqrt(p * (1 - p) * sum(n * (x - x_bar) ** 2 for x, _, n in pts))
    return num / den if den else 0.0


def min_detectable_effect(n_per_group, power_z=0.84, alpha_z=1.96):
    """Smallest difference in proportions (around 50%) this many games per
    group could detect at ~80% power. Reported whenever the verdict is FLAT,
    since "no change" and "cannot tell" are different claims."""
    if not n_per_group:
        return 1.0
    return (alpha_z + power_z) * math.sqrt(2 * 0.25 / n_per_group)


def peak_comparison(rows, group=5):
    """(z of last window vs best window, index of that window's last record,
    selection-corrected |z| threshold). Negative z = the current policy is
    worse than one this run already had on disk.

    This is the verdict's primary test: a linear trend test cannot see
    rise-then-fall, and an early-vs-late split discards ordering and is
    underpowered. The question this answers is whether the policy about to
    keep training is worse than something already saved, not whether it has
    changed since the start.
    """
    pts = [r for r in rows if r.get("games")]
    if len(pts) < 2 * group:
        return 0.0, None, 2.0
    windows = [(i, pts[i:i + group]) for i in range(len(pts) - group + 1)]
    def pool(g):
        return (sum(wins_of(r) for r in g), sum(r.get("games", 0) for r in g))
    rates = [(pool(w), i + group - 1) for i, w in windows]
    (best_w, best_n), best_end = max(rates, key=lambda t: t[0][0] / t[0][1])
    (last_w, last_n), last_end = rates[-1]
    # Sidak-correct the threshold by the number of windows searched: the best
    # of W noisy windows is high by construction, so an uncorrected threshold
    # fires on pure noise.
    crit = NormalDist().inv_cdf(1 - 0.05 / max(1, len(windows)))
    if best_end == last_end:
        return 0.0, best_end, crit  # the most recent window IS the best one
    return two_proportion_z(best_w, best_n, last_w, last_n), best_end, crit


def series_verdict(rows, w_late, n_late):
    """(label, trend z, peak z, peak index)."""
    z = trend_z(rows)
    peak_z, peak_at, crit = peak_comparison(rows)
    if n_late and w_late / n_late >= SATURATED_AT:
        return "SATURATED", z, peak_z, peak_at
    if peak_z <= -crit:
        return "PAST PEAK", z, peak_z, peak_at
    if z <= -2:
        return "REGRESSING", z, peak_z, peak_at
    if z >= 2:
        return "IMPROVING", z, peak_z, peak_at
    return "FLAT", z, peak_z, peak_at


# ---- tick level ---------------------------------------------------------------------------------------------------

def ref_games(anchor_games, current_games):
    """The reference anchor among `anchor_games` for an agent at `current_games`, or None (warm-up)."""
    ok = [g for g in anchor_games if MIN_REF_GAMES <= g <= REF_FRACTION * current_games]
    return max(ok) if ok else None


def assess(own_rows, policy_by_deck, current_games):
    """The league's label at one cadence tick. own_rows: [{"deck", "anchor_games", "wins", "no_winner", "games"}]
    (strength_history's own cells); policy_by_deck: {deck: {"anchors": [{"anchor_games", "js"}], "random":
    {"js"}}} (policy_movement + policy_scale; may be empty).

    Pooled score of the CURRENT agents against their reference selves, with a 95% interval; IMPROVING if the interval
    is above 50%, else CHURNING (policy ratio >= MOVING_FRACTION), STUCK (below), or FLAT (no probe). WARMUP when no
    deck has a reference old enough. Returns {"label", "score", "lo", "hi", "games", "policy_ratio", "decks"}."""
    tally, per_deck, ratios = Tally(), {}, []
    for deck in sorted({r["deck"] for r in own_rows}):
        rows = [r for r in own_rows if r["deck"] == deck]
        ref = ref_games([r["anchor_games"] for r in rows], current_games)
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


def assess_ticks(records):
    """{cumulative_games: assess(...)} for every cadence tick that has strength_history records, re-derived from
    metrics.jsonl records (strength_history, policy_movement, policy_scale) under the CURRENT thresholds."""
    ticks = sorted({r["cumulative_games"] for r in records if r.get("kind") == "strength_history"})
    out = {}
    for tick in ticks:
        at = [r for r in records if r.get("cumulative_games") == tick]
        own = [{"deck": r["deck"], "anchor_games": r["anchor_games"], "wins": wins_of(r), "no_winner": 0,
                "games": r["games"]} for r in at if r.get("kind") == "strength_history"]
        policy = {}
        for r in at:
            if r.get("kind") == "policy_movement":
                policy.setdefault(r["deck"], {"anchors": [], "random": {"js": 0.0}})["anchors"].append(
                    {"anchor_games": r["anchor_games"], "js": r["js"]})
            elif r.get("kind") == "policy_scale" and r["deck"] in policy:
                policy[r["deck"]]["random"] = {"js": r["random_js"]}
        for r in at:  # policy_scale may precede its deck's anchors in file order
            if r.get("kind") == "policy_scale":
                policy.setdefault(r["deck"], {"anchors": [], "random": {"js": 0.0}})["random"] = {"js": r["random_js"]}
        out[tick] = assess(own, policy, tick)
    return out
