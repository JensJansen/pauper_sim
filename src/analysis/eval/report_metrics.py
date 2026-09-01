"""Summarizes checkpoints/<league>/metrics.jsonl -- the per-iteration "ppo" /
"mulligan" / "mulligan_audit" / "vs_history" / "primary_vs_primary_round_robin" /
"primary_vs_training_round_robin" records
rl.league.league_runner's _run_session appends during every training run.
Plain-text, stdlib only.

Leads with the per-record sequence (most recent last), then pools:

  - four verdicts -- IMPROVING / FLAT / REGRESSING (linear decline) / PAST
    PEAK (current window below one this run already reached, regardless of
    overall trend);
  - FLAT is annotated with the minimum effect the sample size could detect;
  - Wilson intervals, which stay inside [0, 1] near saturation;
  - vs_history labels kept strictly separate: `archive_oldest` is permanently
    snapshot_0 (a ~200-game policy), `active_oldest` tracks a rolling
    ~6,400-game-old self -- never pooled into one "vs its past self" number.

Every field is read with .get so records written before a field existed still
parse -- metrics.jsonl is append-only across many schema revisions.

Usage: python analysis/eval/report_metrics.py <league_dir> [--window N]
  e.g. python analysis/eval/report_metrics.py ../checkpoints/4_deck_subleague_test
"""
import json
import math
import sys
from collections import defaultdict
from statistics import NormalDist

SATURATED_AT = 0.95
TREND_WINDOW = 24  # per-record values shown; older ones are pooled into "early"


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wilson(wins, n, z=1.96):
    """(point estimate, lo, hi) as fractions. Wilson interval, not normal
    approximation, since rates near 0/1 would otherwise run off [0, 1]."""
    if not n:
        return float("nan"), 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


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
    pts = [(i, r.get("live_wins", r.get("wins", 0)), r.get("games", 0)) for i, r in enumerate(rows) if r.get("games")]
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
    windows = [(i, pts[i:i + group]) for i in range(0, len(pts) - group + 1)]
    pool = lambda g: (sum(r.get("live_wins", r.get("wins", 0)) for r in g), sum(r.get("games", 0) for r in g))
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


def _verdict(rows, w_late, n_late):
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


def _win_rate_lines(deck, tag, rows, window):
    """Trend-first summary for one (deck, instrument) win-rate series."""
    rows = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))
    total_n = sum(r.get("games", 0) for r in rows)
    if not total_n:
        return [f"  [{tag}] no completed games"]

    shown = rows[-window:]
    trend = " ".join(f"{100 * r.get('live_wins', r.get('wins', 0)) / r['games']:.0f}" for r in shown if r.get("games"))
    half = max(1, len(rows) // 2)
    early, late = rows[:half], rows[half:]
    pool = lambda g: (sum(r.get("live_wins", r.get("wins", 0)) for r in g), sum(r.get("games", 0) for r in g))
    we, ne = pool(early)
    wl, nl = pool(late)
    verdict, z, peak_z, peak_at = _verdict(rows, wl, nl)
    p, lo, hi = wilson(wl, nl)
    pe, _, _ = wilson(we, ne)

    lines = [f"  [{tag}] {len(rows)} records, {total_n} games"]
    if len(rows) > window:
        lines.append(f"    trend (last {window} of {len(rows)}): {trend}")
    else:
        lines.append(f"    trend: {trend}")
    lines.append(f"    early {100 * pe:.1f}% (n={ne})  ->  late {100 * p:.1f}% "
                 f"[{100 * lo:.1f}, {100 * hi:.1f}] (n={nl})   trend z={z:+.2f}   {verdict}")
    if verdict == "PAST PEAK" and peak_at is not None:
        peak_row = [r for r in rows if r.get("games")][peak_at]
        lines.append(f"    !! current window is {abs(peak_z):.1f} sigma BELOW this run's best "
                     f"(peaked around session {peak_row.get('session', '?')}) -- "
                     f"a better policy than the live one is already on disk")
    if verdict == "FLAT":
        mde = min_detectable_effect(min(ne, nl))
        lines.append(f"    (FLAT at this sample size resolves >= {100 * mde:.1f}pp; "
                     f"a smaller real change would read the same)")
    return lines


def _ppo_lines(deck, rows):
    rows = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))
    half = max(1, len(rows) // 10)  # deciles
    mean = lambda g, k: (sum(r[k] for r in g if k in r) / max(1, sum(1 for r in g if k in r)))
    first, last = rows[:half], rows[-half:]

    lines = [f"  [ppo] {len(rows)} iterations (first/last decile means)"]
    for key, fmt in (("entropy", ".4f"), ("value_loss", ".4f"), ("policy_loss", ".4f"),
                     ("approx_kl", ".4f"), ("clip_fraction", ".3f"), ("epochs_run", ".2f"),
                     ("explained_variance", ".3f"), ("adv_std", ".4f")):
        if not any(key in r for r in rows):
            continue  # field absent from this schema revision
        a, b = mean(first, key), mean(last, key)
        note = ""
        if key == "approx_kl":
            target = last[-1].get("target_kl", 0.03)
            note = f"   vs target_kl={target}  {'EXCEEDS' if b > target else 'ok'}"
        lines.append(f"    {key:<20}{a:>10{fmt}} -> {b:>9{fmt}}{note}")
    tail = rows[-1]
    lines.append(f"    latest: buffer={tail.get('buffer_size', '?')} batch={tail.get('batch_size', '?')} "
                 f"ent_coef={tail.get('ent_coef', '?')} games={tail.get('games', '?')} "
                 f"cumulative_games={tail.get('cumulative_games', 'NOT RECORDED')}")
    return lines


def report(records, window=TREND_WINDOW):
    """Returns the report as a list of printed lines (also prints them)."""
    by_deck = defaultdict(lambda: defaultdict(list))
    sessions = []
    for r in records:
        kind = r.get("kind", "?")
        # session_start is league-level, not per-deck: no `deck` field, kept separate.
        if kind == "session_start":
            sessions.append(r)
            continue
        # vs_history's label is part of the series identity, never pooled away.
        tag = f"{kind}:{r['label']}" if kind == "vs_history" and "label" in r else kind
        by_deck[r.get("deck", "?")][tag].append(r)

    lines = []
    if sessions:
        last = sessions[-1]
        rewards = sorted({s.get("reward_fn", "?") for s in sessions})
        lines.append(f"=== run === {len(sessions)} session(s), reward_fn={'/'.join(rewards)}, "
                     f"roster={','.join(last.get('roster', []))}, "
                     f"train_decks={','.join(last.get('train_decks', []))}")
    for deck in sorted(by_deck):
        lines.append(f"=== {deck} ===")
        for tag in sorted(by_deck[deck]):
            rows = by_deck[deck][tag]
            if tag == "ppo":
                lines.extend(_ppo_lines(deck, rows))
            elif tag == "mulligan":
                tail = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))[-1]
                lines.append(f"  [mulligan] {len(rows)} updates -- latest loss={tail.get('loss', float('nan')):.4f} "
                             f"(n={tail.get('n', '?')})")
            elif any(r.get("games") is not None for r in rows):
                lines.extend(_win_rate_lines(deck, tag, rows, window))
    for line in lines:
        print(line)
    return lines


def main():
    args = [a for a in sys.argv[1:]]
    window = TREND_WINDOW
    if "--window" in args:
        i = args.index("--window")
        window = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 1:
        print("usage: python analysis/eval/report_metrics.py <league_dir> [--window N]")
        raise SystemExit(1)
    report(load(f"{args[0]}/metrics.jsonl"), window=window)


if __name__ == "__main__":
    main()
