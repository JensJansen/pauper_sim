"""Summarizes checkpoints/<league>/metrics.jsonl -- the per-iteration "ppo" /
"mulligan" / "vs_history" / "vs_gauntlet" / "vs_heuristic" records
rl.league.league_runner's _run_session appends during every training run (see
_append_metric). Plain-text, stdlib only (no plotting library in
requirements.txt).

WHY THIS WAS REWRITTEN (2026-08-13). The previous version printed win-rate
records ONE AT A TIME, with no pooling, no interval, and no trend test. Every
human summary of its output therefore pooled by hand -- and pooling is exactly
what hides a monotone decline. `dmir_terror` vs its own oldest archived
snapshot ran 60/80/85/65/80/60/60/80/75/55/50/60/60/45 across sessions 24-37;
pooled, that is a bland ~65%, and the regression it actually describes went
unnoticed for thirteen sessions. It was independently confirmed only by a
purpose-built round robin that cost 4,296s
of compute to rediscover what was already sitting in this file.

So this version leads with the PER-RECORD SEQUENCE, and only then pools:

  - the raw trend, most recent last, so a decline is visible before any
    statistic is applied to it;
  - four distinguishable verdicts, because "did it change" is the wrong
    question: IMPROVING / FLAT / REGRESSING (a linear decline) / PAST PEAK
    (the current window is below one this run already reached, whatever the
    overall trend). PAST PEAK is the one that matters here -- every deck in
    this league rose then fell, and a linear trend test reads that as FLAT.
    A gate that only asked "did it change" would pass a deck that is actively
    getting worse (see the same document's Step 1.5);
  - FLAT annotated with the minimum effect the sample size could have
    detected, so "no change" is never confused with "cannot tell";
  - Wilson intervals, which stay inside [0, 1] near the saturated end where
    vs_heuristic has been living at 20/20;
  - vs_history labels kept STRICTLY SEPARATE. `archive_oldest` is permanently
    snapshot_0 (a ~200-game policy) while `active_oldest` tracks a rolling
    ~6,400-game-old self; averaging the two into one "vs its past self" number
    is meaningless, and was done in this project's own reporting for weeks.

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
    """(point estimate, lo, hi) as fractions. Wilson rather than normal-
    approximation because several of these instruments genuinely sit at or
    near 100% (vs_heuristic has been 20/20 for 5+ sessions), where the normal
    interval runs off the end of [0, 1] and reports nonsense."""
    if not n:
        return float("nan"), 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z(w1, n1, w2, n2):
    """z for (group 2 - group 1) under a pooled-proportion null. 0.0 when
    either group is empty or the pooled rate is degenerate, so callers can
    treat "no evidence" and "no data" the same way."""
    if not n1 or not n2:
        return 0.0
    pooled = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return ((w2 / n2) - (w1 / n1)) / se if se else 0.0


def trend_z(rows):
    """Cochran-Armitage trend test across ORDERED records: z for a linear
    change in win rate with record index.

    This, not an early-vs-late split, is the verdict statistic. Splitting an
    ordered series into two halves and comparing them discards the ordering
    and is badly underpowered against a monotone drift -- on the real
    dmir_terror decline (14 records of 20 games, 85% down to 45%) the split
    gives z=-1.63 and calls it FLAT, while the trend test sees the shape.
    The split is still PRINTED, because "early X% -> late Y%" is the readable
    summary; it just does not get to decide.

    Positive z = improving over time. 0.0 when there is no variation to
    regress against (one record, or an all-wins/all-losses series)."""
    pts = [(i, r.get("live_wins", 0), r.get("games", 0)) for i, r in enumerate(rows) if r.get("games")]
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
    because "no change" and "cannot tell" are different claims and this
    project has already conflated them once -- an acceptance criterion of
    'win rate >= 47.4%' was set against an eval that could not resolve 2.4pp."""
    if not n_per_group:
        return 1.0
    return (alpha_z + power_z) * math.sqrt(2 * 0.25 / n_per_group)


def peak_comparison(rows, group=5):
    """(z of last window vs BEST window, index of that window's last record,
    selection-corrected |z| threshold). Negative z = the current policy is worse than one this run
    already had on disk.

    This is the verdict's primary test, and neither of the two obvious
    alternatives can replace it:

      - early-vs-late halves discards ordering and is underpowered;
      - a LINEAR trend test cannot see rise-then-fall, which is the actual
        shape here. Measured on real data: dmir_terror vs archive_oldest rose
        35% -> 85% then fell to 45%, and trend_z reads +0.35, FLAT. The
        round robin independently found
        every deck peaking mid-run -- snapshot 116 / 58 / 289 / 232 -- so
        rise-then-fall is the norm in this project, not an edge case.

    The operational question is not "has it changed since the start" but "is
    what I am about to keep training worse than something I already saved."
    """
    pts = [r for r in rows if r.get("games")]
    if len(pts) < 2 * group:
        return 0.0, None, 2.0
    windows = [(i, pts[i:i + group]) for i in range(0, len(pts) - group + 1)]
    pool = lambda g: (sum(r.get("live_wins", 0) for r in g), sum(r.get("games", 0) for r in g))
    rates = [(pool(w), i + group - 1) for i, w in windows]
    (best_w, best_n), best_end = max(rates, key=lambda t: t[0][0] / t[0][1])
    (last_w, last_n), last_end = rates[-1]
    # Selection correction, and it is not optional. The BEST of W noisy windows
    # is high by construction -- the max of ~23 standard normals sits about
    # 2 sigma up -- so testing "is the last window below the best" against a
    # flat -2 would fire on pure noise routinely. Sidak the threshold by the
    # number of windows actually searched.
    crit = NormalDist().inv_cdf(1 - 0.05 / max(1, len(windows)))
    if best_end == last_end:
        return 0.0, best_end, crit  # the most recent window IS the best one
    return two_proportion_z(best_w, best_n, last_w, last_n), best_end, crit


def _verdict(rows, w_late, n_late):
    """(label, trend z, peak z, peak index). REGRESSING is a distinct outcome
    from FLAT on purpose: a check that only asks "has it changed" reads a deck
    losing ground as a healthy, changing run and lets it keep training."""
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
    trend = " ".join(f"{100 * r['live_wins'] / r['games']:.0f}" for r in shown if r.get("games"))
    half = max(1, len(rows) // 2)
    early, late = rows[:half], rows[half:]
    pool = lambda g: (sum(r.get("live_wins", 0) for r in g), sum(r.get("games", 0) for r in g))
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
        # "No change" and "cannot tell" are different claims, and this project
        # has already conflated them once. Say which one this is.
        mde = min_detectable_effect(min(ne, nl))
        lines.append(f"    (FLAT at this sample size resolves >= {100 * mde:.1f}pp; "
                     f"a smaller real change would read the same)")
    return lines


def _ppo_lines(deck, rows):
    rows = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))
    half = max(1, len(rows) // 10)  # deciles: a 10% window is stable at 10k+ iterations
    mean = lambda g, k: (sum(r[k] for r in g if k in r) / max(1, sum(1 for r in g if k in r)))
    first, last = rows[:half], rows[-half:]

    lines = [f"  [ppo] {len(rows)} iterations (first/last decile means)"]
    for key, fmt in (("entropy", ".4f"), ("value_loss", ".4f"), ("policy_loss", ".4f"),
                     ("approx_kl", ".4f"), ("clip_fraction", ".3f"), ("epochs_run", ".2f"),
                     ("explained_variance", ".3f"), ("adv_std", ".4f")):
        if not any(key in r for r in rows):
            continue  # field predates this schema revision, or postdates these records
        a, b = mean(first, key), mean(last, key)
        note = ""
        if key == "approx_kl":
            # The trust region is the whole point of target_kl; a run sitting
            # above it is early-stopping on most updates, which shows up as
            # epochs_run falling below n_epochs.
            target = last[-1].get("target_kl", 0.03)
            note = f"   vs target_kl={target}  {'EXCEEDS' if b > target else 'ok'}"
        lines.append(f"    {key:<20}{a:>10{fmt}} -> {b:>9{fmt}}{note}")
    tail = rows[-1]
    lines.append(f"    latest: buffer={tail.get('buffer_size', '?')} batch={tail.get('batch_size', '?')} "
                 f"ent_coef={tail.get('ent_coef', '?')} games={tail.get('games', '?')} "
                 f"cumulative_games={tail.get('cumulative_games', 'NOT RECORDED')}")
    return lines


def report(records, window=TREND_WINDOW):
    """Returns the report as a list of printed lines (also prints them) --
    returning them makes this testable without capturing stdout."""
    by_deck = defaultdict(lambda: defaultdict(list))
    sessions = []
    for r in records:
        kind = r.get("kind", "?")
        # session_start is league-level, not per-deck: it has no `deck`, so
        # grouping it with the rest bucketed it under a "?" heading that then
        # rendered empty (no `games`, no ppo/mulligan branch) -- silently
        # hiding the reward_fn/roster it carries, which is the first thing to
        # check when confirming two populations trained under the same rules.
        if kind == "session_start":
            sessions.append(r)
            continue
        # vs_history's label is part of the series identity, never pooled away:
        # archive_oldest is a fixed ~200-game reference, active_oldest a moving
        # ~6,400-game one. They answer different questions.
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
