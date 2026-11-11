"""Summarizes checkpoints/<league>/metrics.jsonl -- the per-iteration "ppo" /
"mulligan" / "mulligan_audit" / "vs_history" (retired check, legacy records) / "primary_vs_primary_round_robin" /
"primary_vs_training_round_robin" records
rl.league.league_runner's _run_session appends during every training run.
Plain-text. The verdict statistics live in evaluation/verdicts.py (shared with the in-run
progress_indicator check), and the history checks' records (strength_history / strength_field per anchor age,
policy_movement / policy_scale) are read here too, plus a league-wide progress label per cadence tick.

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
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/
from evaluation.verdicts import (  # noqa: F401 -- shared with the in-run progress_indicator; re-exported (tests use them)
    SATURATED_AT, assess_ticks, min_detectable_effect, peak_comparison, series_verdict as _verdict, trend_z,
    two_proportion_z, wilson, wins_of,
)

TREND_WINDOW = 24  # per-record values shown; older ones are pooled into "early"


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _win_rate_lines(deck, tag, rows, window):
    """Trend-first summary for one (deck, instrument) win-rate series."""
    rows = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))
    total_n = sum(r.get("games", 0) for r in rows)
    if not total_n:
        return [f"  [{tag}] no completed games"]

    shown = rows[-window:]
    trend = " ".join(f"{100 * wins_of(r) / r['games']:.0f}" for r in shown if r.get("games"))
    half = max(1, len(rows) // 2)
    early, late = rows[:half], rows[half:]
    def pool(g):
        return (sum(wins_of(r) for r in g), sum(r.get("games", 0) for r in g))
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
    def mean(g, k):
        return (sum(r[k] for r in g if k in r) / max(1, sum(1 for r in g if k in r)))
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


def _movement_lines(tag, rows):
    """policy_movement: JS divergence (nats) of the current policy from its snapshot at one anchor age, per tick."""
    rows = sorted(rows, key=lambda r: r.get("cumulative_games", 0))
    js = " ".join(f"{r['js']:.2f}" for r in rows[-8:])
    return [f"  [{tag}] {len(rows)} ticks, JS from that snapshot (last 8): {js}   latest top-1 agreement "
            f"{rows[-1]['top1']:.2f}"]


def _scale_lines(rows):
    rows = sorted(rows, key=lambda r: r.get("cumulative_games", 0))
    last = rows[-1]
    return [f"  [policy_scale] entropy {rows[0]['entropy']:.2f} -> {last['entropy']:.2f}; latest JS one snapshot apart "
            f"{last.get('adjacent_js') or float('nan'):.3f}, random-vs-current {last['random_js']:.3f}"]


def _tick_lines(records):
    """The league-wide progress label at every cadence tick, re-derived from the records (evaluation.verdicts.assess)."""
    ticks = assess_ticks(records)
    if not ticks:
        return []
    lines = ["=== progress indicator by cadence tick (provisional thresholds, re-derived from the records) ==="]
    for tick, r in ticks.items():
        if r["label"] == "WARMUP":
            lines.append(f"  {tick:>9} games/deck  WARMUP")
            continue
        ratio = "n/a" if r["policy_ratio"] is None else f"{100 * r['policy_ratio']:.0f}%"
        lines.append(f"  {tick:>9} games/deck  {r['label']:<9} score {100 * r['score']:.1f}% "
                     f"[{100 * r['lo']:.1f}, {100 * r['hi']:.1f}] n={r['games']}  policy moved {ratio} of the way to random")
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
        if kind in ("strength_history", "strength_field"):  # one series per anchor age: the SAME older self, tick after tick
            tag = f"{kind}@{r['anchor_games']}"
        elif kind == "policy_movement":
            tag = f"policy_movement@{r['anchor_games']}"
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
            elif tag.startswith("policy_movement@"):
                lines.extend(_movement_lines(tag, rows))
            elif tag == "policy_scale":
                lines.extend(_scale_lines(rows))
            elif tag == "progress_indicator":
                continue  # re-derived for every tick below, under the current thresholds
            elif tag == "mulligan":
                tail = sorted(rows, key=lambda r: (r.get("session", 0), r.get("iteration", 0)))[-1]
                lines.append(f"  [mulligan] {len(rows)} updates -- latest loss={tail.get('loss', float('nan')):.4f} "
                             f"(n={tail.get('n', '?')})")
            elif any(r.get("games") is not None for r in rows):
                lines.extend(_win_rate_lines(deck, tag, rows, window))
    lines.extend(_tick_lines(records))
    for line in lines:
        print(line)
    return lines


def main():
    args = list(sys.argv[1:])
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
