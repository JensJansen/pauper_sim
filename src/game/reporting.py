"""Aggregation & output: turn a batch of run_game results into summary
stats and a printable report. No GameState dependency -- pure functions
over the (terminated_turn_or_None, scores) pairs the caller collects."""


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def aggregate_results(results, horizon):
    """results: list of (terminated_turn_or_None, scores) pairs -- scores a
    dict keyed by each score function's own name, the training reward's
    entry always first (dict insertion order). Score means/medians are
    computed across every game, not just terminated ones -- a failed
    game's scores are already zeroed by finalize_scores, so including them
    correctly drags the mean down to reflect overall policy quality, not
    just "how good are the wins." Terminated-turn mean/median has no such
    reading for a failure (there's no turn to average), so that one
    excludes them."""
    n = len(results)
    rows = []
    for turn in range(1, horizon + 1):
        terminated_pct = 100 * sum(1 for t, _s in results if t is not None and t <= turn) / n
        rows.append((turn, terminated_pct))

    terminated_turns = [t for t, _s in results if t is not None]
    score_names = list(results[0][1].keys()) if results else []
    score_summaries = {
        name: {
            "mean": _mean([s[name] for _t, s in results]),
            "median": _median([s[name] for _t, s in results]),
        }
        for name in score_names
    }

    summary = {
        "terminated_mean_turn": _mean(terminated_turns),
        "terminated_median_turn": _median(terminated_turns),
        "never_pct": 100 * (n - len(terminated_turns)) / n,
        "scores": score_summaries,
    }
    return rows, summary


def _fmt(value, spec="{:.2f}"):
    return spec.format(value) if value is not None else "n/a"


def print_report(results, horizon):
    rows, summary = aggregate_results(results, horizon)
    print(f"{'Turn':>4}  {'Terminated %':>13}")
    for turn, terminated_pct in rows:
        print(f"{turn:>4}  {terminated_pct:>12.1f}%")
    print()
    print(f"Terminated: mean turn {_fmt(summary['terminated_mean_turn'])}, "
          f"median {_fmt(summary['terminated_median_turn'], '{:g}')}, "
          f"never by horizon: {summary['never_pct']:.1f}%")
    for name, s in summary["scores"].items():
        print(f"{name}: mean {_fmt(s['mean'])}, median {_fmt(s['median'])}")
