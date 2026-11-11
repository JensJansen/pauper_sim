"""evaluation.verdicts: the shared 'is it still improving' rule, and report_metrics' use of it on the history kinds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "analysis" / "eval"))
import report_metrics

from evaluation import verdicts


def _records(tick, scores, ratio_js):
    """metrics.jsonl records for one cadence tick of a 2-deck league: strength_history cells (anchor 10k = the
    0.6x reference for ticks >= 20k) plus the policy kinds."""
    out = []
    for deck, score in zip(("a", "b"), scores):
        out.append({"kind": "strength_history", "cumulative_games": tick, "deck": deck, "anchor_games": 10080,
                    "games": 400, "score": score})
        out.append({"kind": "policy_movement", "cumulative_games": tick, "deck": deck, "anchor_games": 10080,
                    "js": ratio_js, "top1": 0.5})
        out.append({"kind": "policy_scale", "cumulative_games": tick, "deck": deck, "entropy": 0.4, "adjacent_js": 0.04,
                    "random_js": 0.3})
    return out


def test_wins_of_reads_wins_or_a_score():
    assert verdicts.wins_of({"wins": 7, "games": 10}) == 7
    assert verdicts.wins_of({"live_wins": 3, "wins": 7, "games": 10}) == 3
    assert verdicts.wins_of({"score": 0.62, "games": 400}) == 248


def test_assess_ticks_rederives_a_label_per_tick_from_the_records():
    records = (_records(10000, (0.9, 0.9), 0.2)               # no anchor is 10k old at 0.6 x 10k: warm-up
               + _records(20000, (0.62, 0.64), 0.2)            # clearly beats its reference self
               + _records(30000, (0.50, 0.51), 0.2)            # flat strength, policy moved 0.2/0.3 = 67%
               + _records(40000, (0.50, 0.51), 0.01))          # flat strength, policy frozen (3%)
    labels = {tick: r["label"] for tick, r in verdicts.assess_ticks(records).items()}
    assert labels == {10000: "WARMUP", 20000: "IMPROVING", 30000: "CHURNING", 40000: "STUCK"}


def test_report_metrics_prints_history_series_and_the_tick_table():
    records = _records(20000, (0.55, 0.55), 0.2) + _records(30000, (0.60, 0.60), 0.2)
    text = "\n".join(report_metrics.report(records))
    assert "[strength_history@10080]" in text and "[policy_movement@10080]" in text and "[policy_scale]" in text
    assert "progress indicator by cadence tick" in text and "20000 games/deck  IMPROVING" in text
