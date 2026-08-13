"""Self-check for report_metrics.py: synthetic metrics.jsonl in, readable
summary lines out. Pure stdlib logic (no torch/game engine involved), so
unlike most rl.* tests this isn't marked slow.

The load-bearing test here is test_a_monotone_decline_is_reported_as_REGRESSING,
built from the REAL dmir_terror vs archive_oldest sequence (sessions 24-37).
The previous version of report_metrics printed those fourteen records one at a
time with no trend test, so the decline they describe went unnoticed for
thirteen sessions and was rediscovered only by a purpose-built round robin at
4,296s of compute. That is the failure this module now exists to prevent, so it
gets a test with the actual numbers in it.
"""
import json

import analysis.report_metrics as report_metrics


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _vs_history(deck, label, wins_by_session, games=20):
    return [{"kind": "vs_history", "session": s, "iteration": 1, "deck": deck, "label": label,
             "games": games, "live_wins": w, "snapshot_wins": games - w, "no_winner": 0}
            for s, w in enumerate(wins_by_session)]


def test_report_summarizes_ppo_mulligan_and_vs_history(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(path, [
        {"kind": "ppo", "session": 0, "iteration": 0, "deck": "elves", "games": 6, "buffer_size": 400,
         "batch_size": 32, "salvaged": 0, "policy_loss": 0.5, "value_loss": 1.0, "entropy": 1.2},
        {"kind": "ppo", "session": 0, "iteration": 1, "deck": "elves", "games": 6, "buffer_size": 410,
         "batch_size": 64, "salvaged": 0, "policy_loss": 0.4, "value_loss": 0.9, "entropy": 0.3},
        {"kind": "mulligan", "session": 0, "iteration": 1, "deck": "elves", "n": 6, "loss": 0.2},
        {"kind": "vs_history", "session": 0, "iteration": 1, "deck": "elves", "label": "archive_oldest",
         "games": 20, "live_wins": 11, "snapshot_wins": 9, "no_winner": 0},
    ])
    records = report_metrics.load(str(path))
    assert len(records) == 4

    lines = report_metrics.report(records)
    text = "\n".join(lines)
    assert "=== elves ===" in text
    assert "[ppo] 2 iterations" in text and "entropy" in text
    assert "[mulligan] 1 updates" in text and "loss=0.2000" in text
    assert "[vs_history:archive_oldest]" in text


def test_a_decline_is_flagged_and_never_reads_as_FLAT(tmp_path):
    """The real dmir_terror vs archive_oldest series, sessions 24-37, as
    percentages of 20 games: 60 80 85 65 80 60 60 80 75 55 50 60 60 45.
    Pooled it is a bland ~65%, which is how it went unnoticed for thirteen
    sessions. Either decline verdict is correct here; being FLAT is not."""
    wins = [int(pct / 100 * 20) for pct in
            (60, 80, 85, 65, 80, 60, 60, 80, 75, 55, 50, 60, 60, 45)]
    lines = report_metrics.report(_vs_history("dmir_terror", "archive_oldest", wins))
    text = "\n".join(lines)
    assert "REGRESSING" in text or "PAST PEAK" in text, \
        f"a 85%->45% decline must not read as FLAT:\n{text}"
    # ...and the raw per-record sequence must be visible BEFORE any statistic,
    # because that is what a human reading this needed and did not have.
    assert "60 80 85 65 80" in text, f"per-record trend missing:\n{text}"


def test_rise_then_fall_is_caught_even_though_the_linear_trend_is_flat():
    """The shape every deck in this league actually has (round robin: peaks at
    snapshot 116 / 58 / 289 / 232). A Cochran-Armitage trend test reads a
    symmetric rise-then-fall as no trend at all, so PAST PEAK -- comparing the
    current window against the best window this run reached -- is what has to
    catch it. Verified on real data: dmir vs archive_oldest reads trend
    z=+0.35 over its full 27 records while being 2.9 sigma below its own peak."""
    up_then_down = [6, 8, 10, 12, 14, 16, 18, 16, 14, 12, 10, 8, 6, 4]
    lines = report_metrics.report(_vs_history("d", "archive_oldest", up_then_down))
    text = "\n".join(lines)
    assert abs(report_metrics.trend_z(
        _vs_history("d", "archive_oldest", up_then_down))) < 2, "fixture is not trend-flat"
    assert "PAST PEAK" in text, f"rise-then-fall must not read as FLAT:\n{text}"
    assert "already on disk" in text  # points the operator at the recoverable checkpoint


def test_peak_comparison_is_corrected_for_how_many_windows_it_searched():
    """The best of W noisy windows is high by construction, so an uncorrected
    threshold would flag flat runs constantly. A pure coin-flip series must
    come back FLAT."""
    coin = [10] * 30  # exactly 50% every record, no trend, no peak
    text = "\n".join(report_metrics.report(_vs_history("d", "archive_oldest", coin)))
    assert "PAST PEAK" not in text and "REGRESSING" not in text, text
    _z, _at, crit = report_metrics.peak_comparison(_vs_history("d", "archive_oldest", coin))
    assert crit > 2.0, f"threshold {crit} was not widened for multiple comparisons"


def test_flat_and_improving_and_saturated_are_distinguished(tmp_path):
    flat = report_metrics.report(_vs_history("d", "archive_oldest", [10] * 12))
    assert "FLAT" in "\n".join(flat)

    improving = report_metrics.report(_vs_history("d", "archive_oldest", [2] * 8 + [18] * 8))
    assert "IMPROVING" in "\n".join(improving)

    # vs_heuristic has genuinely been 20/20 for 5+ sessions; that is saturation,
    # not improvement, and a Wilson interval keeps it inside [0, 1].
    saturated = report_metrics.report(
        [{"kind": "vs_heuristic", "session": s, "deck": "d", "games": 20, "live_wins": 20,
          "heuristic_wins": 0, "no_winner": 0} for s in range(8)])
    assert "SATURATED" in "\n".join(saturated)


def test_vs_history_labels_are_never_pooled_together():
    """archive_oldest is permanently snapshot_0 (~200 games); active_oldest is a
    rolling ~6,400-game-old self. They answer different questions, and merging
    them under one 'vs its past self' number is how elves' 50.9% was read as
    parity for weeks."""
    records = (_vs_history("elves", "archive_oldest", [4] * 10)      # 20%
               + _vs_history("elves", "active_oldest", [16] * 10))   # 80%
    lines = report_metrics.report(records)
    text = "\n".join(lines)
    assert "[vs_history:archive_oldest]" in text and "[vs_history:active_oldest]" in text
    assert "20.0%" in text and "80.0%" in text, f"labels appear pooled:\n{text}"


def test_report_handles_missing_kinds_and_absent_fields_gracefully(tmp_path):
    """metrics.jsonl is append-only across many schema revisions, so a record
    predating a field must still parse rather than KeyError the whole report."""
    lines = report_metrics.report([
        {"kind": "ppo", "deck": "d", "policy_loss": 0.0, "value_loss": 0.0,
         "entropy": 0.0, "buffer_size": 1, "batch_size": 1},  # no approx_kl/ent_coef/session
    ])
    text = "\n".join(lines)
    assert "[ppo] 1 iterations" in text
    assert "cumulative_games=NOT RECORDED" in text  # visibly absent, not silently zero


def test_zero_game_series_does_not_divide_by_zero():
    lines = report_metrics.report(
        [{"kind": "vs_gauntlet", "session": 0, "deck": "d", "games": 0, "live_wins": 0,
          "gauntlet_wins": 0, "no_winner": 0}])
    assert any("no completed games" in l for l in lines)


def test_wilson_stays_inside_the_unit_interval_at_the_ends():
    for wins, n in ((0, 20), (20, 20), (1, 3)):
        _p, lo, hi = report_metrics.wilson(wins, n)
        assert 0.0 <= lo <= hi <= 1.0
