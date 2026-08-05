"""Self-check for report_metrics.py: synthetic metrics.jsonl in, readable
summary lines out. Pure stdlib logic (no torch/game engine involved), so
unlike most rl.* tests this isn't marked slow."""
import json

import report_metrics


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


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
    assert any("elves [ppo]" in l and "1.200 -> 0.300" in l for l in lines), \
        f"expected an entropy trend line for elves, got: {lines}"
    assert any("elves [mulligan]" in l and "loss=0.2000" in l for l in lines)
    assert any("elves [vs_history:archive_oldest]" in l and "11/20" in l and "55%" in l for l in lines)


def test_report_summarizes_gauntlet_and_heuristic(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(path, [
        {"kind": "vs_gauntlet", "session": 3, "deck": "rakdos_madness", "games": 20, "live_wins": 12,
         "gauntlet_wins": 8, "no_winner": 0},
        {"kind": "vs_heuristic", "session": 3, "deck": "mono_red_rally", "games": 20, "live_wins": 15,
         "heuristic_wins": 5, "no_winner": 0},
    ])
    lines = report_metrics.report(report_metrics.load(str(path)))
    assert any("rakdos_madness [vs_gauntlet]" in l and "12/20" in l and "60%" in l for l in lines)
    assert any("mono_red_rally [vs_heuristic]" in l and "15/20" in l and "75%" in l for l in lines)


def test_report_handles_missing_kinds_gracefully(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(path, [{"kind": "ppo", "deck": "d", "policy_loss": 0.0, "value_loss": 0.0,
                          "entropy": 0.0, "buffer_size": 1, "batch_size": 1}])
    lines = report_metrics.report(report_metrics.load(str(path)))
    assert len(lines) == 1
