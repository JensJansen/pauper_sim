"""Snapshot-history addressing, the retro_sweep designs, and the policy probe's arithmetic."""
import json
import math

import numpy as np
import pytest

import retro_sweep
from evaluation import AgentRef, Matchup, history, play, probe


def _league(tmp_path, n_snapshots=4, decks=("d1", "d2"), games_per_snapshot=180):
    """Snapshots 0..n-1 per deck: the oldest half evicted to archive/, the rest in the deck dir."""
    for deck in decks:
        for i in range(n_snapshots):
            sub = tmp_path / deck / ("archive" if i < n_snapshots // 2 else "")
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"snapshot_{i}.pt").write_bytes(b"")
    (tmp_path / "progress.json").write_text(json.dumps({"cumulative_games_per_deck": n_snapshots * games_per_snapshot}))
    return str(tmp_path)


def test_history_maps_ids_to_games_and_finds_snapshots_in_active_and_archive(tmp_path):
    league = _league(tmp_path)
    assert history.snapshot_ids(league, "d1") == [0, 1, 2, 3]
    assert history.verify(league, "d1") == [0, 1, 2, 3]
    assert [history.games_at(i) for i in range(4)] == [180, 360, 540, 720]
    assert history.snapshot_path(league, "d1", 0).endswith("archive\\snapshot_0.pt") or \
        history.snapshot_path(league, "d1", 0).endswith("archive/snapshot_0.pt")
    assert "archive" not in history.snapshot_path(league, "d1", 3)
    with pytest.raises(FileNotFoundError):
        history.snapshot_path(league, "d1", 9)


def test_history_verify_rejects_a_directory_the_mapping_does_not_describe(tmp_path):
    league = _league(tmp_path)
    (tmp_path / "progress.json").write_text(json.dumps({"cumulative_games_per_deck": 999}))
    with pytest.raises(ValueError, match="progress.json"):
        history.verify(league, "d1")
    (tmp_path / "d1" / "snapshot_2.pt").unlink()  # a hole in the ids
    with pytest.raises(ValueError, match="contiguous"):
        history.verify(league, "d1")


def test_history_at_ages_takes_the_nearest_snapshot_and_dedupes(tmp_path):
    league = _league(tmp_path)
    assert history.at_ages(league, "d1", [100, 400, 1000, "final"]) == [("100", 0, 180), ("400", 1, 360),
                                                                      ("1000", 3, 720)]  # 1000 and final are one snapshot
    assert history.at_ages(league, "d1", [500])[0][1] == 2


def test_agent_ref_names_and_distinguishes_snapshots():
    a, b = AgentRef("elves", "x/league", "t", 5), AgentRef("elves", "x/league", "t", 6)
    assert a != b and a.name == "t/elves@5" and AgentRef("elves", "x/league").name == "league/elves"


def test_play_chunk_loads_the_snapshot_each_side_names(monkeypatch):
    seen = []
    monkeypatch.setattr(play, "collect_rollout", lambda *a, **k: None)
    monkeypatch.setattr(play, "_load", lambda *args: seen.append(args) or (None, None))
    monkeypatch.setattr(play, "_pool", lambda: ({"a": [], "b": []}, None, {"a": None, "b": None}, None))
    monkeypatch.setattr(play, "SeatAgent", lambda *a: None)
    play._play_chunk((("d", "a", 7), ("e", "b", None), 1, 1, True, 30))
    play._play_chunk((("d", "a"), ("e", "b"), 1, 1, True, 30))  # the pre-snapshot task shape still works
    assert seen == [("d", "a", 7), ("e", "b", None), ("d", "a"), ("e", "b")]


def test_retro_sweep_designs_cover_every_age_pair_and_every_deck_pair(tmp_path):
    league = _league(tmp_path, n_snapshots=5)
    ages = [180, 540, 900, "final"]  # 900 and final are the same snapshot (id 4): three distinct ages
    matrix = retro_sweep.build_matrix(league, ["d1", "d2"], ages, 180, AgentRef, Matchup, history)
    assert len(matrix) == 2 * 3  # 2 decks x C(3, 2) age pairs
    assert all(m.a.snapshot > m.b.snapshot and m.a.deck == m.b.deck for m, *_ in matrix)
    field = retro_sweep.build_field(league, ["d1", "d2"], ages, 180, AgentRef, Matchup, history)
    assert len(field) == 2 * 3 * 2  # opponent deck x age x final-agent deck
    assert all(m.a.snapshot == 4 for m, *_ in field)  # the final agent is always the newest snapshot


def test_js_divergence_known_values():
    p = np.array([1.0, 0.0])
    assert probe.js_divergence(p, p) == 0.0
    assert probe.js_divergence(p, np.array([0.0, 1.0])) == pytest.approx(math.log(2))
    q = np.array([0.5, 0.5])
    assert probe.js_divergence(p, q) == probe.js_divergence(q, p) > 0


def test_probe_compare_and_entropy():
    a = [np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1])]
    b = [np.array([0.7, 0.2, 0.1]), np.array([0.8, 0.1, 0.1])]
    js, agree = probe.compare(a, b)
    assert agree == 0.5 and js > 0 and probe.compare(a, a) == (0.0, 1.0)
    assert probe.mean_entropy([np.array([0.5, 0.5])]) == pytest.approx(math.log(2))
    assert probe.mean_entropy([np.array([1.0, 0.0])]) == 0.0


def test_probe_selects_only_developed_states_with_a_real_choice():
    mask3, mask2 = np.array([1, 1, 1, 0], dtype=bool), np.array([1, 1, 0, 0], dtype=bool)
    assert probe._selected((probe.MID_TURN, probe.MID_PERMANENTS), mask3)
    assert not probe._selected((probe.MID_TURN - 1, probe.MID_PERMANENTS), mask3)  # opening
    assert not probe._selected((probe.MID_TURN, probe.MID_PERMANENTS - 1), mask3)  # empty board
    assert not probe._selected((probe.MID_TURN, probe.MID_PERMANENTS), mask2)  # two options: forced or a coin flip
