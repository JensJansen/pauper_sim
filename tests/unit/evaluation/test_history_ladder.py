"""evaluation.history's age ladder and the deck listing (plan P0): which snapshot is "the deck at N games", which rungs a periodic
check may use, and which decks a league directory offers."""
import json

import pytest

from evaluation import history as H
from evaluation.matchups import available_decks
from evaluation.probe import table_fingerprint

G = H.GAMES_PER_SNAPSHOT


def _league(tmp_path, n_snapshots, deck="elves", trained=None):
    d = tmp_path / deck
    d.mkdir(parents=True)
    for i in range(n_snapshots):
        (d / f"snapshot_{i}.pt").write_bytes(b"x")
    (tmp_path / "progress.json").write_text(json.dumps({"cumulative_games_per_deck": trained or n_snapshots * G}))
    return str(tmp_path)


def test_games_at_counts_the_snapshot_itself():
    assert H.games_at(0) == G and H.games_at(9) == 10 * G


def test_ladder_points_keep_only_rungs_two_intervals_older_than_the_agent(tmp_path):
    league = _league(tmp_path, 20)
    current = 20 * G
    pts = H.ladder_points(league, "elves", current, [G, 5 * G, 10 * G, 18 * G, 19 * G, 20 * G])
    games = [g for g, _sid in pts]
    assert games == sorted(games) and all(g <= current - 2 * G for g in games)
    assert max(games) == 18 * G, "a rung closer than two intervals would be the agent against itself"


def test_ladder_points_drop_duplicate_snapshots_and_stop_at_the_first_too_young_rung(tmp_path):
    league = _league(tmp_path, 20)
    pts = H.ladder_points(league, "elves", 20 * G, [5 * G, 5 * G + 3, 10 * G, 100 * G, 12 * G])
    assert [sid for _g, sid in pts] == sorted({sid for _g, sid in pts}) and len(pts) == 2


def test_at_ages_picks_the_nearest_snapshot_and_final_is_the_newest(tmp_path):
    league = _league(tmp_path, 10)
    out = H.at_ages(league, "elves", [3 * G + 5, "final"])
    assert out[0] == (str(3 * G + 5), 2, 3 * G) and out[1] == ("final", 9, 10 * G)


def test_at_ages_drops_two_ages_that_resolve_to_the_same_snapshot(tmp_path):
    assert len(H.at_ages(_league(tmp_path, 10), "elves", [3 * G, 3 * G + 2])) == 1


def test_verify_accepts_a_consistent_directory_and_rejects_a_gap_or_a_stale_progress_file(tmp_path):
    assert H.verify(_league(tmp_path / "ok", 5), "elves") == [0, 1, 2, 3, 4]
    bad = tmp_path / "gap"
    league = _league(bad, 4)
    (bad / "elves" / "snapshot_1.pt").unlink()
    with pytest.raises(ValueError, match="not contiguous"):
        H.verify(league, "elves")
    with pytest.raises(ValueError, match="progress.json"):
        H.verify(_league(tmp_path / "stale", 5, trained=7 * G), "elves")


def test_available_decks_lists_decks_with_a_live_checkpoint_sorted(tmp_path):
    for name, live in (("zeta", True), ("alpha", True), ("noweights", False)):
        (tmp_path / name).mkdir()
        if live:
            (tmp_path / name / "live.pt").write_bytes(b"x")
    (tmp_path / "progress.json").write_text("{}")
    assert available_decks(str(tmp_path)) == ["alpha", "zeta"]
    assert available_decks(str(tmp_path / "missing")) == []


def test_table_fingerprint_changes_with_a_label_or_its_order_and_ignores_the_closures():
    a = [("Pass", None, None), ("Cast Bolt", None, None)]
    assert table_fingerprint(a) == table_fingerprint([("Pass", lambda s: 1, lambda s: 2), ("Cast Bolt", 0, 0)])
    assert table_fingerprint(a) != table_fingerprint(a[::-1])
    assert table_fingerprint(a) != table_fingerprint([("Pass", None, None), ("Cast Burst", None, None)])
