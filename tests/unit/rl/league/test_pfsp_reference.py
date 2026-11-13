"""PFSP opponent weighting and the opponent-stats file, against the formula and the spec (plan P4).

test_league.py covers the behaviours (cold start, hardest-matchup bias, persistence); this adds the exact weight formula, the
sampling frequencies against the weights, the key round trip, isolation between decks, and the corrupt-file contract."""
import json
import random

import pytest

from rl.league import league as L

DECKS = ["a", "b", "c", "d"]


def _pool(tmp_path, power=0.5):
    return L.LeaguePool(str(tmp_path), DECKS, pfsp_power=power)


@pytest.mark.parametrize("wins,games", [(0, 1), (1, 1), (3, 10), (7, 10), (0, 50), (50, 50)])
@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_weight_is_floor_plus_loss_rate_to_the_power(tmp_path, wins, games, power):
    pool = _pool(tmp_path, power)
    pool.opponent_stats["a"][("deck", "b")] = (wins, games)
    assert pool._pfsp_weight("a", ("deck", "b")) == pytest.approx(L.PFSP_FLOOR + (1 - wins / games) ** power)


def test_an_unseen_opponent_has_the_neutral_half_win_rate_weight(tmp_path):
    pool = _pool(tmp_path, 1.0)
    assert pool._pfsp_weight("a", ("deck", "b")) == pytest.approx(L.PFSP_FLOOR + 0.5)
    assert pool._pfsp_weight("zzz", ("deck", "b")) == pytest.approx(L.PFSP_FLOOR + 0.5), "a deck with no stats at all"


def test_weight_never_increases_as_the_training_deck_wins_more(tmp_path):
    pool = _pool(tmp_path)
    weights = []
    for wins in range(11):
        pool.opponent_stats["a"][("deck", "b")] = (wins, 10)
        weights.append(pool._pfsp_weight("a", ("deck", "b")))
    assert weights == sorted(weights, reverse=True) and weights[-1] == pytest.approx(L.PFSP_FLOOR)


def test_sampled_deck_frequencies_follow_the_weights(tmp_path):
    pool = _pool(tmp_path, 1.0)
    for name, (w, g) in {"a": (5, 10), "b": (9, 10), "c": (1, 10), "d": (0, 0)}.items():
        pool.opponent_stats["a"][("deck", name)] = (w, g)
    weights = [pool._pfsp_weight("a", ("deck", n)) for n in DECKS]
    rng, n = random.Random(0), 20000
    counts = dict.fromkeys(DECKS, 0)
    for _ in range(n):
        counts[pool.sample_opponent("a", rng)[0]] += 1
    for name, weight in zip(DECKS, weights, strict=True):
        expect = weight / sum(weights)
        assert counts[name] / n == pytest.approx(expect, abs=0.015), name


def test_pfsp_off_samples_every_deck_equally_whatever_the_record(tmp_path):
    pool = _pool(tmp_path)
    pool.opponent_stats["a"][("deck", "b")] = (10, 10)
    rng, n = random.Random(1), 8000
    counts = dict.fromkeys(DECKS, 0)
    for _ in range(n):
        counts[pool.sample_opponent("a", rng, pfsp=False)[0]] += 1
    assert all(c / n == pytest.approx(0.25, abs=0.02) for c in counts.values())


@pytest.mark.parametrize("key", [("deck", "mono_red_madness"), ("snapshot", "elves", 0), ("snapshot", "rakdos_madness", 137)])
def test_opponent_key_round_trips_through_its_string_form(key):
    assert L._parse_opponent_key(L._format_opponent_key(key)) == key


def test_outcomes_are_recorded_against_the_right_deck_and_key_only(tmp_path):
    pool = _pool(tmp_path)
    pool.record_outcome("a", ("deck", "b"), True)
    pool.record_outcome("a", ("deck", "b"), False)
    pool.record_outcome("a", ("snapshot", "b", 3), True)
    assert pool.opponent_stats["a"] == {("deck", "b"): (1, 2), ("snapshot", "b", 3): (1, 1)}
    assert pool.opponent_stats["b"] == {} and pool.opponent_stats["c"] == {}


def test_each_decks_stats_are_stored_in_its_own_file_and_reload_unmixed(tmp_path):
    pool = _pool(tmp_path)
    pool.record_outcome("a", ("deck", "b"), True)
    pool.record_outcome("c", ("deck", "a"), False)
    pool.save_opponent_stats()
    again = _pool(tmp_path)
    assert again.opponent_stats["a"] == {("deck", "b"): (1, 1)} and again.opponent_stats["c"] == {("deck", "a"): (0, 1)}
    assert again.opponent_stats["b"] == {} and again.opponent_stats["d"] == {}


def test_a_corrupt_stats_file_fails_loudly_instead_of_silently_resetting_the_league(tmp_path):
    """Owner default (plan Q9): a damaged opponent_stats.json must stop the run, not quietly restart PFSP from its prior."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "opponent_stats.json").write_text('{"deck:b": [3, ')
    with pytest.raises(json.JSONDecodeError):
        _pool(tmp_path)
