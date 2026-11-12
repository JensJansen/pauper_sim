"""evaluation/: pairings, seeds, win counting and statistics (fast), plus play over real tiny leagues (slow)."""
import math
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import pytest
import torch

import evaluation
from evaluation import (AgentRef, Matchup, Tally, cross, diff_interval, gauntlet, head_to_head, play_matchups,
                        round_robin, tally, wilson_interval)
from evaluation import play
from rl.league import league_runner

ROSTER = ["elves", "boggles"]


def _fake_league(tmp_path, name, decks):
    for d in decks:
        (tmp_path / name / d).mkdir(parents=True)
        (tmp_path / name / d / "live.pt").write_text("x")
    return str(tmp_path / name)


# ---- stats ----------------------------------------------------------------------------------------------

def test_tally_scores_a_timeout_as_half_a_win_and_adds():
    t = Tally(wins=6, ties=2, games=10) + Tally(wins=1, ties=0, games=10)
    assert (t.wins, t.ties, t.games) == (7, 2, 20) and t.score == 0.4


def test_tally_sums_results_from_either_side():
    rs = [SimpleNamespace(a_wins=3, b_wins=5, no_winner=2, games=10), SimpleNamespace(a_wins=1, b_wins=1, no_winner=0, games=2)]
    assert tally(rs, "a") == Tally(4, 2, 12) and tally(rs, "b") == Tally(6, 2, 12)


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(Tally(50, 0, 100))
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(Tally(0, 0, 20))  # never leaves [0, 1], unlike the normal interval
    assert lo == 0.0 and 0.0 < hi < 0.2
    assert math.isnan(wilson_interval(Tally())[0])


def test_diff_interval_is_centred_on_the_difference_and_shrinks_with_games():
    d, lo, hi = diff_interval(Tally(60, 0, 100), Tally(50, 0, 100))
    assert d == pytest.approx(0.1) and lo < d < hi and (lo + hi) / 2 == pytest.approx(d)
    _, lo2, hi2 = diff_interval(Tally(600, 0, 1000), Tally(500, 0, 1000))
    assert hi2 - lo2 < hi - lo


# ---- pairings -------------------------------------------------------------------------------------------

def test_designs_enumerate_the_expected_pairings(tmp_path):
    a = _fake_league(tmp_path, "a", ["d1", "d2", "d3"])
    b = _fake_league(tmp_path, "b", ["d2", "d3", "d4"])
    g = gauntlet(a, b, subject_label="A", field_label="B")
    assert len(g) == 9 and all(m.a.label == "A" and m.b.label == "B" for m in g)
    assert [m.a.deck for m in head_to_head(a, b)] == ["d2", "d3"]  # only decks both have
    assert [m.a.deck for m in head_to_head(a, b, decks=["d3"])] == ["d3"]
    assert len(round_robin(a)) == 6  # 3 decks: 3 mirrors + 3 pairs
    assert len(cross([AgentRef("d1", a)], [AgentRef("d2", b), AgentRef("d3", b)])) == 2
    assert gauntlet(a, b, subject_decks=["d1"], field_decks=["d4"]) == [Matchup(AgentRef("d1", a), AgentRef("d4", b))]


def test_agent_ref_normalizes_the_path_and_names_by_label():
    r = AgentRef("elves", "some/dir/../league", "S")
    assert r.league_dir.endswith("league") and ".." not in r.league_dir and r.name == "S/elves"
    assert AgentRef("elves", "x/league").name == "league/elves"


def test_matchup_seed_ignores_where_the_agents_live_but_not_the_pairing_or_chunk():
    s = play.matchup_seed
    assert s(7, "elves", "boggles", 0) == s(7, "elves", "boggles", 0)
    assert len({s(7, "elves", "boggles", 0), s(7, "elves", "boggles", 1), s(7, "boggles", "elves", 0),
                s(8, "elves", "boggles", 0)}) == 4
    assert 0 <= s(7, "elves", "boggles", 0) < 2 ** 31


def test_play_counts_winners_by_seat(monkeypatch):
    def fake_rollout(pairing, n, horizon, rng, on_game_end=None, **_):
        for w in (0, 0, 1, None)[:n]:
            on_game_end(SimpleNamespace(winner=w))

    monkeypatch.setattr(play, "collect_rollout", fake_rollout)
    monkeypatch.setattr(play, "_load", lambda *_: (None, None))
    monkeypatch.setattr(play, "_pool", lambda: ({"a": [], "b": []}, None, {"a": None, "b": None}, None))
    monkeypatch.setattr(play, "SeatAgent", lambda *a: None)
    assert play._play_chunk((("d", "a"), ("d", "b"), 4, 1, True, 30)) == (2, 1, 1)


# ---- play over real checkpoints -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def leagues(tmp_path_factory):
    """Two tiny trained-for-one-iteration leagues (different seeds, so different weights)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(league_runner, "HORIZON", 30)
    mp.chdir(Path(evaluation.__file__).resolve().parents[1])  # the per-test chdir does not cover a module fixture
    out = {}
    try:
        for name, seed in (("A", 1), ("B", 2)):
            d = tmp_path_factory.mktemp(name)
            torch.manual_seed(seed), random.seed(seed)
            league_runner._run_session(1, 2, 100, None, 1, league_dir=str(d), seed=seed, roster=ROSTER,
                                       trunk_hidden=(8, 8), ppo_hparams={"n_epochs": 1})
            out[name] = str(d)
    finally:
        mp.undo()
    return out


def test_play_matchups_accounts_for_every_game_and_ignores_labels_and_executors(leagues):
    A, B = leagues["A"], leagues["B"]
    ms = [Matchup(AgentRef("elves", A, "x"), AgentRef("boggles", B)), Matchup(AgentRef("elves", A, "y"), AgentRef("boggles", B)),
          Matchup(AgentRef("boggles", A), AgentRef("boggles", B))]
    r = play_matchups(ms, games=8, seed=3, horizon=30)
    assert all(x.a_wins + x.b_wins + x.no_winner == 8 for x in r)
    assert (r[0].a_wins, r[0].b_wins, r[0].no_winner) == (r[1].a_wins, r[1].b_wins, r[1].no_winner), \
        "labels must not change play: same weights, same seed, same games"
    with ProcessPoolExecutor(max_workers=2) as ex:
        pooled = play_matchups(ms, games=8, seed=3, horizon=30, executor=ex)
    assert [(x.a_wins, x.b_wins, x.no_winner) for x in pooled] == [(x.a_wins, x.b_wins, x.no_winner) for x in r], \
        "results must not depend on the executor"


def test_head_to_head_of_a_league_with_itself_plays_and_accounts_for_every_game(leagues):
    A = leagues["A"]
    r = play_matchups(head_to_head(A, A, decks=["elves"]), games=10, seed=1, horizon=30)[0]
    assert r.a_wins + r.b_wins + r.no_winner == 10
