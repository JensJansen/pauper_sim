"""The numbers decisions rest on, against scipy and by simulation (plan P8).

The 2026-08 plateau reading and the S-vs-L1 A/B both used evaluation.stats / evaluation.verdicts, so these are checked against an
independent implementation (scipy) and against their own stated error rates (coverage, false-positive rate, power)."""
import math

import numpy as np
import pytest
from scipy import stats as sp

from evaluation import verdicts as V
from evaluation.play import MatchupResult
from evaluation.stats import Tally, diff_interval, tally, wilson_interval

CASES = [(0, 1), (1, 1), (0, 10), (10, 10), (3, 10), (50, 100), (7, 100), (93, 100), (480, 1000)]


@pytest.mark.parametrize("wins,n", CASES)
def test_wilson_interval_equals_scipys(wins, n):
    ref = sp.binomtest(wins, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = wilson_interval(Tally(wins, 0, n))
    assert lo == pytest.approx(ref.low, abs=1e-5) and hi == pytest.approx(ref.high, abs=1e-5)
    p, vlo, vhi = V.wilson(wins, n)
    assert p == wins / n and vlo == pytest.approx(max(0.0, ref.low), abs=1e-5) and vhi == pytest.approx(min(1.0, ref.high), abs=1e-5)


def test_wilson_with_no_games_is_the_uninformative_interval():
    p, lo, hi = V.wilson(0, 0)
    assert math.isnan(p) and (lo, hi) == (0.0, 1.0)
    assert all(math.isnan(x) for x in wilson_interval(Tally()))


def test_a_tie_scores_half_and_tallies_add():
    t = Tally(3, 2, 10) + Tally(1, 0, 4)
    assert t == Tally(4, 2, 14) and t.score == pytest.approx(5 / 14)


def test_tally_sums_matchup_results_from_either_side():
    rs = [MatchupResult("a", "b", 10, 6, 3, 1), MatchupResult("a", "b", 10, 2, 8, 0)]
    assert tally(rs, "a") == Tally(8, 1, 20) and tally(rs, "b") == Tally(11, 1, 20)


@pytest.mark.parametrize("p,n", [(0.1, 30), (0.3, 100), (0.5, 50), (0.5, 400), (0.77, 200)])
def test_the_wilson_interval_covers_the_true_rate_about_95_percent_of_the_time(p, n):
    k = np.arange(n + 1)
    pmf = sp.binom.pmf(k, n, p)
    covered = 0.0
    for wins, w in zip(k, pmf, strict=True):
        lo, hi = wilson_interval(Tally(int(wins), 0, n))
        covered += w * (lo <= p <= hi)
    assert 0.93 <= covered <= 0.985, f"exact coverage {covered:.4f}"


@pytest.mark.parametrize("w1,n1,w2,n2", [(40, 100, 55, 100), (500, 1000, 480, 1000), (3, 20, 9, 30), (0, 50, 10, 50)])
def test_two_proportion_z_is_the_pooled_z_scipy_agrees_with(w1, n1, w2, n2):
    chi2 = sp.chi2_contingency([[w1, n1 - w1], [w2, n2 - w2]], correction=False)[0]
    z = V.two_proportion_z(w1, n1, w2, n2)
    assert z * z == pytest.approx(chi2, rel=1e-6)
    assert (z > 0) == (w2 / n2 > w1 / n1), "positive means group 2 is higher"


@pytest.mark.parametrize("args", [(0, 0, 5, 10), (5, 10, 0, 0), (0, 10, 0, 10), (10, 10, 10, 10)])
def test_two_proportion_z_is_zero_for_empty_or_degenerate_groups(args):
    assert V.two_proportion_z(*args) == 0.0


def test_two_proportion_z_false_positive_rate_is_about_the_nominal_five_percent():
    rng = np.random.default_rng(0)
    n, reps = 300, 20000
    a, b = rng.binomial(n, 0.5, reps), rng.binomial(n, 0.5, reps)
    z = np.array([V.two_proportion_z(int(x), n, int(y), n) for x, y in zip(a[:4000], b[:4000], strict=True)])
    assert 0.035 <= (np.abs(z) > 1.96).mean() <= 0.065


def test_the_minimum_detectable_effect_really_is_detected_about_80_percent_of_the_time():
    rng = np.random.default_rng(1)
    n, reps = 400, 4000
    delta = V.min_detectable_effect(n)
    a, b = rng.binomial(n, 0.5, reps), rng.binomial(n, 0.5 + delta, reps)
    z = np.array([V.two_proportion_z(int(x), n, int(y), n) for x, y in zip(a, b, strict=True)])
    assert 0.74 <= (z > 1.96).mean() <= 0.86


def test_min_detectable_effect_shrinks_with_n_and_is_one_without_games():
    assert V.min_detectable_effect(0) == 1.0
    sizes = [V.min_detectable_effect(n) for n in (50, 200, 800, 3200)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[1] == pytest.approx(sizes[0] / 2) and sizes[2] == pytest.approx(sizes[1] / 2)


def test_diff_interval_is_the_normal_interval_for_a_difference_of_proportions():
    d, lo, hi = diff_interval(Tally(60, 0, 100), Tally(45, 0, 100))
    se = math.sqrt(0.6 * 0.4 / 100 + 0.45 * 0.55 / 100)
    assert d == pytest.approx(0.15) and lo == pytest.approx(0.15 - 1.959964 * se) and hi == pytest.approx(0.15 + 1.959964 * se)


# ---- the trend and verdict logic -----------------------------------------------------------------------------------------------

def _rows(rates, games=200):
    return [{"games": games, "wins": round(r * games)} for r in rates]


def test_trend_z_is_positive_for_a_rising_series_negative_for_a_falling_one_and_small_for_noise():
    assert V.trend_z(_rows(np.linspace(0.40, 0.60, 10))) > 3
    assert V.trend_z(_rows(np.linspace(0.60, 0.40, 10))) < -3
    assert abs(V.trend_z(_rows([0.5, 0.52, 0.48, 0.5, 0.51, 0.49, 0.5, 0.5]))) < 1


def test_trend_z_matches_a_weighted_regression_of_win_rate_on_index():
    """Cochran-Armitage is the score test of a logistic slope; for a mild trend it must agree with the slope t-statistic of a
    weighted least-squares fit up to the variance estimate -- check sign, order of magnitude, and exactness of the closed form."""
    rates = [0.42, 0.47, 0.5, 0.49, 0.55, 0.58, 0.6]
    rows = _rows(rates, 300)
    x = np.arange(len(rows))
    w = np.array([r["wins"] for r in rows])
    n = np.array([r["games"] for r in rows])
    p = w.sum() / n.sum()
    xbar = (x * n).sum() / n.sum()
    closed = ((x - xbar) * (w - n * p)).sum() / math.sqrt(p * (1 - p) * (n * (x - xbar) ** 2).sum())
    assert V.trend_z(rows) == pytest.approx(closed)
    slope = np.polyfit(x, np.array(rates), 1)[0]
    assert slope > 0 and V.trend_z(rows) > 2


@pytest.mark.parametrize("rows", [[], _rows([0.5]), _rows([1.0, 1.0, 1.0]), _rows([0.0, 0.0]), [{"games": 0}] * 4])
def test_trend_z_is_zero_when_there_is_nothing_to_regress_against(rows):
    assert V.trend_z(rows) == 0.0


def test_series_verdict_labels():
    up = _rows(np.linspace(0.40, 0.62, 12))
    assert V.series_verdict(up, 130, 200)[0] == "IMPROVING"
    flat = _rows([0.5, 0.51, 0.49, 0.5, 0.5, 0.52, 0.48, 0.5, 0.5, 0.51, 0.5, 0.49])
    assert V.series_verdict(flat, 100, 200)[0] == "FLAT"
    down = _rows(np.linspace(0.62, 0.40, 12))
    assert V.series_verdict(down, 80, 200)[0] in ("REGRESSING", "PAST PEAK")
    assert V.series_verdict(up, 196, 200)[0] == "SATURATED", "a late score at 95%+ is saturated whatever the trend"


def test_a_rise_then_fall_is_labelled_past_peak_and_that_outranks_regressing():
    rates = [0.40, 0.48, 0.56, 0.64, 0.70, 0.70, 0.64, 0.56, 0.48, 0.40, 0.40, 0.40]
    label, z, peak_z, peak_at = V.series_verdict(_rows(rates, 400), 160, 400)
    assert z < -2, "the falling tail also makes the trend test fire; the verdict must still be the more specific PAST PEAK"
    assert label == "PAST PEAK" and peak_z < 0 and peak_at is not None


def test_peak_comparison_needs_two_windows_and_is_zero_when_the_latest_window_is_the_best():
    assert V.peak_comparison(_rows([0.5] * 5))[0] == 0.0
    z, at, _crit = V.peak_comparison(_rows(np.linspace(0.4, 0.7, 12)))
    assert z == 0.0 and at == 11
