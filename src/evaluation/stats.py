"""Counting wins and putting error bars on them. Stdlib only."""
import math
from dataclasses import dataclass

Z95 = 1.959964


@dataclass(frozen=True)
class Tally:
    """Games from one side's point of view. A game nobody won (horizon timeout) is `ties` and scores 0.5."""
    wins: int = 0
    ties: int = 0
    games: int = 0

    @property
    def score(self):
        return (self.wins + 0.5 * self.ties) / self.games if self.games else float("nan")

    def __add__(self, other):
        return Tally(self.wins + other.wins, self.ties + other.ties, self.games + other.games)


def tally(results, side="a"):
    """Sum MatchupResults (anything with a_wins/b_wins/no_winner/games) into one Tally for `side` ("a" or "b")."""
    t = Tally()
    for r in results:
        t += Tally(r.a_wins if side == "a" else r.b_wins, r.no_winner, r.games)
    return t


def wilson_interval(t, z=Z95):
    """Wilson score interval for t.score. Treats the score as a binomial proportion, which is exact without ties
    and a close approximation with them (ties are rare: they need a horizon timeout)."""
    n = t.games
    if not n:
        return (float("nan"), float("nan"))
    p = t.score
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def diff_interval(t1, t2, z=Z95):
    """(score1 - score2, low, high): normal-approximation interval for the difference of two independent tallies.
    Independent is conservative for a paired design (shared seeds make the arms mildly positively correlated,
    which only tightens the true interval)."""
    d = t1.score - t2.score
    se = math.sqrt(t1.score * (1 - t1.score) / t1.games + t2.score * (1 - t2.score) / t2.games)
    return (d, d - z * se, d + z * se)
