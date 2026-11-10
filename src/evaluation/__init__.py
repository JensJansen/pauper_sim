"""Strength evaluation of trained agents, independent of any league's training loop.

An agent is "the policy a deck's live.pt/mulligan.pt pair holds, in some directory" (AgentRef), so the two sides
of a game can come from DIFFERENT checkpoint directories -- two training arms, a past and a present version of one
league, or a candidate against a hand-built reference field. That is what validation/ (one league, one directory)
cannot express.

  matchups.py  AgentRef / Matchup and the standard designs (gauntlet, head_to_head, round_robin, cross)
  play.py      play_matchups: plays them (greedy by default) over an optional ProcessPoolExecutor
  stats.py     Tally + Wilson / difference confidence intervals
  history.py   a league's archived snapshots addressed by games trained (AgentRef.snapshot)
  probe.py     policy-movement probe: how far a checkpoint's decisions move over a fixed set of mid-game states

CLI: `python -m evaluation strength|history|probe` (cli_strength.py: strength; cli_history.py: strength over a
league's history; cli_probe.py: policy movement). A typical use is a paired comparison of training variants: evaluate each variant's
decks against the SAME frozen field with the SAME seed (common random numbers -- see play.matchup_seed), then
compare the variants' scores with stats.diff_interval.
"""
from .matchups import AgentRef, Matchup, cross, gauntlet, head_to_head, round_robin
from .play import MatchupResult, play_matchups
from .stats import Tally, diff_interval, tally, wilson_interval

__all__ = ["AgentRef", "Matchup", "cross", "gauntlet", "head_to_head", "round_robin", "MatchupResult",
           "play_matchups", "Tally", "diff_interval", "tally", "wilson_interval"]
