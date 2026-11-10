"""What to play: agents and the standard ways of pairing them up."""
import itertools
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRef:
    """The agent for `deck` stored under `league_dir`/<deck>/ (live.pt + mulligan.pt). `label` only names the
    agent in reports (e.g. "S", "L1", "start"); it never affects play."""
    deck: str
    league_dir: str
    label: str = ""

    def __post_init__(self):
        object.__setattr__(self, "league_dir", os.path.abspath(self.league_dir))

    @property
    def name(self):
        return f"{self.label or os.path.basename(self.league_dir)}/{self.deck}"


@dataclass(frozen=True)
class Matchup:
    """`a` plays `b`. Who goes first is decided per game by the game loop's own coin flip, so neither side is
    systematically on the play."""
    a: AgentRef
    b: AgentRef


def available_decks(league_dir):
    """Decks that have a saved live.pt under league_dir, sorted."""
    return sorted(d for d in os.listdir(league_dir) if os.path.exists(os.path.join(league_dir, d, "live.pt")))


def _refs(league_dir, decks, label):
    return [AgentRef(d, league_dir, label) for d in (decks if decks is not None else available_decks(league_dir))]


def cross(subjects, field):
    """Every subject against every field agent (subject is `a`)."""
    return [Matchup(s, f) for s in subjects for f in field]


def gauntlet(subject_dir, field_dir, subject_decks=None, field_decks=None, subject_label="", field_label=""):
    """Each subject deck against each field deck (mirrors included) -- the subject league measured against a fixed
    reference field. Compare several subject directories against the same field for a paired comparison."""
    return cross(_refs(subject_dir, subject_decks, subject_label), _refs(field_dir, field_decks, field_label))


def head_to_head(dir_a, dir_b, decks=None, label_a="", label_b=""):
    """The same deck from two directories against itself, for each deck both have."""
    decks = decks if decks is not None else sorted(set(available_decks(dir_a)) & set(available_decks(dir_b)))
    return [Matchup(AgentRef(d, dir_a, label_a), AgentRef(d, dir_b, label_b)) for d in decks]


def round_robin(league_dir, decks=None, label=""):
    """Every pair from one directory, mirrors included (what validation/'s round robin plays)."""
    refs = _refs(league_dir, decks, label)
    return [Matchup(a, b) for a, b in itertools.combinations_with_replacement(refs, 2)]
