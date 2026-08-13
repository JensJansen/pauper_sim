"""Helpers shared by several analysis scripts.

Extracted 2026-08-13 from analyze_mana_burn_by_turn.py, which was DELETED for
carrying a hardcoded `CURVES` table of reward constants that had silently
drifted out of sync with rl.rewards (the drift that motivated
analyze_burn_saturation.py, which replays the reward's OWN
charge_single_pip_burn instead of a copy of its numbers). The table deserved to
go; these three helpers did not, and three sibling scripts import them --
analyze_decision_entropy, analyze_mana_burn_turns, analyze_target_fizzle.

Nothing here duplicates a constant that lives in rl.*: DEFAULT_ROSTER is a
convenience default for ad-hoc analysis (the real roster is whatever a config's
own `roster` says), and the other two only read the engine's own event log.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `repo_paths` / `rl.*`

from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet
from rl import checkpoint as ckpt_io
from rl.league_runner import D_MODEL

# The 4-deck sub-league this project's analysis mostly targets. A DEFAULT for
# scripts run by hand, never authoritative -- training reads its roster from
# training_configs/*.json, and league_runner.league_roster reads what is
# actually on disk.
DEFAULT_ROSTER = ["rakdos_madness", "dmir_terror", "elves", "mono_red_rally"]


def _load_deck(league_dir, name, shared, fixed_tables):
    """(DeckNetwork, MulliganNet) for one deck's CURRENT live checkpoint, both
    in eval mode. For a historical snapshot instead, use
    rl.league_runner.load_vintage_agent, which also resolves archive/ paths."""
    net = DeckNetwork(shared, film_condition_dim=D_MODEL, non_targeting_n_actions=len(fixed_tables[name]))
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/live.pt", net)
    net.eval()
    mnet = MulliganNet(shared)
    ckpt_io.load_deck_checkpoint(f"{league_dir}/{name}/mulligan.pt", mnet)
    mnet.eval()
    return net, mnet


def _per_turn_tagged_burn(event_log, seat):
    """turn -> single-pip-tagged pips burnt that turn, for `seat`, read off the
    engine's own mana_emptied events (rule 500.4's pool empty). Tagged-only:
    board-state-scaled burst sources (Priest of Titania, Overgrown Battlement)
    are excluded pip by pip upstream, so this matches what the reward charges."""
    by_turn = {}
    for e in event_log:
        if e["kind"] != "mana_emptied":
            continue
        amt = e["pools_single_pip"].get(seat, 0)
        if amt:
            by_turn[e["turn"]] = by_turn.get(e["turn"], 0) + amt
    return by_turn
