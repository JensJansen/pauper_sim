"""Opponent pool for league-style training: historical checkpoint snapshots
per deck (not latest-only), so a deck can't drift away from skills it needed
against an opponent's earlier self.

Sampling is two-level: pick a deck from the whole roster (including the
training deck itself, for mirror play), then decide live vs. checkpoint via
`checkpoint_rate` (default 0.0 -- always live), and if a checkpoint is drawn,
pick among that deck's held snapshots. checkpoint_rate is independent of how
many snapshots a deck has banked, so the live/checkpoint split stays a
stable, explicit number rather than drifting as the window fills.

Both levels are PFSP-weighted by default (pfsp=True): a candidate opponent
the training deck currently loses to more often is sampled more, per
`_pfsp_weight`'s (1 - win_rate) weighting (AlphaStar's PFSP). PFSP_FLOOR
keeps even a thoroughly-beaten opponent sampleable at a low rate rather than
starved to zero. A never-played candidate gets a neutral 0.5 win-rate
prior."""

import json
import os

from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork
from rl.decision.agent import AlwaysKeep, SeatAgent
from rl.model.mulligan import MulliganNet
from rl import checkpoint as ckpt_io

# ponytail: fixed rolling window per deck (32), oldest evicted (archived, see
# register_snapshot) first -- a real cap, so most of a deck's history stays
# reachable only via the vs_history eval spot-check.
DEFAULT_MAX_SNAPSHOTS_PER_DECK = 32

# PFSP weighting: weight(win_rate) = PFSP_FLOOR + (1 - win_rate) ** PFSP_POWER.
# Since (1 - win_rate) < 1, a LARGER exponent sharpens concentration onto the
# hardest opponent; a smaller one flattens it. PFSP_FLOOR keeps every
# opponent sampleable, never starved to zero.
PFSP_FLOOR = 0.1
PFSP_POWER = 0.5


def _format_opponent_key(key):
    """opponent_stats keys are ("deck", name) or ("snapshot", name, id) tuples
    -- JSON object keys must be strings, so join with ":" (deck names can't
    contain ":", so this round-trips exactly)."""
    return ":".join(str(part) for part in key)


def _parse_opponent_key(key_str):
    parts = key_str.split(":")
    if parts[0] == "deck":
        return ("deck", parts[1])
    return ("snapshot", parts[1], int(parts[2]))


class LeaguePool:
    def __init__(self, root_dir, deck_names, max_snapshots_per_deck=DEFAULT_MAX_SNAPSHOTS_PER_DECK,
                 pfsp_power=PFSP_POWER):
        self.root_dir = root_dir
        self.pfsp_power = pfsp_power  # config-driven so re-tuning doesn't need a code edit
        self.deck_names = list(deck_names)
        self.max_snapshots_per_deck = max_snapshots_per_deck
        self.snapshots = {name: [] for name in self.deck_names}  # per deck: [(id, path), ...] oldest first
        self._net_cache = {}  # snapshot path -> loaded, frozen DeckNetwork
        # opponent_stats[training_deck][opponent_key] = (wins, games), opponent_key
        # ("deck", name) or ("snapshot", name, id) -- see sample_opponent/_pfsp_weight.
        self.opponent_stats = {name: {} for name in self.deck_names}

        for name in self.deck_names:
            deck_dir = os.path.join(root_dir, name)
            os.makedirs(deck_dir, exist_ok=True)
            ids = sorted(
                int(fn[len("snapshot_"):-len(".pt")]) for fn in os.listdir(deck_dir)
                if fn.startswith("snapshot_") and fn.endswith(".pt")
            )
            self.snapshots[name] = [(i, os.path.join(deck_dir, f"snapshot_{i}.pt")) for i in ids]
            self.opponent_stats[name] = self._load_opponent_stats(name)

    def _stats_path(self, deck_name):
        return os.path.join(self.root_dir, deck_name, "opponent_stats.json")

    def _load_opponent_stats(self, deck_name):
        path = self._stats_path(deck_name)
        if not os.path.exists(path):
            return {}
        raw = json.load(open(path))
        return {_parse_opponent_key(k): tuple(v) for k, v in raw.items()}

    def save_opponent_stats(self):
        """Persists every deck's PFSP win/loss tallies to disk (one JSON file
        per deck). Called by league_runner._run_session at the same cadence
        as its other checkpoints, not every game -- without this, PFSP
        weighting would reset to its cold-start prior every session instead
        of accumulating across a run."""
        for name in self.deck_names:
            path = self._stats_path(name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            raw = {_format_opponent_key(k): list(v) for k, v in self.opponent_stats[name].items()}
            with open(path, "w") as f:
                json.dump(raw, f)

    def record_outcome(self, training_deck_name, opponent_key, won):
        """Updates training_deck_name's running (wins, games) tally against
        opponent_key. Called once per real training game, never a separate
        eval pass."""
        stats = self.opponent_stats.setdefault(training_deck_name, {})
        wins, games = stats.get(opponent_key, (0, 0))
        stats[opponent_key] = (wins + int(won), games + 1)

    def _pfsp_weight(self, training_deck_name, opponent_key):
        wins, games = self.opponent_stats.get(training_deck_name, {}).get(opponent_key, (0, 0))
        win_rate = wins / games if games else 0.5  # cold start: neutral prior, not "beats everyone" or "loses to everyone"
        return PFSP_FLOOR + (1.0 - win_rate) ** self.pfsp_power

    def register_snapshot(self, deck_name, net, mulligan_net=None):
        """Freezes net's current weights as a new historical opponent for
        deck_name, evicting the oldest snapshot once the window is full. A
        snapshot is the whole frozen agent: the DeckNetwork plus the paired
        MulliganNet (when given), so a historical opponent plays with its own
        era-matched pregame policy. mulligan_net=None writes a deck-only
        snapshot; load_snapshot_agent then falls back to AlwaysKeep for it."""
        deck_dir = os.path.join(self.root_dir, deck_name)
        os.makedirs(deck_dir, exist_ok=True)
        next_id = (self.snapshots[deck_name][-1][0] + 1) if self.snapshots[deck_name] else 0
        path = os.path.join(deck_dir, f"snapshot_{next_id}.pt")
        ckpt_io.save_snapshot(path, net, mulligan_net)
        self.snapshots[deck_name].append((next_id, path))
        while len(self.snapshots[deck_name]) > self.max_snapshots_per_deck:
            evicted_id, evicted_path = self.snapshots[deck_name].pop(0)
            # Moved into deck_dir/archive/, not deleted -- lets
            # league_runner._run_eval_vs_history measure a deck against its
            # own older selves after they age out of the active window.
            archive_dir = os.path.join(deck_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            os.replace(evicted_path, os.path.join(archive_dir, f"snapshot_{evicted_id}.pt"))
            self._net_cache.pop(evicted_path, None)
            # An evicted snapshot is no longer sampleable -- drop its stats
            # from every deck's opponent_stats too.
            evicted_key = ("snapshot", deck_name, evicted_id)
            for stats in self.opponent_stats.values():
                stats.pop(evicted_key, None)

    def sample_opponent(self, training_deck_name, rng, checkpoint_rate=0.0, pfsp=True):
        """Returns (opponent_deck_name, snapshot_path_or_None). None means
        "use that deck's current live net" -- when opponent_deck_name ==
        training_deck_name and snapshot_path is None, that's a true mirror,
        not a frozen copy.

        checkpoint_rate: the live/checkpoint split, independent of how many
        snapshots that deck has banked. 0.0 (default): always live. A deck
        with an empty snapshot list always resolves to live regardless.

        pfsp: weight both the deck choice and (if a checkpoint is drawn) the
        snapshot choice by _pfsp_weight instead of drawing uniformly.
        pfsp=False restores plain-uniform behavior."""
        if pfsp:
            weights = [self._pfsp_weight(training_deck_name, ("deck", name)) for name in self.deck_names]
            opponent_deck_name = rng.choices(self.deck_names, weights=weights, k=1)[0]
        else:
            opponent_deck_name = rng.choice(self.deck_names)
        snaps = self.snapshots[opponent_deck_name]
        if snaps and rng.random() < checkpoint_rate:
            if pfsp:
                weights = [self._pfsp_weight(training_deck_name, ("snapshot", opponent_deck_name, sid))
                           for sid, _path in snaps]
                _sid, chosen_path = rng.choices(snaps, weights=weights, k=1)[0]
                return opponent_deck_name, chosen_path
            return opponent_deck_name, rng.choice([path for _id, path in snaps])
        return opponent_deck_name, None

    def load_snapshot_agent(self, snapshot_path, deck_ctx):
        """Builds (or returns a cached) frozen SeatAgent for a historical
        snapshot -- the DeckNetwork plus its era-matched pregame decider (a
        frozen MulliganNet, or AlwaysKeep for a deck-only snapshot). Never
        trained itself, only ever a rollout opponent. Cached by path, so the
        same SeatAgent object plays many games and its recurrent state must
        be cleared between them (collect_rollout's game loop does this).

        Takes no encoder: a snapshot carries the perception encoder its
        policy was trained with inside its own state_dict. A fresh
        SetTransformer is built here only as a shape to load those weights
        into, sized from deck_ctx's own vocab -- so a historical opponent is
        an honest snapshot of the era it came from."""
        if snapshot_path in self._net_cache:
            return self._net_cache[snapshot_path]
        vocab, fixed_table = deck_ctx
        saved = ckpt_io.load_snapshot(snapshot_path)
        encoder = SetTransformer(vocab.size)
        # SetTransformer's default architecture is the only one this repo
        # ever trains. Checked explicitly so a mismatch fails with a clear
        # message instead of a wall of unrelated torch size-mismatch errors.
        saved_d_model = saved["state_dict"]["encoder.embedding.weight"].shape[1]
        assert saved_d_model == encoder.d_model, (
            f"{snapshot_path} was saved with a d_model={saved_d_model} encoder, but SetTransformer's "
            f"current default is {encoder.d_model} -- the architecture changed since this snapshot was "
            f"written, so its weights cannot be loaded"
        )
        net = DeckNetwork(encoder, film_condition_dim=encoder.d_model, non_targeting_n_actions=len(fixed_table),
                           trunk_hidden=saved["trunk_hidden"])
        net.load_state_dict(saved["state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        if "mulligan_state_dict" in saved:
            mull = MulliganNet(net.encoder, hidden=saved.get("mulligan_hidden", 64))
            mull.load_state_dict(saved["mulligan_state_dict"])
            mull.eval()
            for p in mull.parameters():
                p.requires_grad = False
        else:
            mull = AlwaysKeep()  # no mulligan state saved -> neutral pregame

        agent = SeatAgent(net, mull, deck_ctx)
        self._net_cache[snapshot_path] = agent
        return agent
