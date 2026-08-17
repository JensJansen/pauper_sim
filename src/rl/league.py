"""Opponent pool for league-style training: historical checkpoint snapshots
per deck (not latest-only, so a deck can't quietly drift away from skills it
needed against an opponent's earlier self).

Sampling is two-level: pick a deck from the whole roster (including the
training deck itself, for mirror play), THEN decide live vs. checkpoint for
that deck via `checkpoint_rate` (a live/checkpoint coin flip with a
caller-set probability, default 0.0 -- always live), and if a checkpoint is
drawn, pick among that deck's currently-held snapshots. checkpoint_rate makes
the live/checkpoint split an explicit, stable number, deliberately
independent of how many snapshots a deck happens to have banked -- deriving
that ratio from snapshot-window occupancy instead (e.g. uniformly among
{live} union {every snapshot}) would let the live-net probability silently
shrink as the window filled (1/(N+1), so 80% checkpoint odds once a deck had
4 snapshots), a side effect of window size rather than a deliberate choice.

Early training uses NO checkpoint opponents at all (checkpoint_rate=0.0, the
default): an early snapshot is barely-trained and teaches little as an
opponent, so paying collection cost against one is close to wasted relative
to a live opponent that's also improving. Reintroducing checkpoint diversity
later is an explicit, deliberate choice (a nonzero rate), not an automatic
side effect of the pool filling up.

BOTH levels are PFSP-weighted by default (pfsp=True), not uniform: within
each level, a candidate opponent (a deck, or one of its snapshots) the
training deck currently loses to more often is sampled MORE, per
`_pfsp_weight`'s (1 - win_rate) weighting -- "compete against whoever's most
likely to beat you right now" (AlphaStar's PFSP), rather than every
opponent getting equal airtime regardless of whether the training deck has
already solved it. A PFSP_FLOOR keeps even a thoroughly-beaten opponent
sampleable at a low rate -- driving its weight to exactly zero would starve
it and reintroduce the catastrophic-forgetting failure this pool's snapshot
history exists to prevent in the first place. A never-yet-played candidate
(0 recorded games) gets a neutral 0.5 win-rate prior, which is why every
existing test that never calls record_outcome still observes effectively
uniform sampling: every untried candidate carries the same weight."""

import json
import os

from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.agent import AlwaysKeep, SeatAgent
from rl.mulligan import MulliganNet
from rl import checkpoint as ckpt_io

# ponytail: fixed rolling window per deck, oldest evicted (archived, see
# register_snapshot) first. Raised 8->32 (2026-08-06): at snapshot_every_
# games=200 (training_configs/run_default.json), 8 slots capped the ACTIVE
# (actually-sampleable-as-a-training-opponent) window at 1,600 games no
# matter how long a run continued -- confirmed on a real 34,579-games/deck
# run to be 95.4% of that deck's own history permanently unreachable except
# via the vs_history eval spot-check. 32 gives 6,400 games of reachable
# memory instead -- still finite, still a real design tradeoff (an older
# opponent is a weaker training signal per the module docstring above), but
# a meaningfully deeper one. See TRAINING_IMPROVEMENT_OPTIONS.md section 3.
DEFAULT_MAX_SNAPSHOTS_PER_DECK = 32

# PFSP weighting: weight(win_rate) = PFSP_FLOOR + (1 - win_rate) ** PFSP_POWER.
#
# POWER 2.0 -> 0.5 (2026-08-13). THE 2026-08-06 CHANGE WENT THE WRONG WAY.
# It was made to REDUCE concentration on unwinnable matchups and did the exact
# opposite: since (1 - win_rate) < 1, raising the exponent SHARPENS the
# weighting onto the hardest opponent. Lowering it flattens. The comment that
# stood here claimed the opposite for a week.
#
# Predicted shares reproduce what was actually observed on disk to within 1pp,
# so this formula is the mechanism, not a guess:
#
#   POWER  elves: mono_red / rakdos / dmir / mirror   hardest:mirror
#   0.5         28.1 / 27.5 / 23.7 / 20.7                  1.36
#   1.0         31.1 / 29.6 / 22.2 / 17.1                  1.82
#   2.0         36.3 / 33.0 / 18.9 / 11.8                  3.09
#   observed    36.9 / 33.0 / 18.4 / 11.8                    --
#
# Why this matters more than a mis-set knob. Measured from the real
# opponent_stats.json at 60,001 games/deck, the share of each deck's training
# games spent in matchups it wins <25% of, against the Elo it gained over the
# whole run:
#
#   mono_red_rally   0.0% unwinnable, 51.3% mirror  ->  +241 Elo
#   rakdos_madness  58.0% unwinnable, 25.5% mirror  ->   -66 Elo
#   elves           69.8% unwinnable, 11.8% mirror  ->   -57 Elo
#   dmir_terror     76.5% unwinnable, 14.7% mirror  ->     0 Elo
#
# The one deck with a balanced training distribution is the one deck that
# improved. Elves spent 36.9% of its games at a 1.1% win rate. Those games do
# not merely waste compute: nearly every trajectory returns -1, so the raw
# advantage spread is almost pure noise, and rl.ppo's unconditional
# `adv = (adv - adv.mean()) / (adv.std() + 1e-8)` rescales that noise to UNIT
# VARIANCE before ~67 Adam steps are taken on it.
#
# 0.5 is a first estimate exactly as 2.0 was. The table above is now a
# calibration instrument: re-check the observed shares at the next checkpoint
# and re-tune. PFSP_FLOOR keeps even a thoroughly-beaten opponent sampleable,
# so nothing is ever starved to zero.
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
        # Config-driven rather than a module constant: 0.5 is a first estimate
        # and re-tuning it must not require a code edit (see PFSP_POWER above).
        self.pfsp_power = pfsp_power
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
        """Persists every deck's running PFSP win/loss tallies to disk (JSON,
        one file per deck alongside its live.pt/snapshot_*.pt) -- called by
        rl.league_runner._run_session on the SAME cadence as its other checkpoints
        (snapshot points + session end), not on every game: each real
        training run is many SEPARATE short-lived process invocations (no
        long-running daemon), so without this the PFSP weighting would
        silently reset to its cold-start uniform prior every single session
        instead of accumulating across a run."""
        for name in self.deck_names:
            path = self._stats_path(name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            raw = {_format_opponent_key(k): list(v) for k, v in self.opponent_stats[name].items()}
            with open(path, "w") as f:
                json.dump(raw, f)

    def record_outcome(self, training_deck_name, opponent_key, won):
        """Updates training_deck_name's running (wins, games) tally against
        opponent_key. Called once per REAL training game (never a separate
        eval pass) via the on_game_end hook threaded through
        rl.train.collect_rollout(_league(_parallel)) -- rides along with
        normal rollout collection at zero extra compute cost."""
        stats = self.opponent_stats.setdefault(training_deck_name, {})
        wins, games = stats.get(opponent_key, (0, 0))
        stats[opponent_key] = (wins + int(won), games + 1)

    def _pfsp_weight(self, training_deck_name, opponent_key):
        wins, games = self.opponent_stats.get(training_deck_name, {}).get(opponent_key, (0, 0))
        win_rate = wins / games if games else 0.5  # cold start: neutral prior, not "beats everyone" or "loses to everyone"
        return PFSP_FLOOR + (1.0 - win_rate) ** self.pfsp_power

    def register_snapshot(self, deck_name, net, mulligan_net=None):
        """Freezes net's CURRENT weights as a new historical opponent for
        deck_name, evicting the oldest snapshot once the window is full. A
        snapshot is the whole frozen AGENT: the DeckNetwork PLUS the paired
        MulliganNet (when given), so a historical opponent plays with its own
        era-matched pregame policy rather than borrowing whatever the current
        mulligan net happens to be. Saves trunk_hidden alongside the state_dict
        (derived from the net's own trunk_layers, not assumed) -- load has no
        other way to know what shape to reconstruct. mulligan_net=None writes a
        deck-only snapshot with no mulligan state; load_snapshot_agent then
        falls back to AlwaysKeep for any snapshot missing one."""
        deck_dir = os.path.join(self.root_dir, deck_name)
        os.makedirs(deck_dir, exist_ok=True)
        next_id = (self.snapshots[deck_name][-1][0] + 1) if self.snapshots[deck_name] else 0
        path = os.path.join(deck_dir, f"snapshot_{next_id}.pt")
        ckpt_io.save_snapshot(path, net, mulligan_net)
        self.snapshots[deck_name].append((next_id, path))
        while len(self.snapshots[deck_name]) > self.max_snapshots_per_deck:
            evicted_id, evicted_path = self.snapshots[deck_name].pop(0)
            # Moved into deck_dir/archive/, not deleted: the active sampling
            # window (max_snapshots_per_deck) stays small on purpose (a stale
            # snapshot is a weak training opponent), but permanently discarding
            # it left no way to ever measure a deck's win rate against its own
            # older selves once they aged out -- see rl.league_runner._run_eval_vs_history,
            # the reason this archive exists. Cheap: ~1MB/snapshot (checked on
            # disk), so archiving every eviction for a whole run is a few GB at most.
            archive_dir = os.path.join(deck_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            os.replace(evicted_path, os.path.join(archive_dir, f"snapshot_{evicted_id}.pt"))
            self._net_cache.pop(evicted_path, None)
            # An evicted snapshot is no longer sample-able by ANYONE, so its
            # per-training-deck PFSP stats are dead weight -- drop them from
            # every deck's opponent_stats (not just deck_name's own), bounded
            # cost (at most len(deck_names) small-dict pops).
            evicted_key = ("snapshot", deck_name, evicted_id)
            for stats in self.opponent_stats.values():
                stats.pop(evicted_key, None)

    def sample_opponent(self, training_deck_name, rng, checkpoint_rate=0.0, pfsp=True):
        """Returns (opponent_deck_name, snapshot_path_or_None). None means
        "use that deck's current live net" (the caller already holds every
        deck's live net; when opponent_deck_name == training_deck_name AND
        snapshot_path is None, that IS the training net itself -- true
        mirror, not a frozen copy of it).

        checkpoint_rate: the live/checkpoint split for whichever deck gets
        picked, decided independently of how many snapshots that deck happens
        to have banked (see this module's own docstring for why). 0.0
        (default): always live, structurally -- the rng.random() draw and
        comparison never happen, so this is exact, not "very unlikely," even
        before any snapshots exist. A deck with an EMPTY snapshot list always
        resolves to live regardless of checkpoint_rate (nothing to draw
        yet).

        pfsp: weight BOTH the deck choice and (if a checkpoint is drawn) the
        snapshot choice by _pfsp_weight instead of drawing uniformly -- see
        this module's own docstring. True by default; pfsp=False restores
        the plain-uniform behavior this replaced, kept as an escape hatch."""
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
        snapshot -- the DeckNetwork PLUS its era-matched pregame decider (a
        frozen MulliganNet if the snapshot carries one, else AlwaysKeep for a
        deck-only snapshot with no mulligan state). Never trained itself, only
        ever a rollout opponent (collect_rollout's own inference-mode forward
        covers inference; requires_grad=False here is defensive). Cached by
        path so repeat samples reuse it -- which means the SAME SeatAgent
        object plays many different games, so its per-game recurrent state has
        to be cleared between them. rl.train.collect_rollout's own game loop
        does that for every agent a pairing hands it; see SeatAgent.reset for
        why it is load-bearing rather than hygiene.

        Takes no encoder: a snapshot is SELF-CONTAINED, carrying the perception
        encoder its policy was trained with inside its own state_dict
        (rl.deck.DeckNetwork registers it). A fresh SetTransformer is built
        here purely as a shape to load those weights into, sized from
        deck_ctx's own vocab. That is what makes a historical opponent an
        honest snapshot of the era it came from -- under the previous shared-
        frozen-stack design every vintage was replayed on TODAY's encoder, and
        the caller had to supply one (rl.train even passed the TRAINING deck's
        stack to build its OPPONENT)."""
        if snapshot_path in self._net_cache:
            return self._net_cache[snapshot_path]
        vocab, fixed_table = deck_ctx
        saved = ckpt_io.load_snapshot(snapshot_path)
        encoder = SetTransformer(vocab.size)
        # Built at SetTransformer's own default architecture, which is the ONE
        # architecture anything in this repo ever trains (unlike trunk_hidden,
        # which is per-league configurable and therefore recorded in the
        # snapshot). Checked explicitly because the raw failure is a wall of
        # torch size-mismatch lines naming an unrelated tensor
        # ("pointer_query.bias") rather than the actual cause.
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
