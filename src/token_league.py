"""Opponent pool for league-style training (replaces the old pairwise
Stage 1/Stage 2 scripts) -- per confirmed design: historical checkpoint
snapshots per deck (not latest-only, so a deck can't quietly drift away
from skills it needed against an opponent's earlier self), sampled
uniformly (not skill-weighted -- no meaningful signal to weight by yet
with only a couple of decks in the roster; revisit once the league is
bigger).

Sampling is two-level uniform: pick a deck uniformly from the whole
roster (including the training deck itself, for mirror play), then pick
a snapshot of THAT deck uniformly (its current live weights count as one
option, each historical snapshot another) -- keeps one deck's snapshot
count from skewing the overall distribution for reasons unrelated to
which decks exist."""

import os

import torch

from token_deck import DeckNetwork

# ponytail: fixed rolling window per deck, oldest evicted first. Revisit
# (e.g. keep-most-diverse instead of keep-most-recent) if a shallow window
# turns out to lose opponent diversity that actually mattered.
DEFAULT_MAX_SNAPSHOTS_PER_DECK = 8


class LeaguePool:
    def __init__(self, root_dir, deck_names, max_snapshots_per_deck=DEFAULT_MAX_SNAPSHOTS_PER_DECK):
        self.root_dir = root_dir
        self.deck_names = list(deck_names)
        self.max_snapshots_per_deck = max_snapshots_per_deck
        self.snapshots = {name: [] for name in self.deck_names}  # per deck: [(id, path), ...] oldest first
        self._net_cache = {}  # snapshot path -> loaded, frozen DeckNetwork

        for name in self.deck_names:
            deck_dir = os.path.join(root_dir, name)
            os.makedirs(deck_dir, exist_ok=True)
            ids = sorted(
                int(fn[len("snapshot_"):-len(".pt")]) for fn in os.listdir(deck_dir)
                if fn.startswith("snapshot_") and fn.endswith(".pt")
            )
            self.snapshots[name] = [(i, os.path.join(deck_dir, f"snapshot_{i}.pt")) for i in ids]

    def register_snapshot(self, deck_name, net):
        """Freezes net's CURRENT weights as a new historical opponent for
        deck_name, evicting the oldest snapshot once the window is full.
        Saves trunk_hidden alongside the state_dict (derived from the
        net's own trunk_layers, not assumed) -- load_snapshot_net has no
        other way to know what shape to reconstruct, and silently
        defaulting to DeckNetwork's own default would break the moment a
        caller ever builds decks with a non-default trunk_hidden (caught
        the hard way: this module's own self-check missed it because it
        never actually round-tripped a snapshot with a non-default trunk;
        token_train.py's smoke test, which does, is what caught it)."""
        deck_dir = os.path.join(self.root_dir, deck_name)
        os.makedirs(deck_dir, exist_ok=True)
        next_id = (self.snapshots[deck_name][-1][0] + 1) if self.snapshots[deck_name] else 0
        path = os.path.join(deck_dir, f"snapshot_{next_id}.pt")
        trunk_hidden = tuple(layer.out_features for layer in net.trunk_layers)
        torch.save({"state_dict": net.state_dict(), "trunk_hidden": trunk_hidden}, path)
        self.snapshots[deck_name].append((next_id, path))
        while len(self.snapshots[deck_name]) > self.max_snapshots_per_deck:
            _evicted_id, evicted_path = self.snapshots[deck_name].pop(0)
            os.remove(evicted_path)
            self._net_cache.pop(evicted_path, None)

    def sample_opponent(self, training_deck_name, rng):
        """Returns (opponent_deck_name, snapshot_path_or_None). None means
        "use that deck's current live net" (the caller already holds every
        deck's live net; when opponent_deck_name == training_deck_name AND
        snapshot_path is None, that IS the training net itself -- true
        mirror, not a frozen copy of it)."""
        opponent_deck_name = rng.choice(self.deck_names)
        candidates = [None] + [path for _id, path in self.snapshots[opponent_deck_name]]
        return opponent_deck_name, rng.choice(candidates)

    def load_snapshot_net(self, snapshot_path, shared_stack, deck_ctx):
        """Builds (or returns a cached) frozen DeckNetwork for a historical
        snapshot -- never trained itself, only ever used as a rollout
        opponent (collect_rollout's own no_grad forward pass already
        covers inference; requires_grad=False here is defensive, not load-
        bearing, matching the frozen shared stack's own convention)."""
        if snapshot_path in self._net_cache:
            return self._net_cache[snapshot_path]
        _vocab, fixed_table, _pending_kinds = deck_ctx
        saved = torch.load(snapshot_path, weights_only=True)
        net = DeckNetwork(shared_stack, film_condition_dim=shared_stack.d_model, non_targeting_n_actions=len(fixed_table),
                           trunk_hidden=saved["trunk_hidden"])
        net.load_state_dict(saved["state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self._net_cache[snapshot_path] = net
        return net


if __name__ == "__main__":
    # ponytail self-check: run via `python token_league.py` from src/. No
    # real game simulation needed here -- this module's own logic (sampling
    # distribution, snapshot eviction, cache invalidation) is what's under
    # test, not the token architecture (already covered by token_deck.py/
    # token_arch.py's own self-checks).
    import random
    import shutil
    import tempfile

    from token_arch import SetTransformer

    tmp_dir = tempfile.mkdtemp()
    try:
        deck_names = ["deck_a", "deck_b"]
        pool = LeaguePool(tmp_dir, deck_names, max_snapshots_per_deck=3)
        assert pool.snapshots == {"deck_a": [], "deck_b": []}

        rng = random.Random(0)
        # With no snapshots yet, every sample must be (some deck, None) -- only "live" is available.
        for _ in range(20):
            name, path = pool.sample_opponent("deck_a", rng)
            assert name in deck_names
            assert path is None

        shared = SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)
        fake_net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)

        # Register more snapshots than the cap -- oldest must be evicted, cache invalidated with it.
        for _ in range(5):
            pool.register_snapshot("deck_a", fake_net)
        assert len(pool.snapshots["deck_a"]) == 3, "eviction window must cap at max_snapshots_per_deck"
        remaining_ids = [i for i, _p in pool.snapshots["deck_a"]]
        assert remaining_ids == [2, 3, 4], f"oldest snapshots (0, 1) must be evicted first, got ids {remaining_ids}"
        for i in (0, 1):
            assert not os.path.exists(os.path.join(tmp_dir, "deck_a", f"snapshot_{i}.pt")), "evicted snapshot file must be deleted from disk"

        # Sampling now must be able to return real snapshot paths for deck_a.
        saw_snapshot = False
        for _ in range(200):
            name, path = pool.sample_opponent("deck_a", rng)
            if name == "deck_a" and path is not None:
                saw_snapshot = True
                assert os.path.exists(path)
        assert saw_snapshot, "expected at least one snapshot sample across 200 draws once deck_a has 3 snapshots"

        # load_snapshot_net must actually load real weights and be cached (same object on repeat calls).
        deck_ctx = (None, [("Pass", None, None)] * 4, ())
        _sid, path = pool.snapshots["deck_a"][0]
        loaded = pool.load_snapshot_net(path, shared, deck_ctx)
        loaded_again = pool.load_snapshot_net(path, shared, deck_ctx)
        assert loaded is loaded_again, "repeat loads of the same snapshot must hit the cache, not reconstruct"
        assert all(not p.requires_grad for p in loaded.parameters()), "a snapshot net must be fully frozen"

        # A freshly-registered snapshot that evicts `path` must invalidate its cache entry.
        for _ in range(1):
            pool.register_snapshot("deck_a", fake_net)
        assert path not in pool._net_cache, "evicting a snapshot must drop it from the load cache too"

        # A fresh LeaguePool pointed at the same root_dir must rediscover existing snapshots from disk.
        pool2 = LeaguePool(tmp_dir, deck_names, max_snapshots_per_deck=3)
        assert pool2.snapshots["deck_a"] == pool.snapshots["deck_a"], "snapshot discovery from disk must match what's actually there"
    finally:
        shutil.rmtree(tmp_dir)

    print("token_league.py self-check: OK (sampling, eviction, disk rediscovery, cache invalidation)")
