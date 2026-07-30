"""Opponent pool for league-style training (in place of the old pairwise
Stage 1/Stage 2 curriculum) -- per confirmed design: historical checkpoint
snapshots per deck (not latest-only, so a deck can't quietly drift away
from skills it needed against an opponent's earlier self).

Sampling is two-level: pick a deck uniformly from the whole roster
(including the training deck itself, for mirror play), THEN decide live vs.
checkpoint for that deck via `checkpoint_rate` (a live/checkpoint coin flip
with a caller-set probability, default 0.0 -- always live), and if a
checkpoint is drawn, pick uniformly among that deck's currently-held
snapshots. This replaced an earlier "uniform among {live} union {every
snapshot}" scheme: that scheme let the live-net probability silently shrink
as the snapshot window filled (1/(N+1), so 80% checkpoint odds once a deck
had 4 snapshots) -- nobody chose that ratio, it just fell out of window
size. checkpoint_rate makes the live/checkpoint split an explicit, stable
number instead of a side effect of how many snapshots happen to exist.
Owner directive (2026-07-30): early training should see NO checkpoint
opponents at all (checkpoint_rate=0.0, the default) -- an early snapshot is
barely-trained and teaches little as an opponent, so paying collection cost
against one is close to wasted relative to a live opponent that's ALSO
improving. Reintroducing checkpoint diversity later is an explicit,
deliberate choice (a nonzero rate), not an automatic side effect of the pool
filling up."""

import os

import torch

from rl.deck import DeckNetwork
from rl.agent import AlwaysKeep, SeatAgent
from rl.mulligan import MulliganNet

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

    def register_snapshot(self, deck_name, net, mulligan_net=None):
        """Freezes net's CURRENT weights as a new historical opponent for
        deck_name, evicting the oldest snapshot once the window is full. A
        snapshot is the whole frozen AGENT: the DeckNetwork PLUS the paired
        MulliganNet (when given), so a historical opponent plays with its own
        era-matched pregame policy rather than borrowing whatever the current
        mulligan net happens to be. Saves trunk_hidden alongside the state_dict
        (derived from the net's own trunk_layers, not assumed) -- load has no
        other way to know what shape to reconstruct. mulligan_net=None writes an
        old-style deck-only snapshot; load_snapshot_agent then falls back to
        AlwaysKeep for it (keeps pre-refactor snapshot files loadable)."""
        deck_dir = os.path.join(self.root_dir, deck_name)
        os.makedirs(deck_dir, exist_ok=True)
        next_id = (self.snapshots[deck_name][-1][0] + 1) if self.snapshots[deck_name] else 0
        path = os.path.join(deck_dir, f"snapshot_{next_id}.pt")
        trunk_hidden = tuple(layer.out_features for layer in net.trunk_layers)
        saved = {"state_dict": net.state_dict(), "trunk_hidden": trunk_hidden}
        if mulligan_net is not None:
            saved["mulligan_state_dict"] = mulligan_net.state_dict()
            saved["mulligan_hidden"] = mulligan_net.trunk[0].out_features  # restore the exact hidden width on load
        torch.save(saved, path)
        self.snapshots[deck_name].append((next_id, path))
        while len(self.snapshots[deck_name]) > self.max_snapshots_per_deck:
            _evicted_id, evicted_path = self.snapshots[deck_name].pop(0)
            os.remove(evicted_path)
            self._net_cache.pop(evicted_path, None)

    def sample_opponent(self, training_deck_name, rng, checkpoint_rate=0.0):
        """Returns (opponent_deck_name, snapshot_path_or_None). None means
        "use that deck's current live net" (the caller already holds every
        deck's live net; when opponent_deck_name == training_deck_name AND
        snapshot_path is None, that IS the training net itself -- true
        mirror, not a frozen copy of it).

        checkpoint_rate: the live/checkpoint split for whichever deck gets
        picked, decided independently of how many snapshots that deck
        happens to have banked (see this module's own docstring for why --
        the old "uniform over {live} union {snapshots}" scheme let this
        ratio drift with window occupancy instead of being a deliberate
        choice). 0.0 (default): always live, structurally -- the rng.random()
        draw and comparison never happen, so this is exact, not "very
        unlikely," even before any snapshots exist. A deck with an EMPTY
        snapshot list always resolves to live regardless of checkpoint_rate
        (nothing to draw yet), same fallback the old code's [None]-first
        candidate list gave for free."""
        opponent_deck_name = rng.choice(self.deck_names)
        snaps = self.snapshots[opponent_deck_name]
        if snaps and rng.random() < checkpoint_rate:
            return opponent_deck_name, rng.choice([path for _id, path in snaps])
        return opponent_deck_name, None

    def load_snapshot_agent(self, snapshot_path, shared_stack, deck_ctx):
        """Builds (or returns a cached) frozen SeatAgent for a historical
        snapshot -- the DeckNetwork PLUS its era-matched pregame decider (a
        frozen MulliganNet if the snapshot carries one, else AlwaysKeep for a
        pre-refactor deck-only snapshot). Never trained itself, only ever a
        rollout opponent (collect_rollout's own inference-mode forward covers
        inference; requires_grad=False here is defensive, matching the frozen
        shared stack's convention). Cached by path so repeat samples reuse it."""
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

        if "mulligan_state_dict" in saved:
            mull = MulliganNet(shared_stack, hidden=saved.get("mulligan_hidden", 64))
            mull.load_state_dict(saved["mulligan_state_dict"])
            mull.eval()
            for p in mull.parameters():
                p.requires_grad = False
        else:
            mull = AlwaysKeep()  # pre-refactor deck-only snapshot -> neutral pregame

        agent = SeatAgent(net, mull, deck_ctx)
        self._net_cache[snapshot_path] = agent
        return agent


if __name__ == "__main__":
    # ponytail self-check: run via `python -m rl.league` from src/. No
    # real game simulation needed here -- this module's own logic (sampling
    # distribution, snapshot eviction, cache invalidation) is what's under
    # test, not the token architecture (already covered by rl.deck/
    # rl.arch's own self-checks).
    import random
    import shutil
    import tempfile

    from rl.arch import SetTransformer

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
        fake_mull = MulliganNet(shared, hidden=8)  # snapshots are whole agents now (deck + mulligan)

        # Register more snapshots than the cap -- oldest must be evicted, cache invalidated with it.
        for _ in range(5):
            pool.register_snapshot("deck_a", fake_net, fake_mull)
        assert len(pool.snapshots["deck_a"]) == 3, "eviction window must cap at max_snapshots_per_deck"
        remaining_ids = [i for i, _p in pool.snapshots["deck_a"]]
        assert remaining_ids == [2, 3, 4], f"oldest snapshots (0, 1) must be evicted first, got ids {remaining_ids}"
        for i in (0, 1):
            assert not os.path.exists(os.path.join(tmp_dir, "deck_a", f"snapshot_{i}.pt")), "evicted snapshot file must be deleted from disk"

        # DEFAULT checkpoint_rate (0.0): must stay live-only EVEN THOUGH deck_a
        # now has 3 real snapshots on disk -- the whole point of this change
        # (see this module's own docstring) is that a full snapshot window no
        # longer silently implies mostly-checkpoint sampling.
        for _ in range(200):
            name, path = pool.sample_opponent("deck_a", rng)
            if name == "deck_a":
                assert path is None, "checkpoint_rate defaults to 0.0 -- must never draw a snapshot"

        # checkpoint_rate=1.0: must always return a real snapshot path when
        # the sampled deck actually has any.
        saw_snapshot = False
        for _ in range(200):
            name, path = pool.sample_opponent("deck_a", rng, checkpoint_rate=1.0)
            if name == "deck_a":
                saw_snapshot = True
                assert path is not None and os.path.exists(path), "checkpoint_rate=1.0 must always draw a snapshot when one exists"
            else:
                assert path is None, "deck_b has zero snapshots -- must fall back to live regardless of checkpoint_rate"
        assert saw_snapshot, "expected at least one deck_a draw across 200 samples"

        # checkpoint_rate=0.5: the live/checkpoint split must roughly track
        # the requested rate, independent of how many snapshots exist (3
        # here) -- confirms the split is no longer 1/(1+N) as it used to be.
        # Filtered to deck_a draws only: deck_b has zero snapshots and would
        # always read as "live" regardless of rate, diluting the measured
        # fraction if mixed in (opponent-deck selection is a SEPARATE uniform
        # draw from the live/checkpoint decision -- see this function's own
        # docstring on the two-level structure).
        deck_a_draws = [p for n, p in (pool.sample_opponent("deck_a", rng, checkpoint_rate=0.5) for _ in range(4000)) if n == "deck_a"]
        checkpoint_frac = sum(1 for p in deck_a_draws if p is not None) / len(deck_a_draws)
        assert abs(checkpoint_frac - 0.5) < 0.05, f"checkpoint_rate=0.5 should draw a snapshot ~50% of the time deck_a is picked, got {checkpoint_frac:.3f}"

        # load_snapshot_agent must load a frozen SeatAgent (deck + mulligan) and be cached.
        deck_ctx = (None, [("Pass", None, None)] * 4, ())
        _sid, path = pool.snapshots["deck_a"][0]
        loaded = pool.load_snapshot_agent(path, shared, deck_ctx)
        loaded_again = pool.load_snapshot_agent(path, shared, deck_ctx)
        assert loaded is loaded_again, "repeat loads of the same snapshot must hit the cache, not reconstruct"
        assert isinstance(loaded, SeatAgent), "a loaded snapshot is a frozen SeatAgent"
        assert isinstance(loaded.mulligan, MulliganNet), "a snapshot saved WITH a mulligan net must round-trip it"
        assert all(not p.requires_grad for p in loaded.main.parameters()), "a snapshot's deck net must be fully frozen"
        assert all(not p.requires_grad for p in loaded.mulligan.parameters()), "a snapshot's mulligan net must be fully frozen"

        # A deck-only (pre-refactor) snapshot must still load, falling back to AlwaysKeep.
        legacy_path = os.path.join(tmp_dir, "deck_b", "snapshot_0.pt")
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        torch.save({"state_dict": fake_net.state_dict(),
                    "trunk_hidden": tuple(l.out_features for l in fake_net.trunk_layers)}, legacy_path)
        legacy = pool.load_snapshot_agent(legacy_path, shared, deck_ctx)
        assert isinstance(legacy, SeatAgent) and isinstance(legacy.mulligan, AlwaysKeep), \
            "a deck-only snapshot must load with an AlwaysKeep pregame decider"

        # A freshly-registered snapshot that evicts `path` must invalidate its cache entry.
        for _ in range(1):
            pool.register_snapshot("deck_a", fake_net, fake_mull)
        assert path not in pool._net_cache, "evicting a snapshot must drop it from the load cache too"

        # A fresh LeaguePool pointed at the same root_dir must rediscover existing snapshots from disk.
        pool2 = LeaguePool(tmp_dir, deck_names, max_snapshots_per_deck=3)
        assert pool2.snapshots["deck_a"] == pool.snapshots["deck_a"], "snapshot discovery from disk must match what's actually there"
    finally:
        shutil.rmtree(tmp_dir)

    print("rl.league self-check: OK (sampling, eviction, disk rediscovery, cache invalidation)")
