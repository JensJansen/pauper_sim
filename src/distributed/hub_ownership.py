"""Deck ownership: a static, git-tracked pool of eligible decks, plus
dynamic, hub-runtime-only state tracking who currently holds each deck, at
what progress, and how recently they've been heard from. See
DISTRIBUTED_TRAINING_PLAN.md's "Node lifecycle" (registration/dynamic
tokens) and "Elastic/bounded drift" for the design this implements.

Ownership operations (claim_deck/record_progress/the drift computation) are
serialized with an explicit `threading.Lock`, independent of whatever
threading model `hub_server.py` ends up running under -- correctness here
should never depend on remembering a server-config detail.

Stdlib-only, no torch import.
"""
import json
import os
import threading

from . import auth


def load_deck_pool(path):
    """The static list of deck names eligible for this distributed run --
    just a JSON array, checked into git. No node identity in it: identity
    is assigned dynamically at registration, not pre-declared here (see
    module docstring)."""
    with open(path) as f:
        return list(json.load(f))


class HubRuntimeState:
    """claims: deck -> {"token": str, "last_cumulative_games": int,
    "last_sync_ts": float, "node_id": str | None, "stop_requested": bool,
    "metrics_source": str | None, "metrics_lines": int} -- the last two are
    the metrics de-dup ack, see accept_metrics.
    seed_progress: deck -> the starting cumulative_games a never-claimed deck
    reports at registration (written by seed_hub; a deck that already has a
    claim uses that claim's own, fresher value).
    node_decks: node_id -> the deck that node most recently claimed, so a
    restarted node (same persistent node_id) gets its own deck back instead
    of the next free one -- see assign_deck. Persisted to `path` (atomic temp-then-replace
    write, same pattern rl.checkpoint._save_with_retry uses for a different
    reason) so a hub process restart doesn't silently invalidate every
    node's token and force a mass re-registration -- a real, foreseeable
    event, not an edge case worth ignoring.

    `now`/timestamps are always passed in by the caller, never read from
    the system clock internally -- keeps every function here testable with
    injected values, no real clock needed (matching this repo's existing
    testing convention for time-dependent logic)."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self.claims = {}
        self.node_decks = {}
        self.seed_progress = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
            # Older files were the bare claims dict (no node ids yet).
            if "claims" in data and "node_decks" in data:
                self.claims, self.node_decks = data["claims"], data["node_decks"]
                self.seed_progress = data.get("seed_progress", {})
            else:
                self.claims, self.node_decks, self.seed_progress = data, {}, {}
        else:
            self.claims, self.node_decks, self.seed_progress = {}, {}, {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        tmp = f"{self._path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"claims": self.claims, "node_decks": self.node_decks, "seed_progress": self.seed_progress}, f)
        os.replace(tmp, self._path)


def claim_deck(state, deck, pool, now, node_id=None):
    """Mints a fresh token for `deck` and records the claim, overwriting
    (and thereby invalidating) any previous claim's token -- a stale writer
    fails its next verify_claim immediately, not just eventually. Preserves
    the deck's last-known progress across a reclaim (a replacement node
    inherits the deck's history, doesn't reset to 0 -- see
    DISTRIBUTED_TRAINING_PLAN.md gap #2's resolution). Raises if `deck`
    isn't in the static pool. `node_id` (optional, the registering node's
    persistent ID) is recorded on the claim and in node_decks; any other
    node's mapping to this deck is dropped, since the deck is now this
    node's."""
    if deck not in pool:
        raise ValueError(f"{deck!r} is not in the ownership pool {sorted(pool)}")
    token = auth.mint_deck_token()
    with state._lock:
        prior = state.claims.get(deck, {})
        prior_progress = prior.get("last_cumulative_games", state.seed_progress.get(deck, 0))
        # The metrics ack carries over a reclaim: it describes what the hub's
        # metrics.jsonl already holds, which a new token doesn't change.
        state.claims[deck] = {"token": token, "last_cumulative_games": prior_progress, "last_sync_ts": now,
                               "stop_requested": False, "node_id": node_id,
                               "metrics_source": prior.get("metrics_source"),
                               "metrics_lines": prior.get("metrics_lines", 0)}
        for other in [n for n, d in state.node_decks.items() if d == deck and n != node_id]:
            del state.node_decks[other]
        if node_id is not None:
            state.node_decks[node_id] = deck
        state._save()
    return token


def pick_available_deck(state, pool, now, liveness_timeout):
    """First deck in `pool` (in pool order -- deterministic, not random)
    that's either never been claimed or whose current claimant has gone
    quiet past liveness_timeout. Covers both "genuinely new deck" and "the
    previous holder crashed" with the same check: a deck not in
    active_decks is available either way. None if every deck in the pool
    currently has a live claimant."""
    active = active_decks(state, now, liveness_timeout)
    for deck in pool:
        if deck not in active:
            return deck
    return None


def assign_deck(state, node_id, pool, now, liveness_timeout):
    """The deck a registering node should get: its own previous deck if
    `node_id` is known and nobody else has claimed that deck since (even if
    its old claim is still live -- a restarted node supersedes its own prior
    claim via claim_deck's token replacement), otherwise the next free deck
    (pick_available_deck). A stale mapping (deck since given to a different
    node, or no longer in the pool) is dropped. None if the pool is full."""
    if node_id is not None:
        with state._lock:
            deck = state.node_decks.get(node_id)
            claim = state.claims.get(deck) if deck is not None else None
            if deck is not None and deck in pool and (claim is None or claim.get("node_id") == node_id):
                return deck
            if deck is not None:
                del state.node_decks[node_id]
                state._save()
    return pick_available_deck(state, pool, now, liveness_timeout)


def verify_claim(state, deck, token):
    """True iff `token` is the CURRENTLY valid token for `deck` -- a token
    from a superseded claim (see claim_deck) always returns False, never a
    stale-but-still-honored credential."""
    with state._lock:
        claim = state.claims.get(deck)
        # a released claim has token None (see release_claim) and must never match
        return claim is not None and claim["token"] is not None and claim["token"] == token


def record_progress(state, deck, cumulative_games, now):
    """Updates deck's last-known progress and last-sync timestamp -- called
    on every push. Raises if deck has no active claim (a push should never
    reach here without a claim; verify_claim is checked first by the
    caller)."""
    with state._lock:
        if deck not in state.claims:
            raise ValueError(f"{deck!r} has no active claim to record progress against")
        state.claims[deck]["last_cumulative_games"] = cumulative_games
        state.claims[deck]["last_sync_ts"] = now
        state._save()


def seed_hub_state(state, deck_games):
    """Records each deck's starting cumulative_games and wipes every claim and
    node mapping (a reseed invalidates all tokens). Used by seed_hub."""
    with state._lock:
        state.claims.clear()
        state.node_decks.clear()
        state.seed_progress = dict(deck_games)
        state._save()


def unseeded_decks(state, league_dir, deck_pool):
    """Pool decks that have trained weights on the hub but no recorded
    progress (never claimed, never seeded) -- they'd register at 0 games and
    restart their schedules. Used for a startup warning."""
    return [d for d in deck_pool
            if os.path.exists(os.path.join(league_dir, d, "live.pt"))
            and d not in state.claims and d not in state.seed_progress]


def metrics_acked(state, deck, source):
    """How many of `source`'s local metrics.jsonl lines the hub already holds
    for `deck` -- 0 if the hub has only seen a different source (a different
    node ID, i.e. a different disk) or nothing. Handed to a (re)registering
    node so it resumes pushing from the right line."""
    with state._lock:
        claim = state.claims.get(deck, {})
        return claim.get("metrics_lines", 0) if claim.get("metrics_source") == source else 0


def accept_metrics(state, deck, source, start_line, lines, rebase=False):
    """De-duplicates one push's metrics batch. `start_line` is the index, in
    the pushing node's own local metrics.jsonl, of lines[0]; `source` is that
    node's persistent node ID (one local file per node ID/disk). The hub keeps
    per deck how many of that source's lines it has accepted, so a replayed
    batch (restart, or a push whose ack was lost) is skipped instead of
    appended twice. Returns (lines_to_append, new_count); the caller appends
    them and then calls record_metrics_ack.

      - different source, a gap (start_line beyond what the hub holds), or
        rebase=True (node found its local file shorter than the hub's count):
        no overlap can be established, so everything is accepted and the
        count re-anchored to start_line + len(lines).
      - same source, overlap: only the lines past the hub's count are new."""
    with state._lock:
        claim = state.claims.get(deck, {})
        have = claim.get("metrics_lines", 0) if claim.get("metrics_source") == source else None
    end = start_line + len(lines)
    if have is None or rebase or start_line > have:
        return list(lines), end
    skip = min(have - start_line, len(lines))
    return list(lines[skip:]), max(have, end)


def record_metrics_ack(state, deck, source, count):
    with state._lock:
        state.claims[deck]["metrics_source"] = source
        state.claims[deck]["metrics_lines"] = count
        state._save()


def active_decks(state, now, liveness_timeout):
    """Decks whose last sync was within liveness_timeout of `now`. A deck
    that's gone quiet longer than this is excluded -- the mechanism that
    keeps a crashed node from becoming a permanent anchor every other node
    waits on forever (see DISTRIBUTED_TRAINING_PLAN.md's "Elastic/bounded
    drift")."""
    with state._lock:
        return {deck for deck, c in state.claims.items() if now - c["last_sync_ts"] <= liveness_timeout}


def slowest_active_progress(state, now, liveness_timeout):
    """The minimum last_cumulative_games among active decks, or None if
    none are active yet (e.g. a brand-new league, nobody has synced)."""
    active = active_decks(state, now, liveness_timeout)
    if not active:
        return None
    with state._lock:
        return min(state.claims[d]["last_cumulative_games"] for d in active)


def drift_signal(state, deck, now, liveness_timeout, threshold):
    """"continue" or "pause" for `deck`, per the elastic/bounded-drift
    design: paused once this deck's own progress is more than `threshold`
    games ahead of the slowest currently-active deck. `threshold` is a
    parameter, not a hardcoded constant -- its real value is
    DISTRIBUTED_TRAINING_PLAN.md gap #15, benchmarked in Phase 3, not
    guessed here. Never pauses a deck that isn't itself active (a deck that
    just went quiet shouldn't also be told to pause -- it's already not
    making progress), and never pauses when there's no other active deck to
    be ahead of."""
    with state._lock:
        claim = state.claims.get(deck)
    if claim is None:
        raise ValueError(f"{deck!r} has no active claim")
    if deck not in active_decks(state, now, liveness_timeout):
        return "continue"
    slowest = slowest_active_progress(state, now, liveness_timeout)
    if slowest is None:
        return "continue"
    if claim["last_cumulative_games"] - slowest > threshold:
        return "pause"
    return "continue"


def request_stop(state, deck):
    """Marks deck's CURRENT claim for a graceful stop -- its next push/poll
    gets "stop" (see sync_signal) instead of "continue"/"pause", regardless
    of drift. Operators reach this through hub_server's POST /admin/stop
    (`python -m distributed.admin stop`). The node exits after its final push
    and calls release_claim (POST /release), which clears the flag; a fresh
    claim_deck also always starts un-stopped. Returns False if the deck has
    no claim."""
    with state._lock:
        if deck not in state.claims:
            return False
        state.claims[deck]["stop_requested"] = True
        state._save()
        return True


def request_stop_all(state):
    """request_stop for every currently-claimed deck at once -- the
    full-batch-teardown case. Returns how many decks were marked."""
    with state._lock:
        for claim in state.claims.values():
            claim["stop_requested"] = True
        state._save()
        return len(state.claims)


def clear_stop(state, deck=None):
    """Cancels a pending stop request (one deck, or every deck if `deck` is
    None) before the node has acted on it. Returns how many were cleared."""
    with state._lock:
        targets = [deck] if deck is not None else list(state.claims)
        cleared = 0
        for d in targets:
            if d in state.claims and state.claims[d].get("stop_requested"):
                state.claims[d]["stop_requested"] = False
                cleared += 1
        state._save()
        return cleared


def release_claim(state, deck):
    """A node's graceful exit (after a stop signal or SIGTERM): invalidates
    its token, clears any stop request and marks the deck immediately
    available (last_sync_ts=0, so it drops out of active_decks and
    pick_available_deck can hand it out without waiting for the liveness
    timeout). Progress, the node_id mapping and the metrics ack are KEPT, so
    the same node (or a replacement) resumes from the deck's real state."""
    with state._lock:
        claim = state.claims.get(deck)
        if claim is not None:
            claim["token"] = None
            claim["stop_requested"] = False
            claim["last_sync_ts"] = 0.0
            state._save()


def status(state, pool, now, liveness_timeout, threshold):
    """One row per pool deck for `distributed.admin status`: state is one of
    unclaimed / active / quiet (claimed but not heard from within
    liveness_timeout) / stopping (stop requested, not yet released) /
    released."""
    with state._lock:
        claims = {d: dict(c) for d, c in state.claims.items()}
        seeds = dict(state.seed_progress)
    rows = []
    for deck in pool:
        claim = claims.get(deck)
        if claim is None:
            rows.append({"deck": deck, "state": "unclaimed", "node_id": None,
                         "cumulative_games": seeds.get(deck, 0), "last_sync_age_s": None, "signal": None})
            continue
        age = now - claim["last_sync_ts"]
        if claim["token"] is None:
            st = "released"
        elif claim.get("stop_requested"):
            st = "stopping"
        elif age <= liveness_timeout:
            st = "active"
        else:
            st = "quiet"
        rows.append({"deck": deck, "state": st, "node_id": claim.get("node_id"),
                     "cumulative_games": claim["last_cumulative_games"],
                     "last_sync_age_s": None if claim["last_sync_ts"] == 0.0 else round(age, 1),
                     "signal": sync_signal(state, deck, now, liveness_timeout, threshold)
                     if st in ("active", "stopping") else None})
    return rows


def sync_signal(state, deck, now, liveness_timeout, threshold):
    """The signal a push response actually carries: "stop" if
    request_stop(_all) was called for this deck since its last claim,
    otherwise drift_signal's "continue"/"pause". Kept separate from
    drift_signal itself so that function stays pure drift logic, easy to
    test in isolation."""
    with state._lock:
        claim = state.claims.get(deck)
    if claim is not None and claim.get("stop_requested"):
        return "stop"
    return drift_signal(state, deck, now, liveness_timeout, threshold)
