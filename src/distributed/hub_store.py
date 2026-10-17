"""Canonical on-hub `checkpoints/<league>/` read/write. Reuses the existing
on-disk schema (`live.pt`/`mulligan.pt`/`snapshot_N.pt`/`opponent_stats.json`,
plus the league-level `metrics.jsonl`) unmodified -- no new on-disk format
invented. `archive/` and `checks/` are never read or written here, per
DISTRIBUTED_TRAINING_PLAN.md's explicit exclusion (archive/ is
retrospective-only and unbounded; checks/ is validation output never read
by training).

Two module-level concerns kept deliberately separate:
  - push_deck_files: per-deck file writes, temp-then-replace-all-at-once
    (DISTRIBUTED_TRAINING_PLAN.md gap #6's resolution -- not a true
    multi-file transaction, see its own docstring for the honest limit of
    what that buys).
  - metrics.jsonl appends: the one resource every deck's push shares, so
    it gets its own lock independent of per-deck file writes, which never
    collide with each other (only one claim is ever valid per deck at a
    time, enforced by hub_ownership.verify_claim before a push reaches
    here at all).
"""
import os
import threading

_METRICS_LOCK = threading.Lock()
_DECK_LOCKS = {}
_DECK_LOCKS_GUARD = threading.Lock()


def _deck_lock(deck):
    """One lock per deck: held while a push swaps/prunes that deck's files and
    while a pull reads them, so a concurrent pull (the hub is threaded) never
    sees e.g. a new live.pt next to the old snapshot set. Held only for the
    fast swap/read, never for the slow part (receiving the upload, serving the
    response)."""
    with _DECK_LOCKS_GUARD:
        return _DECK_LOCKS.setdefault(deck, threading.Lock())

_DECK_FILENAME_PREFIXES = ("snapshot_",)
_DECK_FILENAMES = ("live.pt", "mulligan.pt", "opponent_stats.json")


def _is_deck_file(filename):
    return filename in _DECK_FILENAMES or (
        filename.startswith(_DECK_FILENAME_PREFIXES) and filename.endswith(".pt"))


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def push_deck_files(league_dir, deck, files, metrics_lines, snapshot_names=None):
    """Writes every file in `files` (filename -> raw bytes) under
    league_dir/deck/, then appends `metrics_lines` (already-formatted JSON
    text lines, one per record, no trailing newline required) to the
    league's metrics.jsonl.

    Every file is written to a `.tmp` sibling first; only once *all* writes
    have succeeded does it swap every temp file into place via
    `os.replace`. If any single write fails, every temp file written so far
    for this call is cleaned up and nothing already on disk is touched --
    a connection drop mid-push leaves the hub's prior state intact rather
    than half-updated. This is NOT a true cross-file transaction (a crash
    between two of the `os.replace` calls in the final swap loop could
    still leave the batch partially applied) -- see
    DISTRIBUTED_TRAINING_PLAN.md gap #6 for why that residual gap is
    accepted rather than solved with a transaction log neither this scope
    nor Windows calls for.

    snapshot_names (optional): the complete list of snapshot_*.pt filenames
    the pushing node currently holds for this deck. When given, every OTHER
    snapshot_*.pt in the hub's copy of the deck is deleted -- the node
    evicted it (LeaguePool.register_snapshot's window cap) and the hub must
    mirror that, or the hub tree and every node's opponent pool for this
    deck would grow past the cap forever. Deleted, not archived (owner
    decision 2026-10-17). Runs only after every file swap succeeded, and
    never deletes a snapshot that is part of this same push. None (an older
    client / a partial push) prunes nothing."""
    deck_dir = os.path.join(league_dir, deck)
    os.makedirs(deck_dir, exist_ok=True)
    tmp_pairs = []
    try:
        for name, content in files.items():
            final_path = os.path.join(deck_dir, name)
            tmp_path = f"{final_path}.tmp"
            with open(tmp_path, "wb") as f:
                f.write(content)
            tmp_pairs.append((tmp_path, final_path))
    except Exception:
        for tmp_path, _final in tmp_pairs:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    with _deck_lock(deck):
        for tmp_path, final_path in tmp_pairs:
            os.replace(tmp_path, final_path)
        if snapshot_names is not None:
            keep = set(snapshot_names) | set(files)
            for fn in os.listdir(deck_dir):
                if fn.startswith("snapshot_") and fn.endswith(".pt") and fn not in keep:
                    os.remove(os.path.join(deck_dir, fn))
    if metrics_lines:
        append_metrics(league_dir, metrics_lines)


def append_metrics(league_dir, lines):
    """Appends `lines` (already-formatted JSON text, one record per line)
    to checkpoints/<league>/metrics.jsonl. A plain append is correct here
    because every metrics.jsonl record is already self-describing (carries
    its own deck/kind/cumulative_games -- see report_metrics.py's own
    docstring) -- merging records from many nodes plus the hub's own
    validation-check output needs no special interleaving logic, just
    getting every line into the same file safely under concurrent
    writers."""
    os.makedirs(league_dir, exist_ok=True)
    path = os.path.join(league_dir, "metrics.jsonl")
    with _METRICS_LOCK:
        with open(path, "a") as f:
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")


def pull_full_roster(league_dir, deck_pool):
    """Every deck in `deck_pool`'s current live.pt/mulligan.pt/active
    snapshots/opponent_stats.json, as {deck: {filename: bytes}}. Iterates
    the *given* deck names, never a bare directory listing of league_dir --
    that directory also holds non-deck entries (metrics.jsonl, progress.json,
    session.txt, a league-level checks/ dir) that a naive listdir would
    otherwise misread as deck names. A deck with no directory yet (never
    seeded) comes back with an empty file dict."""
    roster = {}
    for deck in deck_pool:
        deck_dir = os.path.join(league_dir, deck)
        files = {}
        if os.path.isdir(deck_dir):
            with _deck_lock(deck):
                for fn in os.listdir(deck_dir):
                    full = os.path.join(deck_dir, fn)
                    if os.path.isfile(full) and _is_deck_file(fn):
                        files[fn] = _read_bytes(full)
        roster[deck] = files
    return roster


def seed_fresh_deck(league_dir, deck, vocab, fixed_table, trunk_hidden=None):
    """The hub-side half of gap #2's resolution: on a genuinely fresh
    league (no live.pt anywhere for `deck`), writes ONE canonical
    random-initialized live.pt + mulligan.pt, so every node's first pull
    gets the identical starting point instead of each node independently
    building its own differently-random net (and mulligan net -- the same
    divergence risk applies there too, since `_run_session` builds a fresh
    MulliganNet per deck exactly the same way it builds the main net, with
    the same no-op-if-missing checkpoint load).

    A no-op if live.pt already exists for this deck -- never overwrites
    real trained progress with a fresh random seed, even if called twice
    by mistake."""
    deck_dir = os.path.join(league_dir, deck)
    if os.path.exists(os.path.join(deck_dir, "live.pt")):
        return
    from rl import checkpoint as ckpt_io
    from rl.league.league_runner import TRUNK_HIDDEN, build_deck_net
    from rl.model.mulligan import MulliganNet
    net = build_deck_net(vocab.size, len(fixed_table), trunk_hidden or TRUNK_HIDDEN)
    mnet = MulliganNet(net.encoder)
    os.makedirs(deck_dir, exist_ok=True)
    ckpt_io.save_deck_checkpoint(os.path.join(deck_dir, "live.pt"), net)
    ckpt_io.save_deck_checkpoint(os.path.join(deck_dir, "mulligan.pt"), mnet)
