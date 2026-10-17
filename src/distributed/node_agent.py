"""CLI entrypoint for one distributed-training node. Registers with the
hub, then drives its own loop around the *existing, unmodified*
_run_session -- never _train_to. DISTRIBUTED_TRAINING_PLAN.md's "Node
lifecycle" explains why: _train_to's cadence is validation-oriented
(~10,000 games/deck for main_league.json's real settings) and it has no
hook to stop early; this loop owns both call-sizing and stop-checking
directly instead, at whatever granularity the caller chooses.

Reuses run_training_pipeline.py's own config parsing (_resolve_options) so
a node reads the exact same training config file as local training, byte
for byte -- consistent with this repo's read-only config decision (the
operator manages per-node correctness by hand; this code never writes to
that file).
"""
import argparse
import json
import os
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

from config_loader import load_config
from repo_paths import CHECKPOINTS_DIR, REPO_ROOT
from rl.league.league_runner import _load_progress, _run_session, advance_progress
from rl.roster import VOCAB_PATH, build_pool
from run_league import _resolve_device
from run_training_pipeline import _resolve_options

from . import node_sync, version_gate

# DEFAULT_CALL_GRANULARITY_ITERATIONS: one hub sync (push + full-roster pull)
# happens per _run_session call, so this is also the sync interval.
# benchmarking/distributed_call_granularity.py MEASURED _run_session's per-call
# setup overhead at ~47.5ms (DISTRIBUTED_TRAINING_PLAN.md gap #14), so setup
# cost doesn't constrain the choice; bandwidth does. At 1 iteration a node
# moved ~47MB up + ~700MB down per iteration, mostly unchanged files. Owner
# decision 2026-10-17 (gap #16): 3 iterations = main_league.json's
# snapshot_every (200 games / 60 per iteration), so every sync carries about
# one new snapshot -- syncing more often only re-sends unchanged ones. Cuts
# traffic 3x; staleness / unsynced-loss window / stop latency are all ~180
# games. A placeholder-free but unmeasured-against-real-nodes value; revisit
# with real iteration times (and delta sync, which would shrink the cost of a
# sync rather than just making them rarer).
#
# DEFAULT_PAUSE_POLL_INTERVAL_SECONDS is still a placeholder pending
# DISTRIBUTED_TRAINING_PLAN.md gap #15 (needs a real second machine to
# benchmark against, unlike gap #14 which only needed this one).
DEFAULT_CALL_GRANULARITY_ITERATIONS = 3
DEFAULT_PAUSE_POLL_INTERVAL_SECONDS = 7
DEFAULT_NODE_ID_PATH = "/var/lib/pauper-sim/node_id"  # outside the git checkout; systemd StateDirectory=pauper-sim
# After this many CONSECUTIVE iterations whose push/pull failed (hub
# unreachable even after node_sync's own retries), stop training and wait for
# the hub (poll) instead of running further ahead with nothing reaching it --
# owner decision 2026-10-17. An unmeasured placeholder like the other
# timing knobs (gap #15).
DEFAULT_MAX_FAILED_SYNCS = 5
REGISTER_RETRY_BASE_SECONDS = 5
REGISTER_RETRY_MAX_SECONDS = 60
# A version rejection (426) needs the operator (approve the SHA / fix the node),
# not a quick retry; wait long between attempts and keep saying so loudly.
VERSION_RETRY_SECONDS = 300

_DECK_FILENAMES = ("live.pt", "mulligan.pt", "opponent_stats.json")


class StopFlag:
    """A mutable holder both a SIGTERM handler and the main loop touch --
    a signal handler needs a mutable cell to set, since it can't return a
    value the loop would see."""

    def __init__(self):
        self.requested = False

    def install_sigterm_handler(self):
        def _handler(signum, frame):
            self.requested = True
        signal.signal(signal.SIGTERM, _handler)


def load_or_create_node_id(path):
    """This node's persistent ID: read from `path`, or generated and written
    there (temp + fsync + os.replace, so a crash mid-write can't leave a
    half-written ID) on first run. Called BEFORE registering, so a crash
    between registering and saving can never orphan a hub-side claim under
    an ID the node doesn't remember. Survives crashes, restarts and reboots;
    lost only if the instance's disk is -- then the node is simply new."""
    if os.path.exists(path):
        with open(path) as f:
            node_id = f.read().strip()
        if node_id:
            return node_id
    node_id = uuid.uuid4().hex
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(node_id)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return node_id


def _reconcile_vocab(local_path, hub_vocab):
    """Makes the local vocab.json equal the hub's, safely. The hub's file is
    authoritative (append-only history; see hub_server's /vocab). Absent or
    identical: write/no-op. A local file that is a strict subset of the hub's
    with matching indices (the hub only ever appends): replace it. A local file
    that assigns ANY name a different index than the hub is a conflict -- weights
    trained against it would be misread under the hub's indices, so refuse
    rather than overwrite it silently."""
    local = None
    if os.path.exists(local_path):
        with open(local_path) as f:
            local = json.load(f)
        clashes = sorted(n for n, i in local.items() if n in hub_vocab and hub_vocab[n] != i)
        if clashes:
            raise SystemExit(f"local {local_path} assigns different card indices than the hub's vocabulary "
                             f"(e.g. {clashes[:3]}); local weights trained against it would be misread. "
                             f"Remove it (and any local checkpoints) to adopt the hub's.")
        if local == hub_vocab:
            return
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    tmp = f"{local_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(hub_vocab, f, indent=2, sort_keys=True)
    os.replace(tmp, local_path)


def _sync_vocab(hub_url, credential, local_path=VOCAB_PATH):
    _reconcile_vocab(local_path, node_sync.fetch_vocab(hub_url, credential))


def _reported_version(vocab):
    """What this node tells the hub about the code it runs (see version_gate)."""
    sha, dirty = version_gate.local_version(REPO_ROOT)
    return {"reported_sha": sha, "reported_dirty": dirty, "reported_vocab_hash": version_gate.vocab_hash(vocab)}


def register_with_backoff(hub_url, credential, node_id, stop_flag, sleep_fn=time.sleep, version=None):
    """node_sync.register, retrying while the hub answers 409 ("pool fully
    claimed" -- e.g. a recreated node waiting out its predecessor's liveness
    timeout) with capped exponential backoff, so waiting doesn't depend on
    systemd crash-looping the process. A 426 (version rejected: this node's code
    isn't on the hub's accepted list) is retried every VERSION_RETRY_SECONDS with
    a loud message and the node never trains meanwhile -- the operator approves
    the SHA on the hub, or fixes/recreates the node. Any other failure (bad credential,
    unreachable hub after node_sync's own retries) propagates. Returns None
    if a stop was requested while waiting."""
    delay = REGISTER_RETRY_BASE_SECONDS
    while not stop_flag.requested:
        try:
            return node_sync.register(hub_url, credential, node_id=node_id, **(version or {}))
        except node_sync.SyncError as e:
            if e.status_code == 426:
                print(f"VERSION REJECTED by the hub: {e}. NOT training. Approve this code version on the hub "
                      f"(accepted_versions file) or fix/recreate this node; retrying in "
                      f"{VERSION_RETRY_SECONDS}s", flush=True)
                sleep_fn(VERSION_RETRY_SECONDS)
                continue
            if e.status_code != 409:
                raise
            print(f"no unclaimed deck yet ({e}); retrying registration in {delay}s", flush=True)
            sleep_fn(delay)
            delay = min(delay * 2, REGISTER_RETRY_MAX_SECONDS)
    return None


def _is_deck_file(filename):
    return filename in _DECK_FILENAMES or (filename.startswith("snapshot_") and filename.endswith(".pt"))


def _collect_deck_files(league_dir, deck):
    """This node's own deck's current checkpoint files, read straight off
    local disk -- the same files _run_session's own end-of-call
    _save_live_checkpoints just wrote."""
    deck_dir = os.path.join(league_dir, deck)
    files = {}
    if os.path.isdir(deck_dir):
        for fn in os.listdir(deck_dir):
            full = os.path.join(deck_dir, fn)
            if os.path.isfile(full) and _is_deck_file(fn):
                with open(full, "rb") as f:
                    files[fn] = f.read()
    return files


def _read_new_metrics_lines(league_dir, since_offset):
    """New COMPLETE, non-blank metrics.jsonl lines since byte offset
    `since_offset`; a partial trailing line (no newline yet) is left for the
    next call. Returns (lines, new_offset). The hub's de-dup index counts
    non-blank lines only, matching len(lines) here and the count in
    _resume_metrics_position."""
    path = os.path.join(league_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return [], since_offset
    with open(path, "rb") as f:
        f.seek(since_offset)
        chunk = f.read()
    complete = chunk[:chunk.rfind(b"\n") + 1]
    lines = [ln.rstrip(b"\r").decode("utf-8") for ln in complete.split(b"\n")[:-1]]
    return [ln for ln in lines if ln.strip()], since_offset + len(complete)


def _resume_metrics_position(league_dir, acked_lines):
    """(byte_offset, next_line_index, rebase) for resuming pushes after the
    hub says it already holds `acked_lines` (non-blank) lines of this node's
    local file.
    If the local file has fewer complete lines than that (truncated/replaced
    while the node ID survived), no overlap can be established: start from
    the end of the local file and ask the hub to re-anchor (rebase)."""
    path = os.path.join(league_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return 0, 0, acked_lines > 0
    with open(path, "rb") as f:
        data = f.read()
    offset, lines_seen = 0, 0
    for raw in data.split(b"\n")[:-1]:
        if lines_seen >= acked_lines:
            break
        offset += len(raw) + 1
        lines_seen += 1
    if lines_seen < acked_lines:
        return offset, lines_seen, True
    return offset, lines_seen, False


def _write_roster_to_disk(roster, league_dir, exclude=frozenset()):
    """Writes every deck in `roster` (except names in `exclude`) to local
    disk, atomically per file. `exclude` is empty on the very first
    bootstrap pull (this node's own deck gets seeded from the hub too, on
    equal footing with every other deck) and {own_deck} on every pull
    after that -- once training starts, this node is the authoritative
    writer of its own deck's files, and a stale pull must never overwrite
    fresher local work.

    Mirrors evictions too: a local snapshot_*.pt of a pulled deck that the
    hub no longer has was evicted by that deck's owner, so it is deleted --
    otherwise this node's opponent pool for that deck would grow past the
    snapshot cap. On the bootstrap pull this node's own deck is mirrored too
    (hub is canonical): extra local snapshots from work the hub never received
    are deleted along with the older live.pt being overwritten, so the node's
    weights, snapshots and progress counter all describe the same hub moment."""
    for deck, files in roster.items():
        if deck in exclude:
            continue
        deck_dir = os.path.join(league_dir, deck)
        os.makedirs(deck_dir, exist_ok=True)
        for fn in os.listdir(deck_dir):
            if fn.startswith("snapshot_") and fn.endswith(".pt") and fn not in files:
                os.remove(os.path.join(deck_dir, fn))
        for fn, content in files.items():
            final = os.path.join(deck_dir, fn)
            tmp = f"{final}.tmp"
            with open(tmp, "wb") as f:
                f.write(content)
            os.replace(tmp, final)


def _seed_local_progress(league_dir, cumulative_games):
    """Sets this node's local progress.json to the hub's last-known value for
    this deck, on bootstrap -- DISTRIBUTED_TRAINING_PLAN.md gap #1/#2's
    resolution. Always the hub's value, even if the local counter is ahead:
    bootstrap also overwrites the node's own deck files with the hub's
    (_write_roster_to_disk), so a local counter from a crashed, never-pushed
    call would otherwise describe work whose weights were just rolled back
    (hub is canonical; a deliberate `seed_hub --force` reseed must win too)."""
    current = _load_progress(league_dir)
    from rl.league.league_runner import _save_progress
    _save_progress(league_dir, current["last_batch_size"], cumulative_games)


def run(config_path, hub_url, credential_path, call_granularity=DEFAULT_CALL_GRANULARITY_ITERATIONS,
        pause_poll_interval=DEFAULT_PAUSE_POLL_INTERVAL_SECONDS, stop_flag=None, max_calls=None,
        league_dir=None, node_id_path=None, max_failed_syncs=DEFAULT_MAX_FAILED_SYNCS,
        sleep_fn=time.sleep):
    """max_calls: None runs forever (real deployment); a small int lets
    tests/benchmarks stop after a bounded number of loop iterations without
    needing a real stop signal or SIGTERM.

    league_dir: override for this node's local checkpoint directory,
    default `CHECKPOINTS_DIR/<league_name>` (the real, shared checkpoints
    tree, exactly where run_training_pipeline.py itself writes) --
    overridable so a test or benchmark can point this at a throwaway
    tmp_path instead, the same reason _run_session's own league_dir param
    exists.

    node_id_path: where the persistent node ID lives (see
    load_or_create_node_id). None means an ephemeral ID for this process only
    -- tests; real deployments always go through main(), which passes a path.

    The node exits (final push, then release) once its deck reaches the
    config's total_games, same field and meaning run_training_pipeline.py
    honors. It never exits with that last state unsynced: if the final push
    failed it keeps re-pushing without training further.

    Hub failures never lose work (checkpoints are already on local disk):
      - connection failure (node_sync gave up after its own retries): logged,
        the next iteration just pushes again -- `metrics_offset` only
        advances after a successful push, so no metrics line is skipped.
        After `max_failed_syncs` consecutive failed iterations the node
        stops training and polls until the hub answers.
      - 403 (token superseded): re-register with the same node ID and keep
        going if the hub hands back the SAME deck; a different deck means
        this node's local state is for the wrong deck, so exit (systemd
        restarts it into a clean bootstrap). A second 403 before any
        successful sync also exits, so two live processes sharing one node
        ID can't ping-pong the claim forever.
      - anything else (401 bad credential, 400, ...): propagates, retrying
        can't fix it."""
    stop_flag = stop_flag or StopFlag()
    stop_flag.install_sigterm_handler()

    opts = _resolve_options(load_config(config_path))
    device = _resolve_device(opts["device"])
    with open(credential_path) as f:
        credential = json.load(f)["registration_credential"]

    # The hub's vocabulary first, so build_pool indexes cards exactly as the weights
    # this node will pull were trained.
    _sync_vocab(hub_url, credential)
    # Validates the config-driven roster/vocab build once, up front, before registering;
    # the vocab also feeds the version gate's vocab hash.
    _decklists, vocab, _deck_ctxs, _fixed_tables = build_pool()
    version = _reported_version(vocab)
    node_id = load_or_create_node_id(node_id_path) if node_id_path else uuid.uuid4().hex
    registered = register_with_backoff(hub_url, credential, node_id, stop_flag, version=version)
    if registered is None:
        return
    deck, token, hub_cumulative_games, metrics_acked = registered
    print(f"registered: training {deck!r} (hub reports {hub_cumulative_games} cumulative games)", flush=True)

    league_dir = league_dir or str(CHECKPOINTS_DIR / opts["league_name"])
    os.makedirs(league_dir, exist_ok=True)

    # First sync always pulls EVERYTHING, including this node's own deck,
    # before playing a single game -- the same code path handles a fresh
    # league (hub seeded a canonical random init) and a node recovering
    # from lost local disk (hub has this deck's real prior history).
    initial_roster = node_sync.pull(hub_url, token, deck)
    _write_roster_to_disk(initial_roster, league_dir)
    _seed_local_progress(league_dir, hub_cumulative_games)

    executor = ProcessPoolExecutor(max_workers=opts["n_workers"]) if opts["n_workers"] > 1 else None
    metrics_offset, metrics_next_line, metrics_rebase = _resume_metrics_position(league_dir, metrics_acked)
    calls_made = 0
    failed_syncs = 0
    rejected_since_ok = False
    graceful_exit = False  # set on a hub stop signal or SIGTERM -> release the deck on the way out

    def handle_rejection():
        nonlocal token, rejected_since_ok
        if rejected_since_ok:
            raise SystemExit("token rejected again right after re-registering -- another live process may be "
                             "using this node ID")
        rejected_since_ok = True
        print("token superseded (403) -- re-registering with the same node ID", flush=True)
        registered = register_with_backoff(hub_url, credential, node_id, stop_flag, sleep_fn=sleep_fn,
                                           version=version)
        if registered is None:
            return
        new_deck, token, _, _ = registered
        if new_deck != deck:
            raise SystemExit(f"hub reassigned this node from {deck!r} to {new_deck!r}; local state is for "
                             f"the old deck -- exiting for a clean restart")

    def poll_signal():
        """One poll: the signal, or None if the hub is unreachable / the
        token was just re-registered (caller simply polls again)."""
        try:
            return node_sync.poll(hub_url, token, deck)
        except node_sync.SyncError as e:
            if e.status_code == 403:
                handle_rejection()
            elif e.status_code is not None:
                raise
            return None

    def wait_for_hub():
        nonlocal failed_syncs
        print(f"{failed_syncs} consecutive failed syncs -- waiting for the hub before training further",
              flush=True)
        while not stop_flag.requested:
            sleep_fn(pause_poll_interval)
            sig = poll_signal()
            if sig is not None:
                failed_syncs = 0
                return sig
        return "continue"

    try:
        while not stop_flag.requested:
            progress = _load_progress(league_dir)
            remaining = opts["total_games"] - progress["cumulative_games_per_deck"]
            if remaining > 0:
                # Final call is clipped exactly like run_training_pipeline._train_to
                # does (floor to whole iterations, at least 1), so the deck lands
                # within one iteration of total_games instead of a whole call past it.
                n_iterations = min(call_granularity, max(1, remaining // opts["games_per_iteration"]))
                _run_session(n_iterations, opts["games_per_iteration"], opts["snapshot_every"],
                            executor, opts["n_workers"], league_dir=league_dir, seed=opts["seed"],
                            train_decks=[deck], roster=None, pfsp=opts["pfsp"],
                            checkpoint_rate=opts["checkpoint_rate"],
                            cumulative_games=progress["cumulative_games_per_deck"],
                            ppo_hparams=opts["ppo_hparams"], pfsp_power=opts["pfsp_power"],
                            trunk_hidden=opts["trunk_hidden"], device=device,
                            stratify_0land_pct=opts["stratify_0land_pct"])
                advance_progress(league_dir, n_iterations, opts["games_per_iteration"],
                                 auto_sizing=False, session_start_games=progress["cumulative_games_per_deck"])
                calls_made += 1
            # remaining <= 0: target already reached but the last sync failed -- skip
            # training and just re-push below until it lands (never exit unsynced).
            new_progress = _load_progress(league_dir)

            synced_ok = False
            files = _collect_deck_files(league_dir, deck)
            metrics_lines, new_metrics_offset = _read_new_metrics_lines(league_dir, metrics_offset)
            try:
                signal_ = node_sync.push(hub_url, token, deck, files, metrics_lines,
                                         new_progress["cumulative_games_per_deck"], node_id=node_id,
                                         snapshot_names=[n for n in files if n.startswith("snapshot_")],
                                         metrics_start_line=metrics_next_line, metrics_rebase=metrics_rebase)
                metrics_offset = new_metrics_offset
                metrics_next_line += len(metrics_lines)
                metrics_rebase = False
                pulled = node_sync.pull(hub_url, token, deck)
                _write_roster_to_disk(pulled, league_dir, exclude={deck})
                failed_syncs = 0
                rejected_since_ok = False
                synced_ok = True
            except node_sync.SyncError as e:
                signal_ = "continue"
                if e.status_code == 403:
                    handle_rejection()
                elif e.status_code is None:
                    failed_syncs += 1
                    print(f"sync failed ({failed_syncs}/{max_failed_syncs}), will retry next iteration: {e}",
                          flush=True)
                    if failed_syncs >= max_failed_syncs:
                        signal_ = wait_for_hub()
                else:
                    raise

            if signal_ == "stop":
                print("stop signal received -- exiting after this call's push", flush=True)
                graceful_exit = True
                break
            if synced_ok and new_progress["cumulative_games_per_deck"] >= opts["total_games"]:
                print(f"reached total_games={opts['total_games']} -- final state pushed, exiting", flush=True)
                graceful_exit = True
                break
            while signal_ == "pause" and not stop_flag.requested:
                sleep_fn(pause_poll_interval)
                polled = poll_signal()
                if polled is not None:
                    signal_ = polled
            if signal_ == "stop":
                print("stop signal received while paused -- exiting", flush=True)
                graceful_exit = True
                break
            if max_calls is not None and calls_made >= max_calls:
                break
        graceful_exit = graceful_exit or stop_flag.requested
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if graceful_exit:
            try:
                node_sync.release(hub_url, token, deck)
                print("released deck back to the hub", flush=True)
            except node_sync.SyncError as e:
                # best effort: the deck just frees via the liveness timeout instead
                print(f"release failed (deck frees after the liveness timeout instead): {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Same training config schema run_training_pipeline.py reads.")
    ap.add_argument("--hub-url", required=True)
    ap.add_argument("--credential-file", required=True,
                    help="JSON file with {\"registration_credential\": \"...\"} -- gitignored, node-local.")
    ap.add_argument("--node-id-file", default=DEFAULT_NODE_ID_PATH,
                    help="Persistent node ID, so a restarted node gets its own deck back. Keep it OUTSIDE the "
                         "git checkout.")
    ap.add_argument("--call-granularity", type=int, default=DEFAULT_CALL_GRANULARITY_ITERATIONS)
    args = ap.parse_args()
    run(args.config, args.hub_url, args.credential_file, call_granularity=args.call_granularity,
        node_id_path=args.node_id_file)


if __name__ == "__main__":
    main()
