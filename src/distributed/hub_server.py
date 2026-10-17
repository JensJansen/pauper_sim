"""The hub's Flask app -- registration, push, pull. Flask is already a repo
dependency (used by src/webapp/), so this adds nothing new. Runs on Flask's
built-in dev server, which is THREADED by default (app.run sets
threaded=True) -- so requests do run concurrently: ownership state is
guarded by hub_ownership.HubRuntimeState's lock, metrics.jsonl appends by
hub_store's metrics lock, and each deck's file swap/read by hub_store's
per-deck lock. Keeping the dev server is deliberate (see
DISTRIBUTED_TRAINING_PLAN.md gap #9's resolution -- Tailscale-only exposure +
low traffic).

The version gate (version_gate.py) rejects nodes running code that isn't on the
operator's accepted list (426). TLS defaults
to none (`--tls none`, plain HTTP) -- adhoc self-signed is available via
`--tls adhoc` but needs the `cryptography` package (not in
requirements.txt); a Tailscale-issued real cert is also still being
evaluated, not yet chosen -- see the implementation plan's decision #7.
Neither option is defaulted to, since neither has actually been decided.
"""
import argparse
import base64
import json
import os
import time

from flask import Flask, Response, jsonify, request

from . import auth, hub_ownership, hub_store, version_gate

# Both unbenchmarked placeholders (DISTRIBUTED_TRAINING_PLAN.md gap #15) --
# real values come from Phase 3, once a genuine second machine exists to
# measure real speed variance and real sync-interval gaps against.
DEFAULT_LIVENESS_TIMEOUT_SECONDS = 300
DEFAULT_DRIFT_THRESHOLD_GAMES = 1000


def create_app(league_dir, deck_pool, registration_credential, runtime_state_path,
                liveness_timeout=DEFAULT_LIVENESS_TIMEOUT_SECONDS,
                drift_threshold=DEFAULT_DRIFT_THRESHOLD_GAMES,
                vocab=None, fixed_tables=None, admin_credential=None,
                accepted_versions_path=None, expected_vocab_hash=None, vocab_path=None):
    """league_dir: the hub's own local checkpoints/<league> tree (this
    machine's real checkpoints dir in production; a tmp_path in tests).
    deck_pool: the static list from hub_ownership.load_deck_pool.
    vocab/fixed_tables: from rl.roster.build_pool() -- optional (None skips
    seed_fresh_deck on registration, useful for tests that don't need real
    torch nets); production always passes them.
    admin_credential: secret for the /admin/* endpoints (X-Admin-Token header),
    a SEPARATE secret from the registration credential and held only by the
    operator. None disables the admin endpoints entirely (404).
    accepted_versions_path / expected_vocab_hash: the version gate (see
    version_gate.py). The accepted-SHA file is re-read on every registration.
    Both None = gate off (tests, `--no-version-gate`).
    vocab_path: the hub's card-vocabulary file, served at GET /vocab. The hub's
    file is authoritative -- it is append-only history, so a node that rebuilt it
    fresh from the manifest could index cards differently from the weights it
    pulls (silent corruption). None disables /vocab (404)."""
    app = Flask(__name__)
    version_rejections = {}  # node_id (or "?") -> last rejection, for `admin status`
    state = hub_ownership.HubRuntimeState(runtime_state_path)
    unseeded = hub_ownership.unseeded_decks(state, league_dir, deck_pool)
    if unseeded:
        print(f"WARNING: decks with weights but no recorded progress (will register at 0 games, restarting "
              f"their schedules) -- run `python -m distributed.seed_hub`: {unseeded}", flush=True)

    @app.post("/register")
    def register():
        body = request.get_json(force=True, silent=True) or {}
        if not auth.verify_registration_credential(body.get("credential"), registration_credential):
            return jsonify(error="invalid credential"), 401
        gate_on = accepted_versions_path is not None or expected_vocab_hash is not None
        accepted, reason = version_gate.check(
            body.get("reported_sha"), body.get("reported_vocab_hash"),
            reported_dirty=bool(body.get("reported_dirty")),
            accepted_shas=version_gate.load_accepted_shas(accepted_versions_path) if accepted_versions_path else None,
            expected_vocab_hash=expected_vocab_hash) if gate_on else (True, None)
        if not accepted:
            # 426, not 409: 409 means "pool full, retry soon"; this one needs the
            # operator (approve the SHA, or fix/recreate the node).
            rejected_id = body.get("node_id") or "?"
            version_rejections[rejected_id] = {"node_id": rejected_id, "sha": body.get("reported_sha"),
                                               "reason": reason, "ts": time.time()}
            print(f"register: VERSION REJECTED node {rejected_id!r} sha={body.get('reported_sha')!r}: {reason}",
                  flush=True)
            return jsonify(error=f"version rejected: {reason}"), 426
        node_id = body.get("node_id")
        if node_id is not None and (not isinstance(node_id, str) or not 0 < len(node_id) <= 128):
            return jsonify(error="node_id must be a non-empty string of at most 128 characters"), 400
        now = time.time()
        deck = hub_ownership.assign_deck(state, node_id, deck_pool, now, liveness_timeout)
        if deck is None:
            return jsonify(error="no unclaimed deck available in the pool"), 409
        prior = state.claims.get(deck)
        if prior is not None and now - prior["last_sync_ts"] <= liveness_timeout:
            print(f"register: node {node_id!r} superseding live claim on {deck!r} "
                  f"(previous node {prior.get('node_id')!r})", flush=True)
        token = hub_ownership.claim_deck(state, deck, deck_pool, now, node_id=node_id)
        if vocab is not None and fixed_tables is not None:
            hub_store.seed_fresh_deck(league_dir, deck, vocab, fixed_tables[deck])
        cumulative_games = state.claims[deck]["last_cumulative_games"]
        return jsonify(deck=deck, token=token, cumulative_games=cumulative_games,
                       metrics_acked=hub_ownership.metrics_acked(state, deck, node_id))

    @app.post("/push")
    def push():
        token = request.form.get("token")
        deck = request.form.get("deck")
        if not deck or not hub_ownership.verify_claim(state, deck, token):
            return jsonify(error="invalid or superseded token"), 403
        files = {name: storage.read() for name, storage in request.files.items()}
        metrics_text = request.form.get("metrics_lines", "")
        metrics_lines = [ln for ln in metrics_text.splitlines() if ln.strip()]
        try:
            cumulative_games = int(request.form.get("cumulative_games", "0"))
        except ValueError:
            return jsonify(error="cumulative_games must be an integer"), 400
        # Metrics de-dup (see hub_ownership.accept_metrics). A push with no
        # metrics_start_line (older client) is accepted as-is, undeduped.
        source = request.form.get("node_id")
        start_line = request.form.get("metrics_start_line")
        dedupe = source is not None and start_line is not None
        if dedupe:
            try:
                start_line = int(start_line)
            except ValueError:
                return jsonify(error="metrics_start_line must be an integer"), 400
            metrics_lines, new_count = hub_ownership.accept_metrics(
                state, deck, source, start_line, metrics_lines, rebase=request.form.get("metrics_rebase") == "1")
        snapshot_names = None
        if "snapshot_names" in request.form:
            try:
                snapshot_names = json.loads(request.form["snapshot_names"])
                assert isinstance(snapshot_names, list) and all(isinstance(n, str) for n in snapshot_names)
            except (ValueError, AssertionError):
                return jsonify(error="snapshot_names must be a JSON list of filenames"), 400
        hub_store.push_deck_files(league_dir, deck, files, metrics_lines, snapshot_names=snapshot_names)
        now = time.time()
        hub_ownership.record_progress(state, deck, cumulative_games, now)
        # Recorded only after the append: a hub crash between the two replays
        # (duplicates) one batch rather than losing it -- zero-loss over
        # zero-duplicate, since duplicates are rare and detectable.
        if dedupe:
            hub_ownership.record_metrics_ack(state, deck, source, new_count)
        signal = hub_ownership.sync_signal(state, deck, now, liveness_timeout, drift_threshold)
        return jsonify(ack=True, signal=signal)

    @app.get("/pull")
    def pull():
        token = request.args.get("token")
        deck = request.args.get("deck")
        if not deck or not hub_ownership.verify_claim(state, deck, token):
            return jsonify(error="invalid or superseded token"), 403
        roster = hub_store.pull_full_roster(league_dir, deck_pool)
        # base64-in-JSON, not multipart, for the response side -- simplest
        # correct thing for Phase 1 (a multi-file multipart *response* is
        # unusual to hand-build in Flask). Already-tracked, not a new
        # inefficiency: DISTRIBUTED_TRAINING_PLAN.md gap #11 already flags
        # pull's payload as unoptimized (pulls the whole live.pt including
        # optimizer state an opponent-only consumer never needs).
        encoded = {d: {fn: base64.b64encode(content).decode("ascii") for fn, content in files.items()}
                   for d, files in roster.items()}
        return jsonify(roster=encoded)

    @app.get("/poll")
    def poll():
        """Lightweight signal-only check for a paused node -- the 5-10s
        ping from DISTRIBUTED_TRAINING_PLAN.md's "Elastic/bounded drift",
        scoped correctly to only the paused-and-waiting case, not every
        node's every iteration."""
        token = request.args.get("token")
        deck = request.args.get("deck")
        if not deck or not hub_ownership.verify_claim(state, deck, token):
            return jsonify(error="invalid or superseded token"), 403
        now = time.time()
        signal = hub_ownership.sync_signal(state, deck, now, liveness_timeout, drift_threshold)
        return jsonify(signal=signal)

    @app.get("/vocab")
    def vocab_file():
        """The hub's vocab.json, credential-protected like registration. Nodes
        write it locally before building their pool (node_agent._sync_vocab)."""
        if vocab_path is None or not os.path.exists(vocab_path):
            return jsonify(error="hub has no vocab file to serve"), 404
        if not auth.verify_registration_credential(request.headers.get("X-Registration-Credential"),
                                                   registration_credential):
            return jsonify(error="invalid credential"), 401
        with open(vocab_path, "rb") as f:
            return Response(f.read(), mimetype="application/json")

    @app.post("/release")
    def release():
        """A node's graceful exit: gives the deck back immediately (see
        hub_ownership.release_claim). Token-authorized like push."""
        body = request.get_json(force=True, silent=True) or {}
        deck, token = body.get("deck"), body.get("token")
        if not deck or not hub_ownership.verify_claim(state, deck, token):
            return jsonify(error="invalid or superseded token"), 403
        hub_ownership.release_claim(state, deck)
        return jsonify(released=deck)

    def _admin_denied():
        if admin_credential is None:
            return jsonify(error="admin endpoints disabled"), 404
        if not auth.verify_registration_credential(request.headers.get("X-Admin-Token"), admin_credential):
            return jsonify(error="invalid admin token"), 401
        return None

    @app.get("/admin/status")
    def admin_status():
        denied = _admin_denied()
        if denied:
            return denied
        now = time.time()
        rejections = sorted(version_rejections.values(), key=lambda r: -r["ts"])[:20]
        return jsonify(decks=hub_ownership.status(state, deck_pool, now, liveness_timeout, drift_threshold),
                       version_rejections=[{**r, "age_s": round(now - r["ts"], 1)} for r in rejections])

    def _admin_target(body):
        """(deck | None, error response | None); {"all": true} -> (None, None)."""
        if body.get("all"):
            return None, None
        deck = body.get("deck")
        if deck not in deck_pool:
            return None, (jsonify(error=f"unknown deck {deck!r}; pool is {sorted(deck_pool)}"), 404)
        return deck, None

    @app.post("/admin/stop")
    def admin_stop():
        denied = _admin_denied()
        if denied:
            return denied
        deck, err = _admin_target(request.get_json(force=True, silent=True) or {})
        if err:
            return err
        if deck is None:
            return jsonify(stopping=hub_ownership.request_stop_all(state))
        if not hub_ownership.request_stop(state, deck):
            return jsonify(error=f"{deck!r} has no claim to stop"), 409
        return jsonify(stopping=1)

    @app.post("/admin/resume")
    def admin_resume():
        denied = _admin_denied()
        if denied:
            return denied
        deck, err = _admin_target(request.get_json(force=True, silent=True) or {})
        if err:
            return err
        return jsonify(cleared=hub_ownership.clear_stop(state, deck))

    app.config["_hub_state"] = state  # exposed for tests/admin scripts, not part of the HTTP API
    return app


def main():
    """Real hub launcher -- `create_app` above is used directly by tests
    and by anything embedding the hub programmatically; this is the actual
    `python -m distributed.hub_server ...` entrypoint for running it for
    real. Run from src/ (same convention every other entrypoint in this
    repo already follows, e.g. `python run_training_pipeline.py --config
    ...`) so rl.roster.build_pool()'s relative default paths resolve."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league-name", required=True, help="e.g. main-league -- the hub's canonical "
                    "checkpoints/<league-name>/ tree, same directory run_training_pipeline.py itself writes.")
    ap.add_argument("--ownership-pool", required=True,
                    help="distributed_configs/<league>_ownership.json -- the static list of eligible decks.")
    ap.add_argument("--credential-file", required=True,
                    help="A file containing just the registration credential (one line) -- generate one with "
                         "`python -m distributed.auth`.")
    ap.add_argument("--admin-credential-file", default=None,
                    help="A file with the operator's admin token (generate with `python -m distributed.auth`; "
                         "a DIFFERENT value from the registration credential). Omit to disable /admin/*.")
    ap.add_argument("--accepted-versions-file", default="../distributed_configs/accepted_versions.local.json",
                    help="Operator-maintained JSON {\"shas\": [...]} of code versions (commit SHAs or >=7-char "
                         "prefixes) nodes may run. Re-read on every registration; missing/empty rejects every node.")
    ap.add_argument("--no-version-gate", action="store_true",
                    help="DISABLE the code-version check. Nodes on stale engine code would then train under "
                         "different rules than the hub believes -- invalid data under the rules-faithfulness "
                         "mandate. Development only.")
    ap.add_argument("--runtime-state", default=None,
                    help="Defaults to distributed_configs/hub_runtime_state.json (gitignored).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--liveness-timeout", type=int, default=DEFAULT_LIVENESS_TIMEOUT_SECONDS)
    ap.add_argument("--drift-threshold", type=int, default=DEFAULT_DRIFT_THRESHOLD_GAMES)
    ap.add_argument("--no-seed-fresh-decks", action="store_true",
                    help="Skip building vocab/fixed_tables (and therefore seed_fresh_deck on registration). Only "
                         "safe if every deck in the pool already has real live.pt/mulligan.pt on disk.")
    ap.add_argument("--tls", choices=["none", "adhoc"], default="none",
                    help="'none' (default): plain HTTP -- fine ONLY behind Tailscale, never on the open "
                         "internet. 'adhoc': a self-signed cert generated on the fly (needs the `cryptography` "
                         "package, NOT currently in requirements.txt -- `pip install cryptography` first). "
                         "See DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md decision #7 for evaluating a "
                         "Tailscale-issued real cert instead before real deployment -- neither option is "
                         "committed to yet, which is why this isn't defaulted to 'adhoc'.")
    args = ap.parse_args()

    from repo_paths import CHECKPOINTS_DIR
    league_dir = str(CHECKPOINTS_DIR / args.league_name)
    deck_pool = hub_ownership.load_deck_pool(args.ownership_pool)
    with open(args.credential_file) as f:
        credential = f.read().strip()
    runtime_state_path = args.runtime_state or "../distributed_configs/hub_runtime_state.json"

    admin_credential = None
    if args.admin_credential_file:
        with open(args.admin_credential_file) as f:
            admin_credential = f.read().strip()
        if admin_credential == credential:
            raise SystemExit("--admin-credential-file must differ from the registration credential "
                             "(every node holds that one)")

    from rl.roster import VOCAB_PATH  # the file build_pool reads/writes; same relative-to-src convention
    vocab, fixed_tables = None, None
    expected_vocab_hash = None
    if not (args.no_seed_fresh_decks and args.no_version_gate):
        from rl.roster import build_pool
        _decklists, vocab, _deck_ctxs, fixed_tables = build_pool()
        expected_vocab_hash = None if args.no_version_gate else version_gate.vocab_hash(vocab)
    if args.no_seed_fresh_decks:
        vocab, fixed_tables = None, None
    accepted_versions_path = None if args.no_version_gate else args.accepted_versions_file
    if args.no_version_gate:
        print("WARNING: version gate DISABLED (--no-version-gate) -- any node, running any code, can register.",
              flush=True)
    elif not version_gate.load_accepted_shas(accepted_versions_path):
        print(f"WARNING: {accepted_versions_path} is missing or lists no SHAs -- EVERY node will be rejected "
              f"until you add the commit SHA(s) nodes run.", flush=True)

    app = create_app(league_dir, deck_pool, credential, runtime_state_path,
                     liveness_timeout=args.liveness_timeout, drift_threshold=args.drift_threshold,
                     vocab=vocab, fixed_tables=fixed_tables, admin_credential=admin_credential,
                     accepted_versions_path=accepted_versions_path, expected_vocab_hash=expected_vocab_hash,
                     vocab_path=VOCAB_PATH)
    print(f"hub serving league={args.league_name!r} pool={deck_pool} on {args.host}:{args.port} "
          f"(tls={args.tls})", flush=True)
    if args.tls == "none":
        print("WARNING: plain HTTP -- only safe behind Tailscale, never expose this port to the "
              "open internet as-is.", flush=True)
    # Flask's built-in dev server, deliberately -- see this module's own
    # docstring for why (Tailscale-only exposure, low traffic; gap #9's
    # resolution).
    app.run(host=args.host, port=args.port, ssl_context="adhoc" if args.tls == "adhoc" else None)


if __name__ == "__main__":
    main()
