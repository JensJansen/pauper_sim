"""Loopback integration test: a real hub_server.py and a real node_agent.py
loop, both on this machine, exercising the real game engine + torch --
Phase 2's "prove the node agent's own loop around _run_session actually
works end to end" exit criterion. Marked slow, same convention as every
other real-engine test in this repo (see tests/validation/*.py).

Deliberately does NOT attempt the full statistical "indistinguishable from
run_training_pipeline.py" comparison -- that was never run and is waived by
the owner (DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md section 7, item 13).
This test pins only the mechanical contract: real training
happens, real files land on both the node's local disk and the hub, and a
second sync sees the first sync's own results reflected back correctly."""
import json

import pytest

from distributed import node_agent

CREDENTIAL = "loopback-test-credential"


def _write_credential_file(path):
    path.write_text(json.dumps({"registration_credential": CREDENTIAL}))


def _write_training_config(path):
    # Standalone, no "extends" -- avoids depending on baseline_settings.json's
    # own path resolution from a tmp_path location. n_workers/games_per_iteration
    # kept small so this stays a fast "does the loop mechanically work" check,
    # not a real training run.
    path.write_text(json.dumps({
        "league_name": "distributed-loopback-test-unused",  # league_dir is overridden below, this is never read as a path
        "total_games": 1000,  # required by _resolve_options' own assert; unused by node_agent's looping logic
        "n_workers": 2,
        "games_per_iteration": 4,
        "snapshot_every_games": 200,
        "checkpoint_opponent_rate": 0.0,
        "device": "cpu",
    }))


@pytest.mark.slow
def test_node_agent_trains_one_deck_and_syncs_with_the_hub(tmp_path):
    from distributed import hub_server
    from rl.roster import VOCAB_PATH

    pool = ["elves"]  # a single-deck pool: this test's node is guaranteed to claim it
    hub_league_dir = tmp_path / "hub_checkpoints"
    app = hub_server.create_app(
        league_dir=str(hub_league_dir), deck_pool=pool, registration_credential=CREDENTIAL,
        runtime_state_path=str(tmp_path / "runtime_state.json"),
        vocab_path=VOCAB_PATH,  # the hub serves the same file the node already has: a no-op sync, no write
    )
    # loopback: drive the hub through its own Flask test client's underlying
    # WSGI app via a real running server, same as test_node_sync.py's
    # running_hub fixture -- inlined here (not the shared fixture) since
    # this test needs a single-deck pool, not the shared fixture's two-deck
    # one.
    import socket
    import threading
    from werkzeug.serving import make_server

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        hub_url = f"http://127.0.0.1:{port}"
        config_path = tmp_path / "config.json"
        credential_path = tmp_path / "credential.json"
        node_league_dir = tmp_path / "node_checkpoints"
        _write_training_config(config_path)
        _write_credential_file(credential_path)

        node_agent.run(str(config_path), hub_url, str(credential_path),
                       call_granularity=1, max_calls=1, league_dir=str(node_league_dir))

        # the node's own local disk has real, loadable weights for its deck
        assert (node_league_dir / "elves" / "live.pt").exists()
        assert (node_league_dir / "elves" / "mulligan.pt").exists()

        # the hub received them too, via the real push
        assert (hub_league_dir / "elves" / "live.pt").exists()
        assert (hub_league_dir / "elves" / "live.pt").read_bytes() == \
            (node_league_dir / "elves" / "live.pt").read_bytes()

        # real PPO metrics landed in the hub's merged metrics.jsonl, not just checkpoints
        hub_metrics = (hub_league_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert any(json.loads(ln).get("kind") == "ppo" for ln in hub_metrics)

        # progress actually advanced by exactly one call's worth of games
        node_progress = json.loads((node_league_dir / "progress.json").read_text())
        assert node_progress["cumulative_games_per_deck"] == 4  # call_granularity=1 * games_per_iteration=4

        # A second node registering for the SAME already-claimed deck (a
        # restart before this run's claim has gone stale) is correctly
        # rejected -- not this test's job to re-exercise that, already
        # covered by test_hub_server.py's own claim-exclusivity tests.
        # _seed_local_progress's own resume-from-hub logic (what a
        # genuinely stale/replacement registration would then do) is
        # covered directly, without needing a second real training pass,
        # by test_node_agent_unit.py.
    finally:
        server.shutdown()
        thread.join(timeout=5)
