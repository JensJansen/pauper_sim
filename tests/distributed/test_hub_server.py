"""Self-check for distributed.hub_server: the full register -> push -> pull
round trip over Flask's test client -- no real network, no real second
machine, matching DISTRIBUTED_TRAINING_IMPLEMENTATION_PLAN.md's Phase 1
exit criteria. vocab/fixed_tables omitted (None) throughout: these tests
exercise the HTTP/ownership plumbing, not seed_fresh_deck's real-net path
(that's covered directly in test_hub_store.py)."""
import io
import json

from distributed import hub_ownership, hub_server

POOL = ["boggles", "elves"]
CREDENTIAL = "test-registration-credential"


def _app(tmp_path, **kwargs):
    return hub_server.create_app(
        league_dir=str(tmp_path / "checkpoints"),
        deck_pool=POOL,
        registration_credential=CREDENTIAL,
        runtime_state_path=str(tmp_path / "runtime_state.json"),
        **kwargs,
    )


def _register(client, credential=CREDENTIAL):
    return client.post("/register", json={"credential": credential})


def test_register_rejects_a_bad_credential(tmp_path):
    client = _app(tmp_path).test_client()
    resp = _register(client, credential="wrong")
    assert resp.status_code == 401


def test_register_assigns_a_deck_and_token(tmp_path):
    client = _app(tmp_path).test_client()
    resp = _register(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["deck"] in POOL
    assert body["token"]


def test_register_rejects_once_the_pool_is_fully_claimed(tmp_path):
    client = _app(tmp_path).test_client()
    for _ in POOL:
        assert _register(client).status_code == 200
    resp = _register(client)
    assert resp.status_code == 409


def test_second_registration_for_a_stale_deck_invalidates_the_first_token(tmp_path):
    app = _app(tmp_path, liveness_timeout=100)
    client = app.test_client()
    first = _register(client).get_json()
    # simulate the first node going stale by advancing the claim's
    # last_sync_ts far into the past, directly on the shared state object
    state = app.config["_hub_state"]
    state.claims[first["deck"]]["last_sync_ts"] -= 10_000
    state._save()

    # pick_available_deck always returns the first available deck in pool
    # order, and first["deck"] is now stale/available again -- the very
    # next registration reclaims it directly, no need to exhaust the pool.
    second = _register(client).get_json()
    assert second["deck"] == first["deck"]
    assert second["token"] != first["token"]

    push_resp = client.post("/push", data={
        "token": first["token"], "deck": first["deck"], "cumulative_games": "10",
        "live.pt": (io.BytesIO(b"stale-writer"), "live.pt"),
    }, content_type="multipart/form-data")
    assert push_resp.status_code == 403


def test_push_writes_files_and_returns_continue_signal(tmp_path):
    client = _app(tmp_path).test_client()
    reg = _register(client).get_json()
    resp = client.post("/push", data={
        "token": reg["token"], "deck": reg["deck"], "cumulative_games": "50",
        "live.pt": (io.BytesIO(b"weights"), "live.pt"),
        "metrics_lines": '{"kind": "ppo", "iteration": 0}',
    }, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.get_json() == {"ack": True, "signal": "continue"}

    live_path = tmp_path / "checkpoints" / reg["deck"] / "live.pt"
    assert live_path.read_bytes() == b"weights"
    metrics_path = tmp_path / "checkpoints" / "metrics.jsonl"
    assert json.loads(metrics_path.read_text().strip()) == {"kind": "ppo", "iteration": 0}


def test_push_rejects_an_unrecognized_token(tmp_path):
    client = _app(tmp_path).test_client()
    reg = _register(client).get_json()
    resp = client.post("/push", data={
        "token": "not-the-real-token", "deck": reg["deck"], "cumulative_games": "0",
    }, content_type="multipart/form-data")
    assert resp.status_code == 403


def test_pull_returns_every_pushed_deck_file_base64_encoded(tmp_path):
    import base64
    client = _app(tmp_path).test_client()
    reg = _register(client).get_json()
    client.post("/push", data={
        "token": reg["token"], "deck": reg["deck"], "cumulative_games": "0",
        "live.pt": (io.BytesIO(b"weights"), "live.pt"),
    }, content_type="multipart/form-data")

    resp = client.get(f"/pull?token={reg['token']}&deck={reg['deck']}")
    assert resp.status_code == 200
    roster = resp.get_json()["roster"]
    assert base64.b64decode(roster[reg["deck"]]["live.pt"]) == b"weights"
    # every pool deck is present in the response, even ones never pushed yet
    assert set(roster) == set(POOL)


def test_drift_signal_surfaces_through_the_real_push_endpoint(tmp_path):
    app = _app(tmp_path, drift_threshold=100)
    client = app.test_client()
    fast = _register(client).get_json()
    slow = _register(client).get_json()

    client.post("/push", data={"token": slow["token"], "deck": slow["deck"], "cumulative_games": "10"},
                content_type="multipart/form-data")
    resp = client.post("/push", data={"token": fast["token"], "deck": fast["deck"], "cumulative_games": "5000"},
                       content_type="multipart/form-data")
    assert resp.get_json()["signal"] == "pause"


def test_stop_request_surfaces_through_push_and_poll(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    reg = _register(client).get_json()
    hub_ownership.request_stop(app.config["_hub_state"], reg["deck"])

    push_resp = client.post("/push", data={"token": reg["token"], "deck": reg["deck"], "cumulative_games": "0"},
                            content_type="multipart/form-data")
    assert push_resp.get_json()["signal"] == "stop"

    poll_resp = client.get(f"/poll?token={reg['token']}&deck={reg['deck']}")
    assert poll_resp.get_json()["signal"] == "stop"


def test_register_with_same_node_id_returns_same_deck_and_revokes_old_token(tmp_path):
    client = _app(tmp_path).test_client()
    first = client.post("/register", json={"credential": CREDENTIAL, "node_id": "n1"}).get_json()
    client.post("/register", json={"credential": CREDENTIAL, "node_id": "n2"})
    again = client.post("/register", json={"credential": CREDENTIAL, "node_id": "n1"}).get_json()
    assert again["deck"] == first["deck"] and again["token"] != first["token"]


def test_register_rejects_a_malformed_node_id(tmp_path):
    client = _app(tmp_path).test_client()
    assert client.post("/register", json={"credential": CREDENTIAL, "node_id": 5}).status_code == 400


def test_replayed_metrics_batch_is_not_appended_twice(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    reg = client.post("/register", json={"credential": CREDENTIAL, "node_id": "n1"}).get_json()
    assert reg["metrics_acked"] == 0

    def push(start, lines):
        return client.post("/push", data={
            "token": reg["token"], "deck": reg["deck"], "cumulative_games": "1", "node_id": "n1",
            "metrics_start_line": str(start), "metrics_lines": "\n".join(lines)})
    assert push(0, ['{"i": 0}', '{"i": 1}']).status_code == 200
    assert push(0, ['{"i": 0}', '{"i": 1}']).status_code == 200      # lost-ack replay
    assert push(1, ['{"i": 1}', '{"i": 2}']).status_code == 200      # overlapping retry
    got = (tmp_path / "checkpoints" / "metrics.jsonl").read_text().splitlines()
    assert got == ['{"i": 0}', '{"i": 1}', '{"i": 2}']
    # a restarted node learns what the hub holds
    again = client.post("/register", json={"credential": CREDENTIAL, "node_id": "n1"}).get_json()
    assert again["metrics_acked"] == 3
