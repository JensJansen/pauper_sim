"""Operator control: /admin/{status,stop,resume}, the node's /release on
graceful exit, and the admin CLI's formatting."""
import json

import pytest

from distributed import admin, hub_ownership, hub_server
from test_node_agent_unit import _drive_run, push_kwargs  # noqa: F401  (shared run() harness)

POOL = ["boggles", "elves"]
CRED, ADMIN = "node-credential", "admin-token"
H = {"X-Admin-Token": ADMIN}


def _client(tmp_path, admin_credential=ADMIN):
    app = hub_server.create_app(str(tmp_path / "c"), POOL, CRED, str(tmp_path / "s.json"),
                                admin_credential=admin_credential)
    return app.test_client(), app


def _register(client, node_id):
    return client.post("/register", json={"credential": CRED, "node_id": node_id}).get_json()


def _push(client, reg):
    return client.post("/push", data={"token": reg["token"], "deck": reg["deck"],
                                      "cumulative_games": "10"}).get_json()["signal"]


def test_admin_endpoints_need_the_admin_token_and_are_off_without_one(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/admin/status").status_code == 401
    assert client.get("/admin/status", headers={"X-Admin-Token": CRED}).status_code == 401  # node secret != admin
    assert client.get("/admin/status", headers=H).status_code == 200
    off, _ = _client(tmp_path / "off", admin_credential=None)
    assert off.get("/admin/status", headers=H).status_code == 404


def test_stop_one_deck_reaches_only_that_node_then_release_clears_it(tmp_path):
    client, app = _client(tmp_path)
    a, b = _register(client, "n1"), _register(client, "n2")
    assert client.post("/admin/stop", json={"deck": a["deck"]}, headers=H).get_json() == {"stopping": 1}
    assert _push(client, a) == "stop" and _push(client, b) == "continue"
    states = {r["deck"]: r["state"] for r in client.get("/admin/status", headers=H).get_json()["decks"]}
    assert states[a["deck"]] == "stopping" and states[b["deck"]] == "active"
    # node exits gracefully -> releases: token dead, stop cleared, deck free at once
    assert client.post("/release", json={"deck": a["deck"], "token": a["token"]}).status_code == 200
    assert client.post("/push", data={"token": a["token"], "deck": a["deck"],
                                      "cumulative_games": "1"}).status_code == 403
    states = {r["deck"]: r["state"] for r in client.get("/admin/status", headers=H).get_json()["decks"]}
    assert states[a["deck"]] == "released"
    again = _register(client, "n3")  # a replacement gets the released deck immediately, with its progress
    assert again["deck"] == a["deck"] and again["cumulative_games"] == 10


def test_stop_all_and_resume(tmp_path):
    client, _ = _client(tmp_path)
    a, b = _register(client, "n1"), _register(client, "n2")
    assert client.post("/admin/stop", json={"all": True}, headers=H).get_json() == {"stopping": 2}
    assert client.post("/admin/resume", json={"deck": a["deck"]}, headers=H).get_json() == {"cleared": 1}
    assert _push(client, a) == "continue" and _push(client, b) == "stop"
    assert client.post("/admin/resume", json={"all": True}, headers=H).get_json() == {"cleared": 1}
    assert _push(client, b) == "continue"


def test_stop_unknown_or_unclaimed_deck_is_an_error(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/admin/stop", json={"deck": "nope"}, headers=H).status_code == 404
    assert client.post("/admin/stop", json={"deck": "elves"}, headers=H).status_code == 409  # never claimed


def test_release_with_a_dead_or_missing_token_is_rejected(tmp_path):
    client, _ = _client(tmp_path)
    reg = _register(client, "n1")
    assert client.post("/release", json={"deck": reg["deck"], "token": "bad"}).status_code == 403
    client.post("/release", json={"deck": reg["deck"], "token": reg["token"]})
    assert client.post("/release", json={"deck": reg["deck"], "token": None}).status_code == 403  # released token is None


def test_status_marks_unclaimed_and_quiet(tmp_path):
    state = hub_ownership.HubRuntimeState(str(tmp_path / "s.json"))
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0, node_id="n1")
    rows = {r["deck"]: r for r in hub_ownership.status(state, POOL, now=1000.0, liveness_timeout=300, threshold=1000)}
    assert rows["boggles"]["state"] == "quiet" and rows["elves"]["state"] == "unclaimed"


def test_cli_formats_a_status_table():
    out = admin.format_status([{"deck": "elves", "state": "active", "node_id": "n1", "cumulative_games": 5,
                                "last_sync_age_s": 3.2, "signal": "continue"},
                               {"deck": "boggles", "state": "unclaimed", "node_id": None, "cumulative_games": 0,
                                "last_sync_age_s": None, "signal": None}])
    assert "elves" in out and "3.2s" in out and "unclaimed" in out


def test_node_releases_its_deck_after_a_hub_stop_signal(monkeypatch, tmp_path):
    from distributed import node_agent
    released = []
    monkeypatch.setattr(node_agent.node_sync, "release", lambda url, token, deck: released.append((token, deck)))
    _drive_run(monkeypatch, tmp_path, lambda n, t: "stop", lambda: ("elves", "t1", 0))
    assert released == [("t1", "elves")]


def test_node_does_not_release_on_a_bounded_test_run(monkeypatch, tmp_path):
    from distributed import node_agent
    released = []
    monkeypatch.setattr(node_agent.node_sync, "release", lambda *a: released.append(a))
    _drive_run(monkeypatch, tmp_path, lambda n, t: "continue", lambda: ("elves", "t1", 0), max_calls=1)
    assert released == []
