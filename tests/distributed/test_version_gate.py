"""The code-version gate: pure check(), the hub's use of it (426, live-reloaded
accepted list, rejection log in admin status), and the node's wait-and-retry."""
import json
from types import SimpleNamespace

from distributed import hub_server, node_agent, node_sync, version_gate

SHA = "a" * 40
CRED, ADMIN = "cred", "admin"


def test_gate_off_accepts_everything():
    assert version_gate.check(None, None) == (True, None)


def test_accepts_full_sha_and_prefix_rejects_others():
    assert version_gate.check(SHA, "v", accepted_shas=[SHA])[0]
    assert version_gate.check(SHA, "v", accepted_shas=["aaaaaaa"])[0]
    ok, reason = version_gate.check("b" * 40, "v", accepted_shas=[SHA])
    assert not ok and "not on the hub's accepted list" in reason


def test_rejects_missing_sha_dirty_tree_and_empty_list():
    assert not version_gate.check(None, "v", accepted_shas=[SHA])[0]
    ok, reason = version_gate.check(SHA, "v", reported_dirty=True, accepted_shas=[SHA])
    assert not ok and "local modifications" in reason
    assert not version_gate.check(SHA, "v", accepted_shas=[])[0]          # fail closed


def test_vocab_mismatch_rejected_even_when_sha_is_fine():
    assert not version_gate.check(SHA, "mine", accepted_shas=[SHA], expected_vocab_hash="hubs")[0]
    assert version_gate.check(SHA, "same", accepted_shas=[SHA], expected_vocab_hash="same")[0]


def test_load_accepted_shas_fails_closed_and_ignores_too_short_entries(tmp_path):
    assert version_gate.load_accepted_shas(str(tmp_path / "missing.json")) == []
    p = tmp_path / "acc.json"
    p.write_text(json.dumps({"shas": ["ABCDEF1", "abc", 5]}))
    assert version_gate.load_accepted_shas(str(p)) == ["abcdef1"]
    p.write_text("not json")
    assert version_gate.load_accepted_shas(str(p)) == []


def test_vocab_hash_is_stable_and_order_independent_but_content_sensitive():
    a = SimpleNamespace(name_to_index={"x": 1, "y": 2})
    b = SimpleNamespace(name_to_index={"y": 2, "x": 1})
    c = SimpleNamespace(name_to_index={"x": 2, "y": 1})
    assert version_gate.vocab_hash(a) == version_gate.vocab_hash(b) != version_gate.vocab_hash(c)


def test_local_version_reads_this_repo():
    from repo_paths import REPO_ROOT
    sha, dirty = version_gate.local_version(REPO_ROOT)
    assert len(sha) == 40 and isinstance(dirty, bool)


def test_local_version_outside_a_checkout_is_none(tmp_path):
    assert version_gate.local_version(tmp_path) == (None, False)


# --- hub ---

def _client(tmp_path, accepted_file, **kw):
    app = hub_server.create_app(str(tmp_path / "c"), ["elves", "boggles"], CRED, str(tmp_path / "s.json"),
                                admin_credential=ADMIN, accepted_versions_path=str(accepted_file), **kw)
    return app.test_client()


def _reg(client, sha, node_id="n1", **extra):
    return client.post("/register", json={"credential": CRED, "node_id": node_id, "reported_sha": sha, **extra})


def test_hub_rejects_with_426_then_accepts_once_the_sha_is_approved_without_a_restart(tmp_path):
    accepted = tmp_path / "accepted.json"
    client = _client(tmp_path, accepted)                       # file doesn't exist yet: fail closed
    resp = _reg(client, SHA)
    assert resp.status_code == 426 and "accepted list" in resp.get_json()["error"]
    accepted.write_text(json.dumps({"shas": [SHA[:10]]}))      # operator approves
    assert _reg(client, SHA).status_code == 200


def test_hub_rejects_dirty_and_vocab_mismatch(tmp_path):
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps({"shas": [SHA]}))
    client = _client(tmp_path, accepted, expected_vocab_hash="hub-vocab")
    assert _reg(client, SHA, reported_vocab_hash="hub-vocab").status_code == 200
    assert _reg(client, SHA, node_id="n2", reported_vocab_hash="hub-vocab", reported_dirty=True).status_code == 426
    assert _reg(client, SHA, node_id="n3", reported_vocab_hash="other").status_code == 426


def test_rejections_show_up_in_admin_status(tmp_path):
    client = _client(tmp_path, tmp_path / "accepted.json")
    _reg(client, "b" * 40, node_id="stale-node")
    body = client.get("/admin/status", headers={"X-Admin-Token": ADMIN}).get_json()
    assert [(r["node_id"], r["sha"]) for r in body["version_rejections"]] == [("stale-node", "b" * 40)]


def test_gate_is_off_when_no_gate_is_configured(tmp_path):
    app = hub_server.create_app(str(tmp_path / "c"), ["elves"], CRED, str(tmp_path / "s.json"))
    assert app.test_client().post("/register", json={"credential": CRED}).status_code == 200


# --- node ---

def test_node_waits_and_retries_on_426_then_registers(monkeypatch):
    calls, sleeps = [], []

    def fake_register(url, cred, node_id=None, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise node_sync.SyncError("426: version rejected", status_code=426)
        return "elves", "tok", 0, 0
    monkeypatch.setattr(node_agent.node_sync, "register", fake_register)
    out = node_agent.register_with_backoff("u", "c", "n1", node_agent.StopFlag(), sleep_fn=sleeps.append,
                                           version={"reported_sha": SHA})
    assert out[0] == "elves" and sleeps == [node_agent.VERSION_RETRY_SECONDS]
    assert calls[0] == {"reported_sha": SHA}      # the version is passed through on every attempt
