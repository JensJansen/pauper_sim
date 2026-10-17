"""The hub serves its (append-only, authoritative) vocab.json; a node adopts it
before building its pool, refusing only when its local file conflicts."""
import json

import pytest

from distributed import hub_server, node_agent, node_sync

CRED = "cred"
HUB_VOCAB = {"Forest": 1, "Island": 2, "Mountain": 3}


def _hub_client(tmp_path, vocab_path):
    app = hub_server.create_app(str(tmp_path / "c"), ["elves"], CRED, str(tmp_path / "s.json"),
                                vocab_path=str(vocab_path) if vocab_path else None)
    return app.test_client()


def test_vocab_endpoint_serves_the_file_to_the_credential_holder_only(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(HUB_VOCAB))
    client = _hub_client(tmp_path, p)
    assert client.get("/vocab").status_code == 401
    assert client.get("/vocab", headers={"X-Registration-Credential": "wrong"}).status_code == 401
    resp = client.get("/vocab", headers={"X-Registration-Credential": CRED})
    assert resp.status_code == 200 and resp.get_json() == HUB_VOCAB


def test_vocab_endpoint_404_when_the_hub_has_none(tmp_path):
    assert _hub_client(tmp_path, None).get("/vocab", headers={"X-Registration-Credential": CRED}).status_code == 404
    assert _hub_client(tmp_path, tmp_path / "absent.json").get(
        "/vocab", headers={"X-Registration-Credential": CRED}).status_code == 404


def test_fetch_vocab_over_real_http(running_hub, tmp_path):
    hub_url, app, pool, credential = running_hub
    with pytest.raises(node_sync.SyncError):                      # fixture's hub serves no vocab -> 404
        node_sync.fetch_vocab(hub_url, credential, max_retries=1)


def test_reconcile_writes_when_absent_and_is_a_noop_when_identical(tmp_path):
    p = tmp_path / "x" / "vocab.json"
    node_agent._reconcile_vocab(str(p), HUB_VOCAB)
    assert json.loads(p.read_text()) == HUB_VOCAB
    mtime = p.stat().st_mtime_ns
    node_agent._reconcile_vocab(str(p), HUB_VOCAB)
    assert p.stat().st_mtime_ns == mtime


def test_reconcile_adopts_a_compatible_older_local_file(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"Forest": 1, "Island": 2}))          # the hub only ever appends
    node_agent._reconcile_vocab(str(p), HUB_VOCAB)
    assert json.loads(p.read_text()) == HUB_VOCAB


def test_reconcile_refuses_a_local_file_with_conflicting_indices(tmp_path):
    p = tmp_path / "vocab.json"
    original = json.dumps({"Forest": 2, "Island": 1})             # same names, swapped indices
    p.write_text(original)
    with pytest.raises(SystemExit):
        node_agent._reconcile_vocab(str(p), HUB_VOCAB)
    assert p.read_text() == original                              # untouched
