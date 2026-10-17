"""Evicted snapshots must disappear from the hub tree and from every node's
local mirror, or the 32-snapshot window cap stops holding anywhere but on the
owning node."""
import json
import os

from distributed import hub_server, hub_store, node_agent

CREDENTIAL = "cred"


def _names(d):
    return sorted(os.listdir(d))


def test_push_with_snapshot_names_deletes_the_evicted_ones_only(tmp_path):
    league = str(tmp_path)
    hub_store.push_deck_files(league, "elves", {"live.pt": b"1", "snapshot_1.pt": b"a", "snapshot_2.pt": b"b",
                                                "snapshot_3.pt": b"c"}, [], snapshot_names=["snapshot_1.pt",
                                                "snapshot_2.pt", "snapshot_3.pt"])
    # node evicted snapshot_1 and registered snapshot_4; pushes only the changed/new files
    hub_store.push_deck_files(league, "elves", {"live.pt": b"2", "snapshot_4.pt": b"d"}, [],
                              snapshot_names=["snapshot_2.pt", "snapshot_3.pt", "snapshot_4.pt"])
    assert _names(tmp_path / "elves") == ["live.pt", "snapshot_2.pt", "snapshot_3.pt", "snapshot_4.pt"]


def test_push_without_snapshot_names_prunes_nothing(tmp_path):
    league = str(tmp_path)
    hub_store.push_deck_files(league, "elves", {"snapshot_1.pt": b"a"}, [])
    hub_store.push_deck_files(league, "elves", {"live.pt": b"x"}, [])
    assert _names(tmp_path / "elves") == ["live.pt", "snapshot_1.pt"]


def test_prune_never_touches_other_decks_or_non_snapshot_files(tmp_path):
    league = str(tmp_path)
    hub_store.push_deck_files(league, "boggles", {"snapshot_1.pt": b"a", "opponent_stats.json": b"{}"}, [])
    hub_store.push_deck_files(league, "elves", {"live.pt": b"x", "opponent_stats.json": b"{}"}, [],
                              snapshot_names=[])
    assert _names(tmp_path / "boggles") == ["opponent_stats.json", "snapshot_1.pt"]
    assert _names(tmp_path / "elves") == ["live.pt", "opponent_stats.json"]


def test_server_rejects_malformed_snapshot_names(tmp_path):
    app = hub_server.create_app(str(tmp_path / "c"), ["elves"], CREDENTIAL, str(tmp_path / "s.json"))
    client = app.test_client()
    reg = client.post("/register", json={"credential": CREDENTIAL}).get_json()
    resp = client.post("/push", data={"token": reg["token"], "deck": reg["deck"], "cumulative_games": "1",
                                      "snapshot_names": "not json"})
    assert resp.status_code == 400
    resp = client.post("/push", data={"token": reg["token"], "deck": reg["deck"], "cumulative_games": "1",
                                      "snapshot_names": json.dumps(["snapshot_1.pt"])})
    assert resp.status_code == 200


def test_pull_mirrors_evictions_locally_and_bootstrap_mirrors_the_own_deck_too(tmp_path):
    for deck in ("boggles", "elves"):
        d = tmp_path / deck
        d.mkdir()
        for fn in ("snapshot_1.pt", "snapshot_2.pt"):
            (d / fn).write_bytes(b"old")
    roster = {"boggles": {"live.pt": b"n", "snapshot_2.pt": b"new"}, "elves": {"live.pt": b"e"}}
    # steady state: own deck excluded entirely (this node is its authoritative writer)
    node_agent._write_roster_to_disk(roster, str(tmp_path), exclude={"elves"})
    assert _names(tmp_path / "boggles") == ["live.pt", "snapshot_2.pt"]      # snapshot_1 was evicted upstream
    assert _names(tmp_path / "elves") == ["snapshot_1.pt", "snapshot_2.pt"]   # untouched
    # bootstrap pull: the own deck becomes an exact mirror of the hub's -- snapshots from work the
    # hub never received go away with the older live.pt being overwritten
    node_agent._write_roster_to_disk(roster, str(tmp_path))
    assert _names(tmp_path / "elves") == ["live.pt"]
