"""seed_hub: copies a trained league's sync surface into the hub tree and
records per-deck starting progress, so registering nodes inherit real history."""
import json

import pytest

from distributed import hub_ownership, hub_server, seed_hub

POOL = ["boggles", "elves"]


def _make_source(root, games=92340):
    for deck in POOL:
        d = root / deck
        (d / "archive").mkdir(parents=True)
        (d / "checks").mkdir()
        for fn in ("live.pt", "mulligan.pt", "opponent_stats.json", "snapshot_5.pt", "snapshot_6.pt"):
            (d / fn).write_bytes(f"{deck}:{fn}".encode())
        (d / "archive" / "snapshot_1.pt").write_bytes(b"old")
    (root / "progress.json").write_text(json.dumps({"last_batch_size": 0, "cumulative_games_per_deck": games}))


def _state(tmp_path):
    return hub_ownership.HubRuntimeState(str(tmp_path / "state.json"))


def test_seed_copies_sync_surface_only_and_records_progress(tmp_path):
    src, hub = tmp_path / "src", tmp_path / "hub"
    _make_source(src)
    state = _state(tmp_path)
    assert seed_hub.seed_hub(str(src), str(hub), POOL, state) == {"boggles": 92340, "elves": 92340}
    for deck in POOL:
        assert sorted(p.name for p in (hub / deck).iterdir()) == [
            "live.pt", "mulligan.pt", "opponent_stats.json", "snapshot_5.pt", "snapshot_6.pt"]  # no archive/checks
    assert (hub / "elves" / "snapshot_6.pt").read_bytes() == b"elves:snapshot_6.pt"
    assert _state(tmp_path).seed_progress == {"boggles": 92340, "elves": 92340}  # persisted


def test_registering_node_gets_the_seeded_progress_not_zero(tmp_path):
    src, hub = tmp_path / "src", tmp_path / "hub"
    _make_source(src)
    state = _state(tmp_path)
    seed_hub.seed_hub(str(src), str(hub), POOL, state)
    app = hub_server.create_app(str(hub), POOL, "cred", str(tmp_path / "state.json"))
    reg = app.test_client().post("/register", json={"credential": "cred", "node_id": "n1"}).get_json()
    assert reg["cumulative_games"] == 92340


def test_refuses_to_overwrite_without_force_and_force_leaves_no_stale_snapshots(tmp_path):
    src, hub = tmp_path / "src", tmp_path / "hub"
    _make_source(src)
    state = _state(tmp_path)
    seed_hub.seed_hub(str(src), str(hub), POOL, state)
    with pytest.raises(SystemExit):
        seed_hub.seed_hub(str(src), str(hub), POOL, state)
    (hub / "elves" / "snapshot_99.pt").write_bytes(b"stale")
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0, node_id="n1")
    seed_hub.seed_hub(str(src), str(hub), POOL, state, games=7, force=True)
    assert not (hub / "elves" / "snapshot_99.pt").exists()
    assert state.claims == {} and state.node_decks == {} and state.seed_progress["elves"] == 7


def test_missing_source_deck_is_an_error_before_anything_is_copied(tmp_path):
    src, hub = tmp_path / "src", tmp_path / "hub"
    _make_source(src)
    (src / "elves" / "live.pt").unlink()
    with pytest.raises(SystemExit):
        seed_hub.seed_hub(str(src), str(hub), POOL, _state(tmp_path))
    assert not hub.exists()


def test_unseeded_decks_flags_weights_with_no_recorded_progress(tmp_path):
    hub = tmp_path / "hub"
    (hub / "elves").mkdir(parents=True)
    (hub / "elves" / "live.pt").write_bytes(b"x")
    state = _state(tmp_path)
    assert hub_ownership.unseeded_decks(state, str(hub), POOL) == ["elves"]
    hub_ownership.seed_hub_state(state, {"elves": 5})
    assert hub_ownership.unseeded_decks(state, str(hub), POOL) == []
