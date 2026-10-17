"""Self-check for distributed.hub_ownership: claim exclusivity,
invalidation-on-reclaim, liveness exclusion, drift/stop signal logic.
Every function takes `now` explicitly, so every test here uses injected
timestamps -- no real clock, no real network, matching this repo's existing
convention for time-dependent logic (see tests/test_run_league.py)."""
import json

import pytest

from distributed import hub_ownership

POOL = ["boggles", "elves", "spy_combo"]


def _state(tmp_path):
    return hub_ownership.HubRuntimeState(str(tmp_path / "runtime_state.json"))


def test_load_deck_pool_reads_a_plain_json_array(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(POOL))
    assert hub_ownership.load_deck_pool(str(path)) == POOL


def test_claim_deck_mints_a_token_and_persists_it(tmp_path):
    state = _state(tmp_path)
    token = hub_ownership.claim_deck(state, "boggles", POOL, now=100.0)
    assert hub_ownership.verify_claim(state, "boggles", token) is True
    # persisted -- a fresh HubRuntimeState pointed at the same path sees it
    reloaded = _state(tmp_path)
    assert hub_ownership.verify_claim(reloaded, "boggles", token) is True


def test_claim_deck_rejects_a_deck_outside_the_pool(tmp_path):
    state = _state(tmp_path)
    with pytest.raises(ValueError):
        hub_ownership.claim_deck(state, "not_in_pool", POOL, now=100.0)


def test_reclaiming_a_deck_invalidates_the_previous_token(tmp_path):
    state = _state(tmp_path)
    old_token = hub_ownership.claim_deck(state, "boggles", POOL, now=100.0)
    new_token = hub_ownership.claim_deck(state, "boggles", POOL, now=200.0)
    assert old_token != new_token
    assert hub_ownership.verify_claim(state, "boggles", old_token) is False
    assert hub_ownership.verify_claim(state, "boggles", new_token) is True


def test_reclaiming_preserves_prior_progress_not_reset_to_zero(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=100.0)
    hub_ownership.record_progress(state, "boggles", 5000, now=150.0)
    hub_ownership.claim_deck(state, "boggles", POOL, now=200.0)  # a replacement node registers
    assert state.claims["boggles"]["last_cumulative_games"] == 5000


def test_verify_claim_false_for_unknown_deck(tmp_path):
    state = _state(tmp_path)
    assert hub_ownership.verify_claim(state, "boggles", "whatever") is False


def test_record_progress_raises_without_an_active_claim(tmp_path):
    state = _state(tmp_path)
    with pytest.raises(ValueError):
        hub_ownership.record_progress(state, "boggles", 100, now=100.0)


def test_active_decks_excludes_a_deck_past_the_liveness_timeout(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 100, now=0.0)
    assert hub_ownership.active_decks(state, now=50.0, liveness_timeout=100) == {"boggles"}
    assert hub_ownership.active_decks(state, now=500.0, liveness_timeout=100) == set()


def test_pick_available_deck_skips_active_claims_and_reclaims_stale_ones(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 0, now=0.0)  # still fresh at now=50
    assert hub_ownership.pick_available_deck(state, POOL, now=50.0, liveness_timeout=100) == "elves"
    # boggles goes stale
    assert hub_ownership.pick_available_deck(state, POOL, now=500.0, liveness_timeout=100) == "boggles"


def test_pick_available_deck_none_when_pool_fully_claimed(tmp_path):
    state = _state(tmp_path)
    for i, deck in enumerate(POOL):
        hub_ownership.claim_deck(state, deck, POOL, now=float(i))
        hub_ownership.record_progress(state, deck, 0, now=float(i))
    assert hub_ownership.pick_available_deck(state, POOL, now=10.0, liveness_timeout=1000) is None


def test_drift_signal_continues_when_no_other_active_deck(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 9000, now=0.0)
    assert hub_ownership.drift_signal(state, "boggles", now=1.0, liveness_timeout=100, threshold=500) == "continue"


def test_drift_signal_pauses_once_past_threshold_ahead_of_slowest(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 5000, now=1.0)
    hub_ownership.record_progress(state, "elves", 100, now=1.0)  # far behind
    assert hub_ownership.drift_signal(state, "boggles", now=1.0, liveness_timeout=100, threshold=1000) == "pause"
    assert hub_ownership.drift_signal(state, "elves", now=1.0, liveness_timeout=100, threshold=1000) == "continue"


def test_drift_signal_continues_within_threshold(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 900, now=1.0)
    hub_ownership.record_progress(state, "elves", 100, now=1.0)
    assert hub_ownership.drift_signal(state, "boggles", now=1.0, liveness_timeout=100, threshold=1000) == "continue"


def test_drift_signal_ignores_a_deck_that_has_itself_gone_stale(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 9000, now=0.0)  # goes stale after this
    # asked about long after its own last sync -- shouldn't raise or force a pause on itself
    assert hub_ownership.drift_signal(state, "boggles", now=1000.0, liveness_timeout=100, threshold=1) == "continue"


def test_dead_node_cannot_anchor_the_fleet(tmp_path):
    """The specific failure mode named in the design doc: a crashed node's
    stale progress must not become a permanent floor everyone else waits
    on."""
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 0, now=0.0)      # crashes here, never syncs again
    hub_ownership.record_progress(state, "elves", 5000, now=200.0)   # keeps progressing
    # boggles has gone stale by now=500; elves must not be gated on it
    assert hub_ownership.drift_signal(state, "elves", now=500.0, liveness_timeout=100, threshold=100) == "continue"


def test_stop_request_overrides_drift_signal(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.record_progress(state, "boggles", 0, now=0.0)
    hub_ownership.request_stop(state, "boggles")
    assert hub_ownership.sync_signal(state, "boggles", now=1.0, liveness_timeout=100, threshold=1000) == "stop"


def test_stop_all_marks_every_claimed_deck(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0)
    hub_ownership.request_stop_all(state)
    for deck in ("boggles", "elves"):
        hub_ownership.record_progress(state, deck, 0, now=1.0)
        assert hub_ownership.sync_signal(state, deck, now=1.0, liveness_timeout=100, threshold=1000) == "stop"


def test_a_fresh_claim_clears_any_prior_stop_request(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0)
    hub_ownership.request_stop(state, "boggles")
    hub_ownership.claim_deck(state, "boggles", POOL, now=500.0)  # a replacement node registers
    hub_ownership.record_progress(state, "boggles", 0, now=500.0)
    assert hub_ownership.sync_signal(state, "boggles", now=500.0, liveness_timeout=100, threshold=1000) == "continue"


# --- node_id: a restarted node gets its own deck back ---

def test_returning_node_id_gets_its_deck_back_even_while_old_claim_is_live(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0, node_id="n1")
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0, node_id="n2")
    assert hub_ownership.assign_deck(state, "n2", POOL, now=10.0, liveness_timeout=300) == "elves"
    assert hub_ownership.assign_deck(state, "n1", POOL, now=10.0, liveness_timeout=300) == "boggles"


def test_node_id_mapping_survives_hub_restart(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "elves", POOL, now=0.0, node_id="n1")
    assert hub_ownership.assign_deck(_state(tmp_path), "n1", POOL, now=1.0, liveness_timeout=300) == "elves"


def test_deck_since_given_to_another_node_is_not_returned(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0, node_id="n1")
    hub_ownership.claim_deck(state, "boggles", POOL, now=500.0, node_id="n2")  # n1 timed out, n2 took it
    assert "n1" not in state.node_decks
    # n1 returns: boggles is n2's now, so n1 gets the next free deck
    assert hub_ownership.assign_deck(state, "n1", POOL, now=501.0, liveness_timeout=300) != "boggles"


def test_unknown_or_missing_node_id_falls_back_to_first_free_deck(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, POOL[0], POOL, now=0.0)
    assert hub_ownership.assign_deck(state, "new", POOL, now=1.0, liveness_timeout=300) == POOL[1]
    assert hub_ownership.assign_deck(state, None, POOL, now=1.0, liveness_timeout=300) == POOL[1]


def test_legacy_bare_claims_state_file_still_loads(tmp_path):
    import json
    path = tmp_path / "runtime_state.json"
    path.write_text(json.dumps({"boggles": {"token": "t", "last_cumulative_games": 5, "last_sync_ts": 0.0}}))
    state = hub_ownership.HubRuntimeState(str(path))
    assert state.claims["boggles"]["last_cumulative_games"] == 5 and state.node_decks == {}


# --- metrics de-dup ---

def _acked_state(tmp_path, source="n1", count=3):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0, node_id=source)
    hub_ownership.record_metrics_ack(state, "boggles", source, count)
    return state


def test_accept_metrics_new_source_takes_everything(tmp_path):
    state = _state(tmp_path)
    hub_ownership.claim_deck(state, "boggles", POOL, now=0.0, node_id="n1")
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 0, ["a", "b"]) == (["a", "b"], 2)


def test_accept_metrics_skips_a_replayed_batch_and_partial_overlap(tmp_path):
    state = _acked_state(tmp_path, count=3)
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 0, ["a", "b", "c"]) == ([], 3)
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 1, ["b", "c", "d", "e"]) == (["d", "e"], 5)
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 3, ["d"]) == (["d"], 4)


def test_accept_metrics_gap_rebase_and_other_source_reanchor(tmp_path):
    state = _acked_state(tmp_path, count=3)
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 7, ["x"]) == (["x"], 8)          # gap
    assert hub_ownership.accept_metrics(state, "boggles", "n1", 0, ["x"], rebase=True) == (["x"], 1)
    assert hub_ownership.accept_metrics(state, "boggles", "other-disk", 0, ["x"]) == (["x"], 1)


def test_metrics_acked_is_per_source_and_survives_reclaim(tmp_path):
    state = _acked_state(tmp_path, count=3)
    assert hub_ownership.metrics_acked(state, "boggles", "n1") == 3
    assert hub_ownership.metrics_acked(state, "boggles", "someone-else") == 0
    hub_ownership.claim_deck(state, "boggles", POOL, now=5.0, node_id="n1")  # restart reclaim
    assert hub_ownership.metrics_acked(state, "boggles", "n1") == 3
