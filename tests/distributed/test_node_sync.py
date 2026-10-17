"""Self-check for distributed.node_sync against a REAL running hub_server.py
(see conftest.running_hub) -- exercises actual HTTP via `requests`, not
Flask's test client, since that's what node_sync.py itself uses."""
import pytest
import requests

from distributed import node_sync


def test_register_returns_a_deck_token_and_cumulative_games(running_hub):
    hub_url, app, pool, credential = running_hub
    deck, token, cumulative_games, _acked = node_sync.register(hub_url, credential)
    assert deck in pool
    assert token
    assert cumulative_games == 0  # a fresh claim on an empty pool


def test_register_raises_sync_error_on_bad_credential(running_hub):
    hub_url, app, pool, credential = running_hub
    with pytest.raises(node_sync.SyncError):
        node_sync.register(hub_url, "wrong-credential")


def test_push_then_pull_round_trips_real_bytes_over_http(running_hub):
    hub_url, app, pool, credential = running_hub
    deck, token, _cumulative_games, _acked = node_sync.register(hub_url, credential)

    signal = node_sync.push(hub_url, token, deck,
                            files={"live.pt": b"real-weights-bytes"},
                            metrics_lines=['{"kind": "ppo", "iteration": 0}'],
                            cumulative_games=42)
    assert signal == "continue"

    roster = node_sync.pull(hub_url, token, deck)
    assert roster[deck]["live.pt"] == b"real-weights-bytes"


def test_push_with_a_superseded_token_raises_sync_error(running_hub):
    hub_url, app, pool, credential = running_hub
    deck, old_token, _cumulative_games, _acked = node_sync.register(hub_url, credential)
    # a second registration reclaims deck's slot only once it's
    # stale/unclaimed -- directly invalidate via the app's own state,
    # mirroring test_hub_server.py's own approach
    from distributed import hub_ownership
    state = app.config["_hub_state"]
    hub_ownership.claim_deck(state, deck, pool, now=999999999.0)

    with pytest.raises(node_sync.SyncError):
        node_sync.push(hub_url, old_token, deck, files={}, metrics_lines=[], cumulative_games=0)


def test_poll_reports_stop_after_request_stop(running_hub):
    hub_url, app, pool, credential = running_hub
    deck, token, _cumulative_games, _acked = node_sync.register(hub_url, credential)
    from distributed import hub_ownership
    hub_ownership.request_stop(app.config["_hub_state"], deck)
    assert node_sync.poll(hub_url, token, deck) == "stop"


def test_retry_gives_up_after_max_retries_against_an_unreachable_host():
    calls = []
    # port 1 -- nothing listening, a guaranteed connection failure, no real hub needed
    with pytest.raises(node_sync.SyncError):
        node_sync.register("http://127.0.0.1:1", "cred", max_retries=3, base_delay=0,
                           sleep_fn=lambda s: calls.append(s))
    assert len(calls) == 2  # sleeps between attempts 1->2 and 2->3, not after the last


def test_retry_does_not_sleep_or_retry_a_real_http_error_response(running_hub):
    hub_url, app, pool, credential = running_hub
    calls = []
    with pytest.raises(node_sync.SyncError):
        node_sync.register(hub_url, "wrong-credential", max_retries=5, base_delay=10,
                           sleep_fn=lambda s: calls.append(s))
    assert calls == []  # a 401 is a real rejection, not a transient failure -- must not retry
