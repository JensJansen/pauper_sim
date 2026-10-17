"""HTTP client for a node talking to hub_server.py: register, push, pull,
poll. Uses `requests` (distributed_configs/requirements-cpu.txt only, not
this machine's own requirements.txt -- see the implementation plan's
decision #5) for multipart file uploads on push (decision #6) -- base64 in
JSON for pull's response, matching hub_server.py's own choice there.

Retry/backoff (DISTRIBUTED_TRAINING_PLAN.md gap #5) lives here, once, so
every caller gets it for free rather than each call site re-implementing
it: a transient connection failure (network hiccup, DNS blip) retries with
exponential backoff; a real HTTP error response (401/403/409 -- a
deliberate rejection, not a network problem) is NOT retried, since retrying
a bad-credential or superseded-token rejection can't ever succeed by trying
again.
"""
import base64
import json
import time

import requests

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0


class SyncError(Exception):
    """Raised for both a real HTTP error response (never retried) and
    retry exhaustion after repeated connection failures (retried, then
    given up). `status_code` is the HTTP status for the former, None for
    the latter."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _with_retry(fn, max_retries, base_delay, sleep_fn):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_fn(base_delay * (2 ** attempt))
    raise SyncError(f"request failed after {max_retries} attempts: {last_exc}") from last_exc


def _raise_for_error(resp):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error")
        except ValueError:
            detail = resp.text
        raise SyncError(f"{resp.status_code}: {detail}", status_code=resp.status_code)


def register(hub_url, credential, reported_sha=None, reported_vocab_hash=None, node_id=None,
             reported_dirty=False,
             max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """Returns (deck, token, cumulative_games, metrics_acked) --
    metrics_acked is how many lines of THIS node ID's local metrics.jsonl the
    hub already holds (0 for a new node/disk), so the node resumes pushing
    from the right line instead of re-sending everything. cumulative_games is the
    hub's last-known progress for this deck (0 for a genuinely fresh
    claim), which the caller needs to correctly seed its own local
    progress.json on first bootstrap rather than starting the hyperparameter
    schedule/snapshot cadence from 0 for a deck that already has real
    history. `node_id` is the node's persistent ID (node_agent's
    load_or_create_node_id): a known ID gets its previous deck back."""
    def do():
        return requests.post(f"{hub_url}/register", json={
            "credential": credential, "node_id": node_id, "reported_sha": reported_sha,
            "reported_dirty": reported_dirty, "reported_vocab_hash": reported_vocab_hash,
        }, timeout=30)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)
    body = resp.json()
    return body["deck"], body["token"], body["cumulative_games"], body["metrics_acked"]


def push(hub_url, token, deck, files, metrics_lines, cumulative_games,
         node_id=None, metrics_start_line=None, metrics_rebase=False, snapshot_names=None,
         max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """files: dict[filename, bytes]. metrics_lines: list of already-
    formatted JSON text lines. Returns the signal string ("continue" /
    "pause" / "stop"). node_id + metrics_start_line (index, in this node's
    local metrics.jsonl, of metrics_lines[0]) let the hub drop lines it
    already has; metrics_rebase=True tells it the local file no longer lines
    up with the hub's count, so it should re-anchor instead of skipping.
    snapshot_names: every snapshot_*.pt filename this node currently holds for
    its deck; the hub deletes the rest of that deck's snapshots (evictions).
    None = don't prune."""
    data = {"token": token, "deck": deck, "cumulative_games": str(cumulative_games),
            "metrics_lines": "\n".join(metrics_lines)}
    if snapshot_names is not None:
        data["snapshot_names"] = json.dumps(sorted(snapshot_names))
    if node_id is not None and metrics_start_line is not None:
        data.update(node_id=node_id, metrics_start_line=str(metrics_start_line),
                    metrics_rebase="1" if metrics_rebase else "0")
    file_payload = {name: (name, content) for name, content in files.items()}

    def do():
        return requests.post(f"{hub_url}/push", data=data, files=file_payload, timeout=120)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)
    return resp.json()["signal"]


def pull(hub_url, token, deck,
         max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """Returns dict[deck_name, dict[filename, bytes]] for the whole
    roster."""
    def do():
        return requests.get(f"{hub_url}/pull", params={"token": token, "deck": deck}, timeout=120)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)
    encoded = resp.json()["roster"]
    return {d: {fn: base64.b64decode(content) for fn, content in files.items()}
            for d, files in encoded.items()}


def poll(hub_url, token, deck,
         max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """Lightweight signal-only check for a paused node. Returns the signal
    string."""
    def do():
        return requests.get(f"{hub_url}/poll", params={"token": token, "deck": deck}, timeout=30)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)
    return resp.json()["signal"]


def release(hub_url, token, deck,
            max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """Tells the hub this node is exiting gracefully, so the deck is free
    immediately instead of after the liveness timeout (and any stop request
    is cleared)."""
    def do():
        return requests.post(f"{hub_url}/release", json={"token": token, "deck": deck}, timeout=30)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)


def fetch_vocab(hub_url, credential,
                max_retries=DEFAULT_MAX_RETRIES, base_delay=DEFAULT_BASE_DELAY_SECONDS, sleep_fn=time.sleep):
    """The hub's vocab.json (card name -> index) as a dict. Needs only the
    registration credential -- a node fetches it BEFORE registering/building."""
    def do():
        return requests.get(f"{hub_url}/vocab", headers={"X-Registration-Credential": credential}, timeout=30)
    resp = _with_retry(do, max_retries, base_delay, sleep_fn)
    _raise_for_error(resp)
    return resp.json()
