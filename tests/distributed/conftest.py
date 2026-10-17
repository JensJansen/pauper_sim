"""Shared fixtures for tests/distributed/. `running_hub` starts a real
hub_server.py Flask app on a background thread bound to a free localhost
port -- needed for node_sync.py's tests, which use the `requests` library
and therefore need genuine HTTP, not Flask's test client."""
import os
import pathlib
import socket
import threading

import pytest
from werkzeug.serving import make_server

from distributed import hub_server

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "src"


@pytest.fixture(autouse=True)
def _cwd_in_src():
    """Same convention tests/rl/conftest.py already uses: rl.roster's
    build_pool() default DECK_MANIFEST/VOCAB_PATH are relative paths
    written against cwd=src/ (see repo_paths.py/README's "Run scripts from
    src/"). node_agent.py calls build_pool() with its defaults, exactly as
    run_training_pipeline.py does, so tests exercising it need the same
    cwd."""
    prev = os.getcwd()
    os.chdir(_SRC_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ServerThread(threading.Thread):
    def __init__(self, app, port):
        super().__init__(daemon=True)
        self._server = make_server("127.0.0.1", port, app)

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


@pytest.fixture
def running_hub(tmp_path):
    """Yields (base_url, app, deck_pool, credential). Caller registers
    nodes and drives push/pull/poll against base_url via node_sync."""
    pool = ["boggles", "elves"]
    credential = "test-registration-credential"
    app = hub_server.create_app(
        league_dir=str(tmp_path / "checkpoints"),
        deck_pool=pool,
        registration_credential=credential,
        runtime_state_path=str(tmp_path / "runtime_state.json"),
    )
    port = _free_port()
    thread = _ServerThread(app, port)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", app, pool, credential
    finally:
        thread.shutdown()
        thread.join(timeout=5)
