"""Shared pytest fixtures across the test suite."""
import pytest

import webapp_mirror as wm
import webapp_sync


@pytest.fixture
def fake_webapp(tmp_path, monkeypatch):
    """Points webapp_mirror at tmp_path plus a webapp_ready()-passing fake
    src/webapp/, so a caller's mirror_*/mirror_json calls actually land
    instead of no-op'ing on a CHECKPOINTS_DIR mismatch (see
    webapp_mirror._real_league_name). Returns the mirrored logs/validation/
    dir mirror_json/mirror_metrics_line/mirror_progress write under."""
    monkeypatch.setattr(wm, "CHECKPOINTS_DIR", tmp_path)
    webapp_dir = tmp_path / "fake_webapp"
    (webapp_dir / "logs").mkdir(parents=True)
    (webapp_dir / ".git").write_text("gitdir: whatever")
    monkeypatch.setattr(wm, "WEBAPP_DIR", webapp_dir)
    monkeypatch.setattr(wm, "WEBAPP_VALIDATION_DIR", webapp_dir / "logs" / "validation")
    return webapp_dir / "logs" / "validation"


@pytest.fixture(autouse=True)
def _never_sync_the_real_replay_repo(monkeypatch, request):
    """webapp_sync.sync_async commits and PUSHES src/webapp's data to a public
    repo. The pipeline calls it after every validation tick, and src/webapp is a
    real checked-out submodule on this machine, so no test may reach it: stub it
    for the whole suite. tests/test_webapp_sync.py drives webapp_sync.sync()
    directly against throwaway git repos instead."""
    if request.module.__name__.endswith("test_webapp_sync"):
        return  # its own tests need the real sync_async (with sync() replaced or aimed at throwaway repos)
    monkeypatch.setattr(webapp_sync, "sync_async", lambda label: None)
