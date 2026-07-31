"""Tests for drl_env._seat: _lost, still used directly by rl.train's own
reward attribution. Migrated from drl_env/_selfcheck.py's former
`_run_self_checks()` (run via `python -m drl_env`)."""

from drl_env._seat import _lost


def test_lost_true_only_when_someone_else_won():
    # _lost: true once someone has won and it wasn't seat_idx.
    assert _lost(type("S", (), {"winner": 1})(), 0) is True
    assert _lost(type("S", (), {"winner": 0})(), 0) is False
    assert _lost(type("S", (), {"winner": None})(), 0) is False
