"""Tests for drl_env._seat._lost, used by rl.training.train's reward attribution."""

from drl_env._seat import _lost


def test_lost_true_only_when_someone_else_won():
    assert _lost(type("S", (), {"winner": 1})(), 0) is True
    assert _lost(type("S", (), {"winner": 0})(), 0) is False
    assert _lost(type("S", (), {"winner": None})(), 0) is False
