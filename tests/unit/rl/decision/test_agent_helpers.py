"""The small pure helpers of rl.decision.agent (plan P7): the padded full mask, the Pass test, the previous-pointer tensors, and
the loud all-False failure. The decision flow itself is test_agent.py."""
import numpy as np
import pytest
import torch

from rl.decision import agent as A
from support.scenario import build, seat


def test_padded_full_mask_keeps_exactly_the_legal_positions_and_pads_false():
    raw = np.array([True, False, True, False, True, True, False])  # n_fixed 4, three pointer tokens
    out = A._padded_full_mask(raw, 4, 6, "cpu")
    assert out.dtype == torch.bool and out.shape == (4 + 6,)
    assert out.tolist() == [True, False, True, False, True, True, False, False, False, False]
    assert out.nonzero().flatten().tolist() == np.flatnonzero(raw).tolist(), "padding must not add or move a legal slot"


def test_padded_full_mask_with_no_pointer_tokens_and_with_no_padding():
    assert A._padded_full_mask(np.array([False, True]), 2, 1, "cpu").tolist() == [False, True, False]
    assert A._padded_full_mask(np.array([True, True, False]), 2, 1, "cpu").tolist() == [True, True, False]


def test_padded_full_mask_matches_the_convention_ppo_update_rebuilds_for_a_minibatch():
    """Same rule in two places (rollout and update): the stored raw mask widened with False. Any disagreement would make the
    update's masked distribution differ from the one the action was sampled from."""
    raw = np.array([True, True, False, True, False, True])
    n_fixed, padded = 3, 5
    rebuilt = np.zeros(n_fixed + padded, dtype=bool)  # ppo_update: mask_np[i, :n_fixed] = stored[:n_fixed]; pointer part after
    rebuilt[:n_fixed] = raw[:n_fixed]
    rebuilt[n_fixed:n_fixed + len(raw[n_fixed:])] = raw[n_fixed:]
    assert A._padded_full_mask(raw, n_fixed, padded, "cpu").numpy().tolist() == rebuilt.tolist()


TABLE = [("Cast Lightning Bolt", None, None), ("Pass", None, None), ("Tap Mountain for R", None, None)]


def test_is_pass_is_true_only_for_the_pass_row():
    assert [A._is_pass(i, TABLE) for i in range(3)] == [False, True, False]
    assert A._is_pass(3, TABLE) is False, "a pointer index (past the fixed table) is never Pass"
    assert A._is_pass(10_000, TABLE) is False


def test_prev_pointer_tensors():
    assert A._prev_pointer_tensors(None, "cpu") is None
    vocab_idx, row = A._prev_pointer_tensors((7, [0.5, 1.0, 2.0]), "cpu")
    assert vocab_idx.dtype == torch.long and vocab_idx.tolist() == [7]
    assert row.dtype == torch.float32 and row.shape == (1, 3) and row.tolist() == [[0.5, 1.0, 2.0]]


def test_all_false_failure_is_a_loud_runtime_error_that_names_the_position():
    state = build(seat(bf=["Mountain", ("Goblin Bushwhacker", {"sick": True})], hand=["Lightning Bolt"]), seat())
    state.pending_resolution = {"kind": "pay_cost", "on_complete": lambda s: None, "remaining": {"R": 1}}
    with pytest.raises(RuntimeError) as exc:
        A._raise_all_false(state, 0)
    text = str(exc.value)
    assert "all-False action mask" in text and "'pay_cost'" in text
    assert "Lightning Bolt" in text and "Mountain" in text, "the dump must name the hand and the battlefield"
    assert "on_complete" not in text, "unprintable callables are dropped from the pending dump"
    assert "mana_ability_options" in text and "available_mana_units" in text


def test_all_false_failure_with_nothing_pending_still_raises_cleanly():
    state = build(seat(), seat())
    state.pending_resolution = None
    with pytest.raises(RuntimeError, match="None"):
        A._raise_all_false(state, 0)
