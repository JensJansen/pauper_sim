"""Tests for rl.checkpoint: the centralized save/load helpers for this
repo's ~4 checkpoint schemas (see rl/checkpoint.py's own module docstring).

Focused on the one behavior change this module was explicitly authorized to
make while centralizing: load_deck_checkpoint applies the SAME
optimizer-migration guard to every caller (previously only run_league.py's
live.pt path had it; mulligan.pt would KeyError on a legacy file with no
"optimizer" key) -- plus a round trip per schema so the on-disk shape stays
provably unchanged.
"""
import os

import pytest
import torch

from rl import checkpoint as ckpt_io
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.mulligan import MulliganNet


def _tiny_shared():
    return SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)


def test_save_with_retry_recovers_from_transient_lock_then_gives_up_if_it_never_clears(monkeypatch):
    """Regression (2026-08): this repo lives inside a OneDrive-synced folder,
    and a real multi-hour training run hit torch.save failing mid-write with
    Windows error 1224 (ERROR_USER_MAPPED_FILE) -- OneDrive transiently
    locking the file. _save_with_retry must ride out a transient failure
    (succeed once the lock clears) but still raise if the failure never
    clears (a real problem -- disk full, bad path -- must not be silently
    swallowed forever)."""
    monkeypatch.setattr(ckpt_io.time, "sleep", lambda seconds: None)  # don't actually wait in a test

    calls = []

    def flaky_twice(obj, path):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("open file failed with error code: 1224")

    monkeypatch.setattr(ckpt_io.torch, "save", flaky_twice)
    ckpt_io._save_with_retry({"x": 1}, "irrelevant.pt")
    assert len(calls) == 3  # failed twice, succeeded on the 3rd attempt

    def always_fails(obj, path):
        raise RuntimeError("open file failed with error code: 1224")

    monkeypatch.setattr(ckpt_io.torch, "save", always_fails)
    with pytest.raises(RuntimeError):
        ckpt_io._save_with_retry({"x": 1}, "irrelevant.pt")  # never clears -- must still raise, not hang or swallow


@pytest.mark.slow
def test_load_deck_checkpoint_round_trips_net_and_optimizer(tmp_path):
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    # Move the optimizer's state away from its fresh-init defaults so a load that
    # actually restored it is distinguishable from one that silently skipped it.
    for p in net.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    opt.step()

    path = str(tmp_path / "deck" / "live.pt")
    ckpt_io.save_deck_checkpoint(path, net, opt)
    saved_raw = torch.load(path, weights_only=True)
    assert set(saved_raw) == {"net", "optimizer"}  # exact on-disk schema, unchanged

    net2 = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    opt2 = torch.optim.Adam(net2.parameters(), lr=3e-4)
    assert ckpt_io.load_deck_checkpoint(path, net2, opt2) is True
    for (n1, p1), (n2, p2) in zip(net.state_dict().items(), net2.state_dict().items()):
        assert torch.equal(p1, p2), f"{n1} did not round-trip"
    assert opt2.state_dict()["state"]  # optimizer state actually loaded, not left fresh/empty


@pytest.mark.slow
def test_load_deck_checkpoint_missing_path_is_a_noop(tmp_path):
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    fresh_state = {k: v.clone() for k, v in net.state_dict().items()}

    assert ckpt_io.load_deck_checkpoint(str(tmp_path / "nope.pt"), net, opt) is False
    for k, v in net.state_dict().items():
        assert torch.equal(v, fresh_state[k])  # untouched


@pytest.mark.slow
@pytest.mark.parametrize("net_factory", [
    lambda shared: DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4),
    lambda shared: MulliganNet(shared, hidden=8),
])
def test_load_deck_checkpoint_legacy_file_without_optimizer_key_does_not_raise(tmp_path, net_factory):
    """The authorized fix: BOTH the deck-net and mulligan-net paths now go
    through the same load_deck_checkpoint, so a legacy checkpoint saved
    before the optimizer-migration guard existed (just {"net": ...}, no
    "optimizer" key) must load cleanly on either -- not just on the deck-net
    path, which is all the old hand-rolled code guarded."""
    shared = _tiny_shared()
    net = net_factory(shared)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    path = str(tmp_path / "legacy.pt")
    torch.save({"net": net.state_dict()}, path)  # no "optimizer" key -- the legacy shape

    assert ckpt_io.load_deck_checkpoint(path, net, opt) is True  # must not KeyError


@pytest.mark.slow
def test_snapshot_round_trips_trunk_hidden_and_optional_mulligan(tmp_path):
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    mnet = MulliganNet(shared, hidden=8)

    path = str(tmp_path / "snapshot_0.pt")
    ckpt_io.save_snapshot(path, net, mnet)
    saved = ckpt_io.load_snapshot(path)
    assert set(saved) == {"state_dict", "trunk_hidden", "mulligan_state_dict", "mulligan_hidden"}
    assert saved["trunk_hidden"] == tuple(layer.out_features for layer in net.trunk_layers)
    assert saved["mulligan_hidden"] == 8

    deck_only_path = str(tmp_path / "snapshot_1.pt")
    ckpt_io.save_snapshot(deck_only_path, net)  # mulligan_net=None
    saved_deck_only = ckpt_io.load_snapshot(deck_only_path)
    assert set(saved_deck_only) == {"state_dict", "trunk_hidden"}


@pytest.mark.slow
def test_frozen_stack_round_trip(tmp_path):
    shared = _tiny_shared()
    path = str(tmp_path / "shared_stack_frozen.pt")
    ckpt_io.save_frozen_stack(path, shared, vocab_size=5, d_model=8)
    saved = ckpt_io.load_frozen_stack(path)
    assert saved["vocab_size"] == 5 and saved["d_model"] == 8
    assert set(saved) == {"shared", "vocab_size", "d_model"}


@pytest.mark.slow
def test_pretrain_checkpoint_round_trip_and_missing_returns_none(tmp_path):
    shared = _tiny_shared()
    opt_shared = torch.optim.Adam(shared.parameters(), lr=3e-4)
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    head_opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith("shared_stack.")], lr=3e-4)

    path = str(tmp_path / "pretrain_shared_stack.pt")
    assert ckpt_io.load_pretrain_checkpoint(path) is None  # nothing on disk yet

    ckpt_io.save_pretrain_checkpoint(path, shared, opt_shared, {"deck_a": net}, {"deck_a": head_opt},
                                      session=3, vocab_size=5, d_model=8)
    saved = ckpt_io.load_pretrain_checkpoint(path)
    assert set(saved) == {"shared", "opt_shared", "nets", "head_opts", "session", "vocab_size", "d_model"}
    assert saved["session"] == 3 and set(saved["nets"]) == {"deck_a"}
