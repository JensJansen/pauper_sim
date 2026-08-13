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


# --- shared stack is a plain reference, not a registered child (2026-08-13) ---
#
# DeckNetwork/MulliganNet used to do `self.shared_stack = shared_stack`, which
# nn.Module.__setattr__ registers -- so every checkpoint embedded a full copy of
# the one shared perception stack, and three call sites could silently mutate it
# through a net that was only supposed to be borrowing it. Harmless only while
# the stack stayed frozen and every copy was byte-identical. These pin both the
# new invariant and backward compatibility with the ~800 files already on disk.


def _old_format_state_dict(net, shared):
    """What net.state_dict() USED to return: the net's own tensors plus an
    embedded `shared_stack.*` copy. Reconstructed rather than mocked so the
    backward-compat tests run against the real historical shape."""
    return {**net.state_dict(),
            **{f"shared_stack.{k}": v for k, v in shared.state_dict().items()}}


@pytest.mark.slow
def test_shared_stack_is_not_registered_as_a_child_module():
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    mull = MulliganNet(shared)

    assert not any(k.startswith("shared_stack.") for k in net.state_dict())
    assert not any(k.startswith("shared_stack.") for k in mull.state_dict())
    # ...but still reachable by attribute: rl.ppo's cache_shared check and
    # rl.deck.forward's own d_model read both go through net.shared_stack.
    assert net.shared_stack is shared and mull.shared_stack is shared


@pytest.mark.slow
def test_legacy_checkpoint_with_embedded_stack_still_loads(tmp_path):
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    path = str(tmp_path / "live.pt")
    torch.save({"net": _old_format_state_dict(net, shared)}, path)  # pre-2026-08-13 shape

    fresh = DeckNetwork(_tiny_shared(), film_condition_dim=8, non_targeting_n_actions=4)
    assert ckpt_io.load_deck_checkpoint(path, fresh)  # strict load would reject the extra keys
    assert torch.equal(fresh.critic_head.weight, net.critic_head.weight)


@pytest.mark.slow
def test_loading_one_decks_checkpoint_leaves_the_shared_stack_bit_identical(tmp_path):
    """The actual landmine: deck B's checkpoint carried deck B's era of the
    stack, so loading it rewound the ONE shared instance that deck A was also
    using. Byte-identical copies made it invisible -- until the stack is
    trainable, when it silently corrupts."""
    shared = _tiny_shared()
    before = {k: v.clone() for k, v in shared.state_dict().items()}

    other_stack = _tiny_shared()  # a DIFFERENT stack, as a divergent-era copy would be
    with torch.no_grad():
        for p in other_stack.parameters():
            p.add_(1.0)
    deck_b = DeckNetwork(other_stack, film_condition_dim=8, non_targeting_n_actions=4)
    path = str(tmp_path / "deck_b.pt")
    torch.save({"net": _old_format_state_dict(deck_b, other_stack)}, path)

    deck_a = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    ckpt_io.load_deck_checkpoint(path, deck_a)

    for k, v in shared.state_dict().items():
        assert torch.equal(v, before[k]), f"load rewound the shared stack at {k}"


@pytest.mark.slow
def test_load_snapshot_strips_both_weight_entries(tmp_path):
    shared = _tiny_shared()
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)
    mull = MulliganNet(shared)
    path = str(tmp_path / "snapshot_0.pt")
    torch.save({"state_dict": _old_format_state_dict(net, shared),
                "trunk_hidden": (128, 128),
                "mulligan_state_dict": _old_format_state_dict(mull, shared),
                "mulligan_hidden": 64}, path)

    saved = ckpt_io.load_snapshot(path)
    assert not any(k.startswith("shared_stack.") for k in saved["state_dict"])
    assert not any(k.startswith("shared_stack.") for k in saved["mulligan_state_dict"])
    assert saved["trunk_hidden"] == (128, 128) and saved["mulligan_hidden"] == 64


@pytest.mark.slow
def test_freezing_a_snapshots_parameters_no_longer_freezes_the_shared_stack():
    """rl.league.LeaguePool.load_snapshot_agent sweeps requires_grad=False over
    net.parameters(). While the stack was a child that sweep reached it and made
    the freeze PERMANENT for every other user of the same instance -- the most
    severe of the three call sites, and a hard blocker on ever unfreezing."""
    shared = _tiny_shared()
    for p in shared.parameters():
        p.requires_grad = True
    net = DeckNetwork(shared, film_condition_dim=8, non_targeting_n_actions=4)

    for p in net.parameters():
        p.requires_grad = False

    assert all(p.requires_grad for p in shared.parameters())
