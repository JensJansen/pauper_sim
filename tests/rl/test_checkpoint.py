"""Tests for rl.checkpoint: the centralized save/load helpers for this
repo's checkpoint schemas (see rl/checkpoint.py's own module docstring).

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


def _tiny_encoder():
    return SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)


def test_save_with_retry_recovers_from_transient_lock_then_gives_up_if_it_never_clears(monkeypatch, tmp_path):
    """Regression (2026-08): this repo lives inside a OneDrive-synced folder,
    and a real multi-hour training run hit torch.save failing mid-write with
    Windows error 1224 (ERROR_USER_MAPPED_FILE) -- OneDrive transiently
    locking the file. _save_with_retry must ride out a transient failure
    (succeed once the lock clears) but still raise if the failure never
    clears (a real problem -- disk full, bad path -- must not be silently
    swallowed forever)."""
    monkeypatch.setattr(ckpt_io.time, "sleep", lambda seconds: None)  # don't actually wait in a test

    target = str(tmp_path / "live.pt")
    calls = []

    def flaky_twice(obj, path):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("open file failed with error code: 1224")
        # The save is ATOMIC now: torch.save writes a TEMP file which
        # _save_with_retry then os.replace()s into position, so a stand-in has
        # to actually create the file it was handed or the replace has nothing
        # to move.
        open(path, "wb").close()

    monkeypatch.setattr(ckpt_io.torch, "save", flaky_twice)
    ckpt_io._save_with_retry({"x": 1}, target)
    assert len(calls) == 3  # failed twice, succeeded on the 3rd attempt
    assert os.path.exists(target)

    def always_fails(obj, path):
        raise RuntimeError("open file failed with error code: 1224")

    monkeypatch.setattr(ckpt_io.torch, "save", always_fails)
    with pytest.raises(RuntimeError):
        ckpt_io._save_with_retry({"x": 1}, target)  # never clears -- must still raise, not hang or swallow


@pytest.mark.slow
def test_load_deck_checkpoint_round_trips_net_and_optimizer(tmp_path):
    encoder = _tiny_encoder()
    net = DeckNetwork(encoder, film_condition_dim=8, non_targeting_n_actions=4)
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

    net2 = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)
    opt2 = torch.optim.Adam(net2.parameters(), lr=3e-4)
    assert ckpt_io.load_deck_checkpoint(path, net2, opt2) is True
    for (n1, p1), (n2, p2) in zip(net.state_dict().items(), net2.state_dict().items()):
        assert torch.equal(p1, p2), f"{n1} did not round-trip"
    assert opt2.state_dict()["state"]  # optimizer state actually loaded, not left fresh/empty


@pytest.mark.slow
def test_load_deck_checkpoint_missing_path_is_a_noop(tmp_path):
    encoder = _tiny_encoder()
    net = DeckNetwork(encoder, film_condition_dim=8, non_targeting_n_actions=4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    fresh_state = {k: v.clone() for k, v in net.state_dict().items()}

    assert ckpt_io.load_deck_checkpoint(str(tmp_path / "nope.pt"), net, opt) is False
    for k, v in net.state_dict().items():
        assert torch.equal(v, fresh_state[k])  # untouched


@pytest.mark.slow
@pytest.mark.parametrize("net_factory", [
    lambda enc: DeckNetwork(enc, film_condition_dim=8, non_targeting_n_actions=4),
    lambda enc: MulliganNet(enc, hidden=8),
])
def test_load_deck_checkpoint_legacy_file_without_optimizer_key_does_not_raise(tmp_path, net_factory):
    """The authorized fix: BOTH the deck-net and mulligan-net paths now go
    through the same load_deck_checkpoint, so a legacy checkpoint saved
    before the optimizer-migration guard existed (just {"net": ...}, no
    "optimizer" key) must load cleanly on either -- not just on the deck-net
    path, which is all the old hand-rolled code guarded."""
    encoder = _tiny_encoder()
    net = net_factory(encoder)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    path = str(tmp_path / "legacy.pt")
    torch.save({"net": net.state_dict()}, path)  # no "optimizer" key -- the legacy shape

    assert ckpt_io.load_deck_checkpoint(path, net, opt) is True  # must not KeyError


@pytest.mark.slow
def test_snapshot_round_trips_trunk_hidden_and_optional_mulligan(tmp_path):
    encoder = _tiny_encoder()
    net = DeckNetwork(encoder, film_condition_dim=8, non_targeting_n_actions=4)
    mnet = MulliganNet(encoder, hidden=8)

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
def test_the_encoder_is_a_registered_child_of_its_deck_net():
    """The inverse of what this file used to assert. While the perception
    stack was SHARED across every deck it was deliberately kept OUT of
    net.state_dict() (object.__setattr__), so that one instance could not be
    rewound by loading some other deck's checkpoint. Each deck now owns its
    encoder, and ownership is the point: it must ride along in the checkpoint,
    or a saved policy would come back paired with random perception.

    MulliganNet is the one that still holds a plain reference -- it reads its
    deck's embedding but is trained by its own REINFORCE optimizer, which must
    not also be stepping the encoder PPO owns."""
    net = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)
    mull = MulliganNet(net.encoder)

    assert any(k.startswith("encoder.") for k in net.state_dict())
    assert not any(k.startswith("encoder.") for k in mull.state_dict())
    assert mull.encoder is net.encoder


@pytest.mark.slow
def test_freezing_one_net_leaves_another_nets_encoder_alone():
    """rl.league.LeaguePool.load_snapshot_agent sweeps requires_grad=False over
    net.parameters(), and that sweep now DOES reach the encoder -- which is
    correct, because the encoder belongs to that frozen snapshot alone. What
    must not happen is one net's freeze reaching another's perception, the
    failure that made unfreezing impossible under the shared design."""
    frozen = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)
    other = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)

    for p in frozen.parameters():
        p.requires_grad = False

    assert not any(p.requires_grad for p in frozen.encoder.parameters()),         "a frozen snapshot's own encoder must be frozen with it"
    assert all(p.requires_grad for p in other.encoder.parameters()),         "freezing one net must never reach another net's encoder"


def test_a_save_killed_mid_write_leaves_the_previous_checkpoint_intact(monkeypatch, tmp_path):
    """The reason _save_with_retry is atomic. torch.save streams into its
    destination, truncating it first, so a write that dies partway used to
    leave a truncated file that had ALREADY destroyed the previous good
    checkpoint -- and live.pt carries the entire training history (22,848
    games/deck at the time this was written), so that is unrecoverable, not
    merely a lost session.

    Writing to a temp file and os.replace()ing it means `path` always names
    either the complete old checkpoint or the complete new one."""
    monkeypatch.setattr(ckpt_io.time, "sleep", lambda seconds: None)
    target = str(tmp_path / "live.pt")

    ckpt_io._save_with_retry({"generation": 1}, target)
    assert torch.load(target, weights_only=False)["generation"] == 1

    def killed_mid_write(obj, path):
        # Half a file on disk, then the process dies -- exactly what an OOM
        # kill or Ctrl-C during torch.save looks like.
        with open(path, "wb") as handle:
            handle.write(b"\x00" * 2048)
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(ckpt_io.torch, "save", killed_mid_write)
    with pytest.raises(RuntimeError):
        ckpt_io._save_with_retry({"generation": 2}, target)

    # The whole point: generation 1 is still there and still loadable.
    assert torch.load(target, weights_only=False)["generation"] == 1, (
        "a failed write must not damage the previous checkpoint"
    )
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers, f"a failed save must not leave a temp file behind, found {leftovers}"


def test_atomic_save_does_not_leave_a_temp_file_on_success(tmp_path):
    target = str(tmp_path / "snapshot_1.pt")
    ckpt_io._save_with_retry({"generation": 7}, target)
    assert [p.name for p in tmp_path.iterdir()] == ["snapshot_1.pt"], "os.replace must consume the temp file"
    assert torch.load(target, weights_only=False)["generation"] == 7
