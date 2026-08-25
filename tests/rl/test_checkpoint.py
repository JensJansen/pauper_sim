"""Tests for rl.checkpoint: centralized save/load for this repo's
checkpoint schemas.

Covers load_deck_checkpoint's optimizer-migration guard (tolerates a
checkpoint saved without optimizer state) and round trips for both
schemas."""
import os

import pytest
import torch

from rl import checkpoint as ckpt_io
from rl.model.arch import SetTransformer
from rl.model.deck import DeckNetwork
from rl.model.mulligan import MulliganNet


def _tiny_encoder():
    return SetTransformer(vocab_size=5, d_model=8, n_heads=2, n_layers=1, dim_feedforward=16)


def test_save_with_retry_recovers_from_transient_lock_then_gives_up_if_it_never_clears(monkeypatch, tmp_path):
    """_save_with_retry must ride out a transient failure (succeed once the
    lock clears) but still raise if the failure never clears."""
    monkeypatch.setattr(ckpt_io.time, "sleep", lambda seconds: None)  # don't actually wait in a test

    target = str(tmp_path / "live.pt")
    calls = []

    def flaky_twice(obj, path):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("open file failed with error code: 1224")
        # The save is atomic: torch.save writes a temp file that
        # _save_with_retry then os.replace()s into position.
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
    # Move the optimizer off its fresh-init defaults so a real load is
    # distinguishable from a skipped one.
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
def test_load_optimizer_if_present_preserves_fused_flag_across_a_legacy_resume(tmp_path):
    """This repo's on-disk checkpoints predate fused=True (see
    load_optimizer_if_present's own docstring). Resuming one of those into a
    freshly-built fused=True optimizer must come back out still fused=True --
    load_state_dict alone would silently overwrite it with the checkpoint's
    own saved (unset) flag."""
    net = torch.nn.Linear(4, 4)
    legacy_opt = torch.optim.Adam(net.parameters(), lr=1e-3)  # no fused=True, matches every currently-saved live.pt/mulligan.pt
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    legacy_opt.step()
    path = str(tmp_path / "legacy_optimizer.pt")
    ckpt_io.save_deck_checkpoint(path, net, legacy_opt)

    resumed_opt = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True)
    ckpt = torch.load(path, weights_only=True)
    ckpt_io.load_optimizer_if_present(resumed_opt, ckpt)

    assert resumed_opt.param_groups[0]["fused"] is True, (
        "the live optimizer's fused=True must survive a legacy checkpoint's load_state_dict"
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="device re-homing only diverges from a no-op on a real second device")
def test_load_optimizer_if_present_rehomes_state_tensors_to_the_live_param_device(tmp_path):
    """save_deck_checkpoint always writes optimizer state on CPU (_to_cpu),
    regardless of training device. Resuming that CPU-saved state into a
    fused=True optimizer whose params live on CUDA must re-home every state
    tensor to match -- fused Adam hard-errors on a device-mismatched state
    tensor where eager Adam would have silently tolerated it."""
    net = torch.nn.Linear(4, 4).to("cuda")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True)
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    opt.step()  # populate optimizer.state with CUDA-resident tensors
    path = str(tmp_path / "cuda_optimizer.pt")
    ckpt_io.save_deck_checkpoint(path, net, opt)  # on-disk state is always CPU, per _to_cpu

    resumed_net = torch.nn.Linear(4, 4).to("cuda")
    resumed_opt = torch.optim.Adam(resumed_net.parameters(), lr=1e-3, fused=True)
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    ckpt_io.load_optimizer_if_present(resumed_opt, ckpt)

    for param, state in resumed_opt.state.items():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device == param.device, (
                    "a CPU-saved state tensor must be re-homed to its live CUDA param, or fused Adam.step() hard-errors"
                )
    for p in resumed_net.parameters():
        p.grad = torch.ones_like(p)
    resumed_opt.step()  # would raise RuntimeError ("state_steps is on cpu...") if any state tensor were left on the wrong device


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
    """Both the deck-net and mulligan-net paths go through
    load_deck_checkpoint, so a legacy checkpoint (no "optimizer" key) must
    load cleanly on either."""
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
    """Each deck owns its encoder, so it must ride along in the checkpoint --
    a saved policy would otherwise come back paired with random perception.
    MulliganNet holds a plain reference instead: it reads its deck's
    embedding but trains via its own REINFORCE optimizer, which must not
    also step the encoder PPO owns."""
    net = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)
    mull = MulliganNet(net.encoder)

    assert any(k.startswith("encoder.") for k in net.state_dict())
    assert not any(k.startswith("encoder.") for k in mull.state_dict())
    assert mull.encoder is net.encoder


@pytest.mark.slow
def test_freezing_one_net_leaves_another_nets_encoder_alone():
    """Freezing one net's requires_grad must reach its own encoder (a
    registered child) but never another net's."""
    frozen = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)
    other = DeckNetwork(_tiny_encoder(), film_condition_dim=8, non_targeting_n_actions=4)

    for p in frozen.parameters():
        p.requires_grad = False

    assert not any(p.requires_grad for p in frozen.encoder.parameters()),         "a frozen snapshot's own encoder must be frozen with it"
    assert all(p.requires_grad for p in other.encoder.parameters()),         "freezing one net must never reach another net's encoder"


def test_a_save_killed_mid_write_leaves_the_previous_checkpoint_intact(monkeypatch, tmp_path):
    """Atomicity: torch.save streams into the destination, truncating it
    first, so a write that dies partway would otherwise destroy the
    previous good checkpoint. Writing to a temp file and os.replace()ing it
    means `path` always names either the complete old or complete new
    checkpoint."""
    monkeypatch.setattr(ckpt_io.time, "sleep", lambda seconds: None)
    target = str(tmp_path / "live.pt")

    ckpt_io._save_with_retry({"generation": 1}, target)
    assert torch.load(target, weights_only=False)["generation"] == 1

    def killed_mid_write(obj, path):
        # Half a file on disk, then the process dies mid-write.
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
