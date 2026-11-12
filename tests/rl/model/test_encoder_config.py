"""The encoder is configurable (rl.model.arch.SetTransformer kwargs) and every loader reads its shape off the checkpoint."""
import torch

from rl import checkpoint as ckpt_io
from rl.league.league_runner import build_deck_net
from rl.model.arch import SetTransformer, encoder_kwargs_from_state_dict

BIG = {"d_model": 128, "n_layers": 4, "dim_feedforward": 512, "n_heads": 8}


def test_default_encoder_kwargs_are_the_defaults():
    sd = build_deck_net(20, 5).state_dict()
    assert encoder_kwargs_from_state_dict(sd) == {"d_model": 64, "n_layers": 2, "dim_feedforward": 128, "n_heads": 4}


def test_a_big_checkpoint_round_trips_through_its_own_kwargs(tmp_path):
    net = build_deck_net(20, 5, (32, 32), BIG)
    path = str(tmp_path / "live.pt")
    ckpt_io.save_deck_checkpoint(path, net)
    kw = ckpt_io.encoder_from_deck_checkpoint(path)
    assert kw == BIG
    rebuilt = build_deck_net(20, 5, ckpt_io.trunk_hidden_from_deck_checkpoint(path), kw)
    ckpt_io.load_deck_checkpoint(path, rebuilt)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), rebuilt.state_dict().values()))
    assert ckpt_io.encoder_from_deck_checkpoint(str(tmp_path / "nowhere.pt")) is None


def test_build_deck_net_without_encoder_is_the_default_setransformer():
    assert build_deck_net(20, 5).encoder.d_model == SetTransformer(20).d_model == 64
