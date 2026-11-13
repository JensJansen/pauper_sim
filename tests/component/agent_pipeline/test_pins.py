"""Golden pins for what a saved checkpoint and a metrics line must look like (plan P12).

* the parameter names and shapes of a deck network (default encoder, fixed synthetic vocabulary and action count) and of its
  mulligan network: a change here silently invalidates every saved checkpoint and every snapshot in a running league;
* the keys of a deck checkpoint and of a snapshot file;
* the field names of every metrics.jsonl kind: the readers are .get-based, so a rename would otherwise fail silently.

Regenerating `pins.json` needs the OWNER'S SIGN-OFF (same policy as the encoder and action-table pins): a diff here is a
decision to break compatibility, not a test nuisance.  AGENT_PIPELINE_REGEN=1 python -m pytest tests/component/agent_pipeline/test_pins.py"""
import json
import os
import pathlib

import torch

from rl import checkpoint as ckpt_io
from rl.league import metrics
from rl.league.league_runner import build_deck_net
from rl.model.mulligan import MulliganNet

PINS = pathlib.Path(__file__).with_name("pins.json")
VOCAB, N_ACTIONS = 50, 100  # synthetic: the real vocabulary and action table are pinned by the encoder and action-space pins


def _shapes(module):
    return {k: list(v.shape) for k, v in module.state_dict().items()}


def _current(tmp_path):
    torch.manual_seed(0)
    net = build_deck_net(VOCAB, N_ACTIONS)
    mull = MulliganNet(net.encoder)
    ckpt_io.save_deck_checkpoint(str(tmp_path / "live.pt"), net, torch.optim.Adam(net.parameters()))
    ckpt_io.save_snapshot(str(tmp_path / "snap.pt"), net, mull)
    return {
        "deck_net": _shapes(net),
        "deck_net_params": sum(p.numel() for p in net.parameters()),
        "mulligan_net": _shapes(mull),
        "mulligan_net_params": sum(p.numel() for p in mull.parameters()),
        "live_checkpoint_keys": sorted(torch.load(tmp_path / "live.pt", map_location="cpu", weights_only=True)),
        "snapshot_keys": sorted(torch.load(tmp_path / "snap.pt", map_location="cpu", weights_only=True)),
        "metrics_fields": {k: sorted(spec["fields"]) for k, spec in sorted(metrics.KINDS.items())},
    }


def test_checkpoint_architecture_and_metrics_schema_match_the_pins(tmp_path):
    current = _current(tmp_path)
    if os.environ.get("AGENT_PIPELINE_REGEN") == "1":
        PINS.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    pinned = json.loads(PINS.read_text())
    for section in current:
        assert current[section] == pinned[section], (
            f"{section} changed. If this is intended it breaks saved checkpoints/readers: get the owner's sign-off, then "
            f"regenerate with AGENT_PIPELINE_REGEN=1.")
    assert set(current) == set(pinned)
