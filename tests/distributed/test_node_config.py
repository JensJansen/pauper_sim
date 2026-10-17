"""Pins main_league_node.json to "main_league.json + only the hardware overrides"."""
from config_loader import load_config
from repo_paths import REPO_ROOT

_NODE_OVERRIDES = {"device", "n_workers"}


def test_node_config_differs_from_main_league_only_in_hardware_fields():
    cfgs = REPO_ROOT / "training_configs"
    node = load_config(str(cfgs / "main_league_node.json"))
    main = load_config(str(cfgs / "main_league.json"))
    diff = {k for k in node.keys() | main.keys() if k != "_note" and node.get(k) != main.get(k)}
    assert diff == _NODE_OVERRIDES
    assert node["device"] == "cpu" and node["n_workers"] == 2
