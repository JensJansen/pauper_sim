"""Every committed training config is loadable, uses only keys the launchers read, and describes a league that can run (plan P9).

A key nobody reads is a typo that would be silently ignored for a multi-day run, so the allowed set is not written here: it is
read out of run_league.py and run_training_pipeline.py (every cfg / run_cfg / league_cfg key they look up)."""
import json
import re

import pytest

from config_loader import load_config
from rl.league_cli_spec import build_arg_parser
from support.paths import REPO

CONFIGS = sorted((REPO / "training_configs").glob("*.json"))
LAUNCHERS = [REPO / "src" / "run_league.py", REPO / "src" / "run_training_pipeline.py"]
KEY = re.compile(r"""\b(?:cfg|run_cfg|league_cfg)(?:\.get\(|\[)\s*["'](\w+)["']""")
DECKS = json.loads((REPO / "data" / "league_decks.json").read_text())


def _read_keys():
    keys = set()
    for path in LAUNCHERS:
        keys |= set(KEY.findall(path.read_text(encoding="utf-8")))
    return keys


def _cfg(path):
    return load_config(path)


def test_the_key_scan_finds_the_keys_the_launchers_read():
    keys = _read_keys()
    assert {"league_name", "roster", "total_games", "n_workers", "snapshot_every_games", "pfsp", "device"} <= keys


def test_there_are_configs_to_check():
    assert len(CONFIGS) >= 5


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_key_is_one_a_launcher_reads(path):
    cfg = _cfg(path)
    unknown = sorted(k for k in cfg if not k.startswith("_") and k not in _read_keys())
    assert not unknown, f"{path.name}: nothing reads {unknown} -- a typo, or a retired option"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_the_extends_chain_resolves_and_terminates(path):
    raw = json.loads(path.read_text())
    seen, cur = {path.name}, raw
    while "extends" in cur:
        name = cur["extends"]
        assert name not in seen, f"{path.name}: extends cycle through {name}"
        seen.add(name)
        assert (path.parent / name).exists(), f"{path.name}: extends {name}, which does not exist"
        cur = json.loads((path.parent / name).read_text())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_values_are_in_range(path):
    cfg = _cfg(path)
    for key in ("checkpoint_opponent_rate", "stratify_0land_pct"):
        if key in cfg:
            assert 0.0 <= cfg[key] <= 1.0, key
    for key in ("n_workers", "games_per_iteration", "snapshot_every_games", "total_games", "checks_games"):
        if cfg.get(key) is not None:
            assert isinstance(cfg[key], int) and cfg[key] >= 1, key
    if "device" in cfg:
        assert cfg["device"] == "cpu" or re.fullmatch(r"cuda(:\d+)?", cfg["device"]), cfg["device"]
    for key in ("pfsp", "overlap_collect", "create_training_league", "history_checks"):
        if key in cfg:
            assert isinstance(cfg[key], bool), key
    if "pfsp_power" in cfg:
        assert cfg["pfsp_power"] > 0
    if "checks_cadence_pct" in cfg:
        assert 0 < cfg["checks_cadence_pct"] <= 100


@pytest.mark.parametrize("path", [p for p in CONFIGS if "roster" in _cfg(p)], ids=lambda p: p.name)
def test_a_league_config_names_a_real_distinct_roster_and_a_total(path):
    cfg = _cfg(path)
    roster = cfg["roster"]
    assert roster and len(set(roster)) == len(roster), "empty or duplicated roster"
    assert set(roster) <= set(DECKS), f"unknown decks {sorted(set(roster) - set(DECKS))}"
    assert isinstance(cfg.get("league_name"), str) and cfg["league_name"]
    assert cfg.get("total_games"), "a league config needs total_games (drives auto-sizing and the stop condition)"
    for deck in roster:
        assert (REPO / "data" / DECKS[deck]).exists(), deck


def test_no_two_leagues_share_a_checkpoint_directory():
    names = {}
    for path in CONFIGS:
        cfg = _cfg(path)
        if "league_name" in cfg:
            assert cfg["league_name"] not in names, f"{path.name} and {names[cfg['league_name']]} both write {cfg['league_name']}"
            names[cfg["league_name"]] = path.name


@pytest.mark.parametrize("path", [p for p in CONFIGS if "encoder" in _cfg(p)], ids=lambda p: p.name)
def test_an_encoder_override_is_made_of_real_encoder_arguments(path):
    import inspect

    from rl.model.arch import SetTransformer
    allowed = set(inspect.signature(SetTransformer.__init__).parameters) - {"self", "vocab_size"}
    assert set(_cfg(path)["encoder"]) <= allowed
    assert all(isinstance(w, int) and w > 0 for w in _cfg(path).get("trunk_hidden", [1]))


# ---- the CLI spec ------------------------------------------------------------------------------------------------------------------

def test_the_cli_spec_parses_a_normal_invocation_and_leaves_config_backed_flags_unset():
    args = build_arg_parser().parse_args(["--run-config", "a.json", "--league-config", "b.json"])
    assert args.n_iterations is None and args.n_workers is None and args.pfsp is None
    assert args.checkpoint_opponent_rate is None and args.roster is None and args.device is None
    assert args.games == 50 and args.train_deck_only is False and args.eval is False


def test_the_cli_spec_parses_every_documented_override():
    args = build_arg_parser().parse_args([
        "--n-iterations", "3", "--snapshot-every", "2", "--n-workers", "4", "--device", "cuda", "--matchup", "elves", "boggles",
        "--games", "7", "--train-deck-only", "--eval", "--sampled", "--seed", "9", "--log", "x.json",
        "--checkpoint-opponent-rate", "0.25", "--no-pfsp", "--league-name", "l", "--roster", "elves,boggles",
        "--total-games", "100"])
    assert (args.n_iterations, args.snapshot_every, args.n_workers, args.device) == (3, 2, 4, "cuda")
    assert args.matchup == ["elves", "boggles"] and args.games == 7 and args.seed == 9
    assert args.pfsp is False and args.checkpoint_opponent_rate == 0.25 and args.total_games == 100


def test_the_cli_spec_rejects_unknown_flags_and_bad_types():
    for bad in (["--nonsense"], ["--n-workers", "many"], ["--matchup", "only_one"]):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(bad)


def test_resolve_device_accepts_cpu_and_rejects_nonsense():
    import run_league
    assert run_league._resolve_device("CPU") == "cpu"
    with pytest.raises(SystemExit, match="expected 'cpu' or 'cuda"):
        run_league._resolve_device("tpu")


def test_resolve_device_refuses_cuda_when_it_is_not_there(monkeypatch):
    import run_league
    monkeypatch.setattr(run_league.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no CUDA"):
        run_league._resolve_device("cuda:0")
    monkeypatch.setattr(run_league.torch.cuda, "is_available", lambda: True)
    assert run_league._resolve_device("cuda:1") == "cuda:1"
