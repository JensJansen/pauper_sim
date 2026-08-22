"""Shared JSON-config loader for training scripts (run_league.py,
analysis/mulligan_retrain/train_mulligan.py).

Lives in src/, same convention as repo_paths.py: stdlib-only, so importing it
never pulls in torch, and any caller with src/ already on sys.path can
`import config_loader` directly.
"""
import json
from pathlib import Path


def load_config(path):
    """Loads one training_configs/*.json, resolving an optional top-level
    "extends": "<sibling-file>.json" as a base layer merged underneath this
    file's own keys -- so a value shared with another config (most commonly
    run-mechanics fields duplicated across leagues) is inherited instead of
    retyped, and can never silently drift out of sync the way a manual copy
    can. Resolved relative to the extending file's own directory, and
    recursively (a base may itself extend another base). A path of None (no
    config given at all) reads as {}."""
    if not path:
        return {}
    path = Path(path)
    cfg = json.load(open(path))
    base_name = cfg.pop("extends", None)
    if base_name is None:
        return cfg
    return {**load_config(path.parent / base_name), **cfg}
