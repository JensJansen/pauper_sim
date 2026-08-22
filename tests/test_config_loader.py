"""Self-check for config_loader.load_config -- the shared "extends" resolver
run_league.py and analysis/mulligan_retrain/train_mulligan.py both use.
Stdlib-only, like the module itself, so this needs no torch import.
"""
import json

from config_loader import load_config


def test_extends_merges_base_under_own_keys(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"a": 1, "b": 2}))
    (tmp_path / "child.json").write_text(json.dumps({"extends": "base.json", "b": 3, "c": 4}))

    assert load_config(str(tmp_path / "child.json")) == {"a": 1, "b": 3, "c": 4}


def test_extends_resolves_relative_to_the_extending_files_own_directory(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps({"a": 1}))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "child.json").write_text(json.dumps({"extends": "../base.json", "b": 2}))

    assert load_config(str(sub / "child.json")) == {"a": 1, "b": 2}


def test_extends_chains_recursively(tmp_path):
    (tmp_path / "grandparent.json").write_text(json.dumps({"a": 1, "b": 1}))
    (tmp_path / "parent.json").write_text(json.dumps({"extends": "grandparent.json", "b": 2}))
    (tmp_path / "child.json").write_text(json.dumps({"extends": "parent.json", "c": 3}))

    assert load_config(str(tmp_path / "child.json")) == {"a": 1, "b": 2, "c": 3}


def test_no_path_reads_as_empty_dict():
    assert load_config(None) == {}
    assert load_config("") == {}


def test_no_extends_key_returns_the_file_as_is(tmp_path):
    (tmp_path / "plain.json").write_text(json.dumps({"x": 1}))
    assert load_config(str(tmp_path / "plain.json")) == {"x": 1}
