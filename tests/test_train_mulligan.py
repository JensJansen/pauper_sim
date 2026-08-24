"""Self-check for train_mulligan.py's config/flag resolution -- the
flag > config > hardcoded-default precedence and the two validations
(opponent_mode required, resume_from needs a single deck) that must hold
before any network is built. Exercises _resolve_options directly, so it
never touches build_pool()/collect_rollout.

train_mulligan.py lives in src/analysis/mulligan_retrain/ (not a package --
no __init__.py, same as every other analysis/ script) and does its own
sibling import of _mulligan_common, so importing it here needs that
directory on sys.path first, same trick the script uses on itself when run
directly (its own directory is sys.path[0] in that case).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis" / "mulligan_retrain"))

import pytest

import train_mulligan


def _args(**overrides):
    """A parsed-args stand-in with every train_mulligan flag at its
    CONFIG-BACKED None sentinel, overridden by whatever the caller passes --
    built off the real parser's own defaults so it can't drift from the
    actual flag list."""
    defaults = vars(train_mulligan._build_arg_parser().parse_args([]))
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_flag_overrides_config_overrides_hardcoded_default():
    args = _args(opponent_mode="twin", train_games=500)
    opts = train_mulligan._resolve_options(args, {"train_games": 200, "update_every": 10})
    assert opts["train_games"] == 500  # flag wins over config
    assert opts["update_every"] == 10  # config wins over hardcoded default (50)
    assert opts["eval_games"] == 100  # hardcoded default -- neither flag nor config given


def test_opponent_mode_is_required():
    with pytest.raises(SystemExit):
        train_mulligan._resolve_options(_args(), {})


def test_opponent_mode_resolves_from_config_and_sets_mode_specific_defaults():
    opts = train_mulligan._resolve_options(_args(), {"opponent_mode": "self-mirror"})
    assert opts["opponent_mode"] == "self-mirror"
    assert opts["is_twin"] is False
    assert opts["train_games"] == 3000  # self-mirror's own historical default, not twin's 1000
    assert opts["bootstrap_name"] == "mulligan_bootstrap_selfmirror"


def test_twin_mode_defaults():
    opts = train_mulligan._resolve_options(_args(opponent_mode="twin"), {})
    assert opts["train_games"] == 1000
    assert opts["bootstrap_name"] == "mulligan_bootstrap"
    # Derived from --league's own default (4_deck_subleague_test), matching
    # run_training_pipeline.py's create_training_league naming -- not a
    # hardcoded twin name that can drift from what that pipeline maintains.
    assert opts["twin"] == "4_deck_subleague_test-training"


def test_twin_default_follows_an_explicit_league_override():
    opts = train_mulligan._resolve_options(_args(opponent_mode="twin", league="main-league"), {})
    assert opts["twin"] == "main-league-training"


def test_resume_from_requires_exactly_one_deck():
    with pytest.raises(SystemExit):
        train_mulligan._resolve_options(
            _args(opponent_mode="twin", resume_from="some.pt", decks=["a", "b"]), {})

    opts = train_mulligan._resolve_options(
        _args(opponent_mode="twin", resume_from="some.pt", decks=["a"]), {})
    assert opts["resume_from"] == "some.pt"


def test_config_extends_resolves_before_reaching_resolve_options(tmp_path):
    """train_mulligan.py shares config_loader.load_config with run_league.py
    -- a mulligan config can extend a base the same way a league config can."""
    from config_loader import load_config

    (tmp_path / "base.json").write_text('{"opponent_mode": "twin", "train_games": 777}')
    (tmp_path / "child.json").write_text('{"extends": "base.json", "train_games": 42}')

    cfg = load_config(str(tmp_path / "child.json"))
    opts = train_mulligan._resolve_options(_args(), cfg)
    assert opts["train_games"] == 42
