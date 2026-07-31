import os
import pathlib

import pytest

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "src"


@pytest.fixture(autouse=True)
def _cwd_in_src():
    """Every migrated rl self-check assumes cwd=src/ -- that's how
    `python -m rl.X` ran them originally, and it's what their embedded
    relative paths (e.g. "../data/mono_red_madness.txt") and rl.pool's own
    DECK_MANIFEST/VOCAB_PATH defaults ("../data/league_decks.json",
    "../checkpoints/vocab.json") are written against. pytest itself doesn't
    chdir -- it runs from wherever it's invoked (this repo's own convention:
    the repo root) -- so this fixture reproduces the original cwd for the
    duration of each test in this directory, restoring it after so it never
    leaks into another test directory."""
    prev = os.getcwd()
    os.chdir(_SRC_DIR)
    try:
        yield
    finally:
        os.chdir(prev)
