import os
import pathlib

import pytest

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "src"


@pytest.fixture(autouse=True)
def _cwd_in_src():
    """Every rl test in this directory relies on cwd=src/ (their relative
    paths and rl.roster's DECK_MANIFEST/VOCAB_PATH defaults are written
    against it). pytest runs from the repo root, so this sets cwd=src/ for
    each test here and restores it after."""
    prev = os.getcwd()
    os.chdir(_SRC_DIR)
    try:
        yield
    finally:
        os.chdir(prev)
