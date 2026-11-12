"""Shared fixtures. A tier/component conftest opts in with:  from support.fixtures import _cwd_in_src  # noqa: F401"""
import os

import pytest

from support.paths import SRC


@pytest.fixture(autouse=True)
def _cwd_in_src():
    """Runs each test with cwd=src/. Many tests rely on it: their relative paths ("../data/...") and rl.roster's
    DECK_MANIFEST/VOCAB_PATH defaults are written against src/. pytest runs from the repo root, so this sets it for each
    test and restores it after."""
    prev = os.getcwd()
    os.chdir(SRC)
    try:
        yield
    finally:
        os.chdir(prev)
