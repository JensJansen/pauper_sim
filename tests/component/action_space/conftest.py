"""Fixtures for the action-space suite. Several tests here (the pointer bridge) rely on cwd=src/ (see
support.fixtures._cwd_in_src). The corpus -- real decisions from random-legal self-play over every ordered deck pair,
compared with the independent oracle at decision time -- is built once per session and cached on disk under a key that
covers ALL of src/, the decklists and the corpus/oracle code (support.cache.source_hash), so it can never go stale.
Knobs: ACTION_CORPUS_GAMES (games per ordered deck pair, default 3), ACTION_CORPUS_NOCACHE=1."""
import hashlib
import os
import pickle

import pytest

from support import action_corpus as AC
from support.cache import source_hash
from support.fixtures import _cwd_in_src  # noqa: F401
from support.paths import REPO

CACHE_DIR = REPO / "tests" / ".cache"
_KEY_FILES = ["tests/support/action_corpus.py", "tests/support/action_oracle.py"]


def _cache_path(games):
    key = f"{source_hash(extra=_KEY_FILES)}|{games}"
    return CACHE_DIR / f"action_corpus_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pkl"


@pytest.fixture(scope="session")
def action_corpus():
    games = int(os.environ.get("ACTION_CORPUS_GAMES", "3"))
    path = _cache_path(games)
    use_cache = not os.environ.get("ACTION_CORPUS_NOCACHE")
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    prev = os.getcwd()
    os.chdir(AC.SRC)
    try:
        corpus = AC.build_corpus(games_per_pair=games)
    finally:
        os.chdir(prev)
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(corpus))
        for stale in CACHE_DIR.glob("action_corpus_*.pkl"):  # one live corpus; keys change with every source edit
            if stale != path:
                stale.unlink()
    return corpus
