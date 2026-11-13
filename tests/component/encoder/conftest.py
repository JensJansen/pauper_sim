"""Session fixtures for the encoder suite. The corpus (real decisions from real games, see support/corpus.py) is built
once per session and cached on disk. The cache key covers ALL of src/ (except the web app), the decklists and the corpus
builder (support.cache.source_hash), so a change anywhere that could alter a recorded decision rebuilds it -- a stale
cache would let the oracle keep agreeing with old code. Knobs: ENCODER_CORPUS_GAMES (games per deck pair, default 1),
ENCODER_CORPUS_DECKS (comma list, default every deck), ENCODER_CORPUS_NOCACHE=1."""
import hashlib
import os
import pickle

import pytest

from support import corpus as ec
from support.cache import source_hash
from support.fixtures import _cwd_in_src  # noqa: F401  (these tests rely on cwd=src/)

CACHE_DIR = ec.REPO / "tests" / ".cache"


def _cache_path(games):
    key = f"{source_hash(extra=['tests/support/corpus.py'])}|{games}|{os.environ.get('ENCODER_CORPUS_DECKS', '')}"
    return CACHE_DIR / f"encoder_corpus_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pkl"


@pytest.fixture(scope="session")
def corpus():
    games = int(os.environ.get("ENCODER_CORPUS_GAMES", "1"))
    path = _cache_path(games)
    use_cache = not os.environ.get("ENCODER_CORPUS_NOCACHE")
    if use_cache and path.exists():
        frames = pickle.loads(path.read_bytes())
        prev = os.getcwd()
        os.chdir(ec.SRC)
        try:
            vocab = ec.build_pool(vocab_path=None)[1]
        finally:
            os.chdir(prev)
        return frames, vocab
    frames, vocab, _agents = ec.build_corpus(games_per_pair=games)
    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(frames))
        for stale in CACHE_DIR.glob("encoder_corpus_*.pkl"):  # keys change with every source edit; don't let old corpora pile up
            if stale != path:
                stale.unlink()
    return frames, vocab


@pytest.fixture(scope="session")
def frames(corpus):
    return corpus[0]


@pytest.fixture(scope="session")
def vocab(corpus):
    return corpus[1]
